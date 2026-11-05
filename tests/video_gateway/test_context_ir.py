"""Compiler admission/resource races exercised against actual etcd txn shapes."""
import asyncio
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace
import pytest
from dingo.video_gateway.context_ir.config import BackendConfig, ContextIRConfig, parse_context_ir
from dingo.video_gateway.context_ir.models import ContextIRTask
from dingo.video_gateway.context_ir.task_store import ContextIRStore
from dingo.video_gateway.context_ir.protocol import validate_result
from dingo.video_gateway.context_ir.client import UpstreamRejected
from dingo.video_gateway.context_ir.validation import normalize
from dingo.video_gateway.task_runtime.transactions import encode
from dingo.video_gateway.models import now_ms
from dingo.video_gateway.errors import GatewayError
from tests.video_gateway.test_etcd_task_store import FakeEtcd


async def stack(max_inflight=2):
    c = FakeEtcd()
    cfg = ContextIRConfig(enabled=True, backends={"h3": BackendConfig(url="http://compiler", max_inflight=max_inflight)}, model_bindings={"model": "h3"})
    s = ContextIRStore(c, "test", cfg)
    assert await s.ensure_backend("h3", cfg.backends["h3"])
    for owner in ("a", "b"):
        await c.txn([], [c.put(s.key(f"owners/{owner}"), owner)])
    return c, s


async def task(s, name, *, key=None, digest="digest"):
    await s.reserve_upload(name, name, "a", 100)
    t = ContextIRTask(id=name, deployment_id="test", principal_hash="p", model="model", target_pool_id="pool",
                     backend_id="h3", backend_epoch="initial", route_revision="r", request_digest=digest,
                     idempotency_hash=key, request_path="request", input_manifest_path="manifest", route_snapshot_path="route",
                     reservation_id=name, created_at_ms=now_ms(), queue_deadline_ms=now_ms()+600000,
                     input_bytes=1, reserved_bytes=100)
    return await s.create(t)


async def test_compiler_admission_idempotency_conflict():
    c, s = await stack()
    first, created = await task(s, "one", key="k")
    second, created2 = await task(s, "two", key="k")
    assert created and not created2 and first.task.id == second.task.id
    with pytest.raises(GatewayError) as e:
        await task(s, "three", key="k", digest="changed")
    assert e.value.status == 409
    assert len(await s.records("tasks/")) == 1


async def test_compiler_global_slots_across_owners_and_cancelled_hold():
    c, s = await stack()
    for i in range(4):
        await task(s, str(i))
    claimed = await asyncio.gather(s.claim("h3", "a", "a"), s.claim("h3", "b", "b"))
    claimed = [x for x in claimed if x]
    while len(claimed) < 2:
        claimed.append(await s.claim("h3", "b", "b"))
    assert len({x.task.id for x in claimed}) == 2
    assert await s.claim("h3", "a", "a") is None
    cancelled = await s.finish(claimed[0], "cancelled")
    assert cancelled.task.status == "cancelled"
    assert await s.claim("h3", "a", "a") is None
    assert await s.release_slot(claimed[0].task)
    assert not await s.release_slot(claimed[0].task)
    assert await s.claim("h3", "a", "a") is not None
    assert int((await c.get(s.key("counters/h3/inflight"))).value) == 2


async def test_compiler_unknown_keeps_slot_and_quarantines_only_backend():
    c, s = await stack()
    await task(s, "one")
    claim = await s.claim("h3", "a", "a")
    assert await s.quarantine(claim.task)
    await s.finish(claim, "failed")
    assert len(await s.records("backends/h3/slots/")) == 1
    with pytest.raises(GatewayError) as e:
        await s.claim("h3", "b", "b")
    assert e.value.code == "context_ir_backend_quarantined"


async def test_old_owner_cannot_update_or_release_after_adoption():
    c, s = await stack()
    await task(s, "one")
    old = await s.claim("h3", "a", "a")
    await c.txn([], [c.delete(s.key("owners/a"))])
    new = await s.adopt(old, "b", "b")
    assert new.task.owner_id == "b"
    assert await s.update(old, phase="compiling") is None
    assert not await s.release_slot(old.task)
    assert await s.update(new, phase="compiling")


async def test_compiler_cancel_wins_late_completion():
    c, s = await stack()
    await task(s, "one")
    running = await s.claim("h3", "a", "a")
    cancelled = await s.finish(running, "cancelled")
    assert await s.finish(running, "completed") is None
    assert (await s.get_task("one")).task.status == "cancelled"


async def test_compiler_queue_timeout_and_reservation_accounting():
    c, s = await stack()
    stored, _ = await task(s, "one")
    expired = replace(stored.task, queue_deadline_ms=0)
    await c.txn([], [c.put(s.task_key("one"), encode(expired.to_dict()))])
    assert await s.claim("h3", "a", "a") is None
    assert (await s.get_task("one")).task.error["code"] == "context_ir_queue_timeout"
    await s.release_reservation("one")
    await s.release_reservation("one")
    assert int((await c.get(s.key("counters/reserved_bytes"))).value) == 0
    assert int((await c.get(s.key("counters/h3/reserved_bytes"))).value) == 0


def test_compiler_result_rejects_wrong_canvas_fallback_and_labels():
    normalized = {"mode": "ref2va", "frames": 124, "canvas": [1344, 768],
                  "references": [], "generation_parameters": {"seconds":5, "size":"1344x768"}}
    response = {"status":"ready", "source":"written", "ir":{"mode":"ref2va", "prompt":"hello", "target":{"frames":124,"canvas":[1344,768]},"manifest":[]}}
    assert validate_result(normalized, [], 201, response)["generation_parameters"]["seconds"] == 5
    for bad in ({**response,"status":"degraded"}, {**response,"source":"draft"},
                {**response,"ir":{**response['ir'],"target":{"frames":120,"canvas":[1344,768]}}}):
        with pytest.raises(UpstreamRejected):
            validate_result(normalized, [], 201, bad)


def test_compiler_configuration_rejects_unregistered_backend(make_gateway_config):
    base = make_gateway_config()
    with pytest.raises(ValueError):
        parse_context_ir({"enabled":True,"backends":{"b":{"url":"http://x","adapter":"bb9_context_ir"}},"model_bindings":{"public-fl":"b"}},base.pools_by_model)
    with pytest.raises(ValueError):
        parse_context_ir({"enabled":True,"backends":{"a":{"url":"http://x"},"b":{"url":"http://x/"}},"model_bindings":{"public-fl":"a"}},base.pools_by_model)


def test_context_ir_reuses_video_normalization_and_preserves_duration(make_gateway_config):
    from dingo.video_gateway.adapters.minimax_h3 import MiniMaxH3VideoAdapter
    cfg = make_gateway_config()
    cfg = replace(cfg, context_ir=ContextIRConfig(enabled=True,backends={"h3":BackendConfig(url="http://x")},model_bindings={"public-fl":"h3"}))
    pool = cfg.pools[0]
    adapter = MiniMaxH3VideoAdapter(pool)
    service = SimpleNamespace(config=cfg,resolve_pool=lambda model:pool,adapters={pool.pool_id:adapter})
    fields={"model":["public-fl"],"prompt":["A bird flying."],"seconds":["15"],"size":["1344x768"]}
    result=normalize(service,fields,[])
    assert result['frames']==362
    assert result['generation_parameters']=={"seconds":15.0,"size":"1344x768","fps":24}
    for changed in ({**fields,"seed":["4"]},{**fields,"silent":["invalid"]},{**fields,"reference_options":['[{"index":0,"role":"subject"}]']}):
        with pytest.raises(GatewayError):normalize(service,changed,[])


@pytest.mark.parametrize('role', [None, 'edit_source', 'motion_reference'])
def test_video_role_omission_survives_normalization_and_transport(tmp_path, monkeypatch, role):
    from dingo.video_gateway.context_ir import validation
    from dingo.video_gateway.context_ir.protocol import prepare
    root=tmp_path/'task';path=root/'inputs'/'video.mp4';path.parent.mkdir(parents=True);path.write_bytes(b'fixture')
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    upload=SimpleNamespace(path=path,ordinal=0,manifest_entry=lambda **kw:{'path':'inputs/video.mp4','ordinal':0,'sha256':sha,'bytes':7})
    pool=SimpleNamespace(pool_id='p',adapter=SimpleNamespace(workflow='ref2va'))
    adapter=SimpleNamespace(normalize_request=lambda *a:{'width':1344,'height':768,'num_frames':360,'seconds':15.0,'reference_metadata':[{'kind':'video'}]})
    service=SimpleNamespace(resolve_pool=lambda m:pool,adapters={'p':adapter},config=SimpleNamespace(context_ir=SimpleNamespace(model_bindings={'m':'b'})))
    monkeypatch.setattr(validation,'audio_present',lambda p:False)
    fields={'model':['m'],'prompt':['<Video 1>只给甲转头的动作方向。'],'seconds':['15']}
    if role is not None:fields['reference_options']=[json.dumps([{'index':0,'role':role}])]
    n=normalize(service,fields,[upload]);assert n['references'][0]['role_stated']==(role is not None)
    payload,files,expected=prepare(n,root,root/'attempts/0001',100)
    if role is None:assert 'role' not in payload['assets'][0]
    else:assert payload['assets'][0]['role']==role
    # The compiler may resolve omitted video roles; explicit roles must remain exact.
    resolved='motion_reference' if role is None else role
    response={'status':'ready','source':'written','ir':{'mode':'ref2va','target':{'frames':362,'canvas':[1344,768]},'prompt':'valid prompt','manifest':[{'label':'<Video 1>','sha256':sha,'role':resolved}]}}
    assert validate_result(n,expected,200,response)['references'][0]['role']==resolved
    response['ir']['manifest'][0]['role']='subject'
    with pytest.raises(UpstreamRejected):validate_result(n,expected,200,response)
    if role is not None:
        response['ir']['manifest'][0]['role']='edit_source' if role=='motion_reference' else 'motion_reference'
        with pytest.raises(UpstreamRejected):validate_result(n,expected,200,response)
    for invalid in [None,[],{},'continuation_source']:
        fields['reference_options']=[json.dumps([{'index':0,'role':invalid}])]
        with pytest.raises(GatewayError):normalize(service,fields,[upload])


def test_context_ir_silent_allows_explicit_dialogue(make_gateway_config):
    from dingo.video_gateway.adapters.minimax_h3 import MiniMaxH3VideoAdapter
    cfg=make_gateway_config();cfg=replace(cfg,context_ir=ContextIRConfig(enabled=True,backends={'h':BackendConfig(url='http://x')},model_bindings={'public-fl':'h'}))
    pool=cfg.pools[0];service=SimpleNamespace(config=cfg,resolve_pool=lambda m:pool,adapters={pool.pool_id:MiniMaxH3VideoAdapter(pool)})
    result=normalize(service,{'model':['public-fl'],'prompt':['A person speaks.'],'seconds':['5'],'size':['1344x768'],'silent':['true'],'dialogue':['[{"text":"你好","speaker":"甲"}]']},[])
    assert result['silent'] is True and result['dialogue']==[{'text':'你好','speaker':'甲'}]


def test_ref2va_image_mode_is_pinned_without_inventing_explicit_role(tmp_path):
    from dingo.video_gateway.context_ir.protocol import prepare
    path=tmp_path/'picture.jpg';path.write_bytes(b'image')
    n={'workflow':'ref2va','references':[{'kind':'image','path':'picture.jpg','sha256':hashlib.sha256(b'image').hexdigest(),'bytes':5,'ordinal':0,'role':'subject','role_stated':False}],
       'canvas':[1344,768],'intent':'Animate this photo','seconds':5,'creativity':'balanced','context_ir_seed':7,'dialogue':[],'silent':False}
    payload,_,_=prepare(n,tmp_path,tmp_path/'attempt',100)
    assert 'role' not in payload['assets'][0]
    assert payload['assets'][0]['provenance']=={'intended_role':'reference'}
    n['workflow']='fl2va';n['references'][0].update(role='frame_anchor_first',role_stated=True)
    payload,_,_=prepare(n,tmp_path,tmp_path/'second',100)
    assert payload['assets'][0]['role']=='frame_anchor_first'
    assert 'provenance' not in payload['assets'][0]


async def test_context_ir_idempotency_replays_legacy_without_aliasing_new_roles():
    from dingo.video_gateway.context_ir.service import ContextIRService
    from dingo.video_gateway.context_ir.validation import request_digest
    _,store=await stack()
    original={'intent':'A turns','references':[{'kind':'video','role':'edit_source','sha256':'a'}]}
    old,_=await task(store,'old',key='k',digest=request_digest(original))
    async def read_original(*args):return original
    service=SimpleNamespace(store=store,artifacts=SimpleNamespace(read_bounded=read_original),config=SimpleNamespace(result_max_bytes=10000))
    current={**original,'references':[{**original['references'][0],'role_stated':False}]}
    replay=await ContextIRService.idempotent_request(service,'p','k',current)
    assert replay.task.id==old.task.id and len(await store.records('tasks/'))==1
    for changed in [{**current,'intent':'Different'}, {**current,'references':[{**current['references'][0],'sha256':'b'}]}, {**current,'references':[{**current['references'][0],'role':'motion_reference'}]}]:
        with pytest.raises(GatewayError) as error:await ContextIRService.idempotent_request(service,'p','k',changed)
        assert error.value.code=='idempotency_conflict'
    await task(store,'new',key='newkey',digest=request_digest(current))
    assert (await ContextIRService.idempotent_request(service,'p','newkey',current)).task.id=='new'
    explicit={**current,'references':[{**current['references'][0],'role_stated':True}]}
    with pytest.raises(GatewayError):await ContextIRService.idempotent_request(service,'p','newkey',explicit)


@pytest.mark.parametrize("settings, expected", [({}, 2), ({"max_inflight": 32}, 32), ({"max_inflight": 48}, 48)])
def test_compiler_default_capacity_and_explicit_override(settings, expected):
    config = parse_context_ir({"backends": {"h3": {"url": "http://compiler", **settings}}}, {})
    assert config.backends["h3"].max_inflight == expected


async def test_compiler_explicit_32_slots_shared_across_owners():
    c, s = await stack(max_inflight=32)
    claims = []
    for i in range(32):
        await task(s, str(i))
        owner = "a" if i % 2 == 0 else "b"
        claim = await s.claim("h3", owner, owner)
        assert claim is not None
        claims.append(claim)
    await task(s, "waiting")
    assert len({claim.task.id for claim in claims}) == 32
    assert await s.claim("h3", "a", "a") is None
    assert await s.claim("h3", "b", "b") is None
    assert await s.release_slot(claims[0].task)
    admitted = await s.claim("h3", "b", "b")
    assert admitted is not None and admitted.task.id == "waiting"
    assert int((await c.get(s.key("counters/h3/inflight"))).value) == 32
