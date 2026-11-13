"""Real HTTP multipart -> durable task -> fake compiler integration races."""
import asyncio
from dataclasses import replace
import json
from types import SimpleNamespace
import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest
from dingo.video_gateway.api import error_middleware
from dingo.video_gateway.adapters.minimax_h3 import MiniMaxH3VideoAdapter
from dingo.video_gateway.context_ir.config import BackendConfig, ContextIRConfig
from dingo.video_gateway.context_ir.service import ContextIRService
from dingo.video_gateway.context_ir.api import KEY, register
from tests.video_gateway.test_etcd_task_store import FakeEtcd


class Compiler:
    def __init__(self):
        self.calls = 0
        self.active = 0
        self.peak = 0
        self.gate = asyncio.Event()
        self.gate.set()
        self.status = 201

    async def compile(self, request):
        body = await request.json()
        self.calls += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await self.gate.wait()
            w,h=map(int,body['aspect'].split(':'))
            frames=round(body['seconds']*24);frames+=(5-frames%17)%17
            return web.json_response({'status':getattr(self,'result_status','ready'),
                'source':getattr(self,'source','written'), 'fallback_reason':getattr(self,'fallback_reason',None),
                'usage':getattr(self,'usage',None),
                'ir':{'mode':'t2va','prompt':getattr(self,'prompt','A bird moves through the sky.'),
                      'provenance':{'rejected_written':'Unvalidated candidate; do not return.'},
                      'target':{'frames':frames,'canvas':[w,h]},'manifest':[]}},status=self.status)
        finally:
            self.active -= 1

    async def serve(self):
        app=web.Application()
        async def capabilities(request):return web.json_response({'gateway_contract':{'asset_leases':True}})
        async def lease(request):return web.json_response({'ok':True})
        app.router.add_get('/v1/capabilities',capabilities)
        app.router.add_put('/v1/asset-leases/{token}',lease)
        app.router.add_delete('/v1/asset-leases/{token}',lease)
        app.router.add_post('/v1/briefs',self.compile)
        server=TestServer(app);await server.start_server();return server


@pytest.fixture
async def setup(make_gateway_config):
    compiler=Compiler();up=await compiler.serve()
    cfg=make_gateway_config()
    cc=ContextIRConfig(enabled=True,backends={'h3':BackendConfig(url=str(up.make_url('')).rstrip('/'),max_inflight=2,compile_timeout_s=2)},
        model_bindings={'public-fl':'h3'},sweeper_interval_s=.1)
    cfg=replace(cfg,context_ir=cc)
    async def capacity(required):pass
    video=SimpleNamespace(config=cfg,dispatcher=SimpleNamespace(draining=False),
        resolve_pool=cfg.pool_for_model,adapters={cfg.pools[0].pool_id:MiniMaxH3VideoAdapter(cfg.pools[0])},
        ensure_submission_capacity=capacity)
    etcd=FakeEtcd();service=ContextIRService(video,client=etcd)
    app=web.Application(middlewares=[error_middleware]);app[KEY]=service;register(app)
    client=TestClient(TestServer(app));await client.start_server();await service.start()
    yield compiler,service,client
    compiler.gate.set();await service.stop();await client.close();await up.close()


def form(prompt='A bird flies.'):
    data=aiohttp.FormData(default_to_multipart=True)
    for k,v in {'model':'public-fl','prompt':prompt,'seconds':'5','size':'1344x768'}.items():data.add_field(k,v)
    return data


async def wait_task(client,id,status='completed'):
    for _ in range(200):
        response=await client.get('/v1/context-ir/'+id);value=await response.json()
        if value.get('status')==status:return value
        await asyncio.sleep(.02)
    raise AssertionError(value)


async def test_http_admit_compile_query_replay_delete(setup):
    compiler,s,client=setup
    response=await client.post('/v1/context-ir',data=form(),headers={'Idempotency-Key':'one'})
    assert response.status==202,await response.text()
    task=await response.json();id=task['id']
    result=await wait_task(client,id)
    assert result['content']['prompt'] and result['generation_parameters']['seconds']==5
    response=await client.post('/v1/context-ir',data=form(),headers={'Idempotency-Key':'one'})
    assert response.status==200
    assert (await response.json())['id']==id and compiler.calls==1
    assert (await client.post('/v1/context-ir',data=form('Different'),headers={'Idempotency-Key':'one'})).status==409
    assert (await client.delete('/v1/context-ir/'+id)).status==204
    assert (await client.get('/v1/context-ir/'+id)).status==410
    await s.sweep()
    assert not s.artifacts.task_root(s.video.config.deployment_id,'',id).exists()


async def test_cancel_does_not_free_running_slot_or_allow_late_success(setup):
    compiler,s,client=setup;compiler.gate.clear()
    ids=[]
    for _ in range(3):
        response=await client.post('/v1/context-ir',data=form());assert response.status==202
        ids.append((await response.json())['id'])
    for _ in range(100):
        if compiler.active==2:break
        await asyncio.sleep(.02)
    assert compiler.active==compiler.peak==2
    response=await client.delete('/v1/context-ir/'+ids[0]);assert (await response.json())['status']=='cancelled'
    await asyncio.sleep(.3)
    assert compiler.calls==2 and len(await s.store.records('backends/h3/slots/'))==2
    compiler.gate.set()
    await wait_task(client,ids[2]);assert compiler.calls==3
    assert (await wait_task(client,ids[0],'cancelled'))['status']=='cancelled'


async def test_unknown_compile_response_quarantines_without_retry(setup):
    compiler,s,client=setup;compiler.status=503
    response=await client.post('/v1/context-ir',data=form());id=(await response.json())['id']
    result=await wait_task(client,id,'failed')
    assert result['error']['code']=='context_ir_execution_unknown'
    assert len(await s.store.records('backends/h3/slots/'))==1
    assert compiler.calls==1
    response=await client.post('/v1/context-ir',data=form());assert response.status==503


async def test_unknown_fields_do_not_admit_or_leak_reservation(setup):
    _,s,client=setup
    data=form();data.add_field('seed','4')
    response=await client.post('/v1/context-ir',data=data)
    assert response.status==400
    assert not await s.store.records('tasks/')
    assert not await s.store.records('reservations/')


async def test_admission_commit_with_lost_response_keeps_files_and_same_task(setup,monkeypatch):
    import base64
    from dingo.video_gateway.errors import StoreUnavailable
    compiler,s,client=setup
    original=s.store.client.txn
    lost=False
    async def txn(compares,writes,*args,**kwargs):
        nonlocal lost
        result=await original(compares,writes,*args,**kwargs)
        admitted=any('/tasks/' in base64.b64decode(w['request_put']['key']).decode()
                     and json.loads(base64.b64decode(w['request_put']['value'])).get('status')=='queued'
                     for w in writes if 'request_put' in w and '/tasks/' in base64.b64decode(w['request_put']['key']).decode())
        if admitted and not lost:
            lost=True
            raise StoreUnavailable('response lost after commit')
        return result
    monkeypatch.setattr(s.store.client,'txn',txn)
    response=await client.post('/v1/context-ir',data=form(),headers={'Idempotency-Key':'lost'})
    assert response.status==202,await response.text()
    id=(await response.json())['id']
    assert s.artifacts.task_root(s.video.config.deployment_id,'',id).exists()
    await wait_task(client,id)
    response=await client.post('/v1/context-ir',data=form(),headers={'Idempotency-Key':'lost'})
    assert response.status==200 and (await response.json())['id']==id
    assert compiler.calls==1 and lost


async def test_durable_response_survives_owner_loss_without_recompile(setup,monkeypatch):
    compiler,s,client=setup
    original=s.store.finish;lost=False
    async def finish(stored,status,**kwargs):
        nonlocal lost
        if status=='completed' and not lost:
            lost=True
            await s.store.client.txn([], [s.store.client.delete(s.store.key(f'owners/{s.dispatcher.owner}'))])
            return None
        return await original(stored,status,**kwargs)
    monkeypatch.setattr(s.store,'finish',finish)
    response=await client.post('/v1/context-ir',data=form());id=(await response.json())['id']
    for _ in range(100):
        if lost:break
        await asyncio.sleep(.02)
    assert lost and compiler.calls==1
    s.dispatcher.accepting=False
    other=ContextIRService(s.video,client=s.store.client)
    await other.start()
    try:
        result=await wait_task(client,id)
        assert result['status']=='completed' and compiler.calls==1
    finally:
        await other.stop()


@pytest.mark.parametrize('upstream_status,terminal',[(201,'completed'),(502,'failed')])
async def test_usage_persisted_for_success_failure_and_replay(setup,upstream_status,terminal):
    compiler,s,client=setup
    usage=dict(prompt_tokens=100,completion_tokens=30,total_tokens=130,llm_requests=3,
               requests_with_usage=2,requests_without_usage=1,complete=False)
    usage['prompt_tokens_details']={'cached_tokens':None}
    compiler.usage=usage;compiler.status=upstream_status
    response=await client.post('/v1/context-ir',data=form(),headers={'Idempotency-Key':'usage'})
    task=await response.json();assert task['usage'] is None
    result=await wait_task(client,task['id'],terminal)
    assert result['usage']==usage
    diag=await (await client.get('/v1/context-ir/'+task['id']+'/diagnostics')).json()
    assert diag['usage']==usage
    stored=await s.store.get_task(task['id'])
    assert stored.task.to_dict()['usage']==usage
    replay=await client.post('/v1/context-ir',data=form(),headers={'Idempotency-Key':'usage'})
    assert (await replay.json())['usage']==usage and compiler.calls==1


async def test_complete_cached_usage_survives_gateway_query(setup):
    compiler,s,client=setup
    usage=dict(prompt_tokens=100,completion_tokens=30,total_tokens=130,llm_requests=2,
               requests_with_usage=2,requests_without_usage=0,complete=True,
               prompt_tokens_details={'cached_tokens':80})
    compiler.usage=usage
    response=await client.post('/v1/context-ir',data=form())
    task=await response.json();result=await wait_task(client,task['id'])
    assert result['usage']==usage


async def test_degraded_draft_completes_replays_and_releases_slot_without_new_fields(setup):
    compiler, s, client = setup
    normal_task = await (await client.post('/v1/context-ir', data=form())).json()
    normal = await wait_task(client, normal_task['id'])
    compiler.result_status = 'degraded'
    compiler.source = 'draft'
    compiler.fallback_reason = 'validation_failed'
    compiler.prompt = 'Actual fallback draft.'
    compiler.usage = dict(prompt_tokens=100, completion_tokens=30, total_tokens=130, llm_requests=2,
                          requests_with_usage=2, requests_without_usage=0, complete=True,
                          prompt_tokens_details={'cached_tokens':80})
    response = await client.post('/v1/context-ir', data=form(), headers={'Idempotency-Key':'degraded'})
    assert response.status == 202
    task = await response.json()
    result = await wait_task(client, task['id'])
    assert result.keys() == normal.keys()
    assert result['content'] == {'prompt':'Actual fallback draft.'}
    assert result['warnings'] == [{'code':'context_ir_degraded_result'}]
    assert result['usage'] == compiler.usage
    assert 'error' not in result
    diag = await (await client.get('/v1/context-ir/'+task['id']+'/diagnostics')).json()
    assert diag['upstream'] == {'status':'degraded','source':'draft','rule_codes':[]}
    assert diag['usage'] == compiler.usage
    replay = await client.post('/v1/context-ir', data=form(), headers={'Idempotency-Key':'degraded'})
    assert replay.status == 200
    assert await replay.json() == result
    assert compiler.calls == 2
    for _ in range(100):
        if not await s.store.records('backends/h3/slots/'): break
        await asyncio.sleep(.02)
    assert not await s.store.records('backends/h3/slots/')
    compiler.result_status = 'ready'
    compiler.source = 'written'
    compiler.fallback_reason = None
    next_task = await (await client.post('/v1/context-ir', data=form())).json()
    assert (await wait_task(client, next_task['id']))['warnings'] == []
    assert compiler.calls == 3
