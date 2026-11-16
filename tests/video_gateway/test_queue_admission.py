# SPDX-License-Identifier: Apache-2.0
import time
import pytest
from dingo.common.video_super_resolution import H3_SR_PROFILE_V3
from dingo.common.video_task_protocol import ENVELOPE_KEY
from dingo.video_gateway.errors import GatewayError
from dingo.vllm.omni.detached_tasks import DetachedOmniTaskManager
from dingo.vllm.omni.h3_video_handler import H3VideoHandler
from tests.video_gateway.test_dispatcher import FakeClient,_stack

class SRHandler(H3VideoHandler):
    sr_profile=H3_SR_PROFILE_V3
    @property
    def sr_ready(self):return self.video_ready

class Client(FakeClient):
    def __init__(self,manager):super().__init__();self.manager=manager
    async def direct(self,payload,instance_id,context):
        if payload.get(ENVELOPE_KEY,{}).get('op')=='capabilities':return self.manager.generate(payload,context)
        return await super().direct(payload,instance_id,context)

def setup(make_gateway_config,tmp_path,*,sr=False):
    handler=(SRHandler if sr else H3VideoHandler)(None)
    manager=DetachedOmniTaskManager(handler,tmp_path/'worker',execution_capacity=1)
    options={'super_resolution':{'enabled':True,'profile':H3_SR_PROFILE_V3}} if sr else {}
    pool={'pool_id':'p','served_models':['model'],'backend_model':'worker','backend_target':'dyn://test.backend.generate','execution_mode':'detached','adapter':{'name':'minimax_h3','workflow':'fl2va',**options},'scheduling':{'queue_limit':1,'worker_prefetch_capacity':1,'early_release_slot':True,'accept_without_workers':False}}
    cfg=make_gateway_config(pools=[pool]);client=Client(manager)
    store,artifacts,dispatcher,service=_stack(cfg,{'p':client})
    return handler,manager,client,store,artifacts,dispatcher,service

async def submit(service,artifacts,*,sr=False,key='one'):
    root=await artifacts.create_upload()
    fields={'model':'model','prompt':'pond','seconds':'4','seed':'1','size':'1920x1080' if sr else '1344x768'}
    return await service.submit(fields={k:[v] for k,v in fields.items()},uploads=[],upload_root=root,delivery_mode='async',idempotency_key=key)

@pytest.mark.asyncio
@pytest.mark.parametrize('sr',[False,True])
@pytest.mark.parametrize('pause_source',['wrapper','supervisor'])
async def test_paused_worker_accepts_bounded_queue_but_cannot_dispatch(make_gateway_config,tmp_path,sr,pause_source):
    h,m,c,store,artifacts,d,s=setup(make_gateway_config,tmp_path,sr=sr)
    if pause_source=='wrapper':h.pause()
    else:m.settlement_paused=True
    pool=d.pools['p'];await d._refresh_instances(pool)
    assert not d.has_workers('p') and d.has_admission_workers('p')
    if sr:
        n={'super_resolution':{'profile':H3_SR_PROFILE_V3}}
        assert d.has_sr_admission_workers('p',n) and not d.has_sr_workers('p',n)
    task=(await submit(s,artifacts,sr=sr)).stored.task
    assert task.status.value=='queued' and d._worker_slots(pool)==[] and not d.running_calls
    with pytest.raises(GatewayError) as exc:await submit(s,artifacts,sr=sr,key='two')
    assert exc.value.status==429
    h.resume();m.settlement_paused=False;pool.capacity_cache.clear();await d._refresh_instances(pool)
    assert d.has_workers('p') and d._worker_slots(pool)

@pytest.mark.asyncio
async def test_quarantine_drain_missing_and_expired_evidence_reject(make_gateway_config,tmp_path):
    h,m,c,store,artifacts,d,s=setup(make_gateway_config,tmp_path)
    h.pause();pool=d.pools['p'];await d._refresh_instances(pool)
    assert d.has_admission_workers('p')
    h.quarantine('test');pool.capacity_cache.clear();await d._refresh_instances(pool)
    assert not d.has_admission_workers('p')
    with pytest.raises(GatewayError) as exc:await submit(s,artifacts)
    assert exc.value.status==503
    h.quarantined=False;m._accepting=False;pool.capacity_cache.clear();await d._refresh_instances(pool)
    assert not d.has_admission_workers('p')
    m._accepting=True;pool.capacity_cache.clear();await d._refresh_instances(pool)
    expiry,profiles,shared=pool.queue_admission[c.instance_id]
    pool.queue_admission[c.instance_id]=(time.monotonic()-1,profiles,shared)
    assert not d.has_admission_workers('p')
    c.available=False;await d._refresh_instances(pool)
    assert not pool.queue_admission and not d.has_admission_workers('p')

@pytest.mark.asyncio
async def test_incompatible_profile_does_not_gain_queue_admission(make_gateway_config,tmp_path):
    h,m,c,store,artifacts,d,s=setup(make_gateway_config,tmp_path,sr=True)
    h.pause();h.sr_profile='different-profile';await d._refresh_instances(d.pools['p'])
    assert not d.has_sr_admission_workers('p',{'super_resolution':{'profile':H3_SR_PROFILE_V3}})
    with pytest.raises(GatewayError) as exc:await submit(s,artifacts,sr=True)
    assert exc.value.status==503
