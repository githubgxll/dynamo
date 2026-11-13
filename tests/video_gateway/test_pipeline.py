# SPDX-License-Identifier: Apache-2.0
import asyncio
import copy
import hashlib
from pathlib import Path

import pytest

from dingo.video_gateway.errors import StoreConflict
from dingo.video_gateway.models import TaskStatus, now_ms
from dingo.video_gateway.pipeline import VideoPipelineCoordinator
from dingo.video_gateway.task_store import MemoryTaskStore, EtcdTaskStore
from tests.video_gateway.test_task_store import _task
from tests.video_gateway.test_etcd_task_store import FakeEtcd
from tests.video_gateway.test_dispatcher import _stack, FakeClient
from tests.video_gateway.test_seedvr2_adapter import video


def pools():
    base = {'pool_id':'base','served_models':['base-model'],'backend_model':'base-model',
        'backend_target':'dyn://base.backend.generate','execution_mode':'detached',
        'adapter':{'name':'minimax_h3','workflow':'fl2va'},
        'scheduling':{'accept_without_workers':True}}
    sr = {'pool_id':'sr','served_models':['sr-model'],'backend_model':'sr-model',
        'backend_target':'dyn://sr.backend.generate','execution_mode':'detached',
        'adapter':{'name':'seedvr2','workflow':'sr_only','storage_id':'test-store'},
        'scheduling':{'accept_without_workers':True}}
    parent = copy.deepcopy(base)
    parent.update(pool_id='pipeline',served_models=['public-pipeline'],
        pipeline={'base_pool':'base','sr_pool':'sr'})
    parent['adapter']['super_resolution']={'enabled':True,'profile':'minimax_h3_seedvr2.v3'}
    return [base,sr,parent]


@pytest.fixture(params=['memory','etcd'])
def store(request):
    return (MemoryTaskStore() if request.param=='memory' else
            EtcdTaskStore(FakeEtcd(),prefix='/test',deployment_id='pipeline'))


async def parent_record(store):
    parent = _task('parent',pool_id='pipeline')
    parent.status=TaskStatus.IN_PROGRESS
    parent.normalized_request={'_pipeline':{'phase':'base','children':{}}}
    stored,_=await store.create_task(parent,principal_hash='p',idempotency_hash='p',queue_limit=10)
    return stored


def child_record(name):
    child=_task(name,pool_id='base')
    child.normalized_request={'_pipeline_child':{'parent_id':'parent','phase':'base'}}
    return child


async def test_cancel_fences_child_admission_atomically(store):
    parent=await parent_record(store)
    await store.request_cancel(parent.task.id)
    with pytest.raises(StoreConflict):
        await store.create_task(child_record('child'),principal_hash='p',idempotency_hash='child',
                                queue_limit=10,pipeline_parent=parent)
    assert await store.get_task('child') is None
    assert await store.queue_depth('base') == 0


async def test_two_gateways_admit_one_child_and_link_in_same_transaction(store):
    parent=await parent_record(store)
    results=await asyncio.gather(*[
        store.create_task(child_record(name),principal_hash='p',idempotency_hash='base-child',
                          queue_limit=10,pipeline_parent=parent)
        for name in ['child-a','child-b']], return_exceptions=True)
    admitted=[r for r in results if not isinstance(r,Exception)]
    assert sum(r[1] for r in admitted)==1
    current=await store.get_task('parent')
    cid=current.task.normalized_request['_pipeline']['children']['base']
    assert (await store.get_task(cid)) is not None
    assert await store.queue_depth('base')==1


async def setup_pipeline(make_gateway_config):
    cfg=make_gateway_config(pools=pools())
    store,artifacts,dispatcher,service=_stack(cfg,{p.pool_id:FakeClient() for p in cfg.pools})
    dispatcher.has_workers=lambda pool: True
    dispatcher.has_sr_workers=lambda pool,n: True
    root=await artifacts.create_upload()
    submitted=await service.submit(fields={k:[v] for k,v in {
        'model':'public-pipeline','prompt':'test chart','resolution':'1080P',
        'ratio':'16:9','seconds':'4','seed':'123'}.items()},uploads=[],upload_root=root,
        delivery_mode='async',idempotency_key='parent-test')
    coordinator=VideoPipelineCoordinator(service)
    pool=cfg.pools_by_id['pipeline']
    async def step(coordinator=coordinator):
        await coordinator.advance(await store.get_task(submitted.stored.task.id),pool)
        return await store.get_task(submitted.stored.task.id)
    await step();await step();parent=await step()
    return service,coordinator,step,parent


async def complete_child(service,child_id,path,*,seconds):
    store=service.store
    for state in [TaskStatus.DISPATCHING,TaskStatus.IN_PROGRESS,TaskStatus.FINALIZING,TaskStatus.COMPLETED]:
        current=await store.get_task(child_id)
        await store.transition(child_id,expected={current.task.status},expected_revision=current.revision,
            patch={'status':state,'result_path':str(path),'result_bytes':path.stat().st_size,
                   'result_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                   'inference_time_s':seconds,'queue_wait_s':.1,'completed_at_ms':now_ms()})


async def test_restart_between_stages_reuses_base_and_publishes_same_parent(make_gateway_config,tmp_path):
    service,coordinator,step,parent=await setup_pipeline(make_gateway_config)
    base_id=parent.task.normalized_request['_pipeline']['children']['base']
    base=await service.store.get_task(base_id)
    assert 'super_resolution' not in base.task.normalized_request
    assert (base.task.normalized_request['width'],base.task.normalized_request['height'])==(1344,768)
    base_root=await service.artifacts.resolve_task_root(base.task.deployment_id,base.task.pool_id,base.task.id)
    source=video(base_root).path
    await complete_child(service,base_id,source,seconds=2)
    # Fresh coordinator, as after Gateway restart; no local stage future survives.
    restarted=VideoPipelineCoordinator(service)
    parent=await step(restarted);assert parent.task.normalized_request['_pipeline']['phase']=='sr'
    parent=await step(restarted)
    sr_id=parent.task.normalized_request['_pipeline']['children']['sr']
    assert await service.store.queue_depth('base')==0
    sr=await service.store.get_task(sr_id)
    sr_root=await service.artifacts.resolve_task_root(sr.task.deployment_id,sr.task.pool_id,sr.task.id)
    result=video(sr_root,size=(1920,1080)).path
    await complete_child(service,sr_id,result,seconds=3)
    await step(restarted);parent=await step(restarted)
    assert parent.task.status==TaskStatus.COMPLETED
    assert parent.task.result_sha256==hashlib.sha256(result.read_bytes()).hexdigest()
    assert Path(parent.task.result_path).parent.name==parent.task.id
    assert parent.task.public_dict()['metrics']['pipeline']['base_generation_s']==2
    assert parent.task.public_dict()['metrics']['pipeline']['super_resolution_s']==3
    assert len(await service.store.list_all_tasks())==3


async def test_cancel_at_stage_boundary_does_not_create_sr(make_gateway_config):
    service,coordinator,step,parent=await setup_pipeline(make_gateway_config)
    base_id=parent.task.normalized_request['_pipeline']['children']['base']
    base=await service.store.get_task(base_id)
    root=await service.artifacts.resolve_task_root(base.task.deployment_id,base.task.pool_id,base.task.id)
    await complete_child(service,base_id,video(root).path,seconds=2)
    parent=await step()
    await service.dispatcher.cancel(parent.task.id)
    parent=await step()
    assert parent.task.status==TaskStatus.CANCELLED
    assert 'sr' not in parent.task.normalized_request['_pipeline']['children']
    assert len(await service.store.list_all_tasks())==2


def test_pipeline_rejects_recursive_target(make_gateway_config):
    raw=pools();raw[-1]['pipeline']['base_pool']='pipeline'
    with pytest.raises(ValueError,match='base-only'):
        make_gateway_config(pools=raw)


def test_pipeline_pool_does_not_require_its_own_worker_watch(make_gateway_config):
    cfg=make_gateway_config(pools=pools())
    _,_,dispatcher,_=_stack(cfg,{p.pool_id:FakeClient() for p in cfg.pools})
    dispatcher._ready=True
    dispatcher._gateway_owner_healthy=True
    dispatcher._task_watch_healthy=True
    for pool in dispatcher.pools.values():
        pool.discovery_healthy=pool.config.pipeline is None
        pool.lease_watch_healthy=pool.config.pipeline is None
    assert dispatcher.ready


@pytest.mark.parametrize('phase',['base','sr'])
async def test_failed_stage_propagates_without_restarting_base(make_gateway_config,phase):
    from dingo.video_gateway.models import TaskError
    service,coordinator,step,parent=await setup_pipeline(make_gateway_config)
    base_id=parent.task.normalized_request['_pipeline']['children']['base']
    if phase=='sr':
        base=(await service.store.get_task(base_id)).task
        root=await service.artifacts.resolve_task_root(base.deployment_id,base.pool_id,base.id)
        await complete_child(service,base_id,video(root).path,seconds=2)
        await step();parent=await step()
    child_id=parent.task.normalized_request['_pipeline']['children'][phase]
    child=await service.store.get_task(child_id)
    internal={'schema_version':1,'reason':'resource_exhausted','stage':'super_resolution'}
    await service.store.transition(child_id,expected={TaskStatus.QUEUED},expected_revision=child.revision,
        patch={'status':TaskStatus.FAILED,'error':TaskError('worker_execution_failed','worker failed'),
               'internal_error':internal})
    parent=await step()
    assert parent.task.status==TaskStatus.FAILED
    assert parent.task.internal_error==internal
    assert parent.task.normalized_request['_pipeline']['children']['base']==base_id
    assert len(await service.store.list_all_tasks())==(2 if phase=='base' else 3)


async def test_uncertain_child_commit_keeps_files_and_atomic_link(make_gateway_config,monkeypatch):
    service,coordinator,step,parent=await setup_pipeline(make_gateway_config)
    base_id=parent.task.normalized_request['_pipeline']['children']['base']
    base=(await service.store.get_task(base_id)).task
    root=await service.artifacts.resolve_task_root(base.deployment_id,base.pool_id,base.id)
    await complete_child(service,base_id,video(root).path,seconds=2)
    await step()
    create=service.store.create_task
    async def create_then_disconnect(*args,**kwargs):
        await create(*args,**kwargs)
        async def unavailable_read(*args,**kwargs):return None
        monkeypatch.setattr(service.store,'get_idempotent',unavailable_read)
        raise ConnectionError('lost transaction response')
    monkeypatch.setattr(service.store,'create_task',create_then_disconnect)
    with pytest.raises(ConnectionError):await step()
    parent=await service.store.get_task(parent.task.id)
    sr=await service.store.get_task(parent.task.normalized_request['_pipeline']['children']['sr'])
    assert Path(sr.task.request_path).is_file()
    assert Path(sr.task.input_manifest_path).is_file()
