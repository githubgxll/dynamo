# SPDX-License-Identifier: Apache-2.0
import asyncio
from dataclasses import replace
import pytest

from dingo.video_gateway.errors import GatewayError
from dingo.video_gateway.models import TaskStatus
from dingo.video_gateway.task_store import MemoryTaskStore,EtcdTaskStore
from tests.video_gateway.test_etcd_task_store import FakeEtcd
from tests.video_gateway.test_pipeline import pools,child_record
from tests.video_gateway.test_task_store import _task
from tests.video_gateway.test_dispatcher import _stack,FakeClient


def unified_pools():
    raw=pools()
    parent=raw[-1]
    parent['pool_id']='ref'
    parent['pipeline']={'sr_pool':'sr'}
    parent['scheduling']['queue_limit']=2
    return [parent,raw[1]]


@pytest.mark.parametrize('kind',['memory','etcd'])
async def test_one_queue_atomic_parent_child_and_shared_quota(kind):
    store=MemoryTaskStore() if kind=='memory' else EtcdTaskStore(FakeEtcd(),prefix='/unified',deployment_id='test')
    parent=_task('parent',pool_id='ref')
    parent.status=TaskStatus.IN_PROGRESS
    parent.normalized_request={'_pipeline':{'phase':'base','children':{}}}
    child=child_record('child');child.pool_id='ref'
    saved,created=await store.create_task(child,principal_hash='p',idempotency_hash='key',queue_limit=2,initial_parent=parent)
    assert saved.task.id=='parent' and created
    assert saved.task.normalized_request['_pipeline']['children']=={'base':'child'}
    assert [r.task.id for r in await store.list_queued('ref')]==['child']
    normal=_task('normal',pool_id='ref')
    await store.create_task(normal,principal_hash='p',idempotency_hash=None,queue_limit=2)
    replay,new=await store.create_task(child,principal_hash='p',idempotency_hash='key',queue_limit=2,initial_parent=parent)
    assert replay.task.id=='parent' and not new
    assert await store.queue_depth('ref')==2
    extra=_task('extra',pool_id='ref')
    with pytest.raises(GatewayError) as error:
        await store.create_task(extra,principal_hash='p',idempotency_hash=None,queue_limit=2)
    assert error.value.status==429


@pytest.mark.parametrize('kind',['memory','etcd'])
async def test_queue_full_never_leaves_an_unqueued_parent(kind):
    store=MemoryTaskStore() if kind=='memory' else EtcdTaskStore(FakeEtcd(),prefix='/unified',deployment_id='test')
    await store.create_task(_task('existing',pool_id='ref'),principal_hash='p',idempotency_hash=None,queue_limit=1)
    parent=_task('parent',pool_id='ref');parent.status=TaskStatus.IN_PROGRESS
    parent.normalized_request={'_pipeline':{'phase':'base','children':{}}}
    child=child_record('child');child.pool_id='ref'
    with pytest.raises(GatewayError):
        await store.create_task(child,principal_hash='p',idempotency_hash='key',queue_limit=1,initial_parent=parent)
    assert await store.get_task('parent') is None
    assert await store.get_task('child') is None
    assert await store.get_idempotent('p','key') is None


async def test_service_plain_and_high_resolution_share_same_queue(make_gateway_config):
    cfg=make_gateway_config(pools=unified_pools())
    assert len(cfg.pools)==2 and cfg.pools_by_id['ref'].executes_tasks
    store,artifacts,dispatcher,service=_stack(cfg,{p.pool_id:FakeClient() for p in cfg.pools})
    dispatcher.has_sr_admission_workers=lambda *a:True
    async def submit(tier,key):
        return await service.submit(fields={k:[v] for k,v in {'model':'public-pipeline','prompt':'A bird flies.','resolution':tier,'ratio':'16:9','seconds':'4','seed':'42'}.items()},uploads=[],upload_root=await artifacts.create_upload(),delivery_mode='async',idempotency_key=key)
    plain=await submit('768P','plain')
    assert '_pipeline' not in plain.stored.task.normalized_request
    high=await submit('1080P','high')
    assert high.stored.task.status==TaskStatus.IN_PROGRESS
    child_id=high.stored.task.normalized_request['_pipeline']['children']['base']
    queued=await store.list_queued('ref')
    assert [r.task.id for r in queued]==[plain.stored.task.id,child_id]
    child=await store.get_task(child_id)
    assert '_pipeline' not in child.task.normalized_request
    assert 'super_resolution' not in child.task.normalized_request
    assert (child.task.normalized_request['width'],child.task.normalized_request['height'])==(1344,768)
    replay=await submit('1080P','high')
    assert replay.stored.task.id==high.stored.task.id
    assert await store.queue_depth('ref')==2


@pytest.mark.parametrize('kind',['memory','etcd'])
async def test_concurrent_replay_admits_only_one_parent_and_child(kind):
    store=MemoryTaskStore() if kind=='memory' else EtcdTaskStore(FakeEtcd(),prefix='/race',deployment_id='test')
    async def submit(suffix):
        parent=_task('parent-'+suffix,pool_id='ref');parent.status=TaskStatus.IN_PROGRESS
        parent.normalized_request={'_pipeline':{'phase':'base','children':{}}}
        child=child_record('child-'+suffix);child.pool_id='ref'
        child.normalized_request['_pipeline_child']['parent_id']=parent.id
        return await store.create_task(child,principal_hash='p',idempotency_hash='same',queue_limit=2,initial_parent=parent)
    results=await asyncio.gather(submit('a'),submit('b'))
    assert sum(created for _,created in results)==1
    assert results[0][0].task.id==results[1][0].task.id
    assert await store.queue_depth('ref')==1
    assert len(await store.list_all_tasks())==2


async def unified_service(make_gateway_config):
    cfg=make_gateway_config(pools=unified_pools())
    store,artifacts,dispatcher,service=_stack(cfg,{p.pool_id:FakeClient() for p in cfg.pools})
    dispatcher.has_sr_admission_workers=lambda *a:True
    dispatcher.has_admission_workers=lambda *a:True
    submitted=await service.submit(fields={k:[v] for k,v in {'model':'public-pipeline','prompt':'A bird flies.','resolution':'1080P','ratio':'16:9','seconds':'4','seed':'42'}.items()},uploads=[],upload_root=await artifacts.create_upload(),delivery_mode='async',idempotency_key='high')
    from dingo.video_gateway.pipeline import VideoPipelineCoordinator
    return service,VideoPipelineCoordinator(service),submitted.stored


async def test_unified_parent_cancel_cancels_queued_child(make_gateway_config):
    service,coordinator,parent=await unified_service(make_gateway_config)
    child=parent.task.normalized_request['_pipeline']['children']['base']
    await service.dispatcher.cancel(parent.task.id)
    for _ in range(3):
        parent=await service.store.get_task(parent.task.id)
        if parent.task.status==TaskStatus.CANCELLED:break
        await coordinator.advance(parent,service.config.pools_by_id['ref'])
    assert (await service.store.get_task(parent.task.id)).task.status==TaskStatus.CANCELLED
    assert (await service.store.get_task(child)).task.status==TaskStatus.CANCELLED
    assert await service.store.queue_depth('ref')==0
    assert await service.store.queue_depth('sr')==0


async def test_unified_base_completes_then_joins_shared_sr_queue(make_gateway_config):
    from tests.video_gateway.test_pipeline import complete_child
    from tests.video_gateway.test_seedvr2_adapter import video
    service,coordinator,parent=await unified_service(make_gateway_config)
    child=await service.store.get_task(parent.task.normalized_request['_pipeline']['children']['base'])
    root=await service.artifacts.resolve_task_root(child.task.deployment_id,'ref',child.task.id)
    await complete_child(service,child.task.id,video(root).path,seconds=2)
    for _ in range(2):
        await coordinator.advance(await service.store.get_task(parent.task.id),service.config.pools_by_id['ref'])
    parent=await service.store.get_task(parent.task.id)
    sr_id=parent.task.normalized_request['_pipeline']['children']['sr']
    assert [r.task.id for r in await service.store.list_queued('sr')]==[sr_id]
    assert await service.store.queue_depth('ref')==0
    assert len(await service.store.list_all_tasks())==3


def test_unified_pool_rejects_extra_parent_quota(make_gateway_config):
    raw=unified_pools();raw[0]['pipeline']['max_active_parents']=4
    with pytest.raises(ValueError,match='no active-parent'):
        make_gateway_config(pools=raw)


def test_unified_base_workers_do_not_need_composite_sr_capability(make_gateway_config):
    cfg=make_gateway_config(pools=unified_pools())
    _,_,dispatcher,_=_stack(cfg,{p.pool_id:FakeClient() for p in cfg.pools})
    pool=dispatcher.pools['ref'];pool.instance_ids=[7]
    assert dispatcher._worker_slots(pool,{'width':1344,'height':768})==[(7,0)]
