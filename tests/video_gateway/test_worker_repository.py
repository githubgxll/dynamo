# SPDX-License-Identifier: Apache-2.0
import asyncio

from dingo.video_gateway.context_ir.config import BackendConfig, ContextIRConfig
from dingo.video_gateway.context_ir.task_store import ContextIRStore
from dingo.video_gateway.context_ir.worker_repository import (
    WorkerTaskRepository,
    observed,
)
from dingo.video_gateway.task_runtime.repository import (
    RuntimeKeys,
    WorkerRef,
    WorkerSnapshot,
)
from tests.video_gateway.test_context_ir import task
from tests.video_gateway.test_etcd_task_store import FakeEtcd


async def setup():
    client=FakeEtcd()
    backend=BackendConfig(transport='worker',worker_pool='compiler',backend_target='dyn://test.compiler.compile',storage_id='shared')
    config=ContextIRConfig(enabled=True,backends={'h3':backend},model_bindings={'model':'h3'})
    store=ContextIRStore(client,'test',config)
    await store.ensure_backend('h3',backend)
    for owner in ('a','b'):
        await client.txn([],[client.put(store.key(f'owners/{owner}'),owner)])
    await client.txn([],[client.put('/worker',b'ready')])
    worker=WorkerSnapshot(WorkerRef('compiler','1','boot','qwen'),observed(await client.get('/worker')),16)
    return client,store,worker

async def test_two_owners_share_sixteen_slots_and_queue_isolated():
    client,store,worker=await setup()
    first=WorkerTaskRepository(store,'a','a');second=WorkerTaskRepository(store,'b','b')
    for i in range(17):await task(store,f'task-{i}')
    for i in range(16):
        stored=await store.get_task(f'task-{i}')
        claims=await asyncio.gather(first.claim(stored,worker,i),second.claim(stored,worker,i))
        assert len([c for c in claims if c])==1
    assert await first.claim(await store.get_task('task-16'),worker,0) is None
    assert len(await client.range('/dingo/task-runtime/v2/test/slots/',prefix=True))==16
    assert not await store.records('backends/h3/queue/')

async def test_cancel_settle_and_lease_adoption_preserve_business_shape():
    client,store,worker=await setup();repo=WorkerTaskRepository(store,'a','a')
    original,_=await task(store,'task-1')
    claim=await repo.claim(original,worker,0)
    before=await repo.attempt(claim)
    await client.txn([],[client.delete(store.key('owners/a'))])
    other=WorkerTaskRepository(store,'b','b')
    adopted=await other.adopt(claim)
    after=await other.attempt(adopted)
    assert before.fence==after.fence
    assert after.owner.generation=='b'
    cancelled=await store.finish(adopted,'cancelled')
    result=await other.settle(cancelled,'failed',usage={'complete':False})
    assert result.task.status=='cancelled'
    assert result.task.usage=={'complete':False}
    assert set(result.task.to_dict())==set(original.task.to_dict())
    assert await client.get(RuntimeKeys('test').active(other.ref(original.task))) is None

async def test_stale_result_cannot_overwrite_concurrent_cancellation():
    _client,store,worker=await setup();repo=WorkerTaskRepository(store,'a','a')
    original,_=await task(store,'task-1');claim=await repo.claim(original,worker,0)
    await store.finish(claim,'cancelled')
    assert await repo.settle(claim,'failed') is None
    assert (await store.get_task('task-1')).task.status=='cancelled'

async def test_definite_negative_ack_requeues_without_losing_fifo_or_capacity():
    client,store,worker=await setup();repo=WorkerTaskRepository(store,"a","a")
    original,_=await task(store,"task-1");claim=await repo.claim(original,worker,0)
    retried=await repo.requeue_not_accepted(claim)
    assert retried.task.status=="queued"
    assert retried.task.created_seq==original.task.created_seq
    assert (await client.get(store.queue_key(retried.task))).value==b"task-1"
    assert int((await client.get(store.key("counters/h3/queued"))).value)==1
    assert not await client.range("/dingo/task-runtime/v2/test/slots/",prefix=True)
    again=await repo.claim(retried,worker,0)
    assert again.task.attempt==2
