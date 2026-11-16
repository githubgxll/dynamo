# SPDX-License-Identifier: Apache-2.0
"""Atomic public-parent/queued-child admission against isolated real etcd."""
import asyncio
import pytest
from tests.video_gateway.test_runtime_transactions_etcd import runtime_etcd
from tests.video_gateway.test_task_store import _task
from tests.video_gateway.test_pipeline import child_record
from dingo.video_gateway.models import TaskStatus
from dingo.video_gateway.task_store import EtcdTaskStore
from dingo.video_gateway.errors import GatewayError


async def test_real_etcd_two_gateways_share_single_admission(runtime_etcd):
    stores=[EtcdTaskStore(runtime_etcd,prefix='/unified',deployment_id='test') for _ in range(2)]
    async def submit(index):
        parent=_task('parent-'+str(index),pool_id='ref');parent.status=TaskStatus.IN_PROGRESS
        parent.normalized_request={'_pipeline':{'phase':'base','children':{}}}
        child=child_record('child-'+str(index));child.pool_id='ref'
        child.normalized_request['_pipeline_child']['parent_id']=parent.id
        return await stores[index].create_task(child,principal_hash='p',idempotency_hash='same',queue_limit=1,initial_parent=parent)
    results=await asyncio.gather(submit(0),submit(1))
    assert sum(created for _,created in results)==1
    parent=results[0][0]
    assert parent.task.id==results[1][0].task.id
    child_id=parent.task.normalized_request['_pipeline']['children']['base']
    assert [r.task.id for r in await stores[0].list_queued('ref')]==[child_id]
    assert await stores[0].queue_depth('ref')==1
    assert len(await stores[0].list_all_tasks())==2
    with pytest.raises(GatewayError) as error:
        await stores[1].create_task(_task('normal',pool_id='ref'),principal_hash='p',idempotency_hash=None,queue_limit=1)
    assert error.value.status==429
    assert await stores[1].get_task('normal') is None
