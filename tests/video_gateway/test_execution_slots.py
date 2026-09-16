"""Slot lease identity and physical-instance retry fencing."""

import asyncio
import hashlib
from unittest.mock import AsyncMock

import pytest

from dingo.video_gateway.models import (
    StoredTask,
    TaskStatus,
    VideoTask,
    WorkerLease,
    now_ms,
)
from dingo.video_gateway.task_store import (
    EtcdTaskStore,
    MemoryTaskStore,
    retry_excludes_worker,
    worker_key,
)


def task(name):
    return VideoTask(
        schema_version=1,
        id=name,
        deployment_id="slots",
        pool_id="pool",
        model="model",
        backend_model="model",
        backend_target="dyn://slot.backend.generate",
        configuration_revision="r1",
        delivery_mode="async",
        status=TaskStatus.QUEUED,
        request_digest=name,
        request_path="/test/request",
        input_manifest_path="/test/manifest",
        created_at_ms=now_ms(),
        queued_at_ms=now_ms(),
        expires_at_ms=now_ms() + 60000,
    )


def lease(value, slot):
    return WorkerLease(
        pool_id=value.pool_id,
        worker_key=worker_key(value.backend_target, 7, slot),
        worker_instance_id=7,
        backend_target=value.backend_target,
        task_id=value.id,
        owner_generation="gateway-a",
        state="reserved",
        heartbeat_at_ms=now_ms(),
    )


def test_slot_zero_preserves_legacy_key_and_other_slots_are_distinct():
    target = "dyn://slot.backend.generate"
    legacy = hashlib.sha256(target.encode() + b"\0" + b"7").hexdigest()
    assert worker_key(target, 7) == worker_key(target, "7", 0) == legacy
    assert len({worker_key(target, 7, slot) for slot in range(100)}) == 100


@pytest.mark.parametrize("slot", [-1, True, 1.5, "1"])
def test_invalid_slot_id_rejected(slot):
    with pytest.raises(ValueError, match="slot_id"):
        worker_key("target", 7, slot)


@pytest.mark.parametrize("failed_slot", [0, 1, 9])
def test_retry_excludes_physical_instance_across_all_slots(failed_slot):
    value = task("retry")
    assert not retry_excludes_worker(value, value.backend_target, 7)
    value.attempt = 1
    value.worker_instance_id = 7
    value.worker_key = worker_key(value.backend_target, 7, failed_slot)
    assert retry_excludes_worker(value, value.backend_target, "7")
    assert not retry_excludes_worker(value, value.backend_target, 8)
    value.worker_instance_id = None
    assert retry_excludes_worker(value, value.backend_target, 8)


@pytest.mark.asyncio
async def test_two_slots_reserve_independently_and_same_slot_does_not_oversell():
    store = MemoryTaskStore()
    values = [task(str(i)) for i in range(3)]
    stored = [
        (
            await store.create_task(
                v, principal_hash="p", idempotency_hash=None, queue_limit=4
            )
        )[0]
        for v in values
    ]
    results = await asyncio.gather(
        *[
            store.reserve(
                stored[i], lease(values[i], i % 2), deadline_at_ms=now_ms() + 30000
            )
            for i in range(3)
        ]
    )
    assert sum(r is not None for r in results) == 2
    leases = await store.list_leases("pool")
    assert len(leases) == 2 and {l.worker_instance_id for l in leases} == {7}
    assert len({l.worker_key for l in leases}) == 2
    victim = next(r for r in results if r is not None)
    await store.transition(
        victim.task.id,
        expected={TaskStatus.DISPATCHING},
        patch={"status": TaskStatus.CANCELLED},
        release_lease=True,
    )
    remaining = await store.list_leases("pool")
    assert len(remaining) == 1 and remaining[0].task_id != victim.task.id


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_slot,new_slot", [(0, 1), (1, 0), (1, 2)])
async def test_etcd_reserve_rejects_retry_to_another_slot_of_failed_worker(
    failed_slot, new_slot
):
    value = task("retry")
    value.attempt = 1
    value.worker_instance_id = 7
    value.worker_key = worker_key(value.backend_target, 7, failed_slot)
    store = EtcdTaskStore(object(), prefix="/unit/slots", deployment_id="slots")
    store._counter = AsyncMock(return_value=(1, None))
    store._retry_counter = AsyncMock(
        side_effect=AssertionError("must reject before reserving credits")
    )
    result = await store.reserve(
        StoredTask(value, 1), lease(value, new_slot), deadline_at_ms=now_ms() + 30000
    )
    assert result is None
    store._retry_counter.assert_not_awaited()
