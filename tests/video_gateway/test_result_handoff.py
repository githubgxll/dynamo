import dataclasses
import json

import pytest

from dingo.video_gateway.errors import StoreConflict
from dingo.video_gateway.models import TaskStatus, now_ms
from dingo.video_gateway.result_handoff import HANDOFF_KEY, make_handoff, read_handoff
from dingo.video_gateway.task_store import EtcdTaskStore, MemoryTaskStore
from tests.video_gateway.test_etcd_task_store import FakeEtcd
from tests.video_gateway.test_task_store import _lease, _task

ARTIFACT = dict(
    schema_version=1,
    filename="worker-video-" + "a" * 32 + ".mp4",
    bytes=100,
    sha256="b" * 64,
)


async def setup(kind):
    client = FakeEtcd() if kind == "etcd" else None
    store = (
        EtcdTaskStore(client, prefix="/handoff-test", deployment_id="deployment")
        if client
        else MemoryTaskStore()
    )
    original = _task("first")
    stored, _ = await store.create_task(
        original, principal_hash="p", idempotency_hash=None, queue_limit=8
    )
    lease = _lease(original)
    lease.execution_token = "c" * 32
    if client:
        owner = await store.register_gateway("generation", ttl_s=15)
    else:
        owner = None
    options = {"reserve_retry": True} if client else {}
    stored = await store.reserve(
        stored, lease, deadline_at_ms=now_ms() + 60000, **options
    )
    stored = await store.transition(
        stored.task.id,
        expected={TaskStatus.DISPATCHING},
        patch={"status": TaskStatus.IN_PROGRESS},
    )
    return store, client, stored, owner


def patch_for(stored):
    return {
        "status": TaskStatus.FINALIZING,
        "worker_lease_id": None,
        "normalized_request": {
            **stored.task.normalized_request,
            HANDOFF_KEY: make_handoff(stored.task, ARTIFACT, timeout_s=30),
        },
    }


@pytest.mark.parametrize("kind", ["memory", "etcd"])
async def test_atomic_handoff_releases_slot_and_preserves_result(kind):
    store, client, stored, owner = await setup(kind)
    ready = await store.transition(
        stored.task.id,
        expected={TaskStatus.IN_PROGRESS},
        expected_revision=stored.revision,
        patch=patch_for(stored),
        release_lease=True,
        release_execution=True,
    )
    assert ready.task.status == TaskStatus.FINALIZING
    assert read_handoff(ready.task)["artifact"] == ARTIFACT
    assert not await store.list_leases(stored.task.pool_id)
    if client:
        assert await client.get(store._retry_credit_key(stored.task)) is None
        assert (
            int(
                (
                    await client.get(
                        store._retry_counter_key(stored.task.pool_id, "credits")
                    )
                ).value
            )
            == 0
        )
    # A new task can use exactly the same slot before old postprocessing ends.
    next_task = _task("second")
    next_stored, _ = await store.create_task(
        next_task, principal_hash="p", idempotency_hash=None, queue_limit=8
    )
    next_lease = _lease(next_task)
    next_lease.execution_token = "d" * 32
    reserved = await store.reserve(
        next_stored, next_lease, deadline_at_ms=now_ms() + 60000
    )
    assert reserved is not None
    await store.transition(
        ready.task.id,
        expected={TaskStatus.FINALIZING},
        expected_revision=ready.revision,
        patch={"status": TaskStatus.COMPLETED},
        release_lease=True,
    )
    assert (await store.list_leases(stored.task.pool_id))[0].task_id == "second"


@pytest.mark.parametrize("kind", ["memory", "etcd"])
async def test_cancel_race_cannot_release_lease_or_publish_handoff(kind):
    store, client, stored, owner = await setup(kind)
    await store.request_cancel(stored.task.id)
    with pytest.raises(StoreConflict):
        await store.transition(
            stored.task.id,
            expected={TaskStatus.IN_PROGRESS},
            expected_revision=stored.revision,
            patch=patch_for(stored),
            release_lease=True,
            release_execution=True,
        )
    current = await store.get_task(stored.task.id)
    assert read_handoff(current.task) is None
    assert len(await store.list_leases(stored.task.pool_id)) == 1


async def test_handoff_recovery_does_not_touch_reused_worker_slot():
    store, client, stored, owner = await setup("etcd")
    ready = await store.transition(
        stored.task.id,
        expected={TaskStatus.IN_PROGRESS},
        expected_revision=stored.revision,
        patch=patch_for(stored),
        release_lease=True,
        release_execution=True,
    )
    second = _task("second")
    nxt, _ = await store.create_task(
        second, principal_hash="p", idempotency_hash=None, queue_limit=8
    )
    nxt_lease = _lease(second)
    nxt_lease.execution_token = "d" * 32
    await store.register_gateway("gateway-b", ttl_s=15)
    nxt_lease.owner_generation = "gateway-b"
    reserved = await store.reserve(nxt, nxt_lease, deadline_at_ms=now_ms() + 60000)
    slot_key = store._lease_key(second.pool_id, reserved.task.worker_key)
    slot_before = await client.get(slot_key)
    assert await store.claim_finalizing(ready, new_owner_generation="gateway-b") is None
    await store.unregister_gateway(owner)
    claimed = await store.claim_orphaned_active(ready, new_owner_generation="gateway-b")
    assert claimed.task.owner_generation == "gateway-b"
    assert claimed.task.worker_lease_id is None
    assert await client.get(slot_key) == slot_before
    with pytest.raises(StoreConflict):
        await store.transition(
            ready.task.id,
            expected={TaskStatus.FINALIZING},
            expected_revision=ready.revision,
            patch={"status": TaskStatus.FAILED},
            release_lease=True,
        )
    assert await client.get(slot_key) == slot_before


async def test_conflicting_token_cannot_handoff_someone_elses_reservation():
    store, client, stored, owner = await setup("etcd")
    key = store._lease_key(stored.task.pool_id, stored.task.worker_key)
    value = await client.get(key)
    data = json.loads(value.value)
    data["execution_token"] = "e" * 32
    client.values[key] = dataclasses.replace(value, value=json.dumps(data).encode())
    with pytest.raises(StoreConflict):
        await store.transition(
            stored.task.id,
            expected={TaskStatus.IN_PROGRESS},
            expected_revision=stored.revision,
            patch=patch_for(stored),
            release_lease=True,
            release_execution=True,
        )
    assert read_handoff((await store.get_task(stored.task.id)).task) is None
    assert await client.get(store._retry_credit_key(stored.task)) is not None


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), 0, -1, True, 86401])
async def test_handoff_rejects_invalid_deadline(timeout):
    _, _, stored, _ = await setup("memory")
    with pytest.raises(ValueError):
        make_handoff(stored.task, ARTIFACT, timeout_s=timeout)
