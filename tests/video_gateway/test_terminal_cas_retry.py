"""Terminal writes must survive unrelated pool-ledger CAS contention."""

import asyncio
import json

import pytest

from dingo.video_gateway.errors import StoreConflict
from dingo.video_gateway.models import TaskStatus, now_ms
from dingo.video_gateway.task_store import EtcdTaskStore
from tests.video_gateway.test_etcd_task_store import FakeEtcd, _decode
from tests.video_gateway.test_task_store import _lease, _task


async def prepared():
    client = FakeEtcd()
    store = EtcdTaskStore(client, prefix="/isolated/terminal-cas", deployment_id="cas")
    task = _task("video-cas")
    stored, _ = await store.create_task(
        task, principal_hash="p", idempotency_hash=None, queue_limit=8
    )
    lease = _lease(task)
    await store.register_gateway(lease.owner_generation, ttl_s=15)
    stored = await store.reserve(
        stored,
        lease,
        deadline_at_ms=now_ms() + 60000,
        reserve_retry=True,
        retry_limit=32,
    )
    for status in [TaskStatus.IN_PROGRESS, TaskStatus.FINALIZING]:
        stored = await store.transition(
            task.id,
            expected={stored.task.status},
            expected_revision=stored.revision,
            patch={"status": status},
        )
    return client, store, stored


@pytest.mark.parametrize(
    "terminal", [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
)
async def test_terminal_write_retries_shared_credit_cas(monkeypatch, terminal):
    client, store, stored = await prepared()
    original = client.txn
    credit_key = store._retry_counter_key(stored.task.pool_id, "credits")
    conflicts = 0

    async def racing_txn(compare, success, failure=()):
        nonlocal conflicts
        if conflicts < 2 and any(
            _decode(c["key"]).decode() == credit_key for c in compare
        ):
            # Another task reserves and then releases a retry credit between
            # our counter read and transaction: value unchanged, revision new.
            value = int((await client.get(credit_key)).value)
            await original([], [client.put(credit_key, str(value + 1))])
            await original([], [client.put(credit_key, str(value))])
            conflicts += 1
        return await original(compare, success, failure)

    monkeypatch.setattr(client, "txn", racing_txn)
    result = await store.transition(
        stored.task.id,
        expected={TaskStatus.FINALIZING},
        expected_revision=stored.revision,
        patch={"status": terminal},
        release_lease=True,
    )
    assert conflicts == 2
    assert result.task.status == terminal
    assert await store.list_leases(stored.task.pool_id) == []
    assert int((await client.get(credit_key)).value) == 0


async def test_terminal_retry_never_overwrites_changed_task(monkeypatch):
    client, store, stored = await prepared()
    original = client.txn
    changed = False

    async def racing_txn(compare, success, failure=()):
        nonlocal changed
        if not changed:
            changed = True
            task = stored.task.to_dict()
            task["cancel_requested_at_ms"] = now_ms()
            await original(
                [], [client.put(store._task_key(stored.task.id), store._encode(task))]
            )
        return await original(compare, success, failure)

    monkeypatch.setattr(client, "txn", racing_txn)
    with pytest.raises(StoreConflict):
        await store.transition(
            stored.task.id,
            expected={TaskStatus.FINALIZING},
            expected_revision=stored.revision,
            patch={"status": TaskStatus.COMPLETED},
            release_lease=True,
        )
    current = await store.get_task(stored.task.id)
    assert current.task.cancel_requested_at_ms is not None
    assert current.task.status == TaskStatus.FINALIZING


@pytest.mark.parametrize("release", [False, True])
async def test_terminal_ledger_uses_one_snapshot(monkeypatch, release):
    client, store, stored = await prepared()
    gets, batches = [], []
    original_get, original_many = client.get, client.get_many

    async def get(key):
        gets.append(key)
        return await original_get(key)

    async def many(keys, **kwargs):
        batches.append(list(keys))
        return await original_many(keys, **kwargs)

    monkeypatch.setattr(client, "get", get)
    monkeypatch.setattr(client, "get_many", many)
    result = await store.transition(
        stored.task.id,
        expected={TaskStatus.FINALIZING},
        expected_revision=stored.revision,
        patch={"status": TaskStatus.COMPLETED},
        release_lease=release,
    )
    assert result.task.status == TaskStatus.COMPLETED
    assert gets == [store._task_key(stored.task.id)]
    assert len(batches) == 1
    assert len(batches[0]) == (4 if release else 3)
    assert bool(await store.list_leases(stored.task.pool_id)) is not release


async def test_terminal_snapshot_rejects_changed_hint(monkeypatch):
    client, store, stored = await prepared()
    original_many = client.get_many

    async def many(keys, **kwargs):
        task = stored.task.to_dict()
        task["cancel_requested_at_ms"] = now_ms()
        await client.txn(
            [], [client.put(store._task_key(task["id"]), store._encode(task))]
        )
        return await original_many(keys, **kwargs)

    monkeypatch.setattr(client, "get_many", many)
    with pytest.raises(StoreConflict):
        await store.transition(
            stored.task.id,
            expected={TaskStatus.FINALIZING},
            expected_revision=stored.revision,
            patch={"status": TaskStatus.COMPLETED},
            release_lease=True,
        )
    current = await store.get_task(stored.task.id)
    assert current.task.status == TaskStatus.FINALIZING
    assert current.task.cancel_requested_at_ms is not None
    assert len(await store.list_leases(stored.task.pool_id)) == 1


@pytest.mark.parametrize("count", ["0", "-1", None])
async def test_terminal_snapshot_preserves_counter_invariants(count):
    client, store, stored = await prepared()
    key = store._retry_counter_key(stored.task.pool_id, "credits")
    await client.txn(
        [], [client.delete(key) if count is None else client.put(key, count)]
    )
    with pytest.raises(RuntimeError):
        await store.transition(
            stored.task.id,
            expected={TaskStatus.FINALIZING},
            expected_revision=stored.revision,
            patch={"status": TaskStatus.COMPLETED},
            release_lease=True,
        )
    assert (await store.get_task(stored.task.id)).revision == stored.revision
    assert len(await store.list_leases(stored.task.pool_id)) == 1


async def test_terminal_snapshot_no_credit_and_foreign_lease():
    client, store, stored = await prepared()
    lease_key = store._lease_key(stored.task.pool_id, stored.task.worker_key)
    lease = json.loads((await client.get(lease_key)).value)
    lease["task_id"] = "other-task"
    await client.txn(
        [],
        [
            client.delete(store._retry_credit_key(stored.task)),
            client.put(store._retry_counter_key(stored.task.pool_id, "credits"), "0"),
            client.put(lease_key, store._encode(lease)),
        ],
    )
    result = await store.transition(
        stored.task.id,
        expected={TaskStatus.FINALIZING},
        expected_revision=stored.revision,
        patch={"status": TaskStatus.COMPLETED},
        release_lease=True,
    )
    assert result.task.status == TaskStatus.COMPLETED
    assert json.loads((await client.get(lease_key)).value)["task_id"] == "other-task"


async def test_terminal_snapshot_missing_task(monkeypatch):
    client, store, stored = await prepared()
    original_many = client.get_many

    async def many(keys, **kwargs):
        await client.txn([], [client.delete(store._task_key(stored.task.id))])
        return await original_many(keys, **kwargs)

    monkeypatch.setattr(client, "get_many", many)
    with pytest.raises(KeyError):
        await store.transition(
            stored.task.id,
            expected={TaskStatus.FINALIZING},
            expected_revision=stored.revision,
            patch={"status": TaskStatus.COMPLETED},
            release_lease=True,
        )
    assert len(await store.list_leases(stored.task.pool_id)) == 1


async def test_concurrent_terminal_snapshots_conserve_credits(monkeypatch):
    client = FakeEtcd()
    store = EtcdTaskStore(
        client, prefix="/isolated/batch-concurrency", deployment_id="cas"
    )
    await store.register_gateway("generation", ttl_s=15)
    tasks = []
    for index in range(8):
        task = _task(f"video-batch-{index}")
        stored, _ = await store.create_task(
            task, principal_hash="p", idempotency_hash=None, queue_limit=16
        )
        stored = await store.reserve(
            stored,
            _lease(task, instance_id=index),
            deadline_at_ms=now_ms() + 60000,
            reserve_retry=True,
            retry_limit=16,
        )
        for status in [TaskStatus.IN_PROGRESS, TaskStatus.FINALIZING]:
            stored = await store.transition(
                task.id,
                expected={stored.task.status},
                expected_revision=stored.revision,
                patch={"status": status},
            )
        tasks.append(stored)
    original_many = client.get_many
    barrier = asyncio.Event()
    first_reads = 0

    async def many(keys, **kwargs):
        nonlocal first_reads
        snapshot = await original_many(keys, **kwargs)
        if first_reads < len(tasks):
            first_reads += 1
            if first_reads == len(tasks):
                barrier.set()
            await barrier.wait()  # all eight deliberately see the same counter
        return snapshot

    monkeypatch.setattr(client, "get_many", many)
    results = await asyncio.wait_for(
        asyncio.gather(
            *[
                store.transition(
                    t.task.id,
                    expected={TaskStatus.FINALIZING},
                    expected_revision=t.revision,
                    patch={"status": TaskStatus.COMPLETED},
                    release_lease=True,
                )
                for t in tasks
            ]
        ),
        timeout=5,
    )
    assert all(t.task.status == TaskStatus.COMPLETED for t in results)
    assert await store.retry_budget_used(tasks[0].task.pool_id) == 0
    assert await store.list_leases(tasks[0].task.pool_id) == []
    for task in tasks:
        assert await client.get(store._retry_credit_key(task.task)) is None
