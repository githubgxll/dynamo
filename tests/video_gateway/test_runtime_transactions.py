# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from dataclasses import replace

import pytest

from dingo.video_gateway.task_runtime.repository import (
    AttemptFence,
    AttemptSnapshot,
    CasStatus,
    KeySnapshot,
    OwnerSnapshot,
    RuntimeKeys,
    TaskRef,
    TaskSnapshot,
    ValidatedOutcome,
    WorkerRef,
    WorkerSnapshot,
)
from dingo.video_gateway.task_runtime.transactions import (
    claim_transaction,
    encode,
    execute_runtime_transaction,
    settle_transaction,
)
from tests.video_gateway.test_etcd_task_store import FakeEtcd


async def put(client, key, value):
    await client.txn([], [client.put(key, value)])
    return await snapshot(client, key)


async def snapshot(client, key):
    v = await client.get(key)
    return KeySnapshot(v.key, v.mod_revision, v.value)


async def setup(client, count=1):
    route = await put(client, "/route", b"worker-v2")
    owner = OwnerSnapshot("owner-1", await put(client, "/owner", b"owner-1"))
    worker = WorkerSnapshot(
        WorkerRef("pool", "worker", "boot-1", "qwen"),
        await put(client, "/worker", b"ready-boot-1"),
        16,
    )
    await put(client, "/queued", str(count))
    for i in range(count):
        await put(client, f"/task/{i}", encode({"id": f"task-{i}", "status": "queued"}))
        await put(client, f"/queue/{i}", f"task-{i}")
    return route, owner, worker


async def plan(client, route, owner, worker, i=0, slot=0):
    task = TaskSnapshot(
        TaskRef("deployment", "context_ir", f"task-{i}"),
        await snapshot(client, f"/task/{i}"),
    )
    fence = AttemptFence(task.ref, worker.ref, 1, f"{i+1:032x}")
    compare, writes = claim_transaction(
        client,
        task=task,
        worker=worker,
        owner=owner,
        route=route,
        queue=await snapshot(client, f"/queue/{i}"),
        queued=await snapshot(client, "/queued"),
        slot_id=slot,
        fence=fence,
        task_json={"id": task.ref.task_id, "status": "in_progress"},
    )
    return fence, compare, writes


async def attempt_snapshot(client, fence, owner, i=0, slot=0):
    keys = RuntimeKeys("deployment")
    return AttemptSnapshot(
        fence,
        await snapshot(client, keys.attempt(fence)),
        TaskSnapshot(fence.task, await snapshot(client, f"/task/{i}")),
        await snapshot(client, keys.slot(fence.worker, slot)),
        await snapshot(client, keys.active(fence.task)),
        owner,
    )


async def test_same_task_claimed_once_and_all_writes_are_atomic():
    client = FakeEtcd()
    route, owner, worker = await setup(client)
    fence, compares, writes = await plan(client, route, owner, worker)
    assert not any("lease" in op.get("request_put", {}) for op in writes)
    results = await asyncio.gather(
        execute_runtime_transaction(client, compares, writes),
        execute_runtime_transaction(client, compares, writes),
    )
    assert [r.status for r in results].count(CasStatus.APPLIED) == 1
    assert [r.status for r in results].count(CasStatus.CONFLICT) == 1
    assert await client.get("/queue/0") is None
    assert (await client.get("/queued")).value == b"0"
    keys = RuntimeKeys("deployment")
    records = [
        await client.get(k)
        for k in (
            "/task/0",
            keys.slot(worker.ref, 0),
            keys.attempt(fence),
            keys.active(fence.task),
        )
    ]
    assert len({r.mod_revision for r in records}) == 1


@pytest.mark.parametrize(
    "changed", ["/route", "/worker", "/owner", "/queue/0", "/task/0", "/queued"]
)
async def test_changed_observation_cannot_claim_or_partially_dequeue(changed):
    client = FakeEtcd()
    route, owner, worker = await setup(client)
    _, compare, writes = await plan(client, route, owner, worker)
    await put(client, changed, b"changed")
    before = dict(client.values)
    assert (
        await execute_runtime_transaction(client, compare, writes)
    ).status is CasStatus.CONFLICT
    assert client.values == before


async def test_two_gateway_contenders_share_sixteen_physical_slots():
    client = FakeEtcd()
    route, owner1, worker = await setup(client, 17)
    owner2 = OwnerSnapshot("owner-2", await put(client, "/owner-2", b"owner-2"))
    keys = RuntimeKeys("deployment")

    async def scheduler(owner, alias):
        selected_worker = replace(worker, ref=replace(worker.ref, pool_id=alias))
        for i in range(17):
            while await client.get(f"/queue/{i}") is not None:
                free = [
                    s
                    for s in range(16)
                    if await client.get(keys.slot(worker.ref, s)) is None
                ]
                if not free:
                    return
                _, compares, writes = await plan(
                    client, route, owner, selected_worker, i, free[0]
                )
                await asyncio.sleep(0)  # Force stale snapshots across contenders.
                await execute_runtime_transaction(client, compares, writes)

    await asyncio.gather(scheduler(owner1, "fl"), scheduler(owner2, "ref"))
    assert len(await client.range(keys.root + "/slots/", prefix=True)) == 16
    assert len(await client.range(keys.root + "/attempts/", prefix=True)) == 16
    assert len(await client.range("/queue/", prefix=True)) == 1
    assert (await client.get("/queued")).value == b"1"


async def test_capacity_and_counter_underflow_rejected_before_etcd():
    client = FakeEtcd()
    route, owner, worker = await setup(client)
    with pytest.raises(ValueError, match="slot_id"):
        await plan(client, route, owner, worker, slot=16)
    await put(client, "/queued", b"0")
    with pytest.raises(ValueError, match="underflow"):
        await plan(client, route, owner, worker)
    assert await client.get("/queue/0") is not None


async def test_lost_transaction_ack_is_unknown_and_not_retried():
    client = FakeEtcd()
    route, owner, worker = await setup(client)
    fence, compare, writes = await plan(client, route, owner, worker)
    txn = client.txn
    calls = 0

    async def lost_ack(*args):
        nonlocal calls
        calls += 1
        await txn(*args)
        raise TimeoutError("response lost after commit")

    client.txn = lost_ack
    result = await execute_runtime_transaction(client, compare, writes)
    assert result.status is CasStatus.UNKNOWN
    assert calls == 1
    assert await client.get(RuntimeKeys("deployment").attempt(fence)) is not None
    assert await client.get("/queue/0") is None


async def test_settlement_preserves_cancelled_task_and_deduplicates_release():
    client = FakeEtcd()
    route, owner, worker = await setup(client)
    fence, compare, writes = await plan(client, route, owner, worker)
    await execute_runtime_transaction(client, compare, writes)
    keys = RuntimeKeys("deployment")
    await put(client, "/task/0", encode({"status": "cancelled"}))
    assert await client.get(keys.slot(worker.ref, 0)) is not None
    attempt = await attempt_snapshot(client, fence, owner)
    outcome = ValidatedOutcome("cancelled", None, b'{"complete":false}')
    with pytest.raises(ValueError, match="cancellation"):
        settle_transaction(
            client, attempt=attempt, outcome=outcome, task_json={"status": "completed"}
        )
    compares, writes = settle_transaction(
        client, attempt=attempt, outcome=outcome, task_json={"status": "cancelled"}
    )
    assert (
        await execute_runtime_transaction(client, compares, writes)
    ).status is CasStatus.APPLIED
    assert await client.get(keys.slot(worker.ref, 0)) is None
    assert await client.get(keys.active(fence.task)) is None
    saved = json.loads((await client.get(keys.attempt(fence))).value)
    assert saved["usage"] == {"complete": False}
    before = dict(client.values)
    assert (
        await execute_runtime_transaction(client, compares, writes)
    ).status is CasStatus.CONFLICT
    assert client.values == before


@pytest.mark.parametrize("mutation", ["owner", "slot", "task", "attempt"])
async def test_stale_settlement_cannot_publish_or_release(mutation):
    client = FakeEtcd()
    route, owner, worker = await setup(client)
    fence, compare, writes = await plan(client, route, owner, worker)
    await execute_runtime_transaction(client, compare, writes)
    attempt = await attempt_snapshot(client, fence, owner)
    compares, writes = settle_transaction(
        client,
        attempt=attempt,
        outcome=ValidatedOutcome("failed", None, b"{}"),
        task_json={"status": "failed"},
    )
    target = {
        "owner": owner.record.key,
        "slot": attempt.slot.key,
        "task": attempt.task.record.key,
        "attempt": attempt.record.key,
    }[mutation]
    await put(client, target, b"new-owner-or-execution")
    before = dict(client.values)
    assert (
        await execute_runtime_transaction(client, compares, writes)
    ).status is CasStatus.CONFLICT
    assert client.values == before


async def test_owner_lease_expiry_does_not_delete_execution_slots():
    client = FakeEtcd()
    route, owner, worker = await setup(client)
    lease = await client.lease_grant(30)
    await client.txn([], [client.put("/owner", "owner-1", lease_id=lease.lease_id)])
    owner = OwnerSnapshot("owner-1", await snapshot(client, "/owner"))
    fence, compare, writes = await plan(client, route, owner, worker)
    await execute_runtime_transaction(client, compare, writes)
    await client.lease_revoke(lease.lease_id)
    keys = RuntimeKeys("deployment")
    assert await client.get("/owner") is None
    assert await client.get(keys.slot(worker.ref, 0)) is not None
    assert await client.get(keys.attempt(fence)) is not None
