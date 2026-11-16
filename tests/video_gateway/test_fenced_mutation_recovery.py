# SPDX-License-Identifier: Apache-2.0
"""Real emitted etcd transactions with lost replies and failed read-back."""
from types import SimpleNamespace

import pytest

from dingo.video_gateway.models import TaskStatus, now_ms
from dingo.video_gateway.errors import StoreConflict
from dingo.video_gateway.task_store import EtcdTaskStore
from dingo.video_gateway.task_runtime.transactions import UnconfirmedTransaction
from tests.video_gateway.test_task_store import _task, _lease
from tests.video_gateway.test_context_ir import task as compiler_task
from tests.video_gateway.test_worker_repository import setup as compiler_setup
from dingo.video_gateway.context_ir.worker_repository import WorkerTaskRepository
from tests.video_gateway.test_etcd_task_store import FakeEtcd


async def repository(kind):
    if kind == "compiler":
        client, store, worker = await compiler_setup()
        stored, _ = await compiler_task(store, "task-1")
        repo = WorkerTaskRepository(store, "a", "a")
        return SimpleNamespace(
            client=client,
            store=store,
            stored=stored,
            claim=lambda: repo.claim(stored, worker, 0),
            depth=lambda: client.get(store.key("counters/h3/queued")),
            read=lambda: store.get_task("task-1"),
            repo=repo,
        )
    client = FakeEtcd()
    store = EtcdTaskStore(client, prefix="/test/video", deployment_id="test")
    task = _task("task-1")
    if kind == "sr":
        task.normalized_request["operation"] = "sr_only"
    stored, _ = await store.create_task(
        task, principal_hash="p", idempotency_hash=None, queue_limit=8
    )
    lease = _lease(task)
    lease.execution_token = "a" * 32
    await store.register_gateway(lease.owner_generation, ttl_s=15)
    return SimpleNamespace(
        client=client,
        store=store,
        stored=stored,
        claim=lambda: store.reserve(stored, lease, deadline_at_ms=now_ms() + 60_000),
        depth=lambda: client.get(store._counter_key(task.pool_id)),
        read=lambda: store.get_task(task.id),
    )


def lose_one_reply(client, *, before_commit=False, break_readback=False):
    txn, get_many = client.txn, client.get_many
    count = 0

    async def fault(compare, writes, *args):
        nonlocal count
        count += 1
        if count > 1:
            return await txn(compare, writes, *args)
        if not before_commit:
            await txn(compare, writes, *args)
        raise TimeoutError("injected transaction ACK loss")

    async def unavailable(*args, **kwargs):
        raise ConnectionError("injected read-back outage")

    client.txn = fault
    if break_readback:
        client.get_many = unavailable
    return SimpleNamespace(
        restore=lambda: setattr(client, "get_many", get_many), count=lambda: count
    )


@pytest.mark.parametrize("kind", ["h3", "sr", "compiler"])
async def test_claim_committed_with_lost_ack_is_confirmed_once(kind):
    case = await repository(kind)
    fault = lose_one_reply(case.client)
    claimed = await case.claim()
    assert claimed.task.attempt == 1 and fault.count() == 1
    assert int((await case.depth()).value) == 0
    if kind != "compiler":
        assert claimed.task.worker_lease_id in case.client.native_leases
        await case.store.heartbeat_lease(
            claimed.task.pool_id,
            claimed.task.worker_key,
            claimed.task.id,
            claimed.task.worker_lease_id,
        )


@pytest.mark.parametrize("kind", ["h3", "sr", "compiler"])
@pytest.mark.parametrize("before_commit", [True, False])
async def test_unconfirmed_claim_keeps_identity_and_resolves_without_new_attempt(
    kind, before_commit
):
    case = await repository(kind)
    fault = lose_one_reply(
        case.client, before_commit=before_commit, break_readback=True
    )
    with pytest.raises(UnconfirmedTransaction) as pending:
        await case.claim()
    leases = set(case.client.native_leases)
    before = await case.read()
    assert before.task.attempt == (0 if before_commit else 1)
    fault.restore()
    confirmed = await pending.value.resolve()
    assert confirmed.task.attempt == 1
    assert int((await case.depth()).value) == 0
    assert case.client.native_leases == leases
    assert fault.count() == (2 if before_commit else 1)


@pytest.mark.parametrize("kind", ["h3", "sr"])
async def test_cancel_racing_unknown_claim_cannot_restore_old_task_or_revoke_guard(
    kind,
):
    case = await repository(kind)
    fault = lose_one_reply(case.client, break_readback=True)
    with pytest.raises(UnconfirmedTransaction) as pending:
        await case.claim()
    current = await case.read()
    await case.store.request_cancel(current.task.id)
    fault.restore()
    confirmed = await pending.value.resolve()
    assert confirmed.task.cancel_requested_at_ms is not None
    assert confirmed.task.execution_token == current.task.execution_token
    assert confirmed.task.worker_lease_id in case.client.native_leases


async def test_compiler_unknown_settlement_does_not_report_running_as_settled():
    case = await repository("compiler")
    claimed = await case.claim()
    fault = lose_one_reply(case.client, before_commit=True)
    with pytest.raises(UnconfirmedTransaction) as pending:
        await case.repo.settle(claimed, "failed", usage={"complete": False})
    assert (await case.read()).task.status == "in_progress"
    assert await case.client.range("/dingo/task-runtime/v2/test/slots/", prefix=True)
    settled = await pending.value.resolve()
    assert settled.task.status == "failed" and settled.task.usage == {"complete": False}
    assert not await case.client.range(
        "/dingo/task-runtime/v2/test/slots/", prefix=True
    )
    assert fault.count() == 2


async def test_compiler_lost_settlement_reply_preserves_cancel_and_usage():
    case = await repository("compiler")
    claimed = await case.claim()
    cancelled = await case.store.finish(claimed, "cancelled")
    fault = lose_one_reply(case.client)
    settled = await case.repo.settle(cancelled, "failed", usage={"complete": False})
    assert settled.task.status == "cancelled" and settled.task.usage == {
        "complete": False
    }
    assert not await case.client.range(
        "/dingo/task-runtime/v2/test/slots/", prefix=True
    )
    assert fault.count() == 1


@pytest.mark.parametrize("kind", ["h3", "sr"])
async def test_terminal_transition_lost_reply_does_not_repeat_release(kind):
    case = await repository(kind)
    claimed = await case.claim()
    running = await case.store.transition(
        claimed.task.id,
        expected={TaskStatus.DISPATCHING},
        patch={"status": TaskStatus.IN_PROGRESS},
    )
    fault = lose_one_reply(case.client)
    finished = await case.store.transition(
        running.task.id,
        expected={TaskStatus.IN_PROGRESS},
        expected_revision=running.revision,
        patch={"status": TaskStatus.FAILED},
        release_lease=True,
    )
    assert finished.task.status == TaskStatus.FAILED and fault.count() == 1
    assert not await case.store.list_leases(finished.task.pool_id)


async def test_expired_heartbeat_does_not_hide_a_committed_claim():
    case = await repository("h3")
    fault = lose_one_reply(case.client, break_readback=True)
    with pytest.raises(UnconfirmedTransaction) as pending:
        await case.claim()
    current = await case.read()
    await case.client.lease_revoke(current.task.worker_lease_id)
    fault.restore()
    confirmed = await pending.value.resolve()
    assert confirmed.task.execution_token == current.task.execution_token
    with pytest.raises(StoreConflict, match="heartbeat no longer exists"):
        await case.store.heartbeat_lease(
            confirmed.task.pool_id,
            confirmed.task.worker_key,
            confirmed.task.id,
            confirmed.task.worker_lease_id,
        )


@pytest.mark.parametrize("kind", ["h3", "sr", "compiler"])
async def test_takeover_ack_loss_preserves_attempt_and_reuses_original_execution(kind):
    case = await repository(kind)
    original = await case.claim()
    if kind == "compiler":
        before = await case.repo.attempt(original)
        await case.client.txn([], [case.client.delete(case.store.key("owners/a"))])
        replacement = WorkerTaskRepository(case.store, "b", "b")
        fault = lose_one_reply(case.client)
        adopted = await replacement.adopt(original)
        after = await replacement.attempt(adopted)
        assert before.fence == after.fence
        assert adopted.task.owner_id == "b"
    else:
        owner = await case.client.get(case.store._gateway_key(original.task.owner_generation))
        await case.client.lease_revoke(owner.lease)
        await case.client.lease_revoke(original.task.worker_lease_id)
        await case.store.register_gateway("replacement", ttl_s=15)
        fault = lose_one_reply(case.client)
        adopted = await case.store.claim_orphaned_active(original, new_owner_generation="replacement")
        assert adopted.task.owner_generation == "replacement"
        assert adopted.task.execution_token == original.task.execution_token
        assert adopted.task.worker_lease_id in case.client.native_leases
    assert adopted.task.attempt == original.task.attempt == 1
    assert fault.count() == 1
