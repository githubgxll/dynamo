import asyncio
from types import SimpleNamespace

import pytest

from dingo.video_gateway.dispatcher import VideoDispatcher
from dingo.video_gateway.errors import StoreConflict, StoreUnavailable
from dingo.video_gateway.models import TaskStatus
from dingo.video_gateway.result_handoff import read_handoff
from dingo.video_gateway.telemetry import GatewayTelemetry
from tests.video_gateway.test_result_handoff import ARTIFACT, setup


def dispatcher_for(store, config, monkeypatch):
    import dingo.video_gateway.dispatcher as module

    monkeypatch.setattr(module, "_HANDOFF_RETRY_INITIAL_S", 0.001)
    monkeypatch.setattr(module, "_HANDOFF_RETRY_MAX_S", 0.002)
    obj = object.__new__(VideoDispatcher)
    obj.store = store
    obj.config = config
    obj.generation = "generation"
    obj.telemetry = GatewayTelemetry()
    obj._stop = asyncio.Event()
    return obj, SimpleNamespace(config=config.pools[0])


@pytest.mark.parametrize("kind", ["memory", "etcd"])
@pytest.mark.parametrize(
    "fault",
    [
        "reply_lost",
        "local_error_after_commit",
        "write_unavailable",
        "read_unavailable",
        "cas",
        "failure_reply_lost",
        "cancellation_reply_lost",
        "telemetry",
    ],
)
async def test_reconcile_handoff_and_terminal_outcomes(
    kind, fault, make_gateway_config, monkeypatch
):
    store, client, active, _ = await setup(kind)
    dispatcher, pool = dispatcher_for(store, make_gateway_config(), monkeypatch)
    original = store.transition
    original_get = store.get_task
    writes, reads = 0, 0

    async def read(task_id):
        nonlocal reads
        reads += 1
        if fault == "read_unavailable" and reads <= 3:
            raise StoreUnavailable("read timeout")
        return await original_get(task_id)

    async def transition(*args, **kwargs):
        nonlocal writes
        writes += 1
        if fault in {"failure_reply_lost", "cancellation_reply_lost"} and writes == 1:
            if fault == "cancellation_reply_lost":
                # Use the original writer to avoid recursively invoking this wrapper.
                await original(
                    active.task.id,
                    expected={TaskStatus.IN_PROGRESS},
                    patch={"cancel_requested_at_ms": 1},
                )
            raise TypeError("deterministic error before handoff")
        if fault == "write_unavailable" and writes <= 3:
            raise StoreUnavailable("write timeout")
        if fault == "cas" and writes <= 3:
            raise StoreConflict("contended")
        result = await original(*args, **kwargs)
        if (fault == "reply_lost" and writes == 1) or (
            fault.endswith("reply_lost") and writes == 2
        ):
            raise StoreUnavailable("response lost after commit")
        if fault == "local_error_after_commit" and writes == 1:
            raise TypeError("decode failed after commit")
        return result

    monkeypatch.setattr(store, "transition", transition)
    monkeypatch.setattr(store, "get_task", read)
    if fault == "telemetry":

        def fail(*args, **kwargs):
            raise RuntimeError("telemetry unavailable")

        monkeypatch.setattr(dispatcher.telemetry, "record_transition", fail)
        monkeypatch.setattr(dispatcher.telemetry, "increment", fail)
    result = await dispatcher._commit_result_handoff(pool, active.task, ARTIFACT, 1, {})
    stored = await original_get(active.task.id)
    if fault in {"failure_reply_lost", "cancellation_reply_lost"}:
        assert result is None
        assert stored.task.status == (
            TaskStatus.FAILED if fault == "failure_reply_lost" else TaskStatus.CANCELLED
        )
    else:
        assert result.task.status == TaskStatus.FINALIZING
        assert read_handoff(stored.task)["artifact"] == ARTIFACT
    assert not await store.list_leases(active.task.pool_id)
    if client:
        assert await store.retry_budget_used(active.task.pool_id) == 0
        assert (
            await client.get(
                store._lease_heartbeat_key(active.task.pool_id, active.task.worker_key)
            )
            is None
        )


async def test_broken_terminal_writer_has_recovery_exit(
    make_gateway_config, monkeypatch
):
    store, _, active, _ = await setup("memory")
    dispatcher, pool = dispatcher_for(store, make_gateway_config(), monkeypatch)
    calls = 0

    async def broken(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise TypeError("permanent serializer defect")

    monkeypatch.setattr(store, "transition", broken)
    with pytest.raises(RuntimeError, match="reconciliation failed"):
        await dispatcher._commit_result_handoff(pool, active.task, ARTIFACT, 1, {})
    assert calls == 2  # one handoff, one guarded terminal write; never spin
    assert (await store.get_task(active.task.id)).task.status == TaskStatus.IN_PROGRESS


async def test_storage_outage_retry_is_interruptible(make_gateway_config, monkeypatch):
    store, _, active, _ = await setup("memory")
    dispatcher, pool = dispatcher_for(store, make_gateway_config(), monkeypatch)
    entered = asyncio.Event()

    async def unavailable(*args):
        entered.set()
        raise StoreUnavailable("outage")

    monkeypatch.setattr(store, "get_task", unavailable)
    running = asyncio.create_task(
        dispatcher._commit_result_handoff(pool, active.task, ARTIFACT, 1, {})
    )
    await entered.wait()
    dispatcher._stop.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(running, 1)


@pytest.mark.parametrize("kind", ["memory", "etcd"])
async def test_fenced_terminal_cannot_change_attempt(kind):
    store, _, active, _ = await setup(kind)
    with pytest.raises(ValueError, match="identity"):
        await store.transition(
            active.task.id,
            expected={TaskStatus.IN_PROGRESS},
            expected_revision=active.revision,
            patch={"status": TaskStatus.FAILED, "attempt": 99},
            release_lease=True,
            release_execution=True,
        )
    assert (await store.get_task(active.task.id)).task.status == TaskStatus.IN_PROGRESS
    assert len(await store.list_leases(active.task.pool_id)) == 1
