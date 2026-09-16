"""A saturated pool must not fetch queued task bodies just to reject dispatch."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dingo.video_gateway.adapters import create_adapter
from dingo.video_gateway.artifact_store import FileArtifactStore
from dingo.video_gateway.dispatcher import VideoDispatcher
from dingo.video_gateway.task_store import StoredTask, worker_key
from tests.video_gateway.test_dispatcher import (
    FakeClient,
    FakeContext,
    WatchMemoryTaskStore,
)
from tests.video_gateway.test_task_store import _task


def stack(make_gateway_config):
    config = make_gateway_config()
    store = WatchMemoryTaskStore()
    client = FakeClient()
    dispatcher = VideoDispatcher(
        config,
        store,
        FileArtifactStore(config.artifact_store.root),
        {"fl-pool": client},
        {"fl-pool": create_adapter(config.pools[0])},
        context_factory=FakeContext,
    )
    pool = dispatcher.pools["fl-pool"]
    pool.instance_ids = [7]
    pool.lease_watch_healthy = True
    for name, value in [
        ("list_queued", []),
        ("retry_queue_depth", 0),
        ("retry_budget_used", 0),
        ("queue_depth", 0),
        ("reserve", None),
    ]:
        setattr(store, name, AsyncMock(return_value=value))
    return dispatcher, pool, store, client


@pytest.mark.parametrize(
    "reason", ["no_workers", "lease_watch_unhealthy", "no_free_worker"]
)
async def test_unusable_pool_skips_all_queue_and_counter_reads(
    make_gateway_config, reason
):
    dispatcher, pool, store, client = stack(make_gateway_config)
    if reason == "no_workers":
        pool.instance_ids = []
    elif reason == "lease_watch_unhealthy":
        pool.lease_watch_healthy = False
    else:
        key = worker_key(pool.config.backend_target, 7)
        pool.lease_cache[key] = SimpleNamespace(worker_key=key)
    capacity = (await dispatcher.memory_budget.snapshot()).capacity_bytes
    assert await dispatcher.memory_budget.try_acquire("holder", capacity)
    assert not await dispatcher.memory_budget.try_acquire("stalled", 1)
    pool.budget_waiter_id = "stalled"
    for _ in range(3):
        assert not await dispatcher._dispatch_once(pool)
    for name in [
        "list_queued",
        "retry_queue_depth",
        "retry_budget_used",
        "queue_depth",
        "reserve",
    ]:
        getattr(store, name).assert_not_awaited()
    assert pool.budget_waiter_id is None
    assert (await dispatcher.memory_budget.snapshot()).waiting_tasks == 0
    assert not client.calls


@pytest.mark.parametrize("race", ["leased", "watch_lost", "none"])
async def test_free_worker_still_rechecks_and_uses_reserve_cas(
    make_gateway_config, race
):
    dispatcher, pool, store, client = stack(make_gateway_config)
    task = _task("task", pool_id=pool.config.pool_id)
    task.backend_target = pool.config.backend_target
    stored = StoredTask(task, 1)

    async def read_queue(*_args, **_kwargs):
        if race == "leased":
            key = worker_key(pool.config.backend_target, 7)
            pool.lease_cache[key] = SimpleNamespace(worker_key=key)
        elif race == "watch_lost":
            pool.lease_watch_healthy = False
        return [stored]

    store.list_queued.side_effect = read_queue
    outcome = await dispatcher._dispatch_once(pool)
    if race == "none":
        # None is a lost reservation CAS, not authority to call the Worker.
        store.reserve.assert_awaited_once()
        assert outcome is True
        assert (await dispatcher.memory_budget.snapshot()).used_bytes == 0
    else:
        store.reserve.assert_not_awaited()
        assert outcome is False
    assert not client.calls


async def test_retry_gauges_refresh_without_dispatch(make_gateway_config, monkeypatch):
    from dingo.video_gateway.api import _SERVICE_KEY
    from tests.video_gateway.test_api import _client

    client = await _client(make_gateway_config, FakeClient(available=False))
    try:
        service = client.server.app[_SERVICE_KEY]
        monkeypatch.setattr(
            service.store, "retry_queue_depth", AsyncMock(return_value=2), raising=False
        )
        monkeypatch.setattr(
            service.store, "retry_budget_used", AsyncMock(return_value=3), raising=False
        )
        monkeypatch.setattr(service.store, "queue_depth", AsyncMock(return_value=5))
        response = await client.get("/metrics")
        body = await response.text()
        assert 'dingo_video_retry_waiting_tasks{pool="fl-pool"} 2' in body
        assert 'dingo_video_retry_credits_used{pool="fl-pool"} 3' in body
        assert 'dingo_video_normal_queue_depth{pool="fl-pool"} 3' in body
    finally:
        await client.close()
