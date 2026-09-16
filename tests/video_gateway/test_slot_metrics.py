"""Physical Workers, admission leases and engine concurrency are distinct."""

import dataclasses
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dingo.video_gateway.task_store import worker_key
from tests.video_gateway.test_dispatch_idle import stack


def setup_pool(make_gateway_config, *, n=2, p=1):
    dispatcher, pool, store, _ = stack(make_gateway_config)
    pool.config = dataclasses.replace(
        pool.config,
        scheduling=dataclasses.replace(
            pool.config.scheduling, worker_capacity=n, worker_prefetch_capacity=p
        ),
    )
    pool.capacity_cache = {7: (n + p, float("inf"))}
    pool.prefetch_instances = {7} if p else set()
    pool.discovery_healthy = True
    return dispatcher, pool, store


def lease(pool, slot=0, *, instance=7, state="active"):
    return SimpleNamespace(
        worker_key=worker_key(pool.config.backend_target, instance, slot),
        worker_instance_id=instance,
        backend_target=pool.config.backend_target,
        state=state,
    )


def test_two_slots_are_one_busy_worker(make_gateway_config):
    d, p, _ = setup_pool(make_gateway_config)
    s = d.pool_capacity_snapshot(p.config.pool_id, [lease(p, 0), lease(p, 1)])
    assert s == dict(
        workers=1,
        worker_busy=1,
        worker_execution_capacity=2,
        worker_prefetch_capacity=1,
        worker_admission_capacity=3,
        worker_slots_busy=2,
        worker_slots_quarantined=0,
        worker_slots_free=1,
        worker_unmapped_leases=0,
        worker_capacity_view_healthy=1,
    )


def test_quarantine_and_disappeared_worker_do_not_inflate_capacity(make_gateway_config):
    d, p, _ = setup_pool(make_gateway_config)
    leases = [lease(p, 0), lease(p, 1, state="quarantined"), lease(p, instance=8)]
    s = d.pool_capacity_snapshot(p.config.pool_id, leases)
    assert s["worker_busy"] == 1
    assert s["worker_slots_busy"] == s["worker_slots_quarantined"] == 1
    assert s["worker_slots_free"] == s["worker_unmapped_leases"] == 1
    p.instance_ids = []
    s = d.pool_capacity_snapshot(p.config.pool_id, leases)
    assert (
        s["worker_busy"]
        == s["worker_admission_capacity"]
        == s["worker_slots_free"]
        == 0
    )
    assert s["worker_unmapped_leases"] == 3
    assert not d.has_workers(p.config.pool_id)


@pytest.mark.parametrize("n,prefetch", [(1, 0), (2, 0), (2, 1)])
def test_saturation_does_not_disable_model(make_gateway_config, n, prefetch):
    d, p, _ = setup_pool(make_gateway_config, n=n, p=prefetch)
    s = d.pool_capacity_snapshot(
        p.config.pool_id, [lease(p, i) for i in range(n + prefetch)]
    )
    assert s["worker_slots_free"] == 0
    assert d.has_workers(p.config.pool_id)


@pytest.mark.parametrize("fault", ["discovery", "watch"])
def test_unhealthy_view_never_reports_free_slots(make_gateway_config, fault):
    d, p, _ = setup_pool(make_gateway_config)
    if fault == "discovery":
        p.discovery_healthy = False
    else:
        p.lease_watch_healthy = False
    s = d.pool_capacity_snapshot(p.config.pool_id, [])
    assert s["worker_capacity_view_healthy"] == s["worker_slots_free"] == 0


def test_unknown_and_not_accepting_capacity_then_legacy_fallback(make_gateway_config):
    d, p, _ = setup_pool(make_gateway_config, p=0)
    for cache in ({}, {7: (0, 0)}):
        p.capacity_cache = cache
        assert not d.has_workers(p.config.pool_id)
        assert (
            d.pool_capacity_snapshot(p.config.pool_id, [])["worker_execution_capacity"]
            == 0
        )
    p.capacity_cache = {7: (1, 0)}
    assert d.has_workers(p.config.pool_id)
    assert (
        d.pool_capacity_snapshot(p.config.pool_id, [])["worker_execution_capacity"] == 1
    )


def test_handoff_release_counts_leases_not_historical_task_assignment(
    make_gateway_config,
):
    d, p, _ = setup_pool(make_gateway_config)
    d._finalizing = {"old-task": p.config.pool_id, "elsewhere": "another-pool"}
    s = d.pool_capacity_snapshot(p.config.pool_id, [])
    assert s["worker_busy"] == s["worker_slots_busy"] == 0
    assert s["worker_slots_free"] == 3
    assert d.pool_finalization_pending(p.config.pool_id) == 1


async def test_metrics_endpoint_uses_slot_snapshot(make_gateway_config, monkeypatch):
    from dingo.video_gateway.api import _SERVICE_KEY
    from tests.video_gateway.test_api import _client

    client = await _client(make_gateway_config)
    try:
        service = client.server.app[_SERVICE_KEY]
        d = service.dispatcher
        p = d.pools["fl-pool"]
        p.config = dataclasses.replace(
            p.config,
            scheduling=dataclasses.replace(
                p.config.scheduling, worker_capacity=2, worker_prefetch_capacity=1
            ),
        )
        # Freeze discovery so the legacy fake's next refresh cannot change the snapshot.
        monkeypatch.setattr(d, "_refresh_instances", AsyncMock())
        p.instance_ids = [7]
        p.capacity_cache = {7: (3, float("inf"))}
        p.prefetch_instances = {7}
        monkeypatch.setattr(
            d, "pool_leases", AsyncMock(return_value=[lease(p), lease(p, 1)])
        )
        response = await client.get("/metrics")
        assert response.status == 200
        body = await response.text()
        for metric, value in (
            ("workers", 1),
            ("worker_busy", 1),
            ("worker_slots_busy", 2),
            ("worker_execution_capacity", 2),
            ("worker_prefetch_capacity", 1),
            ("worker_admission_capacity", 3),
        ):
            assert f'dingo_video_{metric}{{pool="fl-pool"}} {value}\n' in body
    finally:
        await client.close()
