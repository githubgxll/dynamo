# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from dataclasses import replace

import dingo.video_gateway.dispatcher as dispatcher_module
from dingo.video_gateway.artifact_store import FileArtifactStore
from dingo.video_gateway.config import DiscoveryWatchdogConfig
from dingo.video_gateway.dispatcher import VideoDispatcher
from dingo.video_gateway.task_store import MemoryTaskStore


class FakeEndpointClient:
    def __init__(self, instance_ids: set[int]) -> None:
        self.ids = instance_ids

    def instance_ids(self) -> list[int]:
        return sorted(self.ids)

    async def direct(self, payload, instance_id, context):  # pragma: no cover
        del payload, instance_id, context
        raise AssertionError("watchdog tests must not dispatch requests")


class DiscoveryMemoryTaskStore(MemoryTaskStore):
    def __init__(self, truth: dict[str, set[int]], *, acquire: bool = True) -> None:
        super().__init__()
        self.truth = truth
        self.acquire = acquire
        self.lock_calls: list[tuple[str, int]] = []

    @property
    def discovery_truth_supported(self) -> bool:
        return True

    async def discovery_instance_snapshot(self, backend_targets):
        return {
            target: set(self.truth.get(target, set()))
            for target in backend_targets
        }

    async def try_acquire_discovery_recovery(self, gateway_id, *, ttl_s):
        self.lock_calls.append((gateway_id, ttl_s))
        return self.acquire


def _dispatcher(make_gateway_config, *, acquire: bool):
    config = make_gateway_config()
    config = replace(
        config,
        runtime=replace(
            config.runtime,
            discovery_watchdog=DiscoveryWatchdogConfig(
                enabled=True,
                interval_s=0.01,
                mismatch_grace_s=0.02,
            ),
        ),
    )
    pool = config.pools[0]
    store = DiscoveryMemoryTaskStore(
        {pool.backend_target: {8}}, acquire=acquire
    )
    dispatcher = VideoDispatcher(
        config,
        store,
        FileArtifactStore(config.artifact_store.root),
        {pool.pool_id: FakeEndpointClient({7})},
        {pool.pool_id: object()},
        generation="watchdog-gateway",
    )
    return dispatcher, store


async def test_watchdog_requests_fenced_restart_after_grace(
    make_gateway_config, monkeypatch
):
    monkeypatch.setattr(dispatcher_module, "_DISCOVERY_RESTART_DRAIN_S", 0)
    dispatcher, store = _dispatcher(make_gateway_config, acquire=True)
    watchdog = asyncio.create_task(dispatcher._discovery_watchdog_loop())
    try:
        reason = await asyncio.wait_for(dispatcher.wait_fatal(), timeout=1)
        assert "Dynamo discovery view remained inconsistent" in reason
        assert dispatcher.draining is True
        assert dispatcher.live is False
        assert store.lock_calls == [("watchdog-gateway", 15)]
        metrics = "\n".join(dispatcher.telemetry.render_prometheus())
        assert 'dingo_video_discovery_consistent{pool="fl-pool"} 0' in metrics
        assert "dingo_video_discovery_watchdog_restarts_total 1" in metrics
    finally:
        dispatcher._stop.set()
        watchdog.cancel()
        await asyncio.gather(watchdog, return_exceptions=True)


async def test_watchdog_stays_live_during_drain_before_fatal_restart(
    make_gateway_config, monkeypatch
):
    monkeypatch.setattr(dispatcher_module, "_DISCOVERY_RESTART_DRAIN_S", 0.05)
    dispatcher, _store = _dispatcher(make_gateway_config, acquire=True)
    watchdog = asyncio.create_task(dispatcher._discovery_watchdog_loop())
    try:
        for _ in range(100):
            if dispatcher.draining:
                break
            await asyncio.sleep(0.001)
        assert dispatcher.draining is True
        assert dispatcher.ready is False
        assert dispatcher.live is True
        assert not dispatcher._fatal_event.is_set()

        await asyncio.wait_for(dispatcher.wait_fatal(), timeout=1)
        assert dispatcher.live is False
    finally:
        dispatcher._stop.set()
        watchdog.cancel()
        await asyncio.gather(watchdog, return_exceptions=True)


async def test_watchdog_keeps_serving_when_recovery_lock_is_unavailable(
    make_gateway_config,
):
    dispatcher, store = _dispatcher(make_gateway_config, acquire=False)
    watchdog = asyncio.create_task(dispatcher._discovery_watchdog_loop())
    try:
        await asyncio.sleep(0.06)
        assert not dispatcher._fatal_event.is_set()
        assert dispatcher.draining is False
        assert len(store.lock_calls) >= 1
    finally:
        dispatcher._stop.set()
        watchdog.cancel()
        await asyncio.gather(watchdog, return_exceptions=True)


async def test_watchdog_clears_transient_mismatch_before_grace(make_gateway_config):
    dispatcher, store = _dispatcher(make_gateway_config, acquire=True)
    pool = dispatcher.config.pools[0]
    watchdog = asyncio.create_task(dispatcher._discovery_watchdog_loop())
    try:
        await asyncio.sleep(0.012)
        store.truth[pool.backend_target] = {7}
        await asyncio.sleep(0.04)
        assert not dispatcher._fatal_event.is_set()
        assert store.lock_calls == []
        metrics = "\n".join(dispatcher.telemetry.render_prometheus())
        assert 'dingo_video_discovery_consistent{pool="fl-pool"} 1' in metrics
    finally:
        dispatcher._stop.set()
        watchdog.cancel()
        await asyncio.gather(watchdog, return_exceptions=True)
