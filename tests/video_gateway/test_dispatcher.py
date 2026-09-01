# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import base64
import os
import threading
from typing import Any

import pytest

from dingo.video_gateway.adapters import create_adapter
from dingo.video_gateway.artifact_store import ArtifactCapacity, FileArtifactStore
from dingo.video_gateway.dispatcher import VideoDispatcher
from dingo.video_gateway.errors import GatewayError, StoreConflict
from dingo.video_gateway.models import TaskStatus, WorkerLease, now_ms
from dingo.video_gateway.service import VideoGatewayService
from dingo.video_gateway.task_store import (
    LeaseWatchEvent,
    MemoryTaskStore,
    worker_key,
)
from dingo.vllm.omni.detached_tasks import DetachedOmniTaskManager
from tests.video_gateway.test_task_store import _task

_MINIMAL_MP4 = b"\x00\x00\x00\x18ftypisomminimal-video"
_REAL_TO_THREAD = asyncio.to_thread


class FakeContext:
    def __init__(self, task_id: str, metadata: dict[str, str]) -> None:
        self.task_id = task_id
        self.metadata = metadata
        self.stopped = asyncio.Event()

    def stop_generating(self) -> None:
        self.stopped.set()


class FakeClient:
    def __init__(
        self,
        *,
        instance_id=7,
        block=False,
        honor_stop=True,
        available=True,
        discovery_error=False,
        b64_json=None,
    ) -> None:
        self.instance_id = instance_id
        self.block = block
        self.honor_stop = honor_stop
        self.available = available
        self.discovery_error = discovery_error
        self.release = asyncio.Event()
        self.calls: list[dict[str, Any]] = []
        self.active = 0
        self.max_active = 0
        self.b64_json = b64_json

    def instance_ids(self):
        if self.discovery_error:
            raise RuntimeError("simulated discovery outage")
        return [self.instance_id] if self.available else []

    async def direct(self, payload, instance_id, context):
        self.calls.append(
            {"payload": payload, "instance_id": instance_id, "context": context}
        )

        async def _stream():
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                if self.block:
                    release_task = asyncio.create_task(self.release.wait())
                    waiters = {release_task}
                    if self.honor_stop:
                        waiters.add(asyncio.create_task(context.stopped.wait()))
                    done, pending = await asyncio.wait(
                        waiters,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    del done
                yield {
                    "status": "completed",
                    "data": [
                        {
                            "output_format": "mp4",
                            "b64_json": self.b64_json
                            or base64.b64encode(_MINIMAL_MP4).decode(),
                        }
                    ],
                    "inference_time_s": 0.01,
                }
            finally:
                self.active -= 1

        return _stream()


class BlockingFinalizeArtifactStore(FileArtifactStore):
    def __init__(self, root) -> None:
        super().__init__(root)
        self.candidate_ready = asyncio.Event()
        self.release_candidate = asyncio.Event()
        self.candidate = None

    async def finalize_b64_mp4(self, *args, **kwargs):
        result = await super().finalize_b64_mp4(*args, **kwargs)
        self.candidate = result[0]
        self.candidate_ready.set()
        await self.release_candidate.wait()
        return result


class WatchMemoryTaskStore(MemoryTaskStore):
    """Exercise the dispatcher's etcd-style lease cache without a live server."""

    def __init__(self) -> None:
        super().__init__()
        self.list_lease_calls = 0
        self.watch_events: asyncio.Queue[LeaseWatchEvent] = asyncio.Queue()

    @property
    def lease_watch_supported(self) -> bool:
        return True

    async def lease_snapshot(self, pool_id: str):
        leases = await MemoryTaskStore.list_leases(self, pool_id)
        return {lease.worker_key: lease for lease in leases}, max(1, self._revision)

    async def watch_leases(self, pool_id: str, *, start_revision: int):
        del pool_id, start_revision
        while True:
            yield await self.watch_events.get()

    async def list_leases(self, pool_id: str):
        self.list_lease_calls += 1
        return await super().list_leases(pool_id)


class CountingMemoryTaskStore(MemoryTaskStore):
    def __init__(self) -> None:
        super().__init__()
        self.get_task_calls = 0

    async def get_task(self, task_id: str):
        self.get_task_calls += 1
        return await super().get_task(task_id)


class IndeterminateLookupMemoryTaskStore(MemoryTaskStore):
    def __init__(self) -> None:
        super().__init__()
        self.lookup_count = 0

    async def get_task(self, task_id: str):
        self.lookup_count += 1
        if self.lookup_count == 2:
            raise RuntimeError("simulated etcd timeout")
        return None


class LeaseLosingMemoryTaskStore(MemoryTaskStore):
    def __init__(self, *, fail_after: int = 1) -> None:
        super().__init__()
        self.fail_after = fail_after
        self.heartbeat_calls = 0

    async def heartbeat_lease(
        self,
        pool_id,
        worker_key_value,
        task_id,
        lease_id=None,
    ):
        self.heartbeat_calls += 1
        if self.heartbeat_calls > self.fail_after:
            raise StoreConflict("simulated execution lease loss")
        await super().heartbeat_lease(
            pool_id, worker_key_value, task_id, lease_id
        )


def _pool(pool_id, model, target, workflow="fl2va"):
    return {
        "pool_id": pool_id,
        "served_models": [model],
        "backend_model": f"backend-{model}",
        "backend_target": target,
        "adapter": {
            "name": "minimax_h3",
            "workflow": workflow,
            "validate_media": False,
        },
        "scheduling": {
            "worker_capacity": 1,
            "queue_limit": 8,
            "execution_timeout_s": 5,
            "abort_grace_s": 0.1,
            "discovery_interval_s": 0.01,
            "dispatch_interval_s": 0.01,
        },
    }


class _DetachedHandler:
    def __init__(self, *, block: bool = False) -> None:
        self.block = block
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def generate(self, request, context):
        self.calls += 1
        self.started.set()
        if self.block:
            release = asyncio.create_task(self.release.wait())
            stopped = context.async_killed_or_stopped()
            done, pending = await asyncio.wait(
                {release, stopped}, return_when=asyncio.FIRST_COMPLETED
            )
            del done
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        if not context.is_stopped():
            yield {
                "status": "completed",
                "data": [
                    {
                        "output_format": "mp4",
                        "b64_json": base64.b64encode(_MINIMAL_MP4).decode(),
                    }
                ],
                "inference_time_s": 0.01,
            }


class _DetachedClient:
    def __init__(self, manager, instance_id=7):
        self.manager = manager
        self.instance_id = instance_id
        self.calls = []

    def instance_ids(self):
        return [self.instance_id]

    async def direct(self, payload, instance_id, context):
        self.calls.append(payload)
        return self.manager.generate(payload, context)


class _LegacyDetachedManager(DetachedOmniTaskManager):
    """Detached Worker shape from before terminal wait capability support."""

    def _base_status(self, identity, state):
        status = super()._base_status(identity, state)
        status.pop("capabilities", None)
        return status


def _stack(config, clients):
    store = MemoryTaskStore()
    artifacts = FileArtifactStore(config.artifact_store.root)
    adapters = {pool.pool_id: create_adapter(pool) for pool in config.pools}
    dispatcher = VideoDispatcher(
        config,
        store,
        artifacts,
        clients,
        adapters,
        context_factory=FakeContext,
        generation="test-generation",
    )
    service = VideoGatewayService(config, store, artifacts, dispatcher, adapters)
    return store, artifacts, dispatcher, service


async def _submit(service, model):
    upload_root = await service.artifacts.create_upload()
    return await service.submit(
        fields={
            "model": [model],
            "prompt": [f"request for {model}"],
            "seconds": ["5"],
            "size": ["1344x768"],
            "seed": ["42"],
        },
        uploads=[],
        upload_root=upload_root,
        delivery_mode="async",
        idempotency_key=None,
    )


async def test_two_pools_with_same_numeric_instance_never_cross_route(
    make_gateway_config,
):
    pools = [
        _pool("pool-a", "model-a", "dyn://scope-a.backend.generate"),
        _pool("pool-b", "model-b", "dyn://scope-b.backend.generate"),
    ]
    config = make_gateway_config(pools=pools)
    clients = {"pool-a": FakeClient(instance_id=7), "pool-b": FakeClient(instance_id=7)}
    store, _artifacts, dispatcher, service = _stack(config, clients)
    await dispatcher.start()
    try:
        left = await _submit(service, "model-a")
        right = await _submit(service, "model-b")
        left_done, right_done = await asyncio.gather(
            dispatcher.wait_terminal(left.stored.task.id, 2),
            dispatcher.wait_terminal(right.stored.task.id, 2),
        )

        assert left_done.task.status == TaskStatus.COMPLETED
        assert right_done.task.status == TaskStatus.COMPLETED
        assert [call["payload"]["model"] for call in clients["pool-a"].calls] == [
            "backend-model-a"
        ]
        assert [call["payload"]["model"] for call in clients["pool-b"].calls] == [
            "backend-model-b"
        ]
        assert left_done.task.worker_key != right_done.task.worker_key
        assert await store.queue_depth("pool-a") == 0
        assert await store.queue_depth("pool-b") == 0
    finally:
        await dispatcher.stop()


async def test_worker_capacity_one_serializes_async_requests(make_gateway_config):
    config = make_gateway_config()
    client = FakeClient(block=True)
    _store, _artifacts, dispatcher, service = _stack(config, {"fl-pool": client})
    await dispatcher.start()
    try:
        submissions = [await _submit(service, "public-fl") for _ in range(3)]
        await asyncio.sleep(0.1)
        states = [
            (await service.store.get_task(item.stored.task.id)).task.status
            for item in submissions
        ]

        assert states.count(TaskStatus.IN_PROGRESS) == 1
        assert states.count(TaskStatus.QUEUED) == 2
        assert client.max_active == 1

        client.release.set()
        completed = await asyncio.gather(
            *(dispatcher.wait_terminal(item.stored.task.id, 2) for item in submissions)
        )
        assert [item.task.status for item in completed] == [
            TaskStatus.COMPLETED,
            TaskStatus.COMPLETED,
            TaskStatus.COMPLETED,
        ]
        assert client.max_active == 1
    finally:
        await dispatcher.stop()


async def test_drain_stops_claiming_queued_tasks_but_keeps_gateway_live(
    make_gateway_config,
):
    config = make_gateway_config(accept_without_workers=True)
    client = FakeClient(available=False)
    store, _artifacts, dispatcher, service = _stack(
        config, {"fl-pool": client}
    )
    await dispatcher.start()
    try:
        submission = await _submit(service, "public-fl")
        assert (await store.get_task(submission.stored.task.id)).task.status == (
            TaskStatus.QUEUED
        )

        assert dispatcher.begin_drain() is True
        assert dispatcher.begin_drain() is False
        client.available = True
        await asyncio.sleep(0.1)

        stored = await store.get_task(submission.stored.task.id)
        assert stored is not None
        assert stored.task.status == TaskStatus.QUEUED
        assert client.calls == []
        assert dispatcher.ready is False
        assert dispatcher.live is True
    finally:
        await dispatcher.stop()


async def test_detached_pool_acknowledges_then_finishes_from_shared_response(
    make_gateway_config,
):
    pool = _pool("fl-pool", "public-fl", "dyn://scope-a.backend.generate")
    pool["execution_mode"] = "detached"
    pool["scheduling"]["abort_grace_s"] = 1
    config = make_gateway_config(pools=[pool])
    handler = _DetachedHandler(block=True)
    manager = DetachedOmniTaskManager(
        handler, config.artifact_store.root, drain_timeout_s=1
    )
    client = _DetachedClient(manager)
    store, _artifacts, dispatcher, service = _stack(
        config, {"fl-pool": client}
    )
    await dispatcher.start()
    try:
        submitted = await _submit(service, "public-fl")
        await asyncio.wait_for(handler.started.wait(), timeout=1)
        active = await store.get_task(submitted.stored.task.id)
        assert active is not None
        assert active.task.status == TaskStatus.IN_PROGRESS
        assert active.task.execution_token is not None
        assert client.calls[0]["_dingo_video_task"]["op"] == "submit"

        handler.release.set()
        terminal = await dispatcher.wait_terminal(submitted.stored.task.id, 2)
        assert terminal.task.status == TaskStatus.COMPLETED
        assert terminal.task.result_bytes == len(_MINIMAL_MP4)
        assert handler.calls == 1
        operations = [call["_dingo_video_task"]["op"] for call in client.calls]
        assert operations == ["submit", "wait"]
        metrics = "\n".join(dispatcher.telemetry.render_prometheus())
        assert "dingo_video_detached_wait_connections_total" in metrics
        assert 'outcome="attached"' in metrics
        assert "dingo_video_detached_status_fallback_reads_total" not in metrics
    finally:
        handler.release.set()
        await dispatcher.stop()
        await manager.shutdown()


async def test_detached_terminal_wait_respects_execution_deadline(
    make_gateway_config,
):
    pool = _pool("fl-pool", "public-fl", "dyn://scope-a.backend.generate")
    pool["execution_mode"] = "detached"
    pool["scheduling"]["execution_timeout_s"] = 0.2
    config = make_gateway_config(pools=[pool])
    handler = _DetachedHandler(block=True)
    manager = DetachedOmniTaskManager(
        handler,
        config.artifact_store.root,
        drain_timeout_s=1,
        cancel_poll_interval_s=0.01,
        cancel_grace_s=0.01,
    )
    client = _DetachedClient(manager)
    store, _artifacts, dispatcher, service = _stack(
        config, {"fl-pool": client}
    )
    await dispatcher.start()
    try:
        submitted = await _submit(service, "public-fl")
        await asyncio.wait_for(handler.started.wait(), timeout=1)
        terminal = await dispatcher.wait_terminal(submitted.stored.task.id, 2)

        assert terminal.task.status == TaskStatus.FAILED
        assert terminal.task.error is not None
        assert terminal.task.error.code == "execution_timeout"
        operations = [call["_dingo_video_task"]["op"] for call in client.calls]
        assert operations == ["submit", "wait"]
        metrics = "\n".join(dispatcher.telemetry.render_prometheus())
        assert 'outcome="attached"' in metrics
    finally:
        handler.release.set()
        await dispatcher.stop()
        await manager.shutdown()


async def test_detached_old_worker_uses_low_frequency_status_fallback(
    make_gateway_config,
    monkeypatch,
):
    monkeypatch.setattr(
        "dingo.video_gateway.dispatcher._DETACHED_STATUS_FALLBACK_S", 0.01
    )
    pool = _pool("fl-pool", "public-fl", "dyn://scope-a.backend.generate")
    pool["execution_mode"] = "detached"
    config = make_gateway_config(pools=[pool])
    handler = _DetachedHandler(block=True)
    manager = _LegacyDetachedManager(
        handler, config.artifact_store.root, drain_timeout_s=1
    )
    client = _DetachedClient(manager)
    store, _artifacts, dispatcher, service = _stack(
        config, {"fl-pool": client}
    )
    await dispatcher.start()
    try:
        submitted = await _submit(service, "public-fl")
        await asyncio.wait_for(handler.started.wait(), timeout=1)
        handler.release.set()
        terminal = await dispatcher.wait_terminal(submitted.stored.task.id, 2)

        assert terminal.task.status == TaskStatus.COMPLETED
        operations = [call["_dingo_video_task"]["op"] for call in client.calls]
        assert operations == ["submit"]
        metrics = "\n".join(dispatcher.telemetry.render_prometheus())
        assert "dingo_video_detached_status_fallback_reads_total" in metrics
        assert 'outcome="unsupported"' in metrics
    finally:
        handler.release.set()
        await dispatcher.stop()
        await manager.shutdown()


async def test_detached_ack_releases_input_portion_of_memory_budget(
    make_gateway_config,
):
    pool = _pool("fl-pool", "public-fl", "dyn://scope-a.backend.generate")
    pool["execution_mode"] = "detached"
    pool["scheduling"]["accept_without_workers"] = True
    config = make_gateway_config(pools=[pool])
    handler = _DetachedHandler(block=True)
    manager = DetachedOmniTaskManager(
        handler, config.artifact_store.root, drain_timeout_s=1
    )
    client = _DetachedClient(manager)
    store, _artifacts, dispatcher, service = _stack(
        config, {"fl-pool": client}
    )
    submitted = await _submit(service, "public-fl")
    input_reservation = 1024 * 1024
    await store.transition(
        submitted.stored.task.id,
        expected={TaskStatus.QUEUED},
        expected_revision=submitted.stored.revision,
        patch={
            "estimated_payload_bytes": (
                config.media.result_task_memory_bytes + input_reservation
            )
        },
    )

    await dispatcher.start()
    try:
        await asyncio.wait_for(handler.started.wait(), timeout=1)
        for _ in range(100):
            snapshot = await dispatcher.memory_budget_snapshot()
            if snapshot.used_bytes == config.media.result_task_memory_bytes:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("input memory budget was not released after ACK")

        assert snapshot.active_tasks == 1
        assert snapshot.peak_bytes == (
            config.media.result_task_memory_bytes + input_reservation
        )

        handler.release.set()
        terminal = await dispatcher.wait_terminal(submitted.stored.task.id, 2)
        assert terminal.task.status == TaskStatus.COMPLETED
        assert (await dispatcher.memory_budget_snapshot()).used_bytes == 0
    finally:
        handler.release.set()
        await dispatcher.stop()
        await manager.shutdown()


async def test_dispatch_uses_watched_lease_cache_without_listing_each_tick(
    make_gateway_config,
):
    config = make_gateway_config()
    client = FakeClient()
    store = WatchMemoryTaskStore()
    artifacts = FileArtifactStore(config.artifact_store.root)
    adapters = {pool.pool_id: create_adapter(pool) for pool in config.pools}
    dispatcher = VideoDispatcher(
        config,
        store,
        artifacts,
        {"fl-pool": client},
        adapters,
        context_factory=FakeContext,
        generation="watch-cache-test",
    )
    service = VideoGatewayService(config, store, artifacts, dispatcher, adapters)

    await dispatcher.start()
    try:
        submission = await _submit(service, "public-fl")
        terminal = await dispatcher.wait_terminal(submission.stored.task.id, 2)

        assert terminal.task.status == TaskStatus.COMPLETED
        assert store.list_lease_calls == 0
        assert len(client.calls) == 1
    finally:
        await dispatcher.stop()


async def test_wait_terminal_uses_task_events_instead_of_periodic_reads(
    make_gateway_config,
):
    config = make_gateway_config()
    client = FakeClient(available=False)
    store = CountingMemoryTaskStore()
    artifacts = FileArtifactStore(config.artifact_store.root)
    adapters = {pool.pool_id: create_adapter(pool) for pool in config.pools}
    dispatcher = VideoDispatcher(
        config,
        store,
        artifacts,
        {"fl-pool": client},
        adapters,
        context_factory=FakeContext,
        generation="task-watch-test",
    )
    pool = config.pools[0]
    task = _task("video-event-wait", pool_id=pool.pool_id)
    task.model = pool.served_models[0]
    task.backend_model = pool.backend_model
    task.backend_target = pool.backend_target
    task.configuration_revision = pool.configuration_revision
    stored, _created = await store.create_task(
        task,
        principal_hash="principal",
        idempotency_hash=None,
        queue_limit=8,
    )

    await dispatcher.start()
    store.get_task_calls = 0
    waiter = asyncio.create_task(dispatcher.wait_terminal(task.id, 2))
    try:
        await asyncio.sleep(0.65)
        await store.transition(
            task.id,
            expected={TaskStatus.QUEUED},
            expected_revision=stored.revision,
            patch={"status": TaskStatus.CANCELLED},
        )
        terminal = await waiter

        assert terminal.task.status == TaskStatus.CANCELLED
        assert store.get_task_calls <= 3
    finally:
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        await dispatcher.stop()


async def test_running_cancel_stops_context_and_releases_after_stream_ends(
    make_gateway_config,
):
    config = make_gateway_config()
    client = FakeClient(block=True)
    store, _artifacts, dispatcher, service = _stack(config, {"fl-pool": client})
    await dispatcher.start()
    try:
        submission = await _submit(service, "public-fl")
        while not client.calls:
            await asyncio.sleep(0.01)

        requested = await dispatcher.cancel(submission.stored.task.id)
        terminal = await dispatcher.wait_terminal(submission.stored.task.id, 2)

        assert requested.task.cancel_requested_at_ms is not None
        assert client.calls[0]["context"].stopped.is_set()
        assert terminal.task.status == TaskStatus.CANCELLED
        assert await store.list_leases("fl-pool") == []
    finally:
        await dispatcher.stop()


async def test_unconfirmed_running_cancel_is_finished_in_background_and_quarantined(
    make_gateway_config,
):
    config = make_gateway_config()
    client = FakeClient(block=True, honor_stop=False)
    store, _artifacts, dispatcher, service = _stack(
        config, {"fl-pool": client}
    )
    await dispatcher.start()
    try:
        submission = await _submit(service, "public-fl")
        while not client.calls:
            await asyncio.sleep(0.01)

        requested = await dispatcher.cancel(submission.stored.task.id)
        terminal = await dispatcher.wait_terminal(submission.stored.task.id, 2)
        leases = await store.list_leases("fl-pool")

        assert requested.task.cancel_requested_at_ms is not None
        assert terminal.task.status == TaskStatus.CANCELLED
        assert len(leases) == 1
        assert leases[0].state == "quarantined"
        assert leases[0].reuse_after_ms is not None
        assert leases[0].reuse_after_ms >= requested.task.deadline_at_ms
    finally:
        client.release.set()
        await dispatcher.stop()


async def test_execution_lease_loss_stops_and_quarantines_worker(
    make_gateway_config,
    monkeypatch,
):
    monkeypatch.setattr(
        "dingo.video_gateway.dispatcher._WORKER_LEASE_HEARTBEAT_INTERVAL_S",
        0.01,
    )
    config = make_gateway_config()
    client = FakeClient(block=True)
    store = LeaseLosingMemoryTaskStore(fail_after=5)
    artifacts = FileArtifactStore(config.artifact_store.root)
    adapters = {pool.pool_id: create_adapter(pool) for pool in config.pools}
    dispatcher = VideoDispatcher(
        config,
        store,
        artifacts,
        {"fl-pool": client},
        adapters,
        context_factory=FakeContext,
        generation="lease-loss-test",
    )
    service = VideoGatewayService(config, store, artifacts, dispatcher, adapters)

    await dispatcher.start()
    try:
        submission = await _submit(service, "public-fl")
        terminal = await dispatcher.wait_terminal(submission.stored.task.id, 2)
        leases = await store.list_leases("fl-pool")

        assert terminal.task.status == TaskStatus.FAILED
        assert terminal.task.error is not None
        assert terminal.task.error.code == "worker_lease_lost"
        assert client.calls[0]["context"].stopped.is_set()
        assert len(leases) == 1
        assert leases[0].state == "quarantined"
        assert leases[0].reuse_after_ms is not None
        assert leases[0].reuse_after_ms >= terminal.task.deadline_at_ms
        metrics = "\n".join(dispatcher.telemetry.render_prometheus())
        assert "dingo_video_worker_lease_lost_total" in metrics
    finally:
        client.release.set()
        await dispatcher.stop()


async def test_payload_build_is_covered_by_execution_lease_heartbeats(
    make_gateway_config,
    monkeypatch,
):
    monkeypatch.setattr(
        "dingo.video_gateway.dispatcher._WORKER_LEASE_HEARTBEAT_INTERVAL_S",
        0.01,
    )
    # The suite normally inlines to_thread for deterministic media tests. This
    # case specifically verifies that the production thread offload leaves the
    # event loop free to renew the execution lease during payload construction.
    monkeypatch.setattr(asyncio, "to_thread", _REAL_TO_THREAD)
    config = make_gateway_config()
    client = FakeClient()
    store = LeaseLosingMemoryTaskStore(fail_after=1000)
    artifacts = FileArtifactStore(config.artifact_store.root)
    adapters = {pool.pool_id: create_adapter(pool) for pool in config.pools}
    adapter = adapters["fl-pool"]
    original_build = adapter.build_worker_payload
    build_started = threading.Event()
    release_build = threading.Event()

    def slow_build(*args, **kwargs):
        build_started.set()
        if not release_build.wait(timeout=2):
            raise TimeoutError("test did not release payload construction")
        return original_build(*args, **kwargs)

    monkeypatch.setattr(adapter, "build_worker_payload", slow_build)
    dispatcher = VideoDispatcher(
        config,
        store,
        artifacts,
        {"fl-pool": client},
        adapters,
        context_factory=FakeContext,
        generation="payload-heartbeat-test",
    )
    service = VideoGatewayService(config, store, artifacts, dispatcher, adapters)

    await dispatcher.start()
    try:
        submission = await _submit(service, "public-fl")
        assert await _REAL_TO_THREAD(build_started.wait, 1)
        await asyncio.sleep(0.05)

        assert store.heartbeat_calls >= 3
        assert client.calls == []

        release_build.set()
        terminal = await dispatcher.wait_terminal(submission.stored.task.id, 2)
        assert terminal.task.status == TaskStatus.COMPLETED
        assert client.calls
    finally:
        release_build.set()
        await dispatcher.stop()


async def test_lease_loss_before_payload_build_never_contacts_or_quarantines_worker(
    make_gateway_config,
):
    config = make_gateway_config()
    client = FakeClient()
    store = LeaseLosingMemoryTaskStore(fail_after=0)
    artifacts = FileArtifactStore(config.artifact_store.root)
    adapters = {pool.pool_id: create_adapter(pool) for pool in config.pools}
    dispatcher = VideoDispatcher(
        config,
        store,
        artifacts,
        {"fl-pool": client},
        adapters,
        context_factory=FakeContext,
        generation="payload-lease-loss-test",
    )
    service = VideoGatewayService(config, store, artifacts, dispatcher, adapters)

    await dispatcher.start()
    try:
        submission = await _submit(service, "public-fl")
        terminal = await dispatcher.wait_terminal(submission.stored.task.id, 2)

        assert terminal.task.status == TaskStatus.FAILED
        assert terminal.task.error is not None
        assert terminal.task.error.code == "worker_lease_lost"
        assert client.calls == []
        assert await store.list_leases("fl-pool") == []
    finally:
        await dispatcher.stop()


async def test_stale_gateway_cannot_publish_after_finalizing_owner_takeover(
    make_gateway_config,
):
    config = make_gateway_config()
    client = FakeClient()
    store = MemoryTaskStore()
    artifacts = BlockingFinalizeArtifactStore(config.artifact_store.root)
    adapters = {pool.pool_id: create_adapter(pool) for pool in config.pools}
    dispatcher = VideoDispatcher(
        config,
        store,
        artifacts,
        {"fl-pool": client},
        adapters,
        context_factory=FakeContext,
        generation="gateway-old",
    )
    service = VideoGatewayService(config, store, artifacts, dispatcher, adapters)

    await dispatcher.start()
    try:
        submission = await _submit(service, "public-fl")
        await asyncio.wait_for(artifacts.candidate_ready.wait(), timeout=2)
        stale_candidate = artifacts.candidate
        assert stale_candidate is not None and stale_candidate.exists()

        current = await store.get_task(submission.stored.task.id)
        assert current is not None
        assert current.task.status == TaskStatus.FINALIZING
        moved = await store.transition(
            current.task.id,
            expected={TaskStatus.FINALIZING},
            expected_revision=current.revision,
            patch={"owner_generation": "gateway-new"},
        )

        artifacts.release_candidate.set()
        for _ in range(100):
            current = await store.get_task(moved.task.id)
            if moved.task.id not in dispatcher.running_calls:
                break
            await asyncio.sleep(0.01)
        assert current is not None
        assert current.task.status == TaskStatus.FINALIZING
        assert current.task.owner_generation == "gateway-new"
        assert not stale_candidate.exists()

        winner_payload = b"\x00\x00\x00\x18ftypisomwinner"
        winner, size, digest, _media = await FileArtifactStore.finalize_b64_mp4(
            artifacts,
            artifacts.task_root(
                current.task.deployment_id,
                current.task.pool_id,
                current.task.id,
            ),
            base64.b64encode(winner_payload).decode(),
            {},
            lambda _path, _normalized: {},
            publication_scope="a1-new-owner",
        )
        completed = await store.transition(
            current.task.id,
            expected={TaskStatus.FINALIZING},
            expected_revision=current.revision,
            patch={
                "status": TaskStatus.COMPLETED,
                "result_path": str(winner),
                "result_bytes": size,
                "result_sha256": digest,
            },
            release_lease=True,
        )

        assert completed.task.status == TaskStatus.COMPLETED
        assert winner.read_bytes() == winner_payload
        assert not stale_candidate.exists()
    finally:
        artifacts.release_candidate.set()
        await dispatcher.stop()


async def test_detached_cancel_acknowledgement_releases_worker_immediately(
    make_gateway_config,
):
    pool = _pool("fl-pool", "public-fl", "dyn://scope-a.backend.generate")
    pool["execution_mode"] = "detached"
    # Gateway observes detached terminal state at 0.5 s intervals. Keep the
    # confirmation window above that interval; production uses 15 seconds.
    pool["scheduling"]["abort_grace_s"] = 1
    config = make_gateway_config(pools=[pool])
    handler = _DetachedHandler(block=True)
    manager = DetachedOmniTaskManager(
        handler,
        config.artifact_store.root,
        drain_timeout_s=1,
        cancel_poll_interval_s=0.01,
        cancel_grace_s=0.01,
    )
    client = _DetachedClient(manager)
    store, _artifacts, dispatcher, service = _stack(
        config, {"fl-pool": client}
    )
    await dispatcher.start()
    try:
        submission = await _submit(service, "public-fl")
        await asyncio.wait_for(handler.started.wait(), timeout=1)
        requested = await dispatcher.cancel(submission.stored.task.id)
        terminal = await dispatcher.wait_terminal(submission.stored.task.id, 2)
        leases = await store.list_leases("fl-pool")

        assert terminal.task.status == TaskStatus.CANCELLED
        assert requested.task.cancel_requested_at_ms is not None
        assert leases == []
    finally:
        handler.release.set()
        await dispatcher.stop()
        await manager.shutdown()


async def test_restart_fails_active_task_and_keeps_old_worker_quarantined(
    make_gateway_config,
):
    config = make_gateway_config()
    client = FakeClient()
    store, _artifacts, dispatcher, _service = _stack(config, {"fl-pool": client})
    pool = config.pools[0]

    active_task = _task("video-before-restart", pool_id=pool.pool_id)
    active_task.model = pool.served_models[0]
    active_task.backend_model = pool.backend_model
    active_task.backend_target = pool.backend_target
    active_task.configuration_revision = pool.configuration_revision
    active, _ = await store.create_task(
        active_task,
        principal_hash="principal",
        idempotency_hash=None,
        queue_limit=8,
    )
    key = worker_key(pool.backend_target, client.instance_id)
    lease = WorkerLease(
        pool_id=pool.pool_id,
        worker_key=key,
        worker_instance_id=client.instance_id,
        backend_target=pool.backend_target,
        task_id=active.task.id,
        owner_generation="old-generation",
        state="reserved",
        heartbeat_at_ms=now_ms(),
        owner_expires_at_ms=now_ms() + 15_000,
    )
    deadline = now_ms() + 5_000
    reserved = await store.reserve(active, lease, deadline_at_ms=deadline)
    assert reserved is not None
    await store.transition(
        active.task.id,
        expected={TaskStatus.DISPATCHING},
        expected_revision=reserved.revision,
        patch={"status": TaskStatus.IN_PROGRESS, "started_at_ms": now_ms()},
    )

    queued_task = _task("video-after-restart", pool_id=pool.pool_id)
    queued_task.model = pool.served_models[0]
    queued_task.backend_model = pool.backend_model
    queued_task.backend_target = pool.backend_target
    queued_task.configuration_revision = pool.configuration_revision
    await store.create_task(
        queued_task,
        principal_hash="principal",
        idempotency_hash=None,
        queue_limit=8,
    )

    await dispatcher.start()
    try:
        recovered = await store.get_task(active.task.id)
        leases = await store.list_leases(pool.pool_id)
        await asyncio.sleep(0.05)

        assert recovered is not None
        assert recovered.task.status == TaskStatus.FAILED
        assert recovered.task.error is not None
        assert recovered.task.error.code == "gateway_restarted"
        assert len(leases) == 1
        assert leases[0].state == "quarantined"
        assert leases[0].reuse_after_ms is not None
        assert leases[0].reuse_after_ms > now_ms()
        assert client.calls == []
        assert await store.queue_depth(pool.pool_id) == 1
    finally:
        await dispatcher.stop()


async def test_restart_does_not_scan_or_rehash_completed_artifacts(
    make_gateway_config,
):
    config = make_gateway_config()
    client = FakeClient(available=False)
    store, artifacts, dispatcher, _service = _stack(config, {"fl-pool": client})
    pool = config.pools[0]
    payload = b"completed-video"
    task = _task("video-completed-before-restart", pool_id=pool.pool_id)
    task.deployment_id = config.deployment_id
    task.status = TaskStatus.COMPLETED
    task.result_bytes = len(payload)
    # Deliberately differs from the file. Recovery only processes nonterminal
    # tasks; the download path performs the cheap path/type/size validation.
    task.result_sha256 = "0" * 64
    path = artifacts.task_root(
        task.deployment_id, task.pool_id, task.id
    ) / "result" / "video.mp4"
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    task.result_path = str(path)
    await store.create_task(
        task,
        principal_hash="principal",
        idempotency_hash=None,
        queue_limit=8,
    )

    await dispatcher.start()
    try:
        recovered = await store.get_task(task.id)
        assert recovered is not None
        assert recovered.task.status == TaskStatus.COMPLETED
        assert recovered.task.error is None
    finally:
        await dispatcher.stop()


async def test_orphan_cleanup_moves_nothing_when_any_store_lookup_is_uncertain(
    make_gateway_config,
):
    config = make_gateway_config()
    store = IndeterminateLookupMemoryTaskStore()
    artifacts = FileArtifactStore(config.artifact_store.root)
    adapters = {pool.pool_id: create_adapter(pool) for pool in config.pools}
    dispatcher = VideoDispatcher(
        config,
        store,
        artifacts,
        {"fl-pool": FakeClient(available=False)},
        adapters,
        context_factory=FakeContext,
    )
    task_roots = []
    for task_id in ("video-orphan-a", "video-orphan-b"):
        upload = await artifacts.create_upload()
        task_root = await artifacts.commit_upload(
            upload,
            config.deployment_id,
            "fl-pool",
            task_id,
            artifact_manifest={
                "schema_version": 1,
                "task_id": task_id,
                "deployment_id": config.deployment_id,
                "pool_id": "fl-pool",
            },
        )
        os.utime(task_root, (0, 0))
        task_roots.append(task_root)

    with pytest.raises(RuntimeError, match="etcd timeout"):
        await dispatcher._cleanup_orphan_tasks()

    assert all(path.exists() for path in task_roots)
    assert list(artifacts.trash_root.iterdir()) == []


async def test_orphan_cleanup_never_moves_directory_with_live_task_record(
    make_gateway_config,
):
    config = make_gateway_config()
    store, artifacts, dispatcher, _service = _stack(
        config, {"fl-pool": FakeClient(available=False)}
    )
    task = _task("video-live-record", pool_id="fl-pool")
    task.deployment_id = config.deployment_id
    await store.create_task(
        task,
        principal_hash="principal",
        idempotency_hash=None,
        queue_limit=8,
    )
    upload = await artifacts.create_upload()
    task_root = await artifacts.commit_upload(
        upload,
        config.deployment_id,
        "fl-pool",
        task.id,
        artifact_manifest={
            "schema_version": 1,
            "task_id": task.id,
            "deployment_id": config.deployment_id,
            "pool_id": "fl-pool",
        },
    )
    os.utime(task_root, (0, 0))

    await dispatcher._cleanup_orphan_tasks()

    assert task_root.exists()
    assert list(artifacts.trash_root.iterdir()) == []


async def test_submission_capacity_rejects_before_parsing_large_upload(
    make_gateway_config,
    monkeypatch,
):
    config = make_gateway_config()
    _store, artifacts, dispatcher, service = _stack(
        config, {"fl-pool": FakeClient(available=False)}
    )
    sweep_calls = 0

    async def _capacity():
        return ArtifactCapacity(total_bytes=1_000, used_bytes=900, free_bytes=100)

    async def _sweep():
        nonlocal sweep_calls
        sweep_calls += 1

    monkeypatch.setattr(artifacts, "capacity", _capacity)
    monkeypatch.setattr(dispatcher, "sweep_now", _sweep)

    with pytest.raises(GatewayError) as raised:
        await service.ensure_submission_capacity(200)

    assert raised.value.status == 507
    assert raised.value.code == "insufficient_artifact_storage"
    assert sweep_calls == 1


async def test_readiness_recovers_after_discovery_outage(make_gateway_config):
    config = make_gateway_config()
    client = FakeClient(discovery_error=True)
    _store, _artifacts, dispatcher, _service = _stack(config, {"fl-pool": client})

    await dispatcher.start()
    try:
        assert dispatcher.ready is False
        client.discovery_error = False
        for _ in range(100):
            if dispatcher.ready:
                break
            await asyncio.sleep(0.01)
        assert dispatcher.ready is True
    finally:
        await dispatcher.stop()


async def test_global_media_budget_waits_before_reserving_second_worker(
    make_gateway_config,
):
    pools = [
        _pool("pool-a", "model-a", "dyn://scope-a.backend.generate"),
        _pool("pool-b", "model-b", "dyn://scope-b.backend.generate"),
    ]
    mebibyte = 1024 * 1024
    config = make_gateway_config(
        pools=pools,
        media={
            "max_encoded_reference_bytes": mebibyte // 4,
            "max_result_bytes": mebibyte,
            "task_memory_overhead_bytes": mebibyte,
            "inflight_memory_budget_bytes": 4 * mebibyte,
        },
    )
    clients = {"pool-a": FakeClient(block=True), "pool-b": FakeClient(block=True)}
    store, _artifacts, dispatcher, service = _stack(config, clients)
    await dispatcher.start()
    try:
        submissions = await asyncio.gather(
            _submit(service, "model-a"),
            _submit(service, "model-b"),
        )
        for _ in range(100):
            if sum(len(client.calls) for client in clients.values()) == 1:
                snapshot = await dispatcher.memory_budget_snapshot()
                if snapshot.waiting_tasks == 1:
                    break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("second task did not wait for the media budget")

        assert sum(len(client.calls) for client in clients.values()) == 1
        assert len(await store.list_leases("pool-a")) + len(
            await store.list_leases("pool-b")
        ) == 1
        for client in clients.values():
            client.release.set()
        completed = await asyncio.gather(
            *(
                dispatcher.wait_terminal(submission.stored.task.id, 2)
                for submission in submissions
            )
        )

        assert all(item.task.status == TaskStatus.COMPLETED for item in completed)
        assert sum(len(client.calls) for client in clients.values()) == 2
        snapshot = await dispatcher.memory_budget_snapshot()
        assert snapshot.used_bytes == 0
        assert snapshot.waiting_tasks == 0
    finally:
        await dispatcher.stop()


async def test_invalid_base64_releases_budget_lease_and_partial_file(
    make_gateway_config,
):
    config = make_gateway_config()
    client = FakeClient(b64_json="====")
    store, artifacts, dispatcher, service = _stack(
        config, {"fl-pool": client}
    )
    await dispatcher.start()
    try:
        submission = await _submit(service, "public-fl")
        terminal = await dispatcher.wait_terminal(submission.stored.task.id, 2)

        assert terminal.task.status == TaskStatus.FAILED
        assert await store.list_leases("fl-pool") == []
        assert (await dispatcher.memory_budget_snapshot()).used_bytes == 0
        task_root = artifacts.task_root(
            terminal.task.deployment_id,
            terminal.task.pool_id,
            terminal.task.id,
        )
        assert list((task_root / "tmp").glob("*.part")) == []
    finally:
        await dispatcher.stop()


async def test_oversize_worker_result_releases_budget_before_decode(
    make_gateway_config,
):
    config = make_gateway_config(media={"max_result_bytes": 1})
    client = FakeClient()
    store, _artifacts, dispatcher, service = _stack(
        config, {"fl-pool": client}
    )
    await dispatcher.start()
    try:
        submission = await _submit(service, "public-fl")
        terminal = await dispatcher.wait_terminal(submission.stored.task.id, 2)

        assert terminal.task.status == TaskStatus.FAILED
        assert await store.list_leases("fl-pool") == []
        assert (await dispatcher.memory_budget_snapshot()).used_bytes == 0
        assert dispatcher.media_runtime_snapshot().result_oversize_count == 1
    finally:
        await dispatcher.stop()
