# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import base64
from typing import Any

from dingo.video_gateway.adapters import create_adapter
from dingo.video_gateway.artifact_store import FileArtifactStore
from dingo.video_gateway.dispatcher import VideoDispatcher
from dingo.video_gateway.models import TaskStatus, WorkerLease, now_ms
from dingo.video_gateway.service import VideoGatewayService
from dingo.video_gateway.task_store import (
    LeaseWatchEvent,
    MemoryTaskStore,
    worker_key,
)
from tests.video_gateway.test_task_store import _task

_MINIMAL_MP4 = b"\x00\x00\x00\x18ftypisomminimal-video"


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
        available=True,
        discovery_error=False,
        b64_json=None,
    ) -> None:
        self.instance_id = instance_id
        self.block = block
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
                    stopped_task = asyncio.create_task(context.stopped.wait())
                    done, pending = await asyncio.wait(
                        {release_task, stopped_task},
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


def _pool(pool_id, model, target, workflow="fl2va"):
    return {
        "pool_id": pool_id,
        "served_models": [model],
        "backend_model": f"backend-{model}",
        "backend_target": target,
        "adapter": {
            "name": "minimax_h3",
            "workflow": workflow,
            "compatibility_version": "test-wire-v1",
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
