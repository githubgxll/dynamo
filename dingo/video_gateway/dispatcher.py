# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-pool FIFO dispatch, sticky direct calls, cancellation and recovery."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from dingo.video_gateway.adapters.base import VideoBackendAdapter
from dingo.video_gateway.artifact_store import FileArtifactStore
from dingo.video_gateway.config import GatewayConfig, PoolConfig
from dingo.video_gateway.dingo_adapter import (
    ContextFactory,
    EndpointClient,
    create_context,
)
from dingo.video_gateway.errors import StoreConflict
from dingo.video_gateway.memory_budget import (
    MemoryBudgetSnapshot,
    WeightedMemoryBudget,
)
from dingo.video_gateway.models import (
    ACTIVE_STATUSES,
    TERMINAL_STATUSES,
    StoredTask,
    TaskStatus,
    WorkerLease,
    now_ms,
)
from dingo.video_gateway.task_store import TaskStore, terminal_error, worker_key

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RunningCall:
    context: Any
    execution: asyncio.Task
    pool_id: str
    worker_key: str


@dataclass(frozen=True, slots=True)
class MediaRuntimeSnapshot:
    legacy_input_encoded_bytes: int
    legacy_output_encoded_bytes: int
    payload_build_count: int
    payload_build_seconds: float
    finalize_count: int
    finalize_seconds: float
    result_oversize_count: int


@dataclass(frozen=True, slots=True)
class ArtifactRuntimeSnapshot:
    sweep_due_tasks: int
    expired_tasks_total: int
    orphan_candidates_total: int
    orphan_trashed_total: int
    cleanup_failures_total: int
    released_bytes_total: int


@dataclass(slots=True)
class PoolRuntime:
    config: PoolConfig
    client: EndpointClient
    adapter: VideoBackendAdapter
    wakeup: asyncio.Event
    instance_ids: list[int]
    cursor: int = 0
    discovery_healthy: bool = False
    budget_waiter_id: str | None = None
    lease_cache: dict[str, WorkerLease] = field(default_factory=dict)
    lease_revision: int = 0
    lease_watch_healthy: bool = True


class VideoDispatcher:
    def __init__(
        self,
        config: GatewayConfig,
        store: TaskStore,
        artifacts: FileArtifactStore,
        clients: Mapping[str, EndpointClient],
        adapters: Mapping[str, VideoBackendAdapter],
        *,
        context_factory: ContextFactory = create_context,
        generation: str | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.artifacts = artifacts
        self.context_factory = context_factory
        self.generation = generation or uuid.uuid4().hex
        self.pools: dict[str, PoolRuntime] = {
            pool.pool_id: PoolRuntime(
                config=pool,
                client=clients[pool.pool_id],
                adapter=adapters[pool.pool_id],
                wakeup=asyncio.Event(),
                instance_ids=[],
                lease_watch_healthy=not store.lease_watch_supported,
            )
            for pool in config.pools
        }
        self.running_calls: dict[str, RunningCall] = {}
        self.memory_budget = WeightedMemoryBudget(
            config.media.inflight_memory_budget_bytes
        )
        self._legacy_input_encoded_bytes = 0
        self._legacy_output_encoded_bytes = 0
        self._payload_build_count = 0
        self._payload_build_seconds = 0.0
        self._finalize_count = 0
        self._finalize_seconds = 0.0
        self._result_oversize_count = 0
        self._loops: list[asyncio.Task] = []
        self._executions: set[asyncio.Task] = set()
        self._stop = asyncio.Event()
        self._ready = False
        self._task_watch_revision = 0
        self._task_watch_healthy = not store.task_watch_supported
        self._task_watch_ready = asyncio.Event()
        self._task_waiters: dict[str, set[asyncio.Future[None]]] = {}
        self._sweep_lock = asyncio.Lock()
        self._next_orphan_scan = 0.0
        self._sweep_due_tasks = 0
        self._expired_tasks_total = 0
        self._orphan_candidates_total = 0
        self._orphan_trashed_total = 0
        self._artifact_cleanup_failures = 0
        self._artifact_released_bytes = 0

    @property
    def ready(self) -> bool:
        return (
            self._ready
            and not self._stop.is_set()
            and self._task_watch_healthy
            and all(
                pool.discovery_healthy and pool.lease_watch_healthy
                for pool in self.pools.values()
            )
        )

    async def start(self) -> None:
        await self.store.health()
        await self.artifacts.health()
        await self.store.prepare()
        await self._recover()
        if self.store.task_watch_supported:
            await self._resync_task_watch()
            task_watch = asyncio.create_task(
                self._task_watch_loop(), name="video-task-watch"
            )
            self._loops.append(task_watch)
            await asyncio.wait_for(self._task_watch_ready.wait(), timeout=10.0)
        for pool in self.pools.values():
            await self._refresh_instances(pool)
            if self.store.lease_watch_supported:
                await self._resync_lease_cache(pool)
                self._loops.append(
                    asyncio.create_task(
                        self._lease_watch_loop(pool),
                        name=f"video-lease-watch-{pool.config.pool_id}",
                    )
                )
            self._loops.append(
                asyncio.create_task(
                    self._pool_loop(pool), name=f"video-dispatch-{pool.config.pool_id}"
                )
            )
        self._loops.append(
            asyncio.create_task(self._sweeper_loop(), name="video-task-sweeper")
        )
        self._ready = True

    async def stop(self) -> None:
        self._ready = False
        self._stop.set()
        self._wake_task_waiters()
        for pool in self.pools.values():
            pool.wakeup.set()
        for running in list(self.running_calls.values()):
            try:
                running.context.stop_generating()
            except Exception:
                logger.exception("failed to stop task during Gateway shutdown")
        for task in self._loops:
            task.cancel()
        if self._loops:
            await asyncio.gather(*self._loops, return_exceptions=True)
        if self._executions:
            done, pending = await asyncio.wait(self._executions, timeout=30.0)
            del done
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        await self.store.close()

    def has_workers(self, pool_id: str) -> bool:
        pool = self.pools[pool_id]
        return bool(pool.instance_ids)

    def pool_instances(self, pool_id: str) -> list[int]:
        return list(self.pools[pool_id].instance_ids)

    async def pool_leases(self, pool_id: str) -> list[WorkerLease]:
        pool = self.pools[pool_id]
        if self.store.lease_watch_supported:
            return list(pool.lease_cache.values())
        return await self.store.list_leases(pool_id)

    def notify(self, pool_id: str) -> None:
        self.pools[pool_id].wakeup.set()

    async def memory_budget_snapshot(self) -> MemoryBudgetSnapshot:
        return await self.memory_budget.snapshot()

    def record_legacy_input(self, encoded_bytes: int) -> None:
        self._legacy_input_encoded_bytes += encoded_bytes

    def media_runtime_snapshot(self) -> MediaRuntimeSnapshot:
        return MediaRuntimeSnapshot(
            legacy_input_encoded_bytes=self._legacy_input_encoded_bytes,
            legacy_output_encoded_bytes=self._legacy_output_encoded_bytes,
            payload_build_count=self._payload_build_count,
            payload_build_seconds=self._payload_build_seconds,
            finalize_count=self._finalize_count,
            finalize_seconds=self._finalize_seconds,
            result_oversize_count=self._result_oversize_count,
        )

    def artifact_runtime_snapshot(self) -> ArtifactRuntimeSnapshot:
        return ArtifactRuntimeSnapshot(
            sweep_due_tasks=self._sweep_due_tasks,
            expired_tasks_total=self._expired_tasks_total,
            orphan_candidates_total=self._orphan_candidates_total,
            orphan_trashed_total=self._orphan_trashed_total,
            cleanup_failures_total=self._artifact_cleanup_failures,
            released_bytes_total=self._artifact_released_bytes,
        )

    async def cancel(self, task_id: str) -> StoredTask:
        stored = await self.store.request_cancel(
            task_id,
            terminal_expires_at_ms=now_ms()
            + int(self.config.lifecycle.cancelled_ttl_s * 1000),
        )
        running = self.running_calls.get(task_id)
        if running is not None:
            running.context.stop_generating()
        self.notify(stored.task.pool_id)
        return stored

    async def force_cancel(self, task_id: str) -> StoredTask:
        """Finish an unconfirmed cancellation while retaining a quarantined lease."""

        stored = await self.store.get_task(task_id)
        if stored is None:
            raise KeyError(task_id)
        if stored.task.status in TERMINAL_STATUSES:
            return stored
        pool = self.pools.get(stored.task.pool_id)
        grace_s = pool.config.scheduling.abort_grace_s if pool is not None else 30.0
        return await self.store.transition(
            task_id,
            expected=ACTIVE_STATUSES,
            expected_revision=stored.revision,
            patch={
                "status": TaskStatus.CANCELLED,
                "completed_at_ms": now_ms(),
                "expires_at_ms": now_ms()
                + int(self.config.lifecycle.cancelled_ttl_s * 1000),
                "error": terminal_error(
                    "cancelled", "video task cancellation was requested"
                ),
            },
            release_lease=True,
            quarantine_until_ms=max(stored.task.deadline_at_ms or now_ms(), now_ms())
            + int(grace_s * 1000),
        )

    async def wait_terminal(self, task_id: str, timeout_s: float) -> StoredTask:
        if not self.store.task_watch_supported:
            return await self._wait_terminal_polling(task_id, timeout_s)

        deadline = time.monotonic() + timeout_s
        while True:
            if self._stop.is_set():
                raise RuntimeError("Gateway stopped while waiting for video task")
            stored = await self.store.get_task(task_id)
            if stored is None:
                raise KeyError(task_id)
            if stored.task.status in TERMINAL_STATUSES:
                return stored
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(task_id)
            waiter = asyncio.get_running_loop().create_future()
            task_waiters = self._task_waiters.setdefault(task_id, set())
            task_waiters.add(waiter)
            try:
                # Close the read/register race: any change before registration
                # is observed by this second linearizable read; any later
                # change is delivered by the already registered watch waiter.
                latest = await self.store.get_task(task_id)
                if latest is None:
                    raise KeyError(task_id)
                if latest.task.status in TERMINAL_STATUSES:
                    return latest
                if latest.revision != stored.revision:
                    continue
                await asyncio.wait_for(waiter, timeout=remaining)
            except asyncio.TimeoutError as exc:
                raise TimeoutError(task_id) from exc
            finally:
                if not waiter.done():
                    waiter.cancel()
                task_waiters.discard(waiter)
                if not task_waiters:
                    self._task_waiters.pop(task_id, None)

    async def _wait_terminal_polling(
        self, task_id: str, timeout_s: float
    ) -> StoredTask:
        deadline = time.monotonic() + timeout_s
        while True:
            stored = await self.store.get_task(task_id)
            if stored is None:
                raise KeyError(task_id)
            if stored.task.status in TERMINAL_STATUSES:
                return stored
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(task_id)
            await asyncio.sleep(min(0.25, remaining))

    def _wake_task_waiters(self, task_id: str | None = None) -> None:
        groups = (
            [self._task_waiters.get(task_id, set())]
            if task_id is not None
            else list(self._task_waiters.values())
        )
        for waiters in groups:
            for waiter in tuple(waiters):
                if not waiter.done():
                    waiter.set_result(None)

    async def _resync_task_watch(self) -> None:
        self._task_watch_healthy = False
        self._task_watch_ready.clear()
        self._task_watch_revision = await self.store.task_watch_revision()
        if self._task_watch_revision < 0:
            raise RuntimeError("task watch snapshot omitted its store revision")
        self._wake_task_waiters()

    async def _task_watch_loop(self) -> None:
        backoff_s = 0.1
        while not self._stop.is_set():
            try:
                async for event in self.store.watch_tasks(
                    start_revision=self._task_watch_revision + 1
                ):
                    self._task_watch_revision = max(
                        self._task_watch_revision, event.revision
                    )
                    if event.created:
                        self._task_watch_healthy = True
                        self._task_watch_ready.set()
                        self._wake_task_waiters()
                    elif event.task_id is not None:
                        self._wake_task_waiters(event.task_id)
                    backoff_s = 0.1
                raise RuntimeError("task watch ended unexpectedly")
            except asyncio.CancelledError:
                raise
            except Exception:
                self._task_watch_healthy = False
                self._task_watch_ready.clear()
                logger.exception("task watch failed and will be rebuilt")
                try:
                    await self._resync_task_watch()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("task watch snapshot rebuild failed")
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=backoff_s)
                except asyncio.TimeoutError:
                    pass
                backoff_s = min(backoff_s * 2.0, 5.0)

    async def _refresh_instances(self, pool: PoolRuntime) -> None:
        try:
            pool.instance_ids = sorted(set(pool.client.instance_ids()))
            pool.discovery_healthy = True
        except Exception:
            logger.exception(
                "failed to refresh Worker instances for pool %s", pool.config.pool_id
            )
            pool.instance_ids = []
            pool.discovery_healthy = False

    async def _resync_lease_cache(self, pool: PoolRuntime) -> None:
        pool.lease_watch_healthy = False
        leases, revision = await self.store.lease_snapshot(pool.config.pool_id)
        if revision <= 0:
            raise RuntimeError("Worker lease snapshot omitted its etcd revision")
        pool.lease_cache = leases
        pool.lease_revision = revision
        pool.lease_watch_healthy = True
        pool.wakeup.set()

    async def _lease_watch_loop(self, pool: PoolRuntime) -> None:
        backoff_s = 0.1
        while not self._stop.is_set():
            try:
                async for event in self.store.watch_leases(
                    pool.config.pool_id,
                    start_revision=pool.lease_revision + 1,
                ):
                    pool.lease_revision = max(pool.lease_revision, event.revision)
                    if event.worker_key is not None:
                        if event.lease is None:
                            pool.lease_cache.pop(event.worker_key, None)
                        else:
                            pool.lease_cache[event.worker_key] = event.lease
                    pool.lease_watch_healthy = True
                    pool.wakeup.set()
                    backoff_s = 0.1
                raise RuntimeError("Worker lease watch ended unexpectedly")
            except asyncio.CancelledError:
                raise
            except Exception:
                pool.lease_watch_healthy = False
                logger.exception(
                    "Worker lease watch failed and will be rebuilt: %s",
                    pool.config.pool_id,
                )
                try:
                    await self._resync_lease_cache(pool)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "Worker lease snapshot rebuild failed: %s",
                        pool.config.pool_id,
                    )
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=backoff_s)
                except asyncio.TimeoutError:
                    pass
                backoff_s = min(backoff_s * 2.0, 5.0)

    async def _pool_loop(self, pool: PoolRuntime) -> None:
        next_discovery = 0.0
        while not self._stop.is_set():
            try:
                now = time.monotonic()
                if now >= next_discovery:
                    await self._refresh_instances(pool)
                    next_discovery = now + pool.config.scheduling.discovery_interval_s
                await self._release_reusable_leases(pool)
                dispatched = await self._dispatch_once(pool)
                if dispatched:
                    continue
                pool.wakeup.clear()
                try:
                    await asyncio.wait_for(
                        pool.wakeup.wait(),
                        timeout=pool.config.scheduling.dispatch_interval_s,
                    )
                except asyncio.TimeoutError:
                    pass
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "pool dispatcher iteration failed and will be retried: %s",
                    pool.config.pool_id,
                )
                pool.wakeup.clear()
                try:
                    await asyncio.wait_for(
                        pool.wakeup.wait(),
                        timeout=max(
                            pool.config.scheduling.dispatch_interval_s,
                            0.1,
                        ),
                    )
                except asyncio.TimeoutError:
                    pass

    async def _release_reusable_leases(self, pool: PoolRuntime) -> None:
        current_time = now_ms()
        for lease in await self.pool_leases(pool.config.pool_id):
            if lease.state != "quarantined":
                continue
            expired = (
                lease.reuse_after_ms is not None
                and lease.reuse_after_ms <= current_time
            )
            # A transiently empty discovery snapshot is not proof that the old
            # inference stopped. A re-registered Worker gets a new instance ID
            # and is usable immediately; the old ID remains isolated until the
            # conservative deadline below.
            if expired:
                await self.store.release_lease(pool.config.pool_id, lease.worker_key)

    async def _dispatch_once(self, pool: PoolRuntime) -> bool:
        queued = await self.store.list_queued(pool.config.pool_id, limit=1)
        if not queued or not pool.instance_ids:
            await self._clear_budget_waiter(pool)
            return False
        task_id = queued[0].task.id
        if pool.budget_waiter_id not in {None, task_id}:
            await self.memory_budget.cancel_waiter(pool.budget_waiter_id)
            pool.budget_waiter_id = None
        if self.store.lease_watch_supported and not pool.lease_watch_healthy:
            await self._clear_budget_waiter(pool)
            return False
        leased = {
            lease.worker_key for lease in await self.pool_leases(pool.config.pool_id)
        }
        available = [
            instance_id
            for instance_id in pool.instance_ids
            if worker_key(pool.config.backend_target, instance_id) not in leased
        ]
        if not available:
            await self._clear_budget_waiter(pool)
            return False
        weight_bytes = (
            queued[0].task.estimated_payload_bytes
            or self.config.media.max_task_memory_bytes
        )
        if not await self.memory_budget.try_acquire(task_id, weight_bytes):
            pool.budget_waiter_id = task_id
            return False
        pool.budget_waiter_id = None
        index = pool.cursor % len(available)
        instance_id = available[index]
        pool.cursor = (index + 1) % len(available)
        key = worker_key(pool.config.backend_target, instance_id)
        lease = WorkerLease(
            pool_id=pool.config.pool_id,
            worker_key=key,
            worker_instance_id=instance_id,
            backend_target=pool.config.backend_target,
            task_id=queued[0].task.id,
            owner_generation=self.generation,
            state="reserved",
            heartbeat_at_ms=now_ms(),
            owner_expires_at_ms=now_ms() + 15_000,
        )
        deadline = now_ms() + int(pool.config.scheduling.execution_timeout_s * 1000)
        try:
            reserved = await self.store.reserve(
                queued[0], lease, deadline_at_ms=deadline
            )
        except Exception:
            await self.memory_budget.release(task_id)
            raise
        if reserved is None:
            await self.memory_budget.release(task_id)
            return True
        if self.store.lease_watch_supported:
            lease.etcd_lease_id = reserved.task.worker_lease_id
            pool.lease_cache[lease.worker_key] = lease
        try:
            execution = asyncio.create_task(
                self._run_reserved(pool, reserved),
                name=f"video-task-{reserved.task.id}",
            )
        except Exception:
            await self.memory_budget.release(task_id)
            raise
        self._executions.add(execution)
        execution.add_done_callback(self._execution_done)
        return True

    async def _clear_budget_waiter(self, pool: PoolRuntime) -> None:
        if pool.budget_waiter_id is None:
            return
        await self.memory_budget.cancel_waiter(pool.budget_waiter_id)
        pool.budget_waiter_id = None

    def _execution_done(self, execution: asyncio.Task) -> None:
        self._executions.discard(execution)
        if execution.cancelled():
            return
        error = execution.exception()
        if error is not None:
            logger.error(
                "video execution task escaped its error handler",
                exc_info=(type(error), error, error.__traceback__),
            )

    async def _run_reserved(self, pool: PoolRuntime, stored: StoredTask) -> None:
        context: Any | None = None
        heartbeat: asyncio.Task | None = None
        worker_stream_finished = False
        task = stored.task
        try:
            normalized = await self.artifacts.read_json(task.request_path)
            manifest = await self.artifacts.read_json(task.input_manifest_path)
            payload_build_started = time.monotonic()
            payload: dict[str, Any] | None = await asyncio.to_thread(
                pool.adapter.build_worker_payload,
                normalized,
                manifest,
                self.artifacts.task_root(task.deployment_id, task.pool_id, task.id),
            )
            self._payload_build_count += 1
            self._payload_build_seconds += time.monotonic() - payload_build_started
            stored = await self.store.transition(
                task.id,
                expected={TaskStatus.DISPATCHING},
                expected_revision=stored.revision,
                patch={
                    "status": TaskStatus.IN_PROGRESS,
                    "started_at_ms": now_ms(),
                    "queue_wait_s": max(0.0, (now_ms() - task.queued_at_ms) / 1000.0),
                },
            )
            context = self.context_factory(
                task.id,
                {
                    "task_id": task.id,
                    "pool_id": task.pool_id,
                    "attempt": str(stored.task.attempt),
                },
            )
            current_execution = asyncio.current_task()
            assert current_execution is not None and stored.task.worker_key is not None
            self.running_calls[task.id] = RunningCall(
                context=context,
                execution=current_execution,
                pool_id=task.pool_id,
                worker_key=stored.task.worker_key,
            )
            latest = await self.store.get_task(task.id)
            if latest is None:
                raise RuntimeError("task disappeared before Worker dispatch")
            if latest.task.status in TERMINAL_STATUSES:
                return
            if latest.task.cancel_requested_at_ms is not None:
                context.stop_generating()
                await self._finish_cancelled(pool, latest, quarantine=False)
                return
            heartbeat = asyncio.create_task(
                self._heartbeat(stored.task), name=f"video-heartbeat-{task.id}"
            )
            response_consumer = pool.adapter.create_worker_stream_consumer()

            async def _consume_worker_stream() -> None:
                nonlocal payload, worker_stream_finished
                assert payload is not None
                stream = await pool.client.direct(
                    payload, int(stored.task.worker_instance_id), context
                )
                payload = None
                async for item in stream:
                    if hasattr(item, "is_error") and item.is_error():
                        comments = item.comments() if hasattr(item, "comments") else []
                        raise RuntimeError(
                            "; ".join(comments) or "Dingo direct call failed"
                        )
                    response_consumer.consume(
                        item.data() if hasattr(item, "data") else item
                    )
                worker_stream_finished = True

            await asyncio.wait_for(
                _consume_worker_stream(),
                timeout=pool.config.scheduling.execution_timeout_s,
            )

            latest = await self.store.get_task(task.id)
            if latest is None:
                raise RuntimeError("task disappeared while Worker was running")
            if latest.task.status in TERMINAL_STATUSES:
                return
            if latest.task.cancel_requested_at_ms is not None:
                await self._finish_cancelled(pool, latest, quarantine=False)
                return
            result = response_consumer.finish()
            self._legacy_output_encoded_bytes += len(result.b64_json)
            if len(result.b64_json) > self.config.media.max_result_encoded_bytes:
                self._result_oversize_count += 1
                raise RuntimeError("Worker base64 result exceeds configured maximum")
            await self.store.transition(
                task.id,
                expected={TaskStatus.IN_PROGRESS},
                expected_revision=latest.revision,
                patch={
                    "status": TaskStatus.FINALIZING,
                    "inference_time_s": result.inference_time_s,
                    "stage_durations": dict(result.stage_durations or {}),
                },
            )
            finalize_started = time.monotonic()
            final_path, size, sha256, _media = await self.artifacts.finalize_b64_mp4(
                self.artifacts.task_root(task.deployment_id, task.pool_id, task.id),
                result.b64_json,
                normalized,
                pool.adapter.validate_artifact,
                pool.adapter.prepare_artifact,
                max_result_bytes=self.config.media.max_result_bytes,
            )
            finalize_seconds = time.monotonic() - finalize_started
            self._finalize_count += 1
            self._finalize_seconds += finalize_seconds
            latest = await self.store.get_task(task.id)
            if latest is None:
                raise RuntimeError("task disappeared during finalization")
            if latest.task.status in TERMINAL_STATUSES:
                await asyncio.to_thread(final_path.unlink, True)
                return
            if latest.task.cancel_requested_at_ms is not None:
                await asyncio.to_thread(final_path.unlink, True)
                await self._finish_cancelled(pool, latest, quarantine=False)
                return
            await self.store.transition(
                task.id,
                expected={TaskStatus.FINALIZING},
                expected_revision=latest.revision,
                patch={
                    "status": TaskStatus.COMPLETED,
                    "completed_at_ms": now_ms(),
                    "expires_at_ms": now_ms()
                    + int(self.config.lifecycle.completed_ttl_s * 1000),
                    "result_path": str(final_path),
                    "result_bytes": size,
                    "result_sha256": sha256,
                    "finalize_time_s": finalize_seconds,
                },
                release_lease=True,
            )
        except asyncio.CancelledError:
            if context is not None:
                context.stop_generating()
            await self._finish_failed(
                pool,
                task.id,
                "gateway_shutdown",
                "Gateway stopped during generation",
                quarantine=True,
            )
            raise
        except asyncio.TimeoutError:
            if context is not None:
                context.stop_generating()
            await self._finish_failed(
                pool,
                task.id,
                "execution_timeout",
                "video generation timed out",
                quarantine=True,
            )
        except Exception as exc:
            if context is not None:
                try:
                    context.stop_generating()
                except Exception:
                    logger.exception("failed to stop Worker after task error")
            logger.exception("video task failed: %s", task.id)
            await self._finish_failed(
                pool,
                task.id,
                "worker_failed",
                str(exc),
                quarantine=context is not None and not worker_stream_finished,
            )
        finally:
            released_budget = await self.memory_budget.release(task.id)
            self.running_calls.pop(task.id, None)
            if heartbeat is not None:
                heartbeat.cancel()
                await asyncio.gather(heartbeat, return_exceptions=True)
            if released_budget:
                for runtime in self.pools.values():
                    runtime.wakeup.set()
            else:
                pool.wakeup.set()

    async def _heartbeat(self, task) -> None:
        assert task.worker_key is not None
        while True:
            await asyncio.sleep(5.0)
            try:
                await self.store.heartbeat_lease(
                    task.pool_id,
                    task.worker_key,
                    task.id,
                    task.worker_lease_id,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("failed to heartbeat video task lease: %s", task.id)

    async def _finish_cancelled(
        self, pool: PoolRuntime, stored: StoredTask, *, quarantine: bool
    ) -> None:
        await self.store.transition(
            stored.task.id,
            expected=ACTIVE_STATUSES,
            expected_revision=stored.revision,
            patch={
                "status": TaskStatus.CANCELLED,
                "completed_at_ms": now_ms(),
                "expires_at_ms": now_ms()
                + int(self.config.lifecycle.cancelled_ttl_s * 1000),
                "error": terminal_error("cancelled", "video task was cancelled"),
            },
            release_lease=True,
            quarantine_until_ms=(
                now_ms() + int(pool.config.scheduling.abort_grace_s * 1000)
                if quarantine
                else None
            ),
        )

    async def _finish_failed(
        self,
        pool: PoolRuntime,
        task_id: str,
        code: str,
        message: str,
        *,
        quarantine: bool,
    ) -> None:
        latest = await self.store.get_task(task_id)
        if latest is None or latest.task.status in TERMINAL_STATUSES:
            return
        if latest.task.cancel_requested_at_ms is not None:
            try:
                await self._finish_cancelled(pool, latest, quarantine=quarantine)
            except StoreConflict:
                pass
            return
        safe_message = message[:1024] or code
        if code == "worker_failed":
            safe_message = "video Worker execution or result validation failed"
        try:
            await self.store.transition(
                task_id,
                expected=ACTIVE_STATUSES,
                expected_revision=latest.revision,
                patch={
                    "status": TaskStatus.FAILED,
                    "completed_at_ms": now_ms(),
                    "expires_at_ms": now_ms()
                    + int(self.config.lifecycle.failed_ttl_s * 1000),
                    "error": terminal_error(code, safe_message),
                },
                release_lease=True,
                quarantine_until_ms=(
                    max(latest.task.deadline_at_ms or now_ms(), now_ms())
                    + int(pool.config.scheduling.abort_grace_s * 1000)
                    if quarantine
                    else None
                ),
            )
        except StoreConflict:
            logger.info("task %s changed state while recording failure", task_id)

    async def _recover(self) -> None:
        # Recovery only needs queued and active tasks. Terminal records remain
        # queryable and their result metadata is checked on download; walking
        # all retained tasks would make startup time grow with task history.
        for status in (TaskStatus.QUEUED, *ACTIVE_STATUSES):
            after: str | None = None
            while True:
                tasks = await self.store.list_tasks(
                    status=status,
                    after=after,
                    limit=512,
                )
                if not tasks:
                    break
                for stored in tasks:
                    task = stored.task
                    pool = self.pools.get(task.pool_id)
                    if task.status == TaskStatus.QUEUED:
                        if (
                            pool is not None
                            and task.configuration_revision
                            == pool.config.configuration_revision
                            and task.backend_target == pool.config.backend_target
                        ):
                            continue
                        try:
                            await self.store.transition(
                                task.id,
                                expected={TaskStatus.QUEUED},
                                expected_revision=stored.revision,
                                patch={
                                    "status": TaskStatus.FAILED,
                                    "completed_at_ms": now_ms(),
                                    "expires_at_ms": now_ms()
                                    + int(
                                        self.config.lifecycle.failed_ttl_s * 1000
                                    ),
                                    "error": terminal_error(
                                        "configuration_changed",
                                        "task pool configuration changed before dispatch",
                                    ),
                                },
                            )
                        except StoreConflict:
                            pass
                        continue
                    try:
                        await self.store.transition(
                            task.id,
                            expected=ACTIVE_STATUSES,
                            expected_revision=stored.revision,
                            patch={
                                "status": TaskStatus.FAILED,
                                "completed_at_ms": now_ms(),
                                "expires_at_ms": now_ms()
                                + int(self.config.lifecycle.failed_ttl_s * 1000),
                                "error": terminal_error(
                                    "gateway_restarted",
                                    "Gateway restarted while the Worker request was active",
                                ),
                            },
                            release_lease=True,
                            quarantine_until_ms=max(
                                task.deadline_at_ms or now_ms(), now_ms()
                            )
                            + int(
                                (
                                    pool.config.scheduling.abort_grace_s
                                    if pool is not None
                                    else 30.0
                                )
                                * 1000
                            ),
                        )
                    except StoreConflict:
                        pass
                after = tasks[-1].task.id
        for pool_id in self.pools:
            await self.store.reconcile_pool(pool_id)

    async def _sweeper_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self.sweep_now()
            except Exception:
                logger.exception("video task sweeper iteration failed")
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=self.config.lifecycle.sweeper_interval_s,
                )
            except asyncio.TimeoutError:
                pass

    async def sweep_now(self) -> None:
        async with self._sweep_lock:
            await self._sweep_once()

    async def expire_terminal(self, stored: StoredTask) -> StoredTask:
        current = now_ms()
        task = stored.task
        if task.status in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }:
            stored = await self.store.transition(
                task.id,
                expected={task.status},
                expected_revision=stored.revision,
                patch={
                    "status": TaskStatus.EXPIRED,
                    "expired_at_ms": current,
                    # Keep the task immediately due until its artifacts are
                    # durably removed and artifact_deleted_at_ms is recorded.
                    "expires_at_ms": current,
                },
            )
            task = stored.task
        if task.status != TaskStatus.EXPIRED:
            return stored
        if task.artifact_deleted_at_ms is not None:
            return stored
        task_root = self.artifacts.task_root(
            task.deployment_id, task.pool_id, task.id
        )
        self._artifact_released_bytes += await self.artifacts.discard(task_root)
        return await self.store.transition(
            task.id,
            expected={TaskStatus.EXPIRED},
            expected_revision=stored.revision,
            patch={
                "artifact_deleted_at_ms": now_ms(),
                "expires_at_ms": now_ms()
                + int(self.config.lifecycle.tombstone_ttl_s * 1000),
            },
        )

    async def _cleanup_orphan_tasks(self) -> None:
        candidates = await self.artifacts.orphan_task_candidates(
            self.config.deployment_id,
            tuple(self.pools),
            minimum_age_s=self.config.lifecycle.orphan_grace_s,
        )
        self._orphan_candidates_total += len(candidates)
        missing = []
        # Resolve the entire candidate set before moving anything. If etcd is
        # unavailable or any lookup is indeterminate, the exception aborts the
        # round and no task directory is touched.
        for candidate in candidates:
            if await self.store.get_task(candidate.task_id) is None:
                missing.append(candidate)
        for candidate in missing:
            try:
                moved = await self.artifacts.trash_orphan(
                    candidate,
                    dry_run=self.config.lifecycle.orphan_cleanup_dry_run,
                )
                if moved is not None and not self.config.lifecycle.orphan_cleanup_dry_run:
                    self._orphan_trashed_total += 1
            except Exception:
                self._artifact_cleanup_failures += 1
                logger.exception(
                    "failed to move orphan task artifact to trash: %s",
                    candidate.task_id,
                )

    async def _sweep_once(self) -> None:
        current = now_ms()
        await self.artifacts.cleanup_orphan_uploads(
            minimum_age_s=self.config.lifecycle.upload_grace_s
        )
        _trash_removed, trash_released = await self.artifacts.cleanup_trash(
            minimum_age_s=self.config.lifecycle.trash_grace_s
        )
        self._artifact_released_bytes += trash_released
        monotonic_now = time.monotonic()
        if monotonic_now >= self._next_orphan_scan:
            await self._cleanup_orphan_tasks()
            self._next_orphan_scan = (
                monotonic_now + self.config.lifecycle.orphan_scan_interval_s
            )
        due_tasks = await self.store.list_due_tasks(
            current, limit=self.config.lifecycle.sweeper_batch_size
        )
        self._sweep_due_tasks = len(due_tasks)
        for stored in due_tasks:
            task = stored.task
            if task.expires_at_ms > current:
                continue
            try:
                if task.status == TaskStatus.QUEUED:
                    await self.store.transition(
                        task.id,
                        expected={TaskStatus.QUEUED},
                        expected_revision=stored.revision,
                        patch={
                            "status": TaskStatus.FAILED,
                            "completed_at_ms": current,
                            "expires_at_ms": current
                            + int(self.config.lifecycle.failed_ttl_s * 1000),
                            "error": terminal_error(
                                "queue_timeout", "video task expired while queued"
                            ),
                        },
                    )
                elif task.status in {
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                    TaskStatus.CANCELLED,
                }:
                    await self.expire_terminal(stored)
                    self._expired_tasks_total += 1
                elif task.status == TaskStatus.EXPIRED:
                    cleaned = await self.expire_terminal(stored)
                    if cleaned.task.expires_at_ms <= current:
                        await self.store.delete_expired(cleaned)
            except StoreConflict:
                continue
            except Exception:
                self._artifact_cleanup_failures += 1
                logger.exception("failed to sweep video task: %s", task.id)
                continue
