# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-pool FIFO dispatch, sticky direct calls, cancellation and recovery."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
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
from dingo.video_gateway.telemetry import GatewayTelemetry
from dingo.common.video_task_protocol import detached_envelope

logger = logging.getLogger(__name__)
_GATEWAY_OWNER_TTL_S = 15
_WORKER_LEASE_HEARTBEAT_INTERVAL_S = 5.0
_DETACHED_STATUS_POLL_S = 0.5
_DETACHED_WORKER_STALE_S = 20.0


class _DetachedWorkerCancelled(RuntimeError):
    pass


class _WorkerLeaseLost(RuntimeError):
    pass


class _TaskOwnershipLost(RuntimeError):
    pass


@dataclass(slots=True)
class RunningCall:
    context: Any
    execution: asyncio.Task
    pool_id: str
    worker_key: str
    detached: bool = False


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
        telemetry: GatewayTelemetry | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.artifacts = artifacts
        self.context_factory = context_factory
        self.generation = generation or uuid.uuid4().hex
        self.telemetry = telemetry or GatewayTelemetry()
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
        self._gateway_lease_id: int | None = None
        self._gateway_owner_healthy = not store.gateway_owner_supported
        self._fatal_error: str | None = None
        self._orphan_recovery_lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        return (
            self._ready
            and not self._stop.is_set()
            and self._gateway_owner_healthy
            and self._task_watch_healthy
            and all(
                pool.discovery_healthy and pool.lease_watch_healthy
                for pool in self.pools.values()
            )
        )

    @property
    def live(self) -> bool:
        return self._fatal_error is None

    async def start(self) -> None:
        await self.store.health()
        await self.artifacts.health()
        await self.store.prepare()
        if self.store.gateway_owner_supported:
            self._gateway_lease_id = await self.store.register_gateway(
                self.generation, ttl_s=_GATEWAY_OWNER_TTL_S
            )
            self._gateway_owner_healthy = True
            self._loops.append(
                asyncio.create_task(
                    self._gateway_owner_loop(), name="video-gateway-owner-lease"
                )
            )
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
        if self.store.gateway_owner_supported:
            self._loops.append(
                asyncio.create_task(
                    self._orphan_recovery_loop(), name="video-orphan-recovery"
                )
            )
        self._ready = True

    async def stop(self) -> None:
        self._ready = False
        self._stop.set()
        self._wake_task_waiters()
        for pool in self.pools.values():
            pool.wakeup.set()
        for running in list(self.running_calls.values()):
            if running.detached:
                running.execution.cancel()
                continue
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
        if self._gateway_lease_id is not None:
            try:
                await self.store.unregister_gateway(self._gateway_lease_id)
            except Exception:
                logger.exception("failed to revoke Gateway owner lease during shutdown")
            self._gateway_lease_id = None
        await self.store.close()

    async def _gateway_owner_loop(self) -> None:
        assert self._gateway_lease_id is not None
        failure_started: float | None = None
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=_GATEWAY_OWNER_TTL_S / 3
                )
                continue
            except asyncio.TimeoutError:
                pass
            try:
                await self.store.keepalive_gateway(self._gateway_lease_id)
                self._gateway_owner_healthy = True
                failure_started = None
            except asyncio.CancelledError:
                raise
            except Exception:
                self._gateway_owner_healthy = False
                failure_started = failure_started or time.monotonic()
                logger.exception("failed to keep Gateway owner lease alive")
                if time.monotonic() - failure_started >= _GATEWAY_OWNER_TTL_S:
                    self.telemetry.increment(
                        "dingo_video_gateway_owner_lease_lost_total"
                    )
                    self._fatal_error = "Gateway owner lease was lost"
                    self._ready = False
                    self._stop.set()
                    self._wake_task_waiters()
                    for pool in self.pools.values():
                        pool.wakeup.set()
                    for running in list(self.running_calls.values()):
                        if running.detached:
                            running.execution.cancel()
                            continue
                        try:
                            running.context.stop_generating()
                        except Exception:
                            logger.exception(
                                "failed to stop task after Gateway owner lease loss"
                            )
                    return

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
        before = await self.store.get_task(task_id)
        stored = await self.store.request_cancel(
            task_id,
            terminal_expires_at_ms=now_ms()
            + int(self.config.lifecycle.cancelled_ttl_s * 1000),
        )
        running = self.running_calls.get(task_id)
        pool = self.pools.get(stored.task.pool_id)
        if (
            pool is not None
            and pool.config.execution_mode == "detached"
            and stored.task.execution_token is not None
            and stored.task.attempt > 0
        ):
            await self.artifacts.request_detached_cancel(
                stored.task.deployment_id,
                stored.task.pool_id,
                stored.task.id,
                stored.task.attempt,
                stored.task.execution_token,
            )
        elif running is not None:
            running.context.stop_generating()
        if before is not None and before.revision != stored.revision:
            if before.task.status != stored.task.status:
                self.telemetry.record_transition(
                    "cancelled",
                    before.task,
                    stored.task,
                    gateway_generation=self.generation,
                    revision=stored.revision,
                )
            else:
                self.telemetry.audit_task(
                    "cancel_requested",
                    stored.task,
                    gateway_generation=self.generation,
                    previous_status=before.task.status.value,
                    revision=stored.revision,
                )
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
        cancelled = await self.store.transition(
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
        self.telemetry.record_transition(
            "cancelled",
            stored.task,
            cancelled.task,
            gateway_generation=self.generation,
            revision=cancelled.revision,
            reason="forced_after_abort_grace",
        )
        return cancelled

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
                self.telemetry.increment(
                    "dingo_video_etcd_watch_rebuilds_total",
                    labels={"watch": "tasks", "pool": "_all"},
                )
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
                self.telemetry.increment(
                    "dingo_video_etcd_watch_rebuilds_total",
                    labels={"watch": "worker_leases", "pool": pool.config.pool_id},
                )
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
            execution_token=secrets.token_hex(16),
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
        self.telemetry.record_transition(
            "reserved",
            queued[0].task,
            reserved.task,
            gateway_generation=self.generation,
            revision=reserved.revision,
        )
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

    async def _run_with_lease_monitor(
        self, operation: Any, heartbeat: asyncio.Task
    ) -> Any:
        operation_task = asyncio.create_task(operation)
        try:
            done, _pending = await asyncio.wait(
                {operation_task, heartbeat},
                return_when=asyncio.FIRST_COMPLETED,
            )
            # A completed Worker stream proves the instance is reusable even
            # when the lease heartbeat failed at the same instant.
            if operation_task in done:
                return await operation_task
            if heartbeat.cancelled():
                raise _WorkerLeaseLost("Worker execution lease monitor stopped")
            error = heartbeat.exception()
            if isinstance(error, _WorkerLeaseLost):
                raise error
            raise _WorkerLeaseLost("Worker execution lease monitor failed") from error
        finally:
            if not operation_task.done():
                operation_task.cancel()
                await asyncio.gather(operation_task, return_exceptions=True)

    @staticmethod
    def _same_execution_owner(current: Any, expected: Any) -> bool:
        return (
            current.owner_generation == expected.owner_generation
            and current.attempt == expected.attempt
            and current.execution_token == expected.execution_token
        )

    def _require_execution_owner(self, current: Any, expected: Any) -> None:
        if not self._same_execution_owner(current, expected):
            raise _TaskOwnershipLost(
                "video task execution ownership moved to another Gateway"
            )

    async def _current_owned_execution(self, expected: Any) -> StoredTask | None:
        current = await self.store.get_task(expected.id)
        if current is None or current.task.status in TERMINAL_STATUSES:
            return None
        if not self._same_execution_owner(current.task, expected):
            return None
        return current

    async def _run_reserved(self, pool: PoolRuntime, stored: StoredTask) -> None:
        context: Any | None = None
        heartbeat: asyncio.Task | None = None
        worker_stream_finished = False
        final_path = None
        task = stored.task
        detached = pool.config.execution_mode == "detached"
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
            if stored.task.status == TaskStatus.DISPATCHING:
                before = stored
                stored = await self.store.transition(
                    task.id,
                    expected={TaskStatus.DISPATCHING},
                    expected_revision=stored.revision,
                    patch={
                        "status": TaskStatus.IN_PROGRESS,
                        "started_at_ms": stored.task.started_at_ms or now_ms(),
                        "queue_wait_s": stored.task.queue_wait_s
                        if stored.task.queue_wait_s is not None
                        else max(
                            0.0, (now_ms() - task.queued_at_ms) / 1000.0
                        ),
                    },
                )
                self.telemetry.record_transition(
                    "execution_started",
                    before.task,
                    stored.task,
                    gateway_generation=self.generation,
                    revision=stored.revision,
                )
                if stored.task.queue_wait_s is not None:
                    self.telemetry.record_stage_duration(
                        task.pool_id, "queue", stored.task.queue_wait_s
                    )
            elif not detached or stored.task.status not in {
                TaskStatus.IN_PROGRESS,
                TaskStatus.FINALIZING,
            }:
                raise RuntimeError(
                    f"cannot run reserved task from {stored.task.status.value}"
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
                detached=detached,
            )
            latest = await self.store.get_task(task.id)
            if latest is None:
                raise RuntimeError("task disappeared before Worker dispatch")
            if latest.task.status in TERMINAL_STATUSES:
                return
            self._require_execution_owner(latest.task, task)
            if latest.task.cancel_requested_at_ms is not None:
                if detached:
                    await self._request_detached_cancel(latest.task)
                else:
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

            if detached:
                await self._run_with_lease_monitor(
                    self._consume_detached_worker(
                        pool,
                        stored,
                        payload,
                        context,
                        response_consumer,
                    ),
                    heartbeat,
                )
                payload = None
                worker_stream_finished = True
            else:
                await asyncio.wait_for(
                    self._run_with_lease_monitor(
                        _consume_worker_stream(), heartbeat
                    ),
                    timeout=pool.config.scheduling.execution_timeout_s,
                )

            # The Worker response stream is terminal. It is now safe for a new
            # task to reuse this instance while this Gateway validates and
            # publishes its own immutable result candidate.
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            heartbeat = None

            latest = await self.store.get_task(task.id)
            if latest is None:
                raise RuntimeError("task disappeared while Worker was running")
            if latest.task.status in TERMINAL_STATUSES:
                return
            self._require_execution_owner(latest.task, task)
            if latest.task.cancel_requested_at_ms is not None:
                await self._finish_cancelled(pool, latest, quarantine=False)
                return
            result = response_consumer.finish()
            self._legacy_output_encoded_bytes += len(result.b64_json)
            if len(result.b64_json) > self.config.media.max_result_encoded_bytes:
                self._result_oversize_count += 1
                raise RuntimeError("Worker base64 result exceeds configured maximum")
            if result.inference_time_s is not None:
                self.telemetry.record_stage_duration(
                    task.pool_id, "execution", result.inference_time_s
                )
            if latest.task.status == TaskStatus.IN_PROGRESS:
                finalizing = await self.store.transition(
                    task.id,
                    expected={TaskStatus.IN_PROGRESS},
                    expected_revision=latest.revision,
                    patch={
                        "status": TaskStatus.FINALIZING,
                        "inference_time_s": result.inference_time_s,
                        "stage_durations": dict(result.stage_durations or {}),
                    },
                )
                self.telemetry.record_transition(
                    "finalization_started",
                    latest.task,
                    finalizing.task,
                    gateway_generation=self.generation,
                    revision=finalizing.revision,
                )
            elif latest.task.status != TaskStatus.FINALIZING:
                raise RuntimeError(
                    f"task reached unexpected status {latest.task.status.value} "
                    "before finalization"
                )
            finalize_started = time.monotonic()
            token_digest = hashlib.sha256(
                (stored.task.execution_token or "legacy").encode()
            ).hexdigest()[:16]
            final_path, size, sha256, _media = await self.artifacts.finalize_b64_mp4(
                self.artifacts.task_root(task.deployment_id, task.pool_id, task.id),
                result.b64_json,
                normalized,
                pool.adapter.validate_artifact,
                pool.adapter.prepare_artifact,
                max_result_bytes=self.config.media.max_result_bytes,
                publication_scope=f"a{stored.task.attempt}-{token_digest}",
            )
            finalize_seconds = time.monotonic() - finalize_started
            self._finalize_count += 1
            self._finalize_seconds += finalize_seconds
            self.telemetry.record_stage_duration(
                task.pool_id, "finalize", finalize_seconds
            )
            latest = await self.store.get_task(task.id)
            if latest is None:
                raise RuntimeError("task disappeared during finalization")
            if latest.task.status in TERMINAL_STATUSES:
                await asyncio.to_thread(final_path.unlink, True)
                final_path = None
                return
            self._require_execution_owner(latest.task, task)
            if latest.task.cancel_requested_at_ms is not None:
                await asyncio.to_thread(final_path.unlink, True)
                final_path = None
                await self._finish_cancelled(pool, latest, quarantine=False)
                return
            try:
                completed = await self.store.transition(
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
            except StoreConflict:
                # Another owner may have published a different immutable
                # candidate. Delete only this Gateway's candidate.
                await asyncio.to_thread(final_path.unlink, True)
                final_path = None
                current = await self.store.get_task(task.id)
                if current is not None and current.task.status in TERMINAL_STATUSES:
                    return
                if current is not None and not self._same_execution_owner(
                    current.task, task
                ):
                    return
                raise
            final_path = None
            self.telemetry.record_transition(
                "completed",
                latest.task,
                completed.task,
                gateway_generation=self.generation,
                revision=completed.revision,
            )
        except _TaskOwnershipLost:
            if final_path is not None:
                await asyncio.to_thread(final_path.unlink, True)
                final_path = None
            logger.info(
                "Gateway relinquished task %s after execution ownership moved",
                task.id,
            )
        except asyncio.CancelledError:
            if detached:
                logger.info(
                    "Gateway relinquished detached task %s; "
                    "Worker execution remains independent",
                    task.id,
                )
                raise
            if await self._current_owned_execution(task) is None:
                raise
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
            if await self._current_owned_execution(task) is None:
                return
            if detached and task.execution_token is not None:
                await self._request_detached_cancel(task)
            elif context is not None:
                context.stop_generating()
            await self._finish_failed(
                pool,
                task.id,
                "execution_timeout",
                "video generation timed out",
                quarantine=True,
            )
        except _WorkerLeaseLost as exc:
            if await self._current_owned_execution(task) is None:
                return
            if detached and task.execution_token is not None:
                try:
                    await self._request_detached_cancel(task)
                except Exception:
                    logger.exception(
                        "failed to cancel detached task after lease loss: %s",
                        task.id,
                    )
            elif context is not None:
                context.stop_generating()
            await self._finish_failed(
                pool,
                task.id,
                "worker_lease_lost",
                str(exc),
                quarantine=True,
            )
        except _DetachedWorkerCancelled:
            latest = await self.store.get_task(task.id)
            if (
                latest is not None
                and latest.task.status not in TERMINAL_STATUSES
                and self._same_execution_owner(latest.task, task)
            ):
                await self._finish_cancelled(pool, latest, quarantine=True)
        except Exception as exc:
            if final_path is not None:
                await asyncio.to_thread(final_path.unlink, True)
                final_path = None
            if await self._current_owned_execution(task) is None:
                logger.info(
                    "ignored stale task failure after execution ownership moved: %s",
                    task.id,
                )
                return
            if detached and task.execution_token is not None:
                try:
                    await self._request_detached_cancel(task)
                except Exception:
                    logger.exception(
                        "failed to request detached Worker cancellation: %s", task.id
                    )
            elif context is not None:
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

    async def _request_detached_cancel(self, task: Any) -> None:
        if task.execution_token is None or task.attempt < 1:
            return
        await self.artifacts.request_detached_cancel(
            task.deployment_id,
            task.pool_id,
            task.id,
            task.attempt,
            task.execution_token,
        )

    async def _consume_detached_worker(
        self,
        pool: PoolRuntime,
        stored: StoredTask,
        payload: dict[str, Any],
        context: Any,
        response_consumer: Any,
    ) -> None:
        task = stored.task
        if (
            task.execution_token is None
            or task.attempt < 1
            or task.worker_instance_id is None
        ):
            raise RuntimeError("detached task reservation metadata is incomplete")

        async def _status() -> dict[str, Any] | None:
            return await self.artifacts.read_detached_status(
                task.deployment_id,
                task.pool_id,
                task.id,
                task.attempt,
                task.execution_token,
            )

        worker_status = await _status()
        if worker_status is None:
            submit = detached_envelope(
                op="submit",
                deployment_id=task.deployment_id,
                pool_id=task.pool_id,
                task_id=task.id,
                attempt=task.attempt,
                execution_token=task.execution_token,
                payload=payload,
            )
            stream = await pool.client.direct(
                submit, int(task.worker_instance_id), context
            )
            acknowledgements: list[dict[str, Any]] = []
            async for item in stream:
                if hasattr(item, "is_error") and item.is_error():
                    comments = item.comments() if hasattr(item, "comments") else []
                    raise RuntimeError(
                        "; ".join(comments) or "detached Worker submit failed"
                    )
                value = item.data() if hasattr(item, "data") else item
                if not isinstance(value, dict):
                    raise RuntimeError("detached Worker acknowledgement is invalid")
                acknowledgements.append(value)
                if len(acknowledgements) > 1:
                    raise RuntimeError(
                        "detached Worker returned multiple acknowledgements"
                    )
            if len(acknowledgements) != 1:
                raise RuntimeError("detached Worker returned no acknowledgement")
            acknowledgement = acknowledgements[0]
            if (
                acknowledgement.get("task_id") != task.id
                or acknowledgement.get("attempt") != task.attempt
                or acknowledgement.get("execution_token") != task.execution_token
                or acknowledgement.get("state")
                not in {"accepted", "running", "completed"}
            ):
                raise RuntimeError("detached Worker acknowledgement identity mismatch")

        while True:
            latest = await self.store.get_task(task.id)
            if latest is None:
                raise RuntimeError("task disappeared while detached Worker was running")
            if latest.task.status in TERMINAL_STATUSES:
                raise _DetachedWorkerCancelled("task became terminal")
            self._require_execution_owner(latest.task, task)
            if latest.task.cancel_requested_at_ms is not None:
                await self._request_detached_cancel(latest.task)
            worker_status = await _status()
            if worker_status is not None:
                state = worker_status.get("state")
                if state == "completed":
                    response_sha256 = worker_status.get("response_sha256")
                    response_bytes = worker_status.get("response_bytes")
                    if (
                        not isinstance(response_sha256, str)
                        or len(response_sha256) != 64
                        or not isinstance(response_bytes, int)
                        or isinstance(response_bytes, bool)
                        or response_bytes < 1
                        or response_bytes
                        > self.config.media.max_result_encoded_bytes + 1024 * 1024
                    ):
                        raise RuntimeError(
                            "detached Worker completed metadata is invalid"
                        )
                    consumed = await self.artifacts.consume_detached_response(
                        task.deployment_id,
                        task.pool_id,
                        task.id,
                        task.attempt,
                        task.execution_token,
                        response_consumer,
                        expected_sha256=response_sha256,
                        max_response_bytes=self.config.media.max_result_encoded_bytes
                        + 1024 * 1024,
                    )
                    if consumed != response_bytes:
                        raise RuntimeError(
                            "detached Worker response size mismatch"
                        )
                    return
                if state == "failed":
                    raise RuntimeError("detached Worker reported execution failure")
                if state == "cancelled":
                    raise _DetachedWorkerCancelled("detached Worker cancelled task")
                updated_at_ms = worker_status.get("updated_at_ms")
                if (
                    state in {"accepted", "running"}
                    and isinstance(updated_at_ms, int)
                    and task.worker_instance_id not in pool.instance_ids
                    and now_ms() - updated_at_ms
                    > int(_DETACHED_WORKER_STALE_S * 1000)
                ):
                    raise RuntimeError(
                        "detached Worker disappeared and its heartbeat is stale"
                    )
            remaining_ms = (task.deadline_at_ms or 0) - now_ms()
            if remaining_ms <= 0:
                raise asyncio.TimeoutError
            await asyncio.sleep(
                min(_DETACHED_STATUS_POLL_S, remaining_ms / 1000.0)
            )

    async def _heartbeat(self, task) -> None:
        assert task.worker_key is not None
        consecutive_failures = 0
        while True:
            await asyncio.sleep(_WORKER_LEASE_HEARTBEAT_INTERVAL_S)
            try:
                await self.store.heartbeat_lease(
                    task.pool_id,
                    task.worker_key,
                    task.id,
                    task.worker_lease_id,
                )
                consecutive_failures = 0
            except asyncio.CancelledError:
                raise
            except StoreConflict as exc:
                self.telemetry.increment(
                    "dingo_video_worker_lease_heartbeat_failures_total",
                    labels={"pool": task.pool_id, "reason": "ownership_lost"},
                )
                self.telemetry.increment(
                    "dingo_video_worker_lease_lost_total",
                    labels={"pool": task.pool_id},
                )
                raise _WorkerLeaseLost(
                    "Worker execution lease ownership was lost"
                ) from exc
            except Exception as exc:
                consecutive_failures += 1
                self.telemetry.increment(
                    "dingo_video_worker_lease_heartbeat_failures_total",
                    labels={"pool": task.pool_id, "reason": "store_unavailable"},
                )
                logger.exception("failed to heartbeat video task lease: %s", task.id)
                # Two failed 5-second heartbeats stop local execution before
                # the 15-second native etcd lease can expire and be reused.
                if consecutive_failures >= 2:
                    self.telemetry.increment(
                        "dingo_video_worker_lease_lost_total",
                        labels={"pool": task.pool_id},
                    )
                    raise _WorkerLeaseLost(
                        "Worker execution lease could not be renewed safely"
                    ) from exc

    async def _finish_cancelled(
        self, pool: PoolRuntime, stored: StoredTask, *, quarantine: bool
    ) -> None:
        cancelled = await self.store.transition(
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
                max(stored.task.deadline_at_ms or now_ms(), now_ms())
                + int(pool.config.scheduling.abort_grace_s * 1000)
                if quarantine
                else None
            ),
        )
        self.telemetry.record_transition(
            "cancelled",
            stored.task,
            cancelled.task,
            gateway_generation=self.generation,
            revision=cancelled.revision,
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
            failed = await self.store.transition(
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
            self.telemetry.record_transition(
                "failed",
                latest.task,
                failed.task,
                gateway_generation=self.generation,
                revision=failed.revision,
                reason=code,
            )
        except StoreConflict:
            logger.info("task %s changed state while recording failure", task_id)

    async def _recover(self) -> None:
        # Recovery only needs queued and active tasks. Terminal records remain
        # queryable and their result metadata is checked on download; walking
        # all retained tasks would make startup time grow with task history.
        for status in (TaskStatus.QUEUED,):
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
                            failed = await self.store.transition(
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
                            self.telemetry.record_transition(
                                "failed",
                                stored.task,
                                failed.task,
                                gateway_generation=self.generation,
                                revision=failed.revision,
                                reason="configuration_changed",
                            )
                        except StoreConflict:
                            pass
                        continue
                after = tasks[-1].task.id
        if self.store.gateway_owner_supported:
            async for stored in self.store.iter_orphaned_active_tasks():
                await self._recover_orphaned_active(stored)
        else:
            for status in ACTIVE_STATUSES:
                after = None
                while True:
                    tasks = await self.store.list_tasks(
                        status=status, after=after, limit=512
                    )
                    if not tasks:
                        break
                    for stored in tasks:
                        await self._recover_orphaned_active(stored)
                    after = tasks[-1].task.id
        for pool_id in self.pools:
            await self.store.reconcile_pool(pool_id)

    async def _recover_orphaned_active(self, stored: StoredTask) -> None:
        task = stored.task
        pool = self.pools.get(task.pool_id)
        if (
            self.store.gateway_owner_supported
            and pool is not None
            and pool.config.execution_mode == "detached"
            and task.execution_token is not None
            and task.worker_key is not None
            and task.worker_instance_id is not None
        ):
            weight_bytes = (
                task.estimated_payload_bytes
                or self.config.media.max_task_memory_bytes
            )
            if not await self.memory_budget.try_acquire(task.id, weight_bytes):
                return
            try:
                claimed = await self.store.claim_orphaned_active(
                    stored, new_owner_generation=self.generation
                )
            except Exception:
                self.telemetry.increment(
                    "dingo_video_ha_takeovers_total",
                    labels={"pool": task.pool_id, "outcome": "failed"},
                )
                await self.memory_budget.release(task.id)
                raise
            if claimed is None:
                self.telemetry.increment(
                    "dingo_video_ha_takeovers_total",
                    labels={"pool": task.pool_id, "outcome": "contended"},
                )
                await self.memory_budget.release(task.id)
                return
            if self.store.lease_watch_supported:
                lease = WorkerLease(
                    pool_id=claimed.task.pool_id,
                    worker_key=claimed.task.worker_key,
                    worker_instance_id=claimed.task.worker_instance_id,
                    backend_target=claimed.task.backend_target,
                    task_id=claimed.task.id,
                    owner_generation=self.generation,
                    execution_token=claimed.task.execution_token,
                    state="running",
                    heartbeat_at_ms=now_ms(),
                    owner_expires_at_ms=now_ms() + 15_000,
                    etcd_lease_id=claimed.task.worker_lease_id,
                )
                pool.lease_cache[lease.worker_key] = lease
            execution = asyncio.create_task(
                self._run_reserved(pool, claimed),
                name=f"video-recovered-{claimed.task.id}",
            )
            self._executions.add(execution)
            execution.add_done_callback(self._execution_done)
            self.telemetry.increment(
                "dingo_video_ha_takeovers_total",
                labels={"pool": claimed.task.pool_id, "outcome": "claimed"},
            )
            self.telemetry.audit_task(
                "ha_takeover",
                claimed.task,
                gateway_generation=self.generation,
                previous_status=stored.task.status.value,
                revision=claimed.revision,
                reason="expired_gateway_owner",
            )
            logger.info(
                "claimed detached task %s from expired Gateway owner",
                claimed.task.id,
            )
            return
        error = (
            terminal_error(
                "gateway_owner_lost",
                "the owning Gateway lease expired during Worker execution",
            )
            if self.store.gateway_owner_supported
            else terminal_error(
                "gateway_restarted",
                "Gateway restarted while the Worker request was active",
            )
        )
        try:
            failed = await self.store.transition(
                task.id,
                expected=ACTIVE_STATUSES,
                expected_revision=stored.revision,
                patch={
                    "status": TaskStatus.FAILED,
                    "completed_at_ms": now_ms(),
                    "expires_at_ms": now_ms()
                    + int(self.config.lifecycle.failed_ttl_s * 1000),
                    "error": error,
                },
                release_lease=True,
                quarantine_until_ms=max(task.deadline_at_ms or now_ms(), now_ms())
                + int(
                    (
                        pool.config.scheduling.abort_grace_s
                        if pool is not None
                        else 30.0
                    )
                    * 1000
                ),
            )
            self.telemetry.record_transition(
                "failed",
                stored.task,
                failed.task,
                gateway_generation=self.generation,
                revision=failed.revision,
                reason=error.code,
            )
        except StoreConflict:
            pass

    async def _orphan_recovery_loop(self) -> None:
        while not self._stop.is_set():
            try:
                async with self._orphan_recovery_lock:
                    async for stored in self.store.iter_orphaned_active_tasks():
                        if self._stop.is_set():
                            return
                        await self._recover_orphaned_active(stored)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("orphaned video task recovery iteration failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass

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
            before = stored
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
            self.telemetry.record_transition(
                "expired",
                before.task,
                stored.task,
                gateway_generation=self.generation,
                revision=stored.revision,
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
        cleaned = await self.store.transition(
            task.id,
            expected={TaskStatus.EXPIRED},
            expected_revision=stored.revision,
            patch={
                "artifact_deleted_at_ms": now_ms(),
                "expires_at_ms": now_ms()
                + int(self.config.lifecycle.tombstone_ttl_s * 1000),
            },
        )
        self.telemetry.audit_task(
            "artifacts_deleted",
            cleaned.task,
            gateway_generation=self.generation,
            previous_status=stored.task.status.value,
            revision=cleaned.revision,
        )
        return cleaned

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
                    failed = await self.store.transition(
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
                    self.telemetry.record_transition(
                        "failed",
                        stored.task,
                        failed.task,
                        gateway_generation=self.generation,
                        revision=failed.revision,
                        reason="queue_timeout",
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
