# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-pool FIFO dispatch, sticky direct calls, cancellation and recovery."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import secrets
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from dingo.common.video_result_file import INLINE_RESULT_FORMAT, normalize_inline_result
from dingo.common.video_task_protocol import (
    ENVELOPE_KEY,
    EXECUTION_CAPACITY_CAPABILITY,
    PREFETCH_CAPABILITY,
    WAIT_TERMINAL_CAPABILITY,
    detached_envelope,
)
from dingo.video_gateway.adapters.base import VideoBackendAdapter
from dingo.video_gateway.artifact_store import FileArtifactStore
from dingo.video_gateway.config import GatewayConfig, PoolConfig
from dingo.video_gateway.dingo_adapter import (
    ContextFactory,
    EndpointClient,
    create_context,
)
from dingo.video_gateway.errors import (
    HandoffReservationLost,
    ResultTooLarge,
    StoreConflict,
    WorkerUnavailable,
    worker_execution_error,
)
from dingo.video_gateway.file_io import run_file_io
from dingo.video_gateway.finalization import ResultFinalizer
from dingo.video_gateway.memory_budget import (
    MemoryBudgetSnapshot,
    WeightedMemoryBudget,
)
from dingo.video_gateway.models import (
    ACTIVE_STATUSES,
    TERMINAL_STATUSES,
    StoredTask,
    TaskError,
    TaskStatus,
    WorkerLease,
    now_ms,
)
from dingo.video_gateway.result_handoff import HANDOFF_KEY, make_handoff, read_handoff
from dingo.video_gateway.task_store import (
    TaskStore,
    retry_excludes_worker,
    terminal_error,
    worker_key,
)
from dingo.video_gateway.telemetry import GatewayTelemetry

logger = logging.getLogger(__name__)
_GATEWAY_OWNER_TTL_S = 15
_WORKER_LEASE_HEARTBEAT_INTERVAL_S = 5.0
_DETACHED_STATUS_FALLBACK_S = 1.0
_DETACHED_WAIT_ATTACH_TIMEOUT_S = 1.0
_DETACHED_WAIT_RETRY_INITIAL_S = 0.2
_DETACHED_WAIT_RETRY_MAX_S = 5.0
_DETACHED_WORKER_STALE_S = 20.0
_WORKER_LIVENESS_CHECK_INTERVAL_S = 5.0
_WORKER_LIVENESS_CHECK_CONCURRENCY = 4
_DISCOVERY_MISMATCH_MIN_CHECKS = 3
_DISCOVERY_RECOVERY_LOCK_TTL_S = 15
_DISCOVERY_RESTART_DRAIN_S = 5.0


class _DetachedWorkerCancelled(RuntimeError):
    pass


_RetryableWorkerFailure = WorkerUnavailable


class _CancellationConfirmationTimedOut(RuntimeError):
    pass


class _WorkerLeaseLost(RuntimeError):
    pass


class _TaskOwnershipLost(RuntimeError):
    pass


class _DetachedWaitUnavailable(RuntimeError):
    pass


class _DetachedWaitProtocolError(RuntimeError):
    pass


@dataclass(slots=True)
class RunningCall:
    context: Any
    execution: asyncio.Task
    pool_id: str
    worker_key: str
    detached: bool = False
    worker_accepted: bool = False
    task_changed: asyncio.Event = field(default_factory=asyncio.Event)


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
    # Fixed per physical registration; failed probes are retried with backoff.
    capacity_cache: dict[int, tuple[int, float]] = field(default_factory=dict)
    prefetch_instances: set[int] = field(default_factory=set)


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
        self._worker_retry_once = os.getenv("DINGO_VIDEO_WORKER_RETRY_ONCE", "0") == "1"
        self._retry_budget_limit = int(os.getenv("DINGO_VIDEO_RETRY_BUDGET", "32"))
        self._retry_wait_timeout_s = float(
            os.getenv("DINGO_VIDEO_RETRY_WAIT_TIMEOUT_S", "600")
        )
        self._failed_instance_backoff_s = float(
            os.getenv("DINGO_VIDEO_RETRY_FAILED_INSTANCE_BACKOFF_S", "30")
        )
        if (
            not 1 <= self._retry_budget_limit <= 1024
            or not 0 < self._retry_wait_timeout_s <= 86400
        ):
            raise ValueError("invalid retry budget or wait timeout")
        if not 0 <= self._failed_instance_backoff_s <= 86400:
            raise ValueError("invalid failed instance backoff")
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
        self._finalizing: dict[str, str] = {}
        self._finalization_slots = {
            pool.pool_id: asyncio.Semaphore(pool.scheduling.finalization_concurrency)
            for pool in config.pools
        }
        self._finalizer = ResultFinalizer(
            store, artifacts, config, self.telemetry, self.generation
        )
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
        self._draining = False
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
        self._fatal_restart_pending = False
        self._fatal_error: str | None = None
        self._fatal_event = asyncio.Event()
        self._orphan_recovery_lock = asyncio.Lock()
        self._worker_liveness_checks = asyncio.Semaphore(
            _WORKER_LIVENESS_CHECK_CONCURRENCY
        )
        self._discovery_mismatch_started: dict[str, float] = {}
        self._discovery_mismatch_checks: dict[str, int] = {}

    @property
    def ready(self) -> bool:
        return (
            self._ready
            and not self._draining
            and not self._stop.is_set()
            and self._gateway_owner_healthy
            and self._task_watch_healthy
            and all(
                pool.discovery_healthy and pool.lease_watch_healthy
                for pool in self.pools.values()
            )
        )

    @property
    def draining(self) -> bool:
        return self._draining

    @property
    def live(self) -> bool:
        return self._fatal_error is None

    def begin_drain(self) -> bool:
        """Stop accepting new work while keeping reads and the listener alive.

        Return True only for the transition into draining. The operation is
        deliberately idempotent because both the Kubernetes preStop hook and
        the SIGTERM fallback may request it.
        """

        if self._draining:
            return False
        self._draining = True
        for pool in self.pools.values():
            pool.wakeup.set()
        return True

    async def wait_fatal(self) -> str:
        await self._fatal_event.wait()
        return self._fatal_error or "video Gateway requested restart"

    async def _request_fatal_restart(
        self, reason: str, *, drain_s: float = 0.0
    ) -> None:
        if self._fatal_restart_pending or self._fatal_error is not None:
            return
        self._fatal_restart_pending = True
        self.begin_drain()
        self._ready = False
        if drain_s > 0:
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=drain_s)
            except asyncio.TimeoutError:
                pass
        if self._stop.is_set():
            return
        self._fatal_error = reason
        self._fatal_event.set()

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
        if self.config.runtime.discovery_watchdog.enabled:
            if not self.store.discovery_truth_supported:
                raise RuntimeError(
                    "Dynamo discovery watchdog requires an etcd discovery truth source"
                )
            self._loops.append(
                asyncio.create_task(
                    self._discovery_watchdog_loop(),
                    name="video-discovery-watchdog",
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
        self._loops.append(
            asyncio.create_task(
                self._owned_finalization_recovery_loop(),
                name="video-finalization-recovery",
            )
        )
        self._ready = True

    async def stop(self) -> None:
        self.begin_drain()
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
                    await self._request_fatal_restart("Gateway owner lease was lost")
                    return

    def has_workers(self, pool_id: str) -> bool:
        pool = self.pools[pool_id]
        # Registration alone is not capacity; saturation still allows queuing.
        return bool(self._worker_slots(pool))

    def pool_instances(self, pool_id: str) -> list[int]:
        return list(self.pools[pool_id].instance_ids)

    def pool_capacity_snapshot(
        self, pool_id: str, leases: list[WorkerLease]
    ) -> dict[str, int]:
        """Local discovery + shared leases, without per-Worker scrape RPCs.

        Admission leases are NOT engine-running tasks. With prefetch enabled,
        a lease can cover execution, Worker queuing or output writing. Slot
        numbers are fungible and cannot identify the prefetch occupant.
        """
        pool = self.pools[pool_id]
        slots = self._worker_slots(pool)
        keys = {
            worker_key(pool.config.backend_target, instance, slot)
            for instance, slot in slots
        }
        registered = {str(i) for i in pool.instance_ids}
        mapped = {
            lease.worker_key: lease for lease in leases if lease.worker_key in keys
        }
        busy = sum(lease.state != "quarantined" for lease in mapped.values())
        quarantined = len(mapped) - busy
        physical_busy = {
            str(lease.worker_instance_id)
            for lease in leases
            if str(lease.worker_instance_id) in registered
            and lease.backend_target == pool.config.backend_target
            and lease.state != "quarantined"
        }
        prefetch = sum(
            instance in pool.prefetch_instances
            for instance in {instance for instance, _ in slots}
        )
        healthy = pool.discovery_healthy and (
            not self.store.lease_watch_supported or pool.lease_watch_healthy
        )
        return {
            "workers": len(registered),
            "worker_busy": len(physical_busy),
            "worker_execution_capacity": len(slots) - prefetch,
            "worker_prefetch_capacity": prefetch,
            "worker_admission_capacity": len(slots),
            "worker_slots_busy": busy,
            "worker_slots_quarantined": quarantined,
            "worker_slots_free": len(keys - mapped.keys()) if healthy else 0,
            "worker_unmapped_leases": len(
                {lease.worker_key for lease in leases} - keys
            ),
            "worker_capacity_view_healthy": int(healthy),
        }

    def pool_finalization_pending(self, pool_id: str) -> int:
        """This Gateway's pending/running finalizers, not a shared pool total."""
        return sum(p == pool_id for p in self._finalizing.values())

    async def pool_leases(self, pool_id: str) -> list[WorkerLease]:
        pool = self.pools[pool_id]
        if self.store.lease_watch_supported:
            return list(pool.lease_cache.values())
        return await self.store.list_leases(pool_id)

    def notify(self, pool_id: str) -> None:
        self.pools[pool_id].wakeup.set()

    async def memory_budget_snapshot(self) -> MemoryBudgetSnapshot:
        return await self.memory_budget.snapshot()

    async def _shrink_to_result_memory_budget(self, task_id: str) -> None:
        released = await self.memory_budget.shrink(
            task_id, self.config.media.result_task_memory_bytes
        )
        if released:
            # The budget is process-global, so an input-heavy task completing
            # submission can unblock a queued task from any configured pool.
            for runtime in self.pools.values():
                runtime.wakeup.set()

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
        if running is not None and not running.detached:
            try:
                running.context.stop_generating()
            except Exception:
                # The durable task record is the source of truth. The owner
                # monitor retries notification outside the HTTP request path.
                logger.exception(
                    "failed to notify local Worker of cancellation: %s", task_id
                )
        if running is not None:
            running.task_changed.set()
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
        for running in self.running_calls.values():
            running.task_changed.set()

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
                        running = self.running_calls.get(event.task_id)
                        if running is not None:
                            running.task_changed.set()
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
        pool.capacity_cache = {
            i: v for i, v in pool.capacity_cache.items() if i in pool.instance_ids
        }
        if hasattr(pool, "prefetch_instances"):
            pool.prefetch_instances.intersection_update(pool.instance_ids)
        if pool.config.scheduling.worker_capacity > 1 or getattr(
            pool.config.scheduling, "worker_prefetch_capacity", 0
        ):
            semaphore = asyncio.Semaphore(8)

            async def probe(instance: int) -> None:
                if pool.capacity_cache.get(instance, (0, 0))[1] > time.monotonic():
                    return
                async with semaphore:
                    pool.capacity_cache[instance] = await self._worker_capacity(
                        pool, instance
                    )

            await asyncio.gather(*(probe(i) for i in pool.instance_ids))

    async def _worker_capacity(
        self, pool: PoolRuntime, instance: int
    ) -> tuple[int, float]:
        pool.prefetch_instances.discard(instance)
        context = self.context_factory(
            f"capacity-{uuid.uuid4().hex}", {"pool_id": pool.config.pool_id}
        )

        async def query() -> int:
            stream = await pool.client.direct(
                {ENVELOPE_KEY: {"schema_version": 1, "op": "capabilities"}},
                instance,
                context,
            )
            response = None
            async for item in stream:
                if hasattr(item, "is_error") and item.is_error():
                    raise RuntimeError("Worker capability query failed")
                value = item.data() if hasattr(item, "data") else item
                if response is not None or not isinstance(value, dict):
                    raise ValueError("invalid Worker capabilities response")
                response = value
            if (
                response is None
                or response.get("schema_version") != 1
                or EXECUTION_CAPACITY_CAPABILITY not in response.get("capabilities", [])
            ):
                raise ValueError("Worker does not advertise execution capacity")
            capacity = response.get("execution_capacity")
            if (
                isinstance(capacity, bool)
                or not isinstance(capacity, int)
                or capacity < 1
            ):
                raise ValueError("invalid Worker execution capacity")
            if response.get("accepting") is not True:
                return 0
            effective = min(capacity, pool.config.scheduling.worker_capacity)
            prefetch = getattr(pool.config.scheduling, "worker_prefetch_capacity", 0)
            if (
                prefetch
                and capacity == pool.config.scheduling.worker_capacity
                and PREFETCH_CAPABILITY in response.get("capabilities", [])
            ):
                advertised = response.get("prefetch_capacity")
                admission = response.get("admission_capacity")
                if (
                    type(advertised) is not int
                    or advertised not in {0, 1}
                    or type(admission) is not int
                    or admission != capacity + advertised
                ):
                    raise ValueError("invalid Worker prefetch capacity")
                if advertised:
                    pool.prefetch_instances.add(instance)
                    effective += min(prefetch, advertised)
            return effective

        try:
            capacity = await asyncio.wait_for(query(), timeout=2.0)
            return capacity, float("inf") if capacity else time.monotonic() + 5.0
        except Exception:
            # Legacy Workers are still usable as single-slot instances. Never
            # infer N slots from the configured ceiling or a failed RPC.
            logger.warning("Worker %s capacity unavailable; using one slot", instance)
            return 1, time.monotonic() + 30.0
        finally:
            context.stop_generating()

    @staticmethod
    def _worker_slots(pool: PoolRuntime) -> list[tuple[int, int]]:
        ceiling = pool.config.scheduling.worker_capacity + getattr(
            pool.config.scheduling, "worker_prefetch_capacity", 0
        )
        return [
            (instance, slot)
            for instance in pool.instance_ids
            for slot in range(
                1
                if ceiling == 1
                else min(ceiling, pool.capacity_cache.get(instance, (0, 0))[0])
            )
        ]

    async def _detached_submit_ack(self, pool, task, submit, context, validate):
        """Busy means not executed: retry admission, not the task attempt."""
        while True:
            if task.deadline_at_ms is not None and now_ms() >= task.deadline_at_ms:
                raise asyncio.TimeoutError("Worker admission deadline exceeded")
            stream = await pool.client.direct(
                submit, int(task.worker_instance_id), context
            )
            response = None
            async for item in stream:
                if hasattr(item, "is_error") and item.is_error():
                    comments = item.comments() if hasattr(item, "comments") else []
                    raise RuntimeError(
                        "; ".join(comments) or "detached Worker submit failed"
                    )
                value = item.data() if hasattr(item, "data") else item
                if response is not None or not isinstance(value, dict):
                    raise RuntimeError("invalid detached Worker acknowledgement")
                response = validate(value)
            if response is None:
                raise RuntimeError("detached Worker returned no acknowledgement")
            if response.get("state") != "busy":
                return response
            if response.get("accepted") is not False:
                raise RuntimeError("busy Worker incorrectly acknowledged execution")
            self.telemetry.increment(
                "dingo_video_worker_admission_busy_total",
                labels={"pool": pool.config.pool_id},
            )
            await asyncio.sleep(0.25)

    def _record_discovery_match(
        self,
        pool: PoolRuntime,
        runtime_ids: set[int],
        truth_ids: set[int],
    ) -> bool:
        missing = truth_ids - runtime_ids
        stale = runtime_ids - truth_ids
        labels = {"pool": pool.config.pool_id}
        self.telemetry.set_gauge(
            "dingo_video_discovery_consistent",
            0 if missing or stale else 1,
            labels=labels,
        )
        self.telemetry.set_gauge(
            "dingo_video_discovery_missing_instances", len(missing), labels=labels
        )
        self.telemetry.set_gauge(
            "dingo_video_discovery_stale_instances", len(stale), labels=labels
        )
        if not missing and not stale:
            self._discovery_mismatch_started.pop(pool.config.pool_id, None)
            self._discovery_mismatch_checks.pop(pool.config.pool_id, None)
            self.telemetry.set_gauge(
                "dingo_video_discovery_last_consistent_timestamp_seconds",
                time.time(),
                labels=labels,
            )
            return True
        if missing:
            self.telemetry.increment(
                "dingo_video_discovery_mismatch_checks_total",
                labels={**labels, "direction": "missing_in_runtime"},
            )
        if stale:
            self.telemetry.increment(
                "dingo_video_discovery_mismatch_checks_total",
                labels={**labels, "direction": "stale_in_runtime"},
            )
        self._discovery_mismatch_started.setdefault(
            pool.config.pool_id, time.monotonic()
        )
        self._discovery_mismatch_checks[pool.config.pool_id] = (
            self._discovery_mismatch_checks.get(pool.config.pool_id, 0) + 1
        )
        return False

    async def _discovery_watchdog_loop(self) -> None:
        watchdog = self.config.runtime.discovery_watchdog
        targets = [pool.config.backend_target for pool in self.pools.values()]
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                truth = await self.store.discovery_instance_snapshot(targets)
                mature_mismatches: list[str] = []
                for pool in self.pools.values():
                    runtime_ids = set(pool.client.instance_ids())
                    truth_ids = truth[pool.config.backend_target]
                    if self._record_discovery_match(pool, runtime_ids, truth_ids):
                        continue
                    mismatch_started = self._discovery_mismatch_started[
                        pool.config.pool_id
                    ]
                    mismatch_checks = self._discovery_mismatch_checks[
                        pool.config.pool_id
                    ]
                    if (
                        mismatch_checks >= _DISCOVERY_MISMATCH_MIN_CHECKS
                        and time.monotonic() - mismatch_started
                        >= watchdog.mismatch_grace_s
                    ):
                        mature_mismatches.append(pool.config.pool_id)
                if mature_mismatches:
                    acquired = await self.store.try_acquire_discovery_recovery(
                        self.generation,
                        ttl_s=_DISCOVERY_RECOVERY_LOCK_TTL_S,
                    )
                    if acquired:
                        pools = ",".join(sorted(mature_mismatches))
                        self.telemetry.increment(
                            "dingo_video_discovery_watchdog_restarts_total"
                        )
                        reason = (
                            "Dynamo discovery view remained inconsistent for pools="
                            f"{pools}; restarting this Video Gateway replica"
                        )
                        logger.error(reason)
                        await self._request_fatal_restart(
                            reason, drain_s=_DISCOVERY_RESTART_DRAIN_S
                        )
                        return
            except asyncio.CancelledError:
                raise
            except Exception:
                self.telemetry.increment("dingo_video_discovery_watchdog_errors_total")
                logger.exception("Dynamo discovery watchdog check failed")
            remaining = watchdog.interval_s - (time.monotonic() - started)
            if remaining <= 0:
                await asyncio.sleep(0)
                continue
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                pass

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
                if self._draining:
                    await self._clear_budget_waiter(pool)
                    await self._stop.wait()
                    continue
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
        if (
            sum(
                p == pool.config.pool_id
                for p in getattr(self, "_finalizing", {}).values()
            )
            >= pool.config.scheduling.finalization_pending_limit
        ):
            await self._clear_budget_waiter(pool)
            return False
        # A healthy lease view can prove that no instance is usable without
        # reading queue indexes, task bodies, or shared retry counters. Keep
        # the later checks and reserve CAS: availability can change while the
        # awaited queue reads are in flight.
        skip_reason = None
        if not pool.instance_ids:
            skip_reason = "no_workers"
        elif self.store.lease_watch_supported and not pool.lease_watch_healthy:
            skip_reason = "lease_watch_unhealthy"
        else:
            occupied = {
                lease.worker_key
                for lease in await self.pool_leases(pool.config.pool_id)
            }
            if all(
                worker_key(pool.config.backend_target, instance, slot) in occupied
                for instance, slot in self._worker_slots(pool)
            ):
                skip_reason = "no_free_worker"
        if skip_reason is not None:
            await self._clear_budget_waiter(pool)
            self.telemetry.increment(
                "dingo_video_dispatch_skips_total",
                labels={"pool": pool.config.pool_id, "reason": skip_reason},
            )
            return False
        ledger = hasattr(self.store, "retry_budget_used")
        budget_limit = getattr(self, "_retry_budget_limit", 32)
        queued = await self.store.list_queued(
            pool.config.pool_id,
            limit=min(10000, pool.config.scheduling.queue_limit + budget_limit),
        )
        used = 0
        if ledger:
            waiting = await self.store.retry_queue_depth(pool.config.pool_id)
            used = await self.store.retry_budget_used(pool.config.pool_id)
            self.telemetry.set_gauge(
                "dingo_video_retry_waiting_tasks",
                waiting,
                labels={"pool": pool.config.pool_id},
            )
            self.telemetry.set_gauge(
                "dingo_video_retry_credits_used",
                used,
                labels={"pool": pool.config.pool_id},
            )
            self.telemetry.set_gauge(
                "dingo_video_normal_queue_depth",
                max(0, await self.store.queue_depth(pool.config.pool_id) - waiting),
                labels={"pool": pool.config.pool_id},
            )
        if not queued or not pool.instance_ids:
            await self._clear_budget_waiter(pool)
            return False
        if self.store.lease_watch_supported and not pool.lease_watch_healthy:
            await self._clear_budget_waiter(pool)
            return False
        leased = {
            lease.worker_key for lease in await self.pool_leases(pool.config.pool_id)
        }
        selected = None
        available = []
        # Retry first, but do not let a retry that excludes the only available
        # instance block unrelated runnable work behind it.
        for candidate in sorted(queued, key=lambda item: item.task.attempt == 0):
            if (
                candidate.task.id in self.running_calls
                or candidate.task.expires_at_ms <= now_ms()
            ):
                continue
            if (
                ledger
                and getattr(self, "_worker_retry_once", False)
                and candidate.task.attempt == 0
                and used >= budget_limit
            ):
                continue
            available = [
                (instance, slot)
                for instance, slot in self._worker_slots(pool)
                if worker_key(pool.config.backend_target, instance, slot) not in leased
                and not retry_excludes_worker(
                    candidate.task, pool.config.backend_target, instance
                )
            ]
            if available:
                selected = candidate
                break
        if selected is None:
            await self._clear_budget_waiter(pool)
            return False
        queued = [selected]
        task_id = selected.task.id
        if pool.budget_waiter_id not in {None, task_id}:
            await self.memory_budget.cancel_waiter(pool.budget_waiter_id)
            pool.budget_waiter_id = None
        weight_bytes = (
            queued[0].task.estimated_payload_bytes
            or self.config.media.max_task_memory_bytes
        )
        if not await self.memory_budget.try_acquire(task_id, weight_bytes):
            pool.budget_waiter_id = task_id
            return False
        pool.budget_waiter_id = None
        if self._draining:
            await self.memory_budget.release(task_id)
            return False
        index = pool.cursor % len(available)
        instance_id, slot_id = available[index]
        pool.cursor = (index + 1) % len(available)
        key = worker_key(pool.config.backend_target, instance_id, slot_id)
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
            retry_options = (
                {
                    "reserve_retry": getattr(self, "_worker_retry_once", False)
                    and pool.config.execution_mode == "detached",
                    "retry_limit": budget_limit,
                }
                if ledger
                else {}
            )
            reserved = await self.store.reserve(
                queued[0], lease, deadline_at_ms=deadline, **retry_options
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

    @staticmethod
    def _raise_if_heartbeat_stopped(heartbeat: asyncio.Task) -> None:
        if not heartbeat.done():
            return
        if heartbeat.cancelled():
            raise _WorkerLeaseLost("Worker execution lease monitor stopped")
        error = heartbeat.exception()
        if isinstance(error, _WorkerLeaseLost):
            raise error
        raise _WorkerLeaseLost("Worker execution lease monitor failed") from error

    async def _run_with_lease_monitor(
        self,
        operation: Any,
        heartbeat: asyncio.Task,
        cancellation: Any | None = None,
        liveness: Any | None = None,
    ) -> Any:
        operation_task = asyncio.create_task(operation)
        cancellation_task = (
            asyncio.create_task(cancellation) if cancellation is not None else None
        )
        liveness_task = asyncio.create_task(liveness) if liveness is not None else None
        try:
            monitored = {operation_task, heartbeat}
            if cancellation_task is not None:
                monitored.add(cancellation_task)
            if liveness_task is not None:
                monitored.add(liveness_task)
            done, _pending = await asyncio.wait(
                monitored,
                return_when=asyncio.FIRST_COMPLETED,
            )
            # A completed Worker stream proves the instance is reusable even
            # when the lease heartbeat failed at the same instant.
            if operation_task in done:
                return await operation_task
            if cancellation_task is not None and cancellation_task in done:
                await cancellation_task
                raise RuntimeError("cancellation monitor ended unexpectedly")
            # Losing execution ownership takes precedence over a simultaneous
            # suspicion of Worker loss. An etcd outage is not a Worker crash.
            if heartbeat in done:
                self._raise_if_heartbeat_stopped(heartbeat)
            if liveness_task is not None and liveness_task in done:
                await liveness_task
                raise RuntimeError("Worker liveness monitor ended unexpectedly")
            raise RuntimeError("execution monitor ended unexpectedly")
        finally:
            children = [operation_task]
            children.extend(
                monitor
                for monitor in (cancellation_task, liveness_task)
                if monitor is not None
            )
            for child in children:
                if not child.done():
                    child.cancel()
            await asyncio.gather(*children, return_exceptions=True)

    def _can_check_missing_worker(self, pool: PoolRuntime, task: Any) -> bool:
        return (
            self.store.discovery_truth_supported
            and self._gateway_owner_healthy
            and self._task_watch_healthy
            and pool.discovery_healthy
            and pool.lease_watch_healthy
            and task.worker_instance_id not in pool.instance_ids
            and (task.deadline_at_ms or 0) > now_ms()
        )

    async def _confirm_worker_loss(self, pool: PoolRuntime, task: Any) -> bool:
        """Read remote evidence only for a locally missing execution instance.

        Unknown/corrupt/unavailable evidence is not positive proof of failure.
        The normal registered-Worker path never reads etcd or artifact files.
        """
        if not self._can_check_missing_worker(pool, task):
            return False
        async with self._worker_liveness_checks:
            # Waiting for the shared concurrency limit may have made the
            # suspicion obsolete, or the control plane may have become sick.
            if not self._can_check_missing_worker(pool, task):
                return False
            try:
                truth = await self.store.discovery_instance_snapshot(
                    [pool.config.backend_target]
                )
                if task.worker_instance_id in truth[pool.config.backend_target]:
                    return False
                current = await self._current_owned_execution(task)
                if current is None:
                    raise _TaskOwnershipLost(
                        "task ownership moved during Worker liveness check"
                    )
                if (
                    current.task.status != TaskStatus.IN_PROGRESS
                    or current.task.cancel_requested_at_ms is not None
                ):
                    return False
                status = await self.artifacts.read_detached_status(
                    task.deployment_id,
                    task.pool_id,
                    task.id,
                    task.attempt,
                    task.execution_token,
                )
                if status is None or status.get("state") not in {"accepted", "running"}:
                    return False
                updated_at_ms = status.get("updated_at_ms")
                if (
                    not isinstance(updated_at_ms, int)
                    or isinstance(updated_at_ms, bool)
                    or now_ms() - updated_at_ms <= int(_DETACHED_WORKER_STALE_S * 1000)
                ):
                    return False
                return self._can_check_missing_worker(pool, task)
            except _TaskOwnershipLost:
                raise
            except asyncio.CancelledError:
                raise
            except Exception:
                self.telemetry.increment(
                    "dingo_video_worker_liveness_checks_total",
                    labels={"pool": pool.config.pool_id, "outcome": "inconclusive"},
                )
                logger.debug(
                    "Worker liveness evidence unavailable for %s",
                    task.id,
                    exc_info=True,
                )
                return False

    async def _monitor_worker_liveness(
        self, pool: PoolRuntime, task: Any, running: RunningCall
    ) -> None:
        """Independent of the possibly stuck result/terminal-wait stream."""
        while True:
            await asyncio.sleep(_WORKER_LIVENESS_CHECK_INTERVAL_S)
            if running.worker_accepted and await self._confirm_worker_loss(pool, task):
                self.telemetry.increment(
                    "dingo_video_worker_liveness_checks_total",
                    labels={"pool": pool.config.pool_id, "outcome": "confirmed_lost"},
                )
                raise _RetryableWorkerFailure(
                    "detached Worker disappeared and its heartbeat is stale"
                )

    async def _monitor_cancellation(
        self,
        pool: PoolRuntime,
        expected: Any,
        context: Any,
        running: RunningCall,
    ) -> None:
        """Notify the Worker and bound unconfirmed cancellation in the owner.

        The task watch wakes this monitor when any Gateway persists a cancel
        request. Memory-store tests are woken directly by ``cancel()``. Until
        then this coroutine performs no polling and adds no etcd read load.
        """

        while True:
            await running.task_changed.wait()
            running.task_changed.clear()
            latest = await self.store.get_task(expected.id)
            if latest is None or latest.task.status in TERMINAL_STATUSES:
                raise _TaskOwnershipLost(
                    "video task became terminal while Worker was running"
                )
            self._require_execution_owner(latest.task, expected)
            requested_at_ms = latest.task.cancel_requested_at_ms
            if requested_at_ms is None:
                continue
            if running.detached:
                await self._request_detached_cancel(latest.task)
            else:
                context.stop_generating()
            deadline_ms = requested_at_ms + int(
                pool.config.scheduling.abort_grace_s * 1000
            )
            remaining_s = (deadline_ms - now_ms()) / 1000.0
            if remaining_s <= 0:
                raise _CancellationConfirmationTimedOut(expected.id)
            try:
                await asyncio.wait_for(running.task_changed.wait(), timeout=remaining_s)
            except asyncio.TimeoutError as exc:
                raise _CancellationConfirmationTimedOut(expected.id) from exc

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
        payload: dict[str, Any] | None = None
        response_consumer: Any | None = None
        result: Any | None = None
        encoded_result: str | None = None
        initial_worker_status: dict[str, Any] | None = None
        running_call: RunningCall | None = None
        task = stored.task
        detached = pool.config.execution_mode == "detached"
        try:
            # Reservation creates a short native etcd lease. Renew it before
            # any request-file reads or Base64 payload construction so a
            # large reference cannot expire the Worker guard before dispatch.
            heartbeat = asyncio.create_task(
                self._heartbeat(stored.task), name=f"video-heartbeat-{task.id}"
            )
            await asyncio.sleep(0)
            self._raise_if_heartbeat_stopped(heartbeat)
            normalized = await self.artifacts.read_json(task.request_path)
            if detached:
                if task.execution_token is None or task.attempt < 1:
                    raise RuntimeError(
                        "detached task reservation metadata is incomplete"
                    )
                initial_worker_status = await self.artifacts.read_detached_status(
                    task.deployment_id,
                    task.pool_id,
                    task.id,
                    task.attempt,
                    task.execution_token,
                )
            if initial_worker_status is None:
                manifest = await self.artifacts.read_json(task.input_manifest_path)
                payload_build_started = time.monotonic()
                payload = await run_file_io(
                    pool.adapter.build_worker_payload,
                    normalized,
                    manifest,
                    await self.artifacts.resolve_task_root(
                        task.deployment_id, task.pool_id, task.id
                    ),
                )
                self._payload_build_count += 1
                self._payload_build_seconds += time.monotonic() - payload_build_started
            self._raise_if_heartbeat_stopped(heartbeat)
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
                        else max(0.0, (now_ms() - task.queued_at_ms) / 1000.0),
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
            running_call = RunningCall(
                context=context,
                execution=current_execution,
                pool_id=task.pool_id,
                worker_key=stored.task.worker_key,
                detached=detached,
            )
            self.running_calls[task.id] = running_call
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
            response_consumer = pool.adapter.create_worker_stream_consumer()

            async def _consume_worker_stream() -> None:
                nonlocal payload, worker_stream_finished
                assert payload is not None
                assert running_call is not None
                running_call.worker_accepted = True
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
                detached_consumer = self._consume_detached_worker(
                    pool,
                    stored,
                    payload,
                    context,
                    response_consumer,
                    running_call,
                    initial_worker_status=initial_worker_status,
                )
                # Ownership has moved into the detached consumer coroutine.
                # Do not keep a second reference in this long-lived frame.
                payload = None
                await self._run_with_lease_monitor(
                    detached_consumer,
                    heartbeat,
                    self._monitor_cancellation(
                        pool, stored.task, context, running_call
                    ),
                    self._monitor_worker_liveness(pool, stored.task, running_call),
                )
                worker_stream_finished = True
            else:
                await asyncio.wait_for(
                    self._run_with_lease_monitor(
                        _consume_worker_stream(),
                        heartbeat,
                        self._monitor_cancellation(
                            pool, stored.task, context, running_call
                        ),
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
            response_consumer = None
            encoded_result = result.b64_json
            binary_artifact = result.artifact
            if binary_artifact is not None and not detached:
                raise RuntimeError(
                    "binary artifact references require detached execution"
                )
            inference_time_s = result.inference_time_s
            stage_durations = dict(result.stage_durations or {})
            result = None
            self._legacy_output_encoded_bytes += len(encoded_result)
            if len(encoded_result) > self.config.media.max_result_encoded_bytes:
                self._result_oversize_count += 1
                raise ResultTooLarge("Worker base64 result exceeds configured maximum")
            if inference_time_s is not None:
                self.telemetry.record_stage_duration(
                    task.pool_id, "execution", inference_time_s
                )
            if (
                detached
                and binary_artifact is not None
                and pool.config.scheduling.early_release_slot
                and latest.task.status == TaskStatus.IN_PROGRESS
            ):
                handed_off = await self._commit_result_handoff(
                    pool, task, binary_artifact, inference_time_s, stage_durations
                )
                if handed_off is not None:
                    self.running_calls.pop(task.id, None)
                    pool.wakeup.set()
                    await self._run_result_finalizer(pool, handed_off)
                return
            if latest.task.status == TaskStatus.IN_PROGRESS:
                finalizing = await self.store.transition(
                    task.id,
                    expected={TaskStatus.IN_PROGRESS},
                    expected_revision=latest.revision,
                    patch={
                        "status": TaskStatus.FINALIZING,
                        "inference_time_s": inference_time_s,
                        "stage_durations": stage_durations,
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
            try:
                if binary_artifact is not None:
                    (
                        final_path,
                        size,
                        sha256,
                        media,
                    ) = await self.artifacts.finalize_worker_mp4(
                        task.deployment_id,
                        task.pool_id,
                        task.id,
                        task.attempt,
                        task.execution_token,
                        binary_artifact,
                        normalized,
                        pool.adapter.validate_artifact,
                        pool.adapter.prepare_artifact,
                        pool.adapter.artifact_requires_processing,
                        inspector=pool.adapter.inspect_artifact_for_publication,
                        max_result_bytes=self.config.media.max_result_bytes,
                    )
                else:
                    (
                        final_path,
                        size,
                        sha256,
                        media,
                    ) = await self.artifacts.finalize_b64_mp4(
                        await self.artifacts.resolve_task_root(
                            task.deployment_id, task.pool_id, task.id
                        ),
                        encoded_result,
                        normalized,
                        pool.adapter.validate_artifact,
                        pool.adapter.prepare_artifact,
                        max_result_bytes=self.config.media.max_result_bytes,
                        publication_scope=f"a{stored.task.attempt}-{token_digest}",
                    )
            finally:
                # Decoding/validation has either published a file candidate or
                # failed.  No later state transition needs the Base64 string.
                encoded_result = None
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
                await run_file_io(final_path.unlink, True)
                final_path = None
                return
            self._require_execution_owner(latest.task, task)
            if latest.task.cancel_requested_at_ms is not None:
                await run_file_io(final_path.unlink, True)
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
                        "normalized_request": {
                            **latest.task.normalized_request,
                            "_result_media": media,
                        },
                        "finalize_time_s": finalize_seconds,
                    },
                    release_lease=True,
                )
            except StoreConflict:
                # Another owner may have published a different immutable
                # candidate. Delete only this Gateway's candidate.
                await run_file_io(final_path.unlink, True)
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
                await run_file_io(final_path.unlink, True)
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
                expected_execution=task,
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
                expected_execution=task,
            )
        except _WorkerLeaseLost as exc:
            if await self._current_owned_execution(task) is None:
                return
            worker_accepted = running_call is not None and running_call.worker_accepted
            if detached and task.execution_token is not None and worker_accepted:
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
                quarantine=worker_accepted,
                expected_execution=task,
            )
        except _DetachedWorkerCancelled:
            latest = await self.store.get_task(task.id)
            if (
                latest is not None
                and latest.task.status not in TERMINAL_STATUSES
                and self._same_execution_owner(latest.task, task)
            ):
                # The Dingo detached wrapper only reports ``cancelled`` after
                # its public Omni generation coroutine has unwound and its
                # partial response has been removed.  That normal terminal
                # state makes the instance reusable; confirmation timeout,
                # lease-loss and transport-error paths remain quarantined.
                await self._finish_cancelled(pool, latest, quarantine=False)
        except _CancellationConfirmationTimedOut:
            latest = await self._current_owned_execution(task)
            if latest is not None and latest.task.cancel_requested_at_ms is not None:
                await self._finish_cancelled(pool, latest, quarantine=True)
        except ResultTooLarge as exc:
            if final_path is not None:
                await run_file_io(final_path.unlink, True)
                final_path = None
            if await self._current_owned_execution(task) is None:
                return
            logger.warning("video task result exceeded policy: %s", task.id)
            await self._finish_failed(
                pool,
                task.id,
                "result_too_large",
                str(exc),
                quarantine=False,
                expected_execution=task,
            )
        except Exception as exc:
            if final_path is not None:
                await run_file_io(final_path.unlink, True)
                final_path = None
            current = await self._current_owned_execution(task)
            if current is None:
                logger.info(
                    "ignored stale task failure after execution ownership moved: %s",
                    task.id,
                )
                return
            if (
                detached
                and task.execution_token is not None
                and running_call is not None
                and running_call.worker_accepted
            ):
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
            if (
                isinstance(exc, StoreConflict)
                and running_call is None
                and task.status == TaskStatus.DISPATCHING
                and current.task.status == TaskStatus.DISPATCHING
                and current.task.cancel_requested_at_ms is not None
            ):
                # A cancellation can advance the reserved task revision before
                # dispatch. Keep the existing cancellation/lease cleanup below;
                # this confirmed pre-dispatch race is not a Worker failure.
                logger.info(
                    "task %s cancelled during dispatch preparation; "
                    "handling expected task revision conflict",
                    task.id,
                )
            elif isinstance(exc, _RetryableWorkerFailure):
                logger.exception("video Worker attempt failed: %s", task.id)
            else:
                logger.exception("video task failed: %s", task.id)
            await self._finish_failed(
                pool,
                task.id,
                "worker_failed",
                str(exc),
                quarantine=(
                    running_call is not None
                    and running_call.worker_accepted
                    and not worker_stream_finished
                ),
                retryable=isinstance(exc, _RetryableWorkerFailure),
                expected_execution=task,
            )
        finally:
            payload = None
            response_consumer = None
            result = None
            encoded_result = None
            released_budget = await self.memory_budget.release(task.id)
            self._finalizing.pop(task.id, None)
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

    async def _commit_result_handoff(
        self, pool, expected, artifact, inference_time_s, stage_durations
    ):
        """Retry ambiguous storage responses by reading the durable task first."""
        reservation_lost = False
        while not self._stop.is_set():
            try:
                latest = await self.store.get_task(expected.id)
                if latest is None or latest.task.status in TERMINAL_STATUSES:
                    return None
                self._require_execution_owner(latest.task, expected)
                if (
                    latest.task.status == TaskStatus.FINALIZING
                    and read_handoff(latest.task) is not None
                ):
                    return latest
                if reservation_lost:
                    # A definite fencing failure is not transient CAS contention.
                    # End only our task, never cancel/release/quarantine the new
                    # occupant of the historical Worker slot.
                    await self.store.transition(
                        expected.id,
                        expected={TaskStatus.IN_PROGRESS},
                        expected_revision=latest.revision,
                        release_lease=False,
                        patch={
                            "status": TaskStatus.FAILED,
                            "error": TaskError(
                                "result_handoff_lost_reservation",
                                "execution reservation changed before result handoff",
                            ),
                            "completed_at_ms": now_ms(),
                            "expires_at_ms": now_ms()
                            + int(self.config.lifecycle.failed_ttl_s * 1000),
                        },
                    )
                    return None
                if latest.task.cancel_requested_at_ms is not None:
                    await self._finish_cancelled(pool, latest, quarantine=False)
                    return None
                reference = make_handoff(
                    latest.task,
                    artifact,
                    timeout_s=pool.config.scheduling.finalization_timeout_s,
                )
                result = await self.store.transition(
                    expected.id,
                    expected={TaskStatus.IN_PROGRESS},
                    expected_revision=latest.revision,
                    release_lease=True,
                    release_execution=True,
                    patch={
                        "status": TaskStatus.FINALIZING,
                        "worker_lease_id": None,
                        "normalized_request": {
                            **latest.task.normalized_request,
                            HANDOFF_KEY: reference,
                        },
                        "inference_time_s": inference_time_s,
                        "stage_durations": stage_durations,
                    },
                )
                self.telemetry.record_transition(
                    "result_handoff",
                    latest.task,
                    result.task,
                    gateway_generation=self.generation,
                    revision=result.revision,
                )
                return result
            except HandoffReservationLost:
                reservation_lost = True
            except (_TaskOwnershipLost, ValueError):
                raise
            except asyncio.CancelledError:
                raise
            except Exception:
                # Includes a lost transaction response. Never discard the source
                # or call the Worker again while the outcome is uncertain.
                logger.warning(
                    "result handoff awaiting storage for task %s",
                    expected.id,
                    exc_info=True,
                )
                await asyncio.sleep(0.2)
        raise asyncio.CancelledError

    async def _run_result_finalizer(self, pool, stored):
        self._finalizing[stored.task.id] = pool.config.pool_id
        try:
            async with self._finalization_slots[pool.config.pool_id]:
                await self._finalizer.run(stored, pool)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Keep the durable handoff for our live-owner recovery loop or HA
            # takeover. Do not route storage failures through model retry/cancel.
            logger.exception("postprocessing suspended for task %s", stored.task.id)

    async def _resume_result_finalizer(self, pool, stored):
        try:
            await self._run_result_finalizer(pool, stored)
        finally:
            await self.memory_budget.release(stored.task.id)
            self._finalizing.pop(stored.task.id, None)
            pool.wakeup.set()

    async def _schedule_result_finalizer(self, pool, stored):
        if stored.task.id in self._finalizing or stored.task.id in self.running_calls:
            return
        if (
            sum(p == pool.config.pool_id for p in self._finalizing.values())
            >= pool.config.scheduling.finalization_pending_limit
        ):
            return
        # Reserve before the first await: owner recovery and the live-owner
        # scanner can otherwise start two finalizers sharing one allocation.
        self._finalizing[stored.task.id] = pool.config.pool_id
        scheduled = False
        try:
            if not await self.memory_budget.try_acquire(
                stored.task.id, self.config.media.result_task_memory_bytes
            ):
                return
            execution = asyncio.create_task(
                self._resume_result_finalizer(pool, stored),
                name=f"video-finalizing-{stored.task.id}",
            )
            scheduled = True
        finally:
            if not scheduled:
                await self.memory_budget.release(stored.task.id)
                self._finalizing.pop(stored.task.id, None)
        self._executions.add(execution)
        execution.add_done_callback(self._execution_done)

    async def _owned_finalization_recovery_loop(self):
        while not self._stop.is_set():
            try:
                after = None
                while not self._stop.is_set():
                    page = await self.store.list_tasks(
                        status=TaskStatus.FINALIZING, after=after, limit=256
                    )
                    if not page:
                        break
                    for stored in page:
                        if (
                            stored.task.owner_generation != self.generation
                            or read_handoff(stored.task) is None
                        ):
                            continue
                        pool = self.pools.get(stored.task.pool_id)
                        if pool is not None:
                            await self._schedule_result_finalizer(pool, stored)
                    after = page[-1].task.id
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("owned finalization recovery failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass

    async def _consume_detached_worker(
        self,
        pool: PoolRuntime,
        stored: StoredTask,
        payload: dict[str, Any] | None,
        context: Any,
        response_consumer: Any,
        running_call: RunningCall,
        *,
        initial_worker_status: dict[str, Any] | None = None,
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

        def _validate_identity(value: Any) -> dict[str, Any]:
            if not isinstance(value, dict):
                raise _DetachedWaitProtocolError(
                    "detached Worker control response is not an object"
                )
            expected = {
                "schema_version": 1,
                "deployment_id": task.deployment_id,
                "pool_id": task.pool_id,
                "task_id": task.id,
                "attempt": task.attempt,
                "execution_token": task.execution_token,
            }
            if any(
                value.get(key) != expected_value
                for key, expected_value in expected.items()
            ):
                raise _DetachedWaitProtocolError(
                    "detached Worker control response identity mismatch"
                )
            return value

        def _supports_wait(value: dict[str, Any] | None) -> bool:
            if value is None:
                return False
            capabilities = value.get("capabilities")
            return (
                isinstance(capabilities, list)
                and WAIT_TERMINAL_CAPABILITY in capabilities
            )

        async def _consume_status(value: dict[str, Any]) -> bool:
            worker_status = _validate_identity(value)
            state = worker_status.get("state")
            if state == "completed":
                queue_wait = worker_status.get("worker_queue_wait_s")
                if (
                    isinstance(queue_wait, (int, float))
                    and not isinstance(queue_wait, bool)
                    and 0 <= queue_wait <= 86400
                ):
                    self.telemetry.record_stage_duration(
                        task.pool_id, "worker_queue", queue_wait
                    )
                if "result_format" in worker_status or "inline_result" in worker_status:
                    if worker_status.get(
                        "result_format"
                    ) != INLINE_RESULT_FORMAT or any(
                        key in worker_status
                        for key in (
                            "response_path",
                            "response_bytes",
                            "response_sha256",
                        )
                    ):
                        raise _DetachedWaitProtocolError(
                            "unsupported or ambiguous inline Worker result"
                        )
                    response_consumer.consume(
                        normalize_inline_result(worker_status.get("inline_result"))
                    )
                    self.telemetry.increment(
                        "dingo_video_detached_inline_results_total",
                        labels={"pool": task.pool_id},
                    )
                    running_call.worker_accepted = False
                    return True
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
                    raise RuntimeError("detached Worker completed metadata is invalid")
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
                    raise RuntimeError("detached Worker response size mismatch")
                running_call.worker_accepted = False
                return True
            if state == "failed":
                running_call.worker_accepted = False
                raise worker_execution_error(worker_status.get("error"))
            if state == "cancelled":
                running_call.worker_accepted = False
                raise _DetachedWorkerCancelled("detached Worker cancelled task")
            if state not in {"accepted", "running", "not_found"}:
                raise _DetachedWaitProtocolError(
                    f"detached Worker returned invalid state {state!r}"
                )
            updated_at_ms = worker_status.get("updated_at_ms")
            if (
                state in {"accepted", "running"}
                and isinstance(updated_at_ms, int)
                and now_ms() - updated_at_ms > int(_DETACHED_WORKER_STALE_S * 1000)
                and await self._confirm_worker_loss(pool, task)
            ):
                raise _RetryableWorkerFailure(
                    "detached Worker disappeared and its heartbeat is stale"
                )
            return False

        async def _wait_event(attached: asyncio.Event) -> dict[str, Any]:
            wait_request = detached_envelope(
                op="wait",
                deployment_id=task.deployment_id,
                pool_id=task.pool_id,
                task_id=task.id,
                attempt=task.attempt,
                execution_token=task.execution_token,
            )
            try:
                stream = await pool.client.direct(
                    wait_request, int(task.worker_instance_id), context
                )
                response: dict[str, Any] | None = None
                watching = False
                count = 0
                async for item in stream:
                    if hasattr(item, "is_error") and item.is_error():
                        comments = item.comments() if hasattr(item, "comments") else []
                        raise _DetachedWaitUnavailable(
                            "; ".join(comments) or "detached Worker wait request failed"
                        )
                    value = _validate_identity(
                        item.data() if hasattr(item, "data") else item
                    )
                    count += 1
                    if count > 2:
                        raise _DetachedWaitProtocolError(
                            "detached Worker wait returned too many responses"
                        )
                    state = value.get("state")
                    if state == "watching":
                        if watching or response is not None:
                            raise _DetachedWaitProtocolError(
                                "detached Worker wait acknowledgement is duplicated"
                            )
                        watching = True
                        attached.set()
                        continue
                    if response is not None:
                        raise _DetachedWaitProtocolError(
                            "detached Worker wait returned multiple terminal responses"
                        )
                    response = value
                if response is None:
                    if watching:
                        raise _DetachedWaitUnavailable(
                            "detached Worker wait ended before terminal status"
                        )
                    raise _DetachedWaitUnavailable(
                        "detached Worker wait returned no response"
                    )
                return response
            except asyncio.CancelledError:
                raise
            except (_DetachedWaitUnavailable, _DetachedWaitProtocolError):
                raise
            except Exception as exc:
                raise _DetachedWaitUnavailable(
                    "detached Worker wait transport failed"
                ) from exc

        async def _wait_once() -> dict[str, Any]:
            remaining_s = ((task.deadline_at_ms or 0) - now_ms()) / 1000.0
            if remaining_s <= 0:
                raise asyncio.TimeoutError
            attached = asyncio.Event()
            wait_task = asyncio.create_task(
                _wait_event(attached), name=f"video-detached-wait-{task.id}"
            )
            attached_task = asyncio.create_task(attached.wait())
            try:
                done, _pending = await asyncio.wait(
                    {wait_task, attached_task},
                    timeout=min(_DETACHED_WAIT_ATTACH_TIMEOUT_S, remaining_s),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if wait_task in done:
                    return await wait_task
                if attached_task in done:
                    self.telemetry.increment(
                        "dingo_video_detached_wait_connections_total",
                        labels={"pool": task.pool_id, "outcome": "attached"},
                    )
                    remaining_s = ((task.deadline_at_ms or 0) - now_ms()) / 1000.0
                    if remaining_s <= 0:
                        raise asyncio.TimeoutError
                    return await asyncio.wait_for(wait_task, timeout=remaining_s)
                raise _DetachedWaitUnavailable(
                    "detached Worker wait attachment timed out"
                )
            finally:
                if not attached_task.done():
                    attached_task.cancel()
                await asyncio.gather(attached_task, return_exceptions=True)
                if not wait_task.done():
                    wait_task.cancel()
                    await asyncio.gather(wait_task, return_exceptions=True)

        worker_status = initial_worker_status
        if worker_status is None:
            if payload is None:
                raise RuntimeError("detached Worker submission payload is unavailable")
            submit = detached_envelope(
                op="submit",
                deployment_id=task.deployment_id,
                pool_id=task.pool_id,
                task_id=task.id,
                attempt=task.attempt,
                execution_token=task.execution_token,
                payload=payload,
                deadline_at_ms=(
                    task.deadline_at_ms
                    if task.worker_instance_id in pool.prefetch_instances
                    else None
                ),
            )
            acknowledgement = await asyncio.wait_for(
                self._detached_submit_ack(
                    pool, task, submit, context, _validate_identity
                ),
                timeout=max(
                    0.0, ((task.deadline_at_ms or now_ms()) - now_ms()) / 1000.0
                ),
            )
            if acknowledgement.get("state") not in {
                "accepted",
                "running",
                "completed",
            }:
                raise RuntimeError("detached Worker acknowledgement identity mismatch")
            worker_status = acknowledgement
            running_call.worker_accepted = True

            # The direct response stream is exhausted and the Worker has
            # durably accepted this detached execution.  Drop the envelope and
            # its Base64 references before entering the long status-poll loop.
            del acknowledgement
            del submit
        else:
            running_call.worker_accepted = worker_status.get("state") in {
                "accepted",
                "running",
            }

        del payload
        await self._shrink_to_result_memory_budget(task.id)

        assert worker_status is not None
        supports_wait = _supports_wait(worker_status)
        if await _consume_status(worker_status):
            return

        retry_delay_s = _DETACHED_WAIT_RETRY_INITIAL_S
        next_wait_retry = time.monotonic()
        if not supports_wait:
            self.telemetry.increment(
                "dingo_video_detached_wait_connections_total",
                labels={"pool": task.pool_id, "outcome": "unsupported"},
            )

        while True:
            if supports_wait and time.monotonic() >= next_wait_retry:
                try:
                    worker_status = await _wait_once()
                    retry_delay_s = _DETACHED_WAIT_RETRY_INITIAL_S
                    if await _consume_status(worker_status):
                        return
                    raise _DetachedWaitUnavailable(
                        "detached Worker could not attach a local terminal waiter"
                    )
                except _DetachedWaitProtocolError:
                    raise
                except _DetachedWaitUnavailable:
                    self.telemetry.increment(
                        "dingo_video_detached_wait_connections_total",
                        labels={"pool": task.pool_id, "outcome": "unavailable"},
                    )
                    next_wait_retry = time.monotonic() + retry_delay_s
                    retry_delay_s = min(retry_delay_s * 2.0, _DETACHED_WAIT_RETRY_MAX_S)

            worker_status = await _status()
            self.telemetry.increment(
                "dingo_video_detached_status_fallback_reads_total",
                labels={"pool": task.pool_id},
            )
            if worker_status is not None:
                supports_wait = supports_wait or _supports_wait(worker_status)
                if await _consume_status(worker_status):
                    return
            remaining_ms = (task.deadline_at_ms or 0) - now_ms()
            if remaining_ms <= 0:
                raise asyncio.TimeoutError
            sleep_s = min(_DETACHED_STATUS_FALLBACK_S, remaining_ms / 1000.0)
            if supports_wait:
                sleep_s = min(sleep_s, max(0.0, next_wait_retry - time.monotonic()))
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)

    async def _heartbeat(self, task) -> None:
        assert task.worker_key is not None
        consecutive_failures = 0
        while True:
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
            await asyncio.sleep(_WORKER_LEASE_HEARTBEAT_INTERVAL_S)

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

    async def _try_worker_retry(
        self, pool: PoolRuntime, stored: StoredTask, *, quarantine: bool
    ) -> bool:
        task = stored.task
        if (
            not getattr(self, "_worker_retry_once", False)
            or pool.config.execution_mode != "detached"
            or task.attempt != 1
            or task.status not in {TaskStatus.DISPATCHING, TaskStatus.IN_PROGRESS}
            or task.cancel_requested_at_ms is not None
            or task.owner_generation != self.generation
            or not hasattr(self.store, "retry_budget_used")
        ):
            return False
        retry = None
        for _ in range(16):
            try:
                retry = await self.store.requeue_failed_attempt(
                    stored,
                    queue_limit=pool.config.scheduling.queue_limit,
                    retry_wait_timeout_s=getattr(self, "_retry_wait_timeout_s", 600),
                    quarantine_until_ms=(
                        max(task.deadline_at_ms or now_ms(), now_ms())
                        + int(pool.config.scheduling.abort_grace_s * 1000)
                    )
                    if quarantine
                    else now_ms()
                    + int(getattr(self, "_failed_instance_backoff_s", 30) * 1000),
                )
                break
            except StoreConflict:
                current = await self.store.get_task(task.id)
                if (
                    current is None
                    or current.task.status
                    not in {TaskStatus.DISPATCHING, TaskStatus.IN_PROGRESS}
                    or not self._same_execution_owner(current.task, task)
                    or current.task.cancel_requested_at_ms is not None
                ):
                    return False
                stored = current
                await asyncio.sleep(0.01)
        if retry is None:
            return False
        self.telemetry.increment(
            "dingo_video_worker_retries_total", labels={"pool": task.pool_id}
        )
        self.telemetry.record_transition(
            "worker_retry_queued",
            task,
            retry.task,
            gateway_generation=self.generation,
            revision=retry.revision,
        )
        logger.warning(
            "queued one Worker retry for task %s after attempt %s; excluding instance %s",
            task.id,
            task.attempt,
            task.worker_instance_id,
        )
        pool.wakeup.set()
        return True

    async def _finish_failed(
        self,
        pool: PoolRuntime,
        task_id: str,
        code: str,
        message: str,
        *,
        quarantine: bool,
        retryable: bool = False,
        expected_execution: Any | None = None,
    ) -> None:
        latest = await self.store.get_task(task_id)
        if latest is None or latest.task.status in TERMINAL_STATUSES:
            return
        if expected_execution is not None and not self._same_execution_owner(
            latest.task, expected_execution
        ):
            return
        if retryable and latest.task.cancel_requested_at_ms is None:
            previous = latest
            try:
                if await self._try_worker_retry(pool, latest, quarantine=quarantine):
                    return
            except Exception:
                # A transaction reply may be ambiguous. Re-read ownership
                # before falling back; never fail an already requeued attempt.
                logger.exception("Worker retry decision failed for task %s", task_id)
            latest = await self.store.get_task(task_id)
            if (
                latest is None
                or latest.task.status in TERMINAL_STATUSES
                or not self._same_execution_owner(latest.task, previous.task)
            ):
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
                                    + int(self.config.lifecycle.failed_ttl_s * 1000),
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
        if task.status == TaskStatus.FINALIZING and read_handoff(task) is not None:
            if pool is None:
                return
            if task.owner_generation == self.generation:
                await self._schedule_result_finalizer(pool, stored)
            elif self.store.gateway_owner_supported:
                claimed = await self.store.claim_finalizing(
                    stored, new_owner_generation=self.generation
                )
                if claimed is not None:
                    await self._schedule_result_finalizer(pool, claimed)
            return
        if (
            self.store.gateway_owner_supported
            and pool is not None
            and pool.config.execution_mode == "detached"
            and task.execution_token is not None
            and task.worker_key is not None
            and task.worker_instance_id is not None
        ):
            weight_bytes = (
                task.estimated_payload_bytes or self.config.media.max_task_memory_bytes
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
                    (pool.config.scheduling.abort_grace_s if pool is not None else 30.0)
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
        task_root = await self.artifacts.resolve_task_root(
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
                if (
                    moved is not None
                    and not self.config.lifecycle.orphan_cleanup_dry_run
                ):
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
                                "retry_wait_timeout"
                                if task.attempt > 0
                                else "queue_timeout",
                                "video retry wait expired"
                                if task.attempt > 0
                                else "video task expired while queued",
                            ),
                        },
                    )
                    self.telemetry.record_transition(
                        "failed",
                        stored.task,
                        failed.task,
                        gateway_generation=self.generation,
                        revision=failed.revision,
                        reason="retry_wait_timeout"
                        if task.attempt > 0
                        else "queue_timeout",
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
