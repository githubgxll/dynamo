# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-pool FIFO dispatch, sticky direct calls, cancellation and recovery."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
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


@dataclass(slots=True)
class PoolRuntime:
    config: PoolConfig
    client: EndpointClient
    adapter: VideoBackendAdapter
    wakeup: asyncio.Event
    instance_ids: list[int]
    cursor: int = 0
    discovery_healthy: bool = False


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
            )
            for pool in config.pools
        }
        self.running_calls: dict[str, RunningCall] = {}
        self._loops: list[asyncio.Task] = []
        self._executions: set[asyncio.Task] = set()
        self._stop = asyncio.Event()
        self._ready = False

    @property
    def ready(self) -> bool:
        return (
            self._ready
            and not self._stop.is_set()
            and all(pool.discovery_healthy for pool in self.pools.values())
        )

    async def start(self) -> None:
        await self.store.health()
        await self.artifacts.health()
        await self._recover()
        for pool in self.pools.values():
            await self._refresh_instances(pool)
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

    def notify(self, pool_id: str) -> None:
        self.pools[pool_id].wakeup.set()

    async def cancel(self, task_id: str) -> StoredTask:
        stored = await self.store.request_cancel(task_id)
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
                "expires_at_ms": now_ms() + 60 * 60 * 1000,
                "error": terminal_error(
                    "cancelled", "video task cancellation was requested"
                ),
            },
            release_lease=True,
            quarantine_until_ms=max(stored.task.deadline_at_ms or now_ms(), now_ms())
            + int(grace_s * 1000),
        )

    async def wait_terminal(self, task_id: str, timeout_s: float) -> StoredTask:
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
        for lease in await self.store.list_leases(pool.config.pool_id):
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
            return False
        leased = {
            lease.worker_key
            for lease in await self.store.list_leases(pool.config.pool_id)
        }
        available = [
            instance_id
            for instance_id in pool.instance_ids
            if worker_key(pool.config.backend_target, instance_id) not in leased
        ]
        if not available:
            return False
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
        reserved = await self.store.reserve(queued[0], lease, deadline_at_ms=deadline)
        if reserved is None:
            return True
        execution = asyncio.create_task(
            self._run_reserved(pool, reserved), name=f"video-task-{reserved.task.id}"
        )
        self._executions.add(execution)
        execution.add_done_callback(self._execution_done)
        return True

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
            payload = await asyncio.to_thread(
                pool.adapter.build_worker_payload,
                normalized,
                manifest,
                self.artifacts.task_root(task.deployment_id, task.pool_id, task.id),
            )
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
            chunks: list[Any] = []

            async def _consume_worker_stream() -> None:
                nonlocal worker_stream_finished
                stream = await pool.client.direct(
                    payload, int(stored.task.worker_instance_id), context
                )
                async for item in stream:
                    if hasattr(item, "is_error") and item.is_error():
                        comments = item.comments() if hasattr(item, "comments") else []
                        raise RuntimeError(
                            "; ".join(comments) or "Dingo direct call failed"
                        )
                    chunks.append(item.data() if hasattr(item, "data") else item)
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
            result = pool.adapter.consume_worker_stream(chunks)
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
            )
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
                    "expires_at_ms": now_ms() + 24 * 60 * 60 * 1000,
                    "result_path": str(final_path),
                    "result_bytes": size,
                    "result_sha256": sha256,
                    "finalize_time_s": time.monotonic() - finalize_started,
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
            self.running_calls.pop(task.id, None)
            if heartbeat is not None:
                heartbeat.cancel()
                await asyncio.gather(heartbeat, return_exceptions=True)
            pool.wakeup.set()

    async def _heartbeat(self, task) -> None:
        assert task.worker_key is not None
        while True:
            await asyncio.sleep(5.0)
            try:
                await self.store.heartbeat_lease(task.pool_id, task.worker_key, task.id)
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
                "expires_at_ms": now_ms() + 60 * 60 * 1000,
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
                    "expires_at_ms": now_ms() + 60 * 60 * 1000,
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
        tasks = await self.store.list_tasks(limit=10000)
        for stored in tasks:
            task = stored.task
            pool = self.pools.get(task.pool_id)
            if task.status == TaskStatus.QUEUED:
                if (
                    pool is None
                    or task.configuration_revision != pool.config.configuration_revision
                    or task.backend_target != pool.config.backend_target
                ):
                    try:
                        await self.store.transition(
                            task.id,
                            expected={TaskStatus.QUEUED},
                            expected_revision=stored.revision,
                            patch={
                                "status": TaskStatus.FAILED,
                                "completed_at_ms": now_ms(),
                                "expires_at_ms": now_ms() + 60 * 60 * 1000,
                                "error": terminal_error(
                                    "configuration_changed",
                                    "task pool configuration changed before dispatch",
                                ),
                            },
                        )
                    except StoreConflict:
                        pass
                continue
            if task.status in ACTIVE_STATUSES:
                try:
                    await self.store.transition(
                        task.id,
                        expected=ACTIVE_STATUSES,
                        expected_revision=stored.revision,
                        patch={
                            "status": TaskStatus.FAILED,
                            "completed_at_ms": now_ms(),
                            "expires_at_ms": now_ms() + 60 * 60 * 1000,
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
                continue
            if task.status == TaskStatus.COMPLETED:
                await self._verify_completed_artifact(stored)
        for pool_id in self.pools:
            await self.store.reconcile_pool(pool_id)

    async def _verify_completed_artifact(self, stored: StoredTask) -> None:
        task = stored.task
        try:
            if task.result_path is None:
                raise FileNotFoundError("completed task has no result path")
            path = self.artifacts.result_path(task.result_path)

            def _digest() -> tuple[int, str]:
                digest = hashlib.sha256()
                size = 0
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        size += len(chunk)
                        digest.update(chunk)
                return size, digest.hexdigest()

            size, sha256 = await asyncio.to_thread(_digest)
            if size != task.result_bytes or sha256 != task.result_sha256:
                raise RuntimeError("completed video artifact checksum mismatch")
        except Exception:
            logger.exception(
                "completed task artifact is missing or corrupt: %s", task.id
            )
            try:
                await self.store.transition(
                    task.id,
                    expected={TaskStatus.COMPLETED},
                    expected_revision=stored.revision,
                    patch={
                        "status": TaskStatus.FAILED,
                        "expires_at_ms": now_ms() + 60 * 60 * 1000,
                        "error": terminal_error(
                            "artifact_missing",
                            "completed video artifact is unavailable",
                        ),
                        "result_path": None,
                        "result_bytes": None,
                        "result_sha256": None,
                    },
                )
            except StoreConflict:
                pass

    async def _sweeper_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self._sweep_once()
            except Exception:
                logger.exception("video task sweeper iteration failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                pass

    async def _sweep_once(self) -> None:
        current = now_ms()
        await self.artifacts.cleanup_orphan_uploads(minimum_age_s=3600.0)
        for stored in await self.store.list_tasks(limit=10_000):
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
                            "expires_at_ms": current + 60 * 60 * 1000,
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
                    expired = await self.store.transition(
                        task.id,
                        expected={task.status},
                        expected_revision=stored.revision,
                        patch={
                            "status": TaskStatus.EXPIRED,
                            "expired_at_ms": current,
                            "expires_at_ms": current + 24 * 60 * 60 * 1000,
                        },
                    )
                    task_root = self.artifacts.task_root(
                        task.deployment_id, task.pool_id, task.id
                    )
                    await self.artifacts.discard(task_root)
                    await self.store.transition(
                        task.id,
                        expected={TaskStatus.EXPIRED},
                        expected_revision=expired.revision,
                        patch={"artifact_deleted_at_ms": now_ms()},
                    )
                elif task.status == TaskStatus.EXPIRED:
                    if task.artifact_deleted_at_ms is None:
                        task_root = self.artifacts.task_root(
                            task.deployment_id, task.pool_id, task.id
                        )
                        await self.artifacts.discard(task_root)
                        stored = await self.store.transition(
                            task.id,
                            expected={TaskStatus.EXPIRED},
                            expected_revision=stored.revision,
                            patch={"artifact_deleted_at_ms": now_ms()},
                        )
                    await self.store.delete_expired(stored)
            except StoreConflict:
                continue
