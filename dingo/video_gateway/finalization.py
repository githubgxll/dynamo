"""Recoverable postprocessing: no Worker calls, slot leases or model retries."""

from __future__ import annotations

import asyncio
import errno
import time

from dingo.video_gateway.artifact_store import ResultTooLarge
from dingo.video_gateway.errors import StoreConflict
from dingo.video_gateway.file_io import run_file_io
from dingo.video_gateway.models import TaskError, TaskStatus, now_ms
from dingo.video_gateway.result_handoff import HANDOFF_KEY, read_handoff


def same_owner(current, expected):
    return (current.owner_generation, current.attempt, current.execution_token) == (
        expected.owner_generation,
        expected.attempt,
        expected.execution_token,
    )


def transient_result_error(exc):
    return isinstance(exc, (TimeoutError, ConnectionError)) or (
        isinstance(exc, OSError)
        and exc.errno
        in {
            errno.EAGAIN,
            errno.EINTR,
            errno.ETIMEDOUT,
            errno.EIO,
            errno.ESTALE,
            errno.EBUSY,
            errno.ENOSPC,
        }
    )


class ResultFinalizer:
    def __init__(self, store, artifacts, config, telemetry, generation):
        self.store, self.artifacts, self.config = store, artifacts, config
        self.telemetry, self.generation = telemetry, generation

    async def _terminal(self, stored, status, error):
        ttl = (
            self.config.lifecycle.cancelled_ttl_s
            if status == TaskStatus.CANCELLED
            else self.config.lifecycle.failed_ttl_s
        )
        result = await self.store.transition(
            stored.task.id,
            expected={TaskStatus.FINALIZING},
            expected_revision=stored.revision,
            patch={
                "status": status,
                "error": TaskError(*error),
                "completed_at_ms": now_ms(),
                "expires_at_ms": now_ms() + int(ttl * 1000),
            },
            release_lease=False,
        )
        self.telemetry.record_transition(
            status.value,
            stored.task,
            result.task,
            gateway_generation=self.generation,
            revision=result.revision,
        )

    async def run(self, stored, pool):
        expected = stored.task
        settings = pool.config.scheduling
        while True:
            current = await self.store.get_task(expected.id)
            if (
                current is None
                or current.task.status != TaskStatus.FINALIZING
                or not same_owner(current.task, expected)
            ):
                return
            task = current.task
            reference = read_handoff(task)
            if reference is None:
                raise ValueError("finalizer requires a durable handoff")
            if task.cancel_requested_at_ms is not None:
                try:
                    await self._terminal(
                        current,
                        TaskStatus.CANCELLED,
                        ("cancelled", "video task was cancelled"),
                    )
                    return
                except StoreConflict:
                    continue
            remaining = (reference["deadline_at_ms"] - now_ms()) / 1000
            if remaining <= 0:
                try:
                    await self._terminal(
                        current,
                        TaskStatus.FAILED,
                        (
                            "finalization_timeout",
                            "result finalization deadline exceeded",
                        ),
                    )
                    return
                except StoreConflict:
                    continue
            final_path = None
            started = time.monotonic()
            try:
                try:
                    final_path, size, digest, media = await asyncio.wait_for(
                        self.artifacts.finalize_worker_mp4(
                            task.deployment_id,
                            task.pool_id,
                            task.id,
                            task.attempt,
                            task.execution_token,
                            reference["artifact"],
                            task.normalized_request,
                            pool.adapter.validate_artifact,
                            pool.adapter.prepare_artifact,
                            pool.adapter.artifact_requires_processing,
                            inspector=pool.adapter.inspect_artifact_for_publication,
                            max_result_bytes=self.config.media.max_result_bytes,
                        ),
                        timeout=remaining,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    latest = await self.store.get_task(task.id)
                    if (
                        latest is None
                        or not same_owner(latest.task, expected)
                        or latest.task.status != TaskStatus.FINALIZING
                    ):
                        return
                    if latest.task.cancel_requested_at_ms is not None:
                        continue
                    ref = read_handoff(latest.task)
                    failures = int(ref.get("failures", 0))
                    if (
                        transient_result_error(exc)
                        and failures < settings.finalization_max_retries
                        and now_ms() < ref["deadline_at_ms"]
                    ):
                        ref["failures"] = failures + 1
                        try:
                            await self.store.transition(
                                task.id,
                                expected={TaskStatus.FINALIZING},
                                expected_revision=latest.revision,
                                patch={
                                    "normalized_request": {
                                        **latest.task.normalized_request,
                                        HANDOFF_KEY: ref,
                                    }
                                },
                            )
                        except StoreConflict:
                            continue
                        self.telemetry.increment(
                            "dingo_video_finalization_retries_total",
                            labels={"pool": task.pool_id},
                        )
                        await asyncio.sleep(
                            min(
                                settings.finalization_retry_delay_s * 2**failures,
                                max(0, (ref["deadline_at_ms"] - now_ms()) / 1000),
                            )
                        )
                        continue
                    if now_ms() >= ref["deadline_at_ms"]:
                        code = "finalization_timeout"
                    else:
                        code = (
                            "result_too_large"
                            if isinstance(exc, ResultTooLarge)
                            else "finalization_failed"
                        )
                    try:
                        await self._terminal(
                            latest,
                            TaskStatus.FAILED,
                            (code, "result validation or finalization failed"),
                        )
                        return
                    except StoreConflict:
                        continue
                duration = time.monotonic() - started
                latest = await self.store.get_task(task.id)
                if (
                    latest is None
                    or latest.task.status != TaskStatus.FINALIZING
                    or not same_owner(latest.task, expected)
                ):
                    return
                if latest.task.cancel_requested_at_ms is not None:
                    continue
                # A synchronous filesystem operation can finish after wait_for's
                # logical deadline. Never publish a result past that deadline.
                if now_ms() >= reference["deadline_at_ms"]:
                    continue
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
                            "result_sha256": digest,
                            "normalized_request": {
                                **latest.task.normalized_request,
                                "_result_media": media,
                            },
                            "finalize_time_s": duration,
                        },
                        release_lease=False,
                    )
                except StoreConflict:
                    continue
                except Exception:
                    # Ambiguous etcd response: do not destroy a possibly committed
                    # candidate. Unique files not referenced after recovery are GC'd.
                    final_path = None
                    raise
                final_path = None
                self.telemetry.record_transition(
                    "completed",
                    latest.task,
                    completed.task,
                    gateway_generation=self.generation,
                    revision=completed.revision,
                )
                self.telemetry.record_stage_duration(task.pool_id, "finalize", duration)
                return
            finally:
                if final_path is not None:
                    await run_file_io(final_path.unlink, missing_ok=True)
