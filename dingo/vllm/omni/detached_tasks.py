# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in durable execution wrapper for long-running Omni video requests.

The normal Omni request path remains untouched.  A caller must send the
private, versioned ``_dingo_video_task`` envelope and the worker must have an
explicit shared task root configured before this code is reachable.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dingo.common.video_result_file import (
    BINARY_RESULT_WRITER,
    INLINE_RESULT_FORMAT,
    BinaryResultWriter,
    normalize_inline_result,
)
from dingo.common.video_task_protocol import (
    ENVELOPE_KEY,
    EXECUTION_CAPACITY_CAPABILITY,
    PREFETCH_CAPABILITY,
    SCHEMA_VERSION,
    WAIT_TERMINAL_CAPABILITY,
    DetachedTaskIdentity,
    detached_attempt_root,
)
from dingo.common.video_task_protocol import (
    detached_envelope as detached_envelope,  # retained compatibility re-export
)

logger = logging.getLogger(__name__)

_TERMINAL = frozenset({"completed", "failed", "cancelled"})


class _DetachedContext:
    """Minimal Dynamo Context surface consumed by ``OmniHandler``."""

    def __init__(self, request_id: str) -> None:
        self._request_id = request_id
        self._cancelled = asyncio.Event()
        self._waiters: set[asyncio.Task[bool]] = set()

    def id(self) -> str:
        return self._request_id

    def stop_generating(self) -> None:
        self._cancelled.set()

    def is_stopped(self) -> bool:
        return self._cancelled.is_set()

    def async_killed_or_stopped(self) -> asyncio.Task[bool]:
        waiter = asyncio.create_task(self._cancelled.wait())
        self._waiters.add(waiter)
        waiter.add_done_callback(self._waiters.discard)
        return waiter

    async def close(self) -> None:
        """Resolve Context waiters orphaned by a cancelled abort monitor."""

        self._cancelled.set()
        waiters = list(self._waiters)
        if waiters:
            await asyncio.gather(*waiters, return_exceptions=True)

    async def wait_stopped(self) -> None:
        await self._cancelled.wait()


@dataclass(slots=True)
class _RunningTask:
    identity: DetachedTaskIdentity
    context: _DetachedContext
    execution: asyncio.Task[None]
    # Published only after the terminal write succeeds. Existing waiters hold
    # this execution-scoped record; completed tasks are not retained globally.
    persisted_terminal: dict[str, Any] | None = None
    started: bool = False


class DetachedOmniTaskManager:
    """Persist Omni response streams independently of the submitting Gateway."""

    def __init__(
        self,
        handler: Any,
        artifact_root: str | Path,
        *,
        drain_timeout_s: float = 1800.0,
        cancel_poll_interval_s: float = 0.25,
        cancel_grace_s: float = 5.0,
        binary_results: bool | None = None,
        inline_results: bool | None = None,
        execution_capacity: int = 1,
        prefetch_capacity: int = 0,
    ) -> None:
        if (
            isinstance(execution_capacity, bool)
            or not isinstance(execution_capacity, int)
            or execution_capacity < 1
        ):
            raise ValueError("detached execution capacity must be a positive integer")
        if drain_timeout_s <= 0:
            raise ValueError("detached drain timeout must be positive")
        if type(prefetch_capacity) is not int or prefetch_capacity not in {0, 1}:
            raise ValueError("detached prefetch capacity must be 0 or 1")
        if cancel_poll_interval_s <= 0:
            raise ValueError("detached cancel poll interval must be positive")
        if cancel_grace_s <= 0:
            raise ValueError("detached cancel grace must be positive")
        self.handler = handler
        self.root = Path(artifact_root).expanduser().resolve()
        self.root.mkdir(mode=0o750, parents=True, exist_ok=True)
        self.drain_timeout_s = drain_timeout_s
        self.cancel_poll_interval_s = cancel_poll_interval_s
        self.cancel_grace_s = cancel_grace_s
        self._running: dict[tuple[str, str, str, int, str], _RunningTask] = {}
        self._lock = asyncio.Lock()
        self._accepting = True
        self.execution_capacity = execution_capacity
        self.prefetch_capacity = prefetch_capacity
        self._execution_slots = asyncio.Semaphore(execution_capacity)
        self._direct_running = 0
        self.binary_results = (
            os.getenv("DINGO_VIDEO_BINARY_RESULTS") == "1"
            if binary_results is None
            else binary_results
        )
        self.inline_results = (
            os.getenv("DINGO_VIDEO_INLINE_RESULT") == "1"
            if inline_results is None
            else inline_results
        )
        if self.inline_results and not self.binary_results:
            raise ValueError("inline results require binary results")

    async def generate(
        self, request: dict[str, Any], context: Any
    ) -> AsyncGenerator[dict[str, Any], None]:
        envelope = request.get(ENVELOPE_KEY)
        if envelope is None:
            async with self._lock:
                if not self._accepting:
                    raise RuntimeError("detached Worker is draining")
                if len(self._running) + self._direct_running >= self.execution_capacity:
                    raise RuntimeError("Worker execution capacity exhausted")
                self._direct_running += 1
            try:
                async with self._execution_slots:
                    async for chunk in self.handler.generate(request, context):
                        yield chunk
            finally:
                self._direct_running -= 1
            return
        if set(request) != {ENVELOPE_KEY} or not isinstance(envelope, Mapping):
            raise ValueError("detached task request must contain only its envelope")
        op = envelope.get("op")
        if op == "capabilities":
            if (
                set(envelope) != {"schema_version", "op"}
                or envelope.get("schema_version") != SCHEMA_VERSION
            ):
                raise ValueError("invalid detached capabilities request")
            yield {
                "schema_version": SCHEMA_VERSION,
                "capabilities": [
                    WAIT_TERMINAL_CAPABILITY,
                    EXECUTION_CAPACITY_CAPABILITY,
                    PREFETCH_CAPABILITY,
                ],
                "execution_capacity": self.execution_capacity,
                "prefetch_capacity": self.prefetch_capacity,
                "admission_capacity": self.execution_capacity + self.prefetch_capacity,
                "accepting": self._accepting,
            }
            return
        identity = DetachedTaskIdentity.from_envelope(envelope)
        if op == "submit":
            yield await self._submit(
                identity,
                envelope.get("payload"),
                deadline_at_ms=envelope.get("deadline_at_ms"),
            )
        elif op == "wait":
            async for status in self._wait_terminal(identity):
                yield status
        elif op == "status" or op == "result":
            yield await self._status(identity)
        elif op == "cancel":
            yield await self._cancel(identity)
        else:
            raise ValueError("unsupported detached task operation")

    def _attempt_root(self, identity: DetachedTaskIdentity) -> Path:
        path = detached_attempt_root(self.root, *identity.key)
        cursor = self.root
        for component in path.relative_to(self.root).parts:
            cursor /= component
            if cursor.is_symlink():
                raise RuntimeError(f"detached task path contains a symlink: {cursor}")
        return path

    def _status_path(self, identity: DetachedTaskIdentity) -> Path:
        return self._attempt_root(identity) / "worker-status.json"

    def _response_path(self, identity: DetachedTaskIdentity) -> Path:
        return self._attempt_root(identity) / "worker-response.jsonl"

    def _cancel_path(self, identity: DetachedTaskIdentity) -> Path:
        return self._attempt_root(identity) / "cancel.requested"

    def _validate_task_manifest(self, identity: DetachedTaskIdentity) -> Path:
        attempt_root = self._attempt_root(identity)
        task_root = attempt_root.parent.parent
        manifest_path = task_root / "_artifact.json"
        if manifest_path.is_symlink():
            raise RuntimeError("detached task manifest is a symlink")
        try:
            if manifest_path.stat().st_size > 64 * 1024:
                raise RuntimeError("detached task manifest exceeds 64 KiB")
            with manifest_path.open("r", encoding="utf-8") as stream:
                manifest = json.load(stream)
        except FileNotFoundError as exc:
            raise RuntimeError("detached task manifest does not exist") from exc
        expected = {
            "schema_version": 1,
            "deployment_id": identity.deployment_id,
            "pool_id": identity.pool_id,
            "task_id": identity.task_id,
        }
        if not isinstance(manifest, dict) or any(
            manifest.get(key) != value for key, value in expected.items()
        ):
            raise RuntimeError("detached task manifest identity mismatch")
        return attempt_root

    @staticmethod
    def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
        temporary = path.with_name(path.name + f".part-{uuid.uuid4().hex}")
        payload = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        renamed = False
        try:
            try:
                stream = temporary.open("xb")
            except FileNotFoundError:
                # Existing attempt directories are the common path. Retain
                # creation behavior without a redundant mkdir on every update.
                path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
                stream = temporary.open("xb")
            with stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            renamed = True
        finally:
            if not renamed:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _read_status(path: Path) -> dict[str, Any] | None:
        try:
            with path.open("r", encoding="utf-8") as stream:
                value = json.load(stream)
        except FileNotFoundError:
            return None
        if not isinstance(value, dict):
            raise RuntimeError("detached Worker status is not an object")
        return value

    def _base_status(
        self, identity: DetachedTaskIdentity, state: str
    ) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "deployment_id": identity.deployment_id,
            "pool_id": identity.pool_id,
            "task_id": identity.task_id,
            "attempt": identity.attempt,
            "execution_token": identity.execution_token,
            "state": state,
            "updated_at_ms": int(time.time() * 1000),
            "capabilities": [WAIT_TERMINAL_CAPABILITY],
        }

    async def _submit(
        self,
        identity: DetachedTaskIdentity,
        payload: Any,
        *,
        deadline_at_ms: int | None = None,
    ) -> dict[str, Any]:
        if not self._accepting:
            raise RuntimeError("detached Worker is draining")
        if not isinstance(payload, dict):
            raise ValueError("detached submit payload must be an object")
        attempt_root = await asyncio.to_thread(self._validate_task_manifest, identity)
        status_path = attempt_root / "worker-status.json"
        async with self._lock:
            running = self._running.get(identity.key)
            if running is not None:
                state = (
                    "accepted"
                    if self.prefetch_capacity and not running.started
                    else "running"
                )
                return {**self._base_status(identity, state), "accepted": False}
            existing = await asyncio.to_thread(self._read_status, status_path)
            if existing is not None and existing.get("state") in _TERMINAL:
                return {**existing, "accepted": False}
            if (
                len(self._running) + self._direct_running
                >= self.execution_capacity + self.prefetch_capacity
            ):
                # No execution lock/status is written for rejected work. A
                # caller can requeue it without mistaking it for an execution.
                return {
                    **self._base_status(identity, "busy"),
                    "accepted": False,
                    "execution_capacity": self.execution_capacity,
                }
            await asyncio.to_thread(attempt_root.mkdir, 0o750, True, True)
            lock_path = attempt_root / "execution.lock"

            def _claim() -> bool:
                try:
                    descriptor = os.open(
                        lock_path,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                    )
                except FileExistsError:
                    return False
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    stream.write(f"pid={os.getpid()}\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                return True

            if not await asyncio.to_thread(_claim):
                current = existing or self._base_status(identity, "running")
                return {**current, "accepted": False}
            initial = self._base_status(identity, "accepted")
            queued_at_ms = int(time.time() * 1000)
            if self.prefetch_capacity:
                initial["queued_at_ms"] = queued_at_ms
            await asyncio.to_thread(self._atomic_json, status_path, initial)
            request_id = (
                f"{identity.task_id}-{identity.attempt}-{identity.execution_token[:12]}"
            )
            detached_context = _DetachedContext(request_id)
            execution = asyncio.create_task(
                self._execute(
                    identity,
                    payload,
                    detached_context,
                    deadline_at_ms=deadline_at_ms,
                    queued_at_ms=queued_at_ms,
                    status_path=status_path,
                ),
                name=f"omni-detached-{identity.task_id}-{identity.attempt}",
            )
            self._running[identity.key] = _RunningTask(
                identity=identity,
                context=detached_context,
                execution=execution,
            )
            execution.add_done_callback(
                lambda completed, key=identity.key: self._execution_done(key, completed)
            )
            return {**initial, "accepted": True}

    def _execution_done(
        self,
        key: tuple[str, str, str, int, str],
        execution: asyncio.Task[None],
    ) -> None:
        self._running.pop(key, None)
        if execution.cancelled():
            return
        error = execution.exception()
        if error is not None:
            logger.error(
                "detached Omni task escaped its terminal recorder",
                exc_info=(type(error), error, error.__traceback__),
            )

    async def _watch_cancel(
        self, identity: DetachedTaskIdentity, context: _DetachedContext
    ) -> None:
        path = await asyncio.to_thread(self._cancel_path, identity)
        while not context.is_stopped():
            if await asyncio.to_thread(path.exists):
                context.stop_generating()
                return
            await asyncio.sleep(self.cancel_poll_interval_s)

    async def _enforce_cancel(
        self, context: _DetachedContext, execution: asyncio.Task[None]
    ) -> None:
        """Give Omni a short abort grace, then stop a stuck response stream."""

        await context.wait_stopped()
        await asyncio.sleep(self.cancel_grace_s)
        if not execution.done():
            execution.cancel()

    async def _heartbeat_status(
        self,
        identity: DetachedTaskIdentity,
        status_path: Path,
        started_at_ms: int,
        stop: asyncio.Event,
    ) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=5.0)
                return
            except asyncio.TimeoutError:
                pass
            await asyncio.to_thread(
                self._atomic_json,
                status_path,
                {
                    **self._base_status(identity, "running"),
                    "started_at_ms": started_at_ms,
                },
            )

    async def _record_terminal(
        self, identity: DetachedTaskIdentity, path: Path, status: dict[str, Any]
    ) -> None:
        """Notify local waiters only after the terminal write completes."""
        await asyncio.to_thread(self._atomic_json, path, status)
        running = self._running.get(identity.key)
        if running is not None:
            running.persisted_terminal = copy.deepcopy(status)

    async def _execute(
        self,
        identity,
        payload,
        context,
        *,
        deadline_at_ms=None,
        queued_at_ms=None,
        status_path=None,
    ):
        if not self.prefetch_capacity:
            return await self._execute_started(identity, payload, context)
        # Submission already resolved this path before publishing accepted.
        # Do not insert a filesystem await before acquiring the FIFO permit:
        # two concurrent metadata reads can complete in the opposite order.
        if status_path is None:
            status_path = await asyncio.to_thread(self._status_path, identity)
        acquire = asyncio.create_task(self._execution_slots.acquire())
        stopped = context.async_killed_or_stopped()
        cancel_watch = asyncio.create_task(self._watch_cancel(identity, context))
        try:
            timeout = (
                None
                if deadline_at_ms is None
                else max(0, (deadline_at_ms - int(time.time() * 1000)) / 1000)
            )
            done, _ = await asyncio.wait(
                {acquire, stopped}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
            if stopped in done or context.is_stopped():
                await self._record_terminal(
                    identity, status_path, self._base_status(identity, "cancelled")
                )
                return
            if acquire not in done or (
                deadline_at_ms is not None and int(time.time() * 1000) >= deadline_at_ms
            ):
                await self._record_terminal(
                    identity,
                    status_path,
                    {
                        **self._base_status(identity, "failed"),
                        "error": {
                            "code": "worker_queue_timeout",
                            "message": "Worker prefetch wait expired",
                        },
                    },
                )
                return
            await self._execute_started(
                identity, payload, context, queued_at_ms=queued_at_ms
            )
        except asyncio.CancelledError:
            context.stop_generating()
            running = self._running.get(identity.key)
            if running is None or running.persisted_terminal is None:
                await self._record_terminal(
                    identity, status_path, self._base_status(identity, "cancelled")
                )
            raise
        finally:
            if not acquire.done():
                acquire.cancel()
            cancel_watch.cancel()
            stopped.cancel()
            await asyncio.gather(acquire, cancel_watch, stopped, return_exceptions=True)
            if (
                not acquire.cancelled()
                and acquire.exception() is None
                and acquire.result()
            ):
                self._execution_slots.release()
            await context.close()

    async def _execute_started(
        self,
        identity: DetachedTaskIdentity,
        payload: dict[str, Any],
        context: _DetachedContext,
        *,
        queued_at_ms: int | None = None,
    ) -> None:
        running = self._running.get(identity.key)
        if running is not None:
            running.started = True
        # Validate the directory once for this operation, off the event loop;
        # deriving sibling filenames does not require rechecking each parent.
        attempt_root = await asyncio.to_thread(self._attempt_root, identity)
        status_path = attempt_root / "worker-status.json"
        response_path = attempt_root / "worker-response.jsonl"
        temporary = response_path.with_name(
            response_path.name + f".part-{uuid.uuid4().hex}"
        )
        execution = asyncio.current_task()
        assert execution is not None
        cancel_watch = asyncio.create_task(self._watch_cancel(identity, context))
        cancel_enforcer = asyncio.create_task(self._enforce_cancel(context, execution))
        started_at_ms = int(time.time() * 1000)
        status_stop = asyncio.Event()
        status_heartbeat = asyncio.create_task(
            self._heartbeat_status(identity, status_path, started_at_ms, status_stop)
        )
        started = time.monotonic()
        await asyncio.to_thread(
            self._atomic_json,
            status_path,
            {**self._base_status(identity, "running"), "started_at_ms": started_at_ms},
        )
        result_token = BINARY_RESULT_WRITER.set(
            BinaryResultWriter(attempt_root) if self.binary_results else None
        )
        try:
            if self.inline_results:
                terminal = None
                async for chunk in self.handler.generate(payload, context):
                    if not isinstance(chunk, dict):
                        raise RuntimeError(
                            "Omni detached response chunk is not an object"
                        )
                    if chunk.get("status") in _TERMINAL:
                        if terminal is not None:
                            raise RuntimeError(
                                "Worker returned multiple terminal responses"
                            )
                        terminal = normalize_inline_result(chunk)
                if terminal is None and not context.is_stopped():
                    raise RuntimeError("Worker stream ended without terminal response")
                response_fields = {
                    "result_format": INLINE_RESULT_FORMAT,
                    "inline_result": terminal,
                }
            else:
                digest = hashlib.sha256()
                written = 0
                with temporary.open("xb") as stream:
                    async for chunk in self.handler.generate(payload, context):
                        if not isinstance(chunk, dict):
                            raise RuntimeError(
                                "Omni detached response chunk is not an object"
                            )
                        encoded = await asyncio.to_thread(
                            lambda value: (
                                json.dumps(
                                    value, ensure_ascii=False, separators=(",", ":")
                                ).encode("utf-8")
                                + b"\n"
                            ),
                            chunk,
                        )
                        await asyncio.to_thread(stream.write, encoded)
                        digest.update(encoded)
                        written += len(encoded)
                    await asyncio.to_thread(stream.flush)
                    await asyncio.to_thread(os.fsync, stream.fileno())
                response_fields = {
                    "response_path": str(response_path),
                    "response_bytes": written,
                    "response_sha256": digest.hexdigest(),
                }
            if context.is_stopped():
                temporary.unlink(missing_ok=True)
                status_stop.set()
                await asyncio.gather(status_heartbeat, return_exceptions=True)
                await asyncio.to_thread(
                    self._atomic_json,
                    status_path,
                    {**self._base_status(identity, "cancelled")},
                )
                return
            if not self.inline_results:
                await asyncio.to_thread(os.replace, temporary, response_path)
            status_stop.set()
            await asyncio.gather(status_heartbeat, return_exceptions=True)
            completed = {
                **self._base_status(identity, "completed"),
                **response_fields,
                "inference_time_s": max(0.0, time.monotonic() - started),
            }
            if queued_at_ms is not None:
                completed["worker_queue_wait_s"] = max(
                    0.0, (started_at_ms - queued_at_ms) / 1000
                )
            await self._record_terminal(identity, status_path, completed)
        except asyncio.CancelledError:
            context.stop_generating()
            temporary.unlink(missing_ok=True)
            status_stop.set()
            await asyncio.gather(status_heartbeat, return_exceptions=True)
            await asyncio.to_thread(
                self._atomic_json,
                status_path,
                {**self._base_status(identity, "cancelled")},
            )
            raise
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            status_stop.set()
            await asyncio.gather(status_heartbeat, return_exceptions=True)
            logger.exception("detached Omni task failed: %s", identity.task_id)
            failed = {
                **self._base_status(identity, "failed"),
                "error": {
                    "code": "worker_failed",
                    "message": str(exc)[:1024] or "detached Omni task failed",
                },
            }
            await self._record_terminal(identity, status_path, failed)
        finally:
            BINARY_RESULT_WRITER.reset(result_token)
            cancel_watch.cancel()
            cancel_enforcer.cancel()
            status_stop.set()
            status_heartbeat.cancel()
            await asyncio.gather(
                cancel_watch,
                cancel_enforcer,
                status_heartbeat,
                return_exceptions=True,
            )
            await context.close()

    async def _status(self, identity: DetachedTaskIdentity) -> dict[str, Any]:
        def read():
            return self._read_status(self._status_path(identity))

        value = await asyncio.to_thread(read)
        if value is None:
            return {**self._base_status(identity, "not_found")}
        return value

    async def _wait_terminal(
        self, identity: DetachedTaskIdentity
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Attach to one local execution without owning its lifetime.

        The initial ``watching`` response lets the Gateway distinguish a
        healthy event channel from a connection attempt that is still
        pending. Cancelling this response stream must never cancel the
        detached inference task, so the execution is always shielded.
        """

        def read():
            path = self._status_path(identity)
            return path, self._read_status(path)

        running = self._running.get(identity.key)
        if running is None:
            # Reconnects and restarted processes still recover from disk. A
            # local execution is already validated against the full identity.
            _, status = await asyncio.to_thread(read)
            yield (
                status
                if status is not None
                else self._base_status(identity, "not_found")
            )
            return

        yield self._base_status(identity, "watching")
        try:
            await asyncio.shield(running.execution)
        except asyncio.CancelledError:
            raise
        except Exception:
            # _execute records its terminal failure before it returns. Read
            # that durable status below instead of leaking an internal task
            # exception across the private protocol.
            pass

        terminal = copy.deepcopy(running.persisted_terminal)
        if terminal is None:
            # Cancellation, legacy subclasses and exceptional recorder paths
            # still read disk. Never invent a successful completion.
            _, terminal = await asyncio.to_thread(read)
        if terminal is None or terminal.get("state") not in _TERMINAL:
            raise RuntimeError(
                "detached execution ended without a durable terminal status"
            )
        yield terminal

    async def _cancel(self, identity: DetachedTaskIdentity) -> dict[str, Any]:
        def _write_cancel() -> None:
            path = self._cancel_path(identity)
            path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
            os.close(descriptor)

        await asyncio.to_thread(_write_cancel)
        running = self._running.get(identity.key)
        if running is not None:
            running.context.stop_generating()
        return {**self._base_status(identity, "cancel_requested")}

    async def shutdown(self) -> None:
        """Drain accepted inference before aborting it at the shutdown deadline."""

        self._accepting = False
        executions = [item.execution for item in self._running.values()]
        if not executions:
            return
        done, pending = await asyncio.wait(executions, timeout=self.drain_timeout_s)
        del done
        if not pending:
            return
        logger.warning(
            "detached Omni drain timed out with %d task(s); aborting them", len(pending)
        )
        for item in list(self._running.values()):
            if item.execution in pending:
                item.context.stop_generating()
                item.execution.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
