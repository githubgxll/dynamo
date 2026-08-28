# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in durable execution wrapper for long-running Omni video requests.

The normal Omni request path remains untouched.  A caller must send the
private, versioned ``_dingo_video_task`` envelope and the worker must have an
explicit shared task root configured before this code is reachable.
"""

from __future__ import annotations

import asyncio
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

from dingo.common.video_task_protocol import (
    ENVELOPE_KEY,
    SCHEMA_VERSION,
    DetachedTaskIdentity,
    detached_attempt_root,
    detached_envelope,
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
    ) -> None:
        if drain_timeout_s <= 0:
            raise ValueError("detached drain timeout must be positive")
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

    async def generate(
        self, request: dict[str, Any], context: Any
    ) -> AsyncGenerator[dict[str, Any], None]:
        envelope = request.get(ENVELOPE_KEY)
        if envelope is None:
            async for chunk in self.handler.generate(request, context):
                yield chunk
            return
        if set(request) != {ENVELOPE_KEY} or not isinstance(envelope, Mapping):
            raise ValueError("detached task request must contain only its envelope")
        identity = DetachedTaskIdentity.from_envelope(envelope)
        op = envelope.get("op")
        if op == "submit":
            yield await self._submit(identity, envelope.get("payload"))
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

    def _validate_task_manifest(self, identity: DetachedTaskIdentity) -> None:
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

    @staticmethod
    def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".part-{uuid.uuid4().hex}")
        payload = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
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
        }

    async def _submit(
        self, identity: DetachedTaskIdentity, payload: Any
    ) -> dict[str, Any]:
        if not self._accepting:
            raise RuntimeError("detached Worker is draining")
        if not isinstance(payload, dict):
            raise ValueError("detached submit payload must be an object")
        await asyncio.to_thread(self._validate_task_manifest, identity)
        attempt_root = self._attempt_root(identity)
        status_path = self._status_path(identity)
        async with self._lock:
            running = self._running.get(identity.key)
            if running is not None:
                return {**self._base_status(identity, "running"), "accepted": False}
            existing = await asyncio.to_thread(self._read_status, status_path)
            if existing is not None and existing.get("state") in _TERMINAL:
                return {**existing, "accepted": False}
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
            await asyncio.to_thread(self._atomic_json, status_path, initial)
            request_id = (
                f"{identity.task_id}-{identity.attempt}-"
                f"{identity.execution_token[:12]}"
            )
            detached_context = _DetachedContext(request_id)
            execution = asyncio.create_task(
                self._execute(identity, payload, detached_context),
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
        path = self._cancel_path(identity)
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

    async def _execute(
        self,
        identity: DetachedTaskIdentity,
        payload: dict[str, Any],
        context: _DetachedContext,
    ) -> None:
        attempt_root = self._attempt_root(identity)
        status_path = self._status_path(identity)
        response_path = self._response_path(identity)
        temporary = response_path.with_name(
            response_path.name + f".part-{uuid.uuid4().hex}"
        )
        execution = asyncio.current_task()
        assert execution is not None
        cancel_watch = asyncio.create_task(self._watch_cancel(identity, context))
        cancel_enforcer = asyncio.create_task(
            self._enforce_cancel(context, execution)
        )
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
        try:
            digest = hashlib.sha256()
            written = 0
            with temporary.open("xb") as stream:
                async for chunk in self.handler.generate(payload, context):
                    if not isinstance(chunk, dict):
                        raise RuntimeError(
                            "Omni detached response chunk is not an object"
                        )
                    encoded = await asyncio.to_thread(
                        lambda value: json.dumps(
                            value,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ).encode("utf-8")
                        + b"\n",
                        chunk,
                    )
                    await asyncio.to_thread(stream.write, encoded)
                    digest.update(encoded)
                    written += len(encoded)
                await asyncio.to_thread(stream.flush)
                await asyncio.to_thread(os.fsync, stream.fileno())
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
            await asyncio.to_thread(os.replace, temporary, response_path)
            status_stop.set()
            await asyncio.gather(status_heartbeat, return_exceptions=True)
            completed = {
                **self._base_status(identity, "completed"),
                "response_path": str(response_path),
                "response_bytes": written,
                "response_sha256": digest.hexdigest(),
                "inference_time_s": max(0.0, time.monotonic() - started),
            }
            await asyncio.to_thread(self._atomic_json, status_path, completed)
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
            await asyncio.to_thread(self._atomic_json, status_path, failed)
        finally:
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
        value = await asyncio.to_thread(
            self._read_status, self._status_path(identity)
        )
        if value is None:
            return {**self._base_status(identity, "not_found")}
        return value

    async def _cancel(self, identity: DetachedTaskIdentity) -> dict[str, Any]:
        path = self._cancel_path(identity)

        def _write_cancel() -> None:
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
