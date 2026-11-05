"""One durable GPU business permit across H3, asynchronous SR and remux."""
from __future__ import annotations

import asyncio
import errno
from dataclasses import dataclass
import inspect
import json
import os
from pathlib import Path
from uuid import uuid4

from .state import FailureLatch, atomic_json, identifier, now, process_birth, read_json


@dataclass(frozen=True)
class TerminalResult:
    action_id: str
    completed: bool
    output_done: bool
    succeeded: bool
    payload: object = None
    job_id: str | None = None

    def require_success(self, action_id):
        if (self.action_id != action_id or self.completed is not True or self.output_done is not True
                or self.succeeded is not True):
            raise RuntimeError("Backend terminal/output completion is not proven; preserve operation")
        return self


class AsyncFileLock:
    """Nonblocking acquisition keeps cancellation responsive; no stale lock deletion."""
    def __init__(self, path):
        self.path, self.stream = Path(path), None

    async def __aenter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise RuntimeError("Operation lock cannot be a symlink")
        self.stream = self.path.open("a+b")
        try:
            if self.path.stat().st_size == 0:
                self.stream.write(b"0")
                self.stream.flush()
            while True:
                try:
                    if os.name == "posix":
                        import fcntl
                        fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    else:
                        import msvcrt
                        self.stream.seek(0)
                        msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
                    return self
                except (BlockingIOError, OSError) as error:
                    if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    await asyncio.sleep(.02)
        except BaseException:
            self.stream.close()
            self.stream = None
            raise

    async def __aexit__(self, *_args):
        try:
            if os.name == "posix":
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            else:
                import msvcrt
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            self.stream.close()
            self.stream = None


async def invoke(callback, *args):
    value = callback(*args)
    return await value if inspect.isawaitable(value) else value


class GatewayExecutor:
    """Gateway callbacks must prove terminal output, including async SR jobs.

    Cancellation and transport failure do not imply GPU completion. The shared
    latch is persisted while ownership is held, and the active lease remains.
    Management cancellation/reconciliation is deliberately outside this permit.
    """
    def __init__(self, root, *, profile, service_epoch, handoff=None, ready_check=None,
                 birth_reader=process_birth):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.profile = profile
        self.epoch = identifier(service_epoch)
        self.handoff = handoff
        self.ready_check = ready_check or self._file_ready
        self.birth_reader = birth_reader
        self.latch = FailureLatch(self.root)
        self.owned = None
        if profile not in ("fl2va-sr", "ref2va-sr"):
            raise ValueError("Unknown H3/SR production profile")
        if profile == "ref2va-sr" and not callable(handoff):
            raise ValueError("Ref profile requires a cache-handoff controller")
        if profile == "fl2va-sr" and handoff is not None:
            raise ValueError("FL profile cannot enable Ref handoff")

    def _file_ready(self):
        value = read_json(self.root / "state.json")
        return (value.get("service_ready") is True and value.get("service_epoch") == self.epoch
                and value.get("profile") == self.profile)

    def operation_headers(self):
        if self.owned is None:
            raise RuntimeError("No owned GPU operation")
        return {"x-h3sr-operation-token": self.owned["operation_token"],
                "x-h3sr-action-id": self.owned["action_id"]}

    def _history(self):
        path = self.root / "history.json"
        return read_json(path) if path.exists() else {"h3_completed": 0, "last_business": "sr-loaded"}

    def _reason(self, history):
        if history.get("h3_completed", 0) == 0:
            return "after-sr-load-before-first-h3"
        if history.get("last_business") == "sr":
            return "after-sr-before-next-h3"
        return "before-consecutive-h3"

    async def _terminal(self, callback, action_id, phase):
        value = await invoke(callback)
        if not isinstance(value, TerminalResult):
            raise RuntimeError(phase + " callback must return TerminalResult, never raw HTTP/202")
        value.require_success(action_id)
        return value

    async def _available(self):
        self.latch.assert_open()
        if not await invoke(self.ready_check):
            raise RuntimeError("An owned backend left Ready before the next GPU phase")

    async def execute_owned(self, action_id, h3_callable=None, sr_callable=None, remux_callable=None):
        action_id = identifier(action_id)
        if h3_callable is None and sr_callable is None:
            raise ValueError("A backend operation is required")
        async with AsyncFileLock(self.root / "operation.lock"):
            self.latch.assert_open()
            if not await invoke(self.ready_check):
                raise RuntimeError("Both owned backend services must be Ready")
            active = self.root / "active.json"
            if active.exists():
                self.latch.trip(epoch=self.epoch, action_id=action_id,
                                error=RuntimeError("Unresolved earlier GPU operation lease"))
                raise RuntimeError("Unresolved earlier GPU operation; no replay")
            result_path = self.root / "actions" / (action_id + ".json")
            if result_path.exists():
                raise RuntimeError("Action identity already exists; observe prior result instead of replay")
            birth = self.birth_reader()
            lease = {"schema": "h3sr-production-operation/1", "service_epoch": self.epoch,
                     "action_id": action_id, "profile": self.profile, "operation_token": uuid4().hex,
                     "controller_birth": birth, "started_at": now(), "status": "RUNNING"}
            atomic_json(active, lease)
            self.owned = lease
            stages = {}
            try:
                history = self._history()
                if h3_callable is not None:
                    if self.profile == "ref2va-sr":
                        from . import worker_extension as worker
                        from .control import handoff_identifier
                        reason = self._reason(history)
                        report = await invoke(self.handoff, reason, lease)
                        expected = read_json(self.root / "state.json").get("sr_worker_identities")
                        stages["handoff"] = worker.validate_report(report, handoff_identifier(self.epoch, action_id),
                            reason, expected, self.epoch, worker.source_sha256(), False, False)
                    await self._available()
                    stages["h3"] = await self._terminal(h3_callable, action_id, "H3")
                    history["h3_completed"] = history.get("h3_completed", 0) + 1
                    history["last_business"] = "h3"
                if sr_callable is not None:
                    # This remains inside operation.lock after restore-long's
                    # HTTP202. Only a complete job AND output receipt passes.
                    await self._available()
                    stages["sr"] = await self._terminal(sr_callable, action_id, "SR")
                    history["last_business"] = "sr"
                if remux_callable is not None:
                    await self._available()
                    stages["remux"] = await self._terminal(remux_callable, action_id, "audio remux")
                self.latch.assert_open()
                if read_json(active) != lease:
                    raise RuntimeError("Owned lease changed before terminal completion")
                atomic_json(self.root / "history.json", history)
                record = {"schema": "h3sr-production-action/1", "service_epoch": self.epoch,
                          "action_id": action_id, "status": "SUCCEEDED", "completed": True,
                          "output_done": True, "finished_at": now(), "phases": list(stages)}
                atomic_json(result_path, record)
                active.unlink()
                return {"receipt": record, "results": stages}
            except BaseException as error:
                # Must precede releasing the cross-process operation lock.
                self.latch.trip(epoch=self.epoch, action_id=action_id, error=error)
                atomic_json(result_path, {"schema": "h3sr-production-action/1", "service_epoch": self.epoch,
                    "action_id": action_id, "status": "BLOCKED", "completed": False,
                    "output_done": False, "error_type": type(error).__name__, "at": now()})
                raise
            finally:
                self.owned = None

    async def management_cancel(self, callback, *args):
        """Do not block the cancellation/status path behind a poisoned GPU lock."""
        return await invoke(callback, *args)

    def record_reconciliation(self, action_id, terminal_result):
        """Record observed completion without silently clearing lease or latch."""
        identifier(action_id)
        if not isinstance(terminal_result, TerminalResult):
            raise TypeError("Reconciliation requires a terminal backend receipt")
        if terminal_result.action_id != action_id or terminal_result.completed is not True or terminal_result.output_done is not True:
            raise RuntimeError("Backend output is still unknown")
        atomic_json(self.root / "reconciliation" / (action_id + ".json"), {
            "schema": "h3sr-production-reconciliation/1", "service_epoch": self.epoch,
            "action_id": action_id, "terminal": True, "output_done": True,
            "succeeded": terminal_result.succeeded, "at": now(), "gate_still_blocked": True})
