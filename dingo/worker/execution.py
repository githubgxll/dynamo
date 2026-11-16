# SPDX-License-Identifier: Apache-2.0
"""Process-local execution ownership shared by finite-task protocol adapters.

Persistence and native execution semantics are supplied by adapters. A control
RPC never owns the task it observes, and a terminal record is published locally
only after its durable writer completes.
"""
import asyncio
import copy
import logging
import threading
from dataclasses import dataclass
from typing import Any


class ExecutionContext:
    """Cancellation signal usable by both async engines and threaded handlers."""

    def __init__(self, request_id):
        self._request_id = request_id
        self.cancelled = threading.Event()
        self._async_cancelled = asyncio.Event()
        self._waiters = set()

    def id(self):
        return self._request_id

    def stop_generating(self):
        self.cancelled.set()
        self._async_cancelled.set()

    def is_stopped(self):
        return self.cancelled.is_set()

    def async_killed_or_stopped(self):
        waiter = asyncio.create_task(self._async_cancelled.wait())
        self._waiters.add(waiter)
        waiter.add_done_callback(self._waiters.discard)
        return waiter

    async def wait_stopped(self):
        await self._async_cancelled.wait()

    async def close(self):
        self.stop_generating()
        if self._waiters:
            await asyncio.gather(*list(self._waiters), return_exceptions=True)


@dataclass(slots=True)
class RunningExecution:
    identity: Any
    context: ExecutionContext
    execution: asyncio.Task
    persisted_terminal: dict | None = None
    started: bool = False


@dataclass
class AdmissionState:
    """Normal settlement pauses execution without claiming component death."""

    draining: bool = False
    recovering: bool = False
    components_ready: bool = True
    settlement_paused: bool = False

    def can_execute(self, handler_ready=True):
        return (
            not self.draining
            and not self.recovering
            and self.components_ready
            and not self.settlement_paused
            and bool(handler_ready)
        )

    def can_queue(self, handler_queueable=True):
        return (
            not self.draining
            and not self.recovering
            and self.components_ready
            and bool(handler_queueable)
        )


class ExecutionKernel:
    def __init__(self, execution_capacity):
        if type(execution_capacity) is not int or execution_capacity < 1:
            raise ValueError("execution capacity must be positive")
        self.records = {}
        self.lock = asyncio.Lock()
        self.slots = asyncio.Semaphore(execution_capacity)
        self.admission = AdmissionState()

    def start(self, key, identity, context, operation, *, name=None):
        """Called under the admission lock, after durable acceptance."""
        if key in self.records:
            operation.close()
            raise ValueError("duplicate local execution")
        task = asyncio.create_task(operation, name=name)
        record = RunningExecution(identity, context, task)
        self.records[key] = record
        task.add_done_callback(lambda done: self._done(key, done))
        return record

    def _done(self, key, task):
        record = self.records.get(key)
        if record is not None and record.execution is task:
            self.records.pop(key)
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                logging.getLogger(__name__).error(
                    "detached execution escaped its terminal recorder",
                    exc_info=(type(error), error, error.__traceback__),
                )

    def cancel(self, key):
        record = self.records.get(key)
        if record is not None:
            record.context.stop_generating()

    async def publish_terminal(self, key, status, durable_write):
        await durable_write()
        record = self.records.get(key)
        if record is not None:
            record.persisted_terminal = copy.deepcopy(status)

    @staticmethod
    async def wait(record, timeout_s=None):
        """Detach waiter cancellation/timeouts from the owned execution."""
        try:
            if timeout_s is None:
                await asyncio.shield(record.execution)
            else:
                await asyncio.wait_for(asyncio.shield(record.execution), timeout_s)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if (
                current is not None and current.cancelling()
            ) or not record.execution.cancelled():
                raise
        except Exception:
            # Read the adapter's durable status after completion/failure. A
            # missing terminal status remains an error in the protocol adapter.
            logging.getLogger(__name__).debug(
                "execution wait ended; read the adapter's durable status", exc_info=True
            )

    async def drain(self, timeout_s, *, abort_on_timeout=False):
        tasks = [r.execution for r in self.records.values()]
        if not tasks:
            return True
        _, pending = await asyncio.wait(tasks, timeout=timeout_s)
        if pending and not abort_on_timeout:
            return False
        if pending:
            for record in list(self.records.values()):
                if record.execution in pending:
                    record.context.stop_generating()
                    record.execution.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        return True
