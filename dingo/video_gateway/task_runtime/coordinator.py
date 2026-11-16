# SPDX-License-Identifier: Apache-2.0
"""Finite-task dispatch and local execution ownership, independent of business data.

Adapters select eligible work and perform fenced storage operations. A claim is
confirmed before execution starts; ambiguous storage operations raise rather than
masquerading as an empty queue. Execution remains owned after an RPC disconnect.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from .transactions import UnconfirmedTransaction

Selection = TypeVar("Selection")
Record = TypeVar("Record")


@dataclass(frozen=True)
class DispatchClaim(Generic[Record]):
    record: Record | None = None
    retry_now: bool = False


class DispatchAdapter(Protocol[Selection, Record]):
    async def candidates(self) -> Iterable[Selection]:
        ...

    def key(self, candidate: Selection) -> str:
        ...

    async def claim(self, candidate: Selection) -> DispatchClaim[Record]:
        ...

    def committed(self, candidate: Selection, record: Record) -> None:
        ...

    async def release_preparation(self, candidate: Selection) -> None:
        ...

    async def execute(self, record: Record) -> None:
        ...


class ExecutionSupervisor:
    """One local controller per task key, shared by new and adopted attempts.

    Completion removes only the matching controller. Exceptions are retrieved
    here so an escaped business error is visible and never an unobserved task.
    The caller decides whether shutdown cancels controllers or waits for them;
    cancelling a controller does not itself prove native computation has ended.
    """

    def __init__(
        self, on_error: Callable[[str, BaseException], None], *, executions=None
    ):
        self.on_error = on_error
        self.tasks: dict[str, asyncio.Task] = {}
        self.executions: set[asyncio.Task] = (
            executions if executions is not None else set()
        )

    def contains(self, key: str) -> bool:
        task = self.tasks.get(key)
        return task is not None and not task.done()

    def spawn(self, key, operation, *, name=None):
        if self.contains(key):
            return False

        async def run():
            await operation()

        coroutine = run()
        try:
            task = asyncio.create_task(coroutine, name=name or key)
        except BaseException:
            coroutine.close()
            raise
        self.tasks[key] = task
        self.executions.add(task)

        def completed(future):
            if self.tasks.get(key) is future:
                self.tasks.pop(key)
            self.executions.discard(future)
            if not future.cancelled() and (error := future.exception()) is not None:
                self.on_error(key, error)

        task.add_done_callback(completed)
        return True


class DispatchCoordinator:
    """The common select/claim/launch flow; policy and atomic writes are injected.

    A confirmed claim whose local launch fails remains pending and is launched
    before more work is claimed. No storage rollback or retry of the model is
    inferred from a local launch error. Prepared resources transfer to the
    execution only after a confirmed claim.
    """

    def __init__(self, supervisor, *, enabled=lambda: True):
        self.supervisor = supervisor
        self.enabled = enabled
        self.pending = {}
        self.uncertain = {}
        self.claiming = set()
        self._published = set()

    def manages(self, key):
        """Include unconfirmed ownership; recovery must not bypass this state."""
        return (
            self.supervisor.contains(key)
            or key in self.pending
            or key in self.uncertain
            or key in self.claiming
        )

    def _launch(self, key, adapter, candidate, record):
        if not self.supervisor.contains(key):
            if key not in self._published:
                adapter.committed(candidate, record)
                self._published.add(key)
            self.supervisor.spawn(key, lambda: adapter.execute(record), name=key)
        self.pending.pop(key, None)
        self._published.discard(key)

    async def claim_one(self, adapter, candidate):
        """Common entry for a queued claim and an orphaned-attempt adoption."""
        key = adapter.key(candidate)
        if self.manages(key):
            return False
        self.claiming.add(key)
        claimed = False
        try:
            try:
                result = await adapter.claim(candidate)
            except UnconfirmedTransaction as error:
                self.uncertain[key] = (adapter, candidate, error.resolve)
                claimed = True
                return False
            if result.record is None:
                return result.retry_now
            claimed = True
            self.pending[key] = (adapter, candidate, result.record)
            self._launch(key, adapter, candidate, result.record)
            return True
        finally:
            self.claiming.discard(key)
            if not claimed:
                await adapter.release_preparation(candidate)

    async def dispatch(self, adapter):
        for key, (owner, candidate, resolve) in tuple(self.uncertain.items()):
            if owner is not adapter:
                continue
            try:
                record = await resolve()
            except UnconfirmedTransaction as error:
                self.uncertain[key] = (owner, candidate, error.resolve)
                # Unknown physical occupancy cannot become a free local permit.
                return False
            self.uncertain.pop(key)
            if record is None:
                await owner.release_preparation(candidate)
            else:
                self.pending[key] = (owner, candidate, record)
        for key, (owner, candidate, record) in tuple(self.pending.items()):
            if owner is adapter:
                # Already-owned work must still settle while admission drains.
                self._launch(key, owner, candidate, record)
        if not self.enabled():
            return False
        again = False
        for candidate in await adapter.candidates():
            if not self.enabled():
                break
            again |= await self.claim_one(adapter, candidate)
            if self.uncertain:
                return False
        return again


async def monitor_execution(operation, *, monitors, on_monitor_finished):
    """Await one operation or an ordered set of failure/cancel observers.

    A completed operation wins simultaneous notifications. Monitor priority is
    supplied by the adapter (ownership loss must precede suspected Worker loss).
    External monitor tasks stay owned by the caller; only this function's own
    children are cancelled and joined when the controller ends.
    """
    work = asyncio.create_task(operation)
    owned = []
    observers = []
    try:
        for name, monitor in monitors:
            if isinstance(monitor, asyncio.Task):
                child = monitor
            else:
                child = asyncio.create_task(monitor)
                owned.append(child)
            observers.append((name, child))
        done, _ = await asyncio.wait(
            [work, *(child for _, child in observers)],
            return_when=asyncio.FIRST_COMPLETED,
        )
        if work in done:
            return await work
        for name, child in observers:
            if child in done:
                await on_monitor_finished(name, child)
                raise RuntimeError(f"execution monitor ended unexpectedly: {name}")
    finally:
        for child in [work, *owned]:
            if not child.done():
                child.cancel()
        await asyncio.gather(work, *owned, return_exceptions=True)
