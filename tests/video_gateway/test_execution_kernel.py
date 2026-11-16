# SPDX-License-Identifier: Apache-2.0
import asyncio

import pytest

from dingo.worker.execution import AdmissionState, ExecutionContext, ExecutionKernel


def test_settlement_queue_permission_is_separate_from_fault_and_drain():
    state = AdmissionState(settlement_paused=True)
    assert state.can_queue() and not state.can_execute()
    state.recovering = True
    assert not state.can_queue() and not state.can_execute()
    state.recovering = False
    state.draining = True
    state.settlement_paused = False
    assert not state.can_queue() and not state.can_execute()
    state.draining = False
    state.components_ready = False
    assert not state.can_queue() and not state.can_execute()


async def test_waiter_cancel_and_timeout_cannot_cancel_execution():
    kernel = ExecutionKernel(1)
    context = ExecutionContext("task")
    entered, release = asyncio.Event(), asyncio.Event()

    async def work():
        entered.set()
        await release.wait()

    record = kernel.start("one", object(), context, work())
    await entered.wait()
    observer = asyncio.create_task(kernel.wait(record))
    await asyncio.sleep(0)
    observer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await observer
    await kernel.wait(record, 0.001)
    assert not record.execution.done() and not context.is_stopped()
    assert not await kernel.drain(0.001)
    release.set()
    assert await kernel.drain(1)


async def test_terminal_cache_requires_durable_write_and_is_execution_scoped():
    kernel = ExecutionKernel(1)
    release = asyncio.Event()
    record = kernel.start("one", object(), ExecutionContext("one"), release.wait())
    entered, written = asyncio.Event(), asyncio.Event()

    async def writer():
        entered.set()
        await written.wait()

    status = {"state": "completed", "nested": {"id": "one"}}
    publish = asyncio.create_task(kernel.publish_terminal("one", status, writer))
    await entered.wait()
    assert record.persisted_terminal is None
    written.set()
    await publish
    status["nested"]["id"] = "mutated"
    assert record.persisted_terminal["nested"]["id"] == "one"
    release.set()
    await kernel.drain(1)
    await asyncio.sleep(0)
    assert not kernel.records and record.persisted_terminal["state"] == "completed"


async def test_failed_persistence_cannot_publish_terminal():
    kernel = ExecutionKernel(1)
    release = asyncio.Event()
    record = kernel.start("one", object(), ExecutionContext("one"), release.wait())

    async def failed():
        raise OSError("disk failure")

    with pytest.raises(OSError):
        await kernel.publish_terminal("one", {"state": "completed"}, failed)
    assert record.persisted_terminal is None
    release.set()
    await kernel.drain(1)


async def test_cancel_signal_reaches_async_and_threaded_workloads_without_force_cancel():
    kernel = ExecutionKernel(1)
    release = asyncio.Event()
    context = ExecutionContext("one")
    record = kernel.start("one", object(), context, release.wait())
    waiter = context.async_killed_or_stopped()
    kernel.cancel("one")
    assert context.cancelled.is_set() and context.is_stopped()
    assert await asyncio.wait_for(waiter, 1)
    assert not record.execution.done()
    release.set()
    await kernel.drain(1)


async def test_abort_is_explicit_and_other_execution_is_not_cancelled():
    kernel = ExecutionKernel(2)
    release = asyncio.Event()
    a = kernel.start("one", object(), ExecutionContext("one"), release.wait())
    b = kernel.start("two", object(), ExecutionContext("two"), release.wait())
    kernel.cancel("one")
    assert a.context.is_stopped() and not b.context.is_stopped()
    assert await kernel.drain(0.001, abort_on_timeout=True)
    assert a.execution.cancelled() and b.execution.cancelled()
