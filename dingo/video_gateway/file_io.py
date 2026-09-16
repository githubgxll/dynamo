# SPDX-License-Identifier: Apache-2.0
"""Run filesystem work off-loop without abandoning in-flight file operations."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from threading import Event
from typing import Any, Callable


async def run_file_io(function: Callable[..., Any], /, *args, **kwargs) -> Any:
    """Drain a started operation before propagating cancellation to its owner.

    Cancelling an asyncio future cannot stop an OS filesystem call. Waiting for
    that call to finish keeps subsequent close/unlink and budget release from
    racing with a still-running read or write. It does not block the event loop.
    A permanently hung filesystem still requires storage/node recovery.
    """
    work = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    return await _drain_on_cancel(work)


async def run_cancellable_file_io(
    function: Callable[..., Any], /, *args, **kwargs
) -> Any:
    """Run a multi-step operation with a cooperative cancellation Event.

    The function receives the Event as its first argument and checks it between
    safe stages. Cancellation still drains the active syscall/thread before
    returning, including when the caller is cancelled repeatedly.
    """
    cancelled = Event()
    work = asyncio.create_task(asyncio.to_thread(function, cancelled, *args, **kwargs))
    return await _drain_on_cancel(work, on_cancel=cancelled.set)


async def _drain_on_cancel(
    work: asyncio.Task, on_cancel: Callable[[], None] | None = None
) -> Any:
    try:
        return await asyncio.shield(work)
    except asyncio.CancelledError:
        if on_cancel is not None:
            on_cancel()
        while not work.done():
            try:
                await asyncio.shield(work)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        if not work.cancelled():
            work.exception()  # retrieve any worker error; preserve cancellation
        raise


@asynccontextmanager
async def opened_file(opener: Callable[..., Any], /, *args, **kwargs):
    """Close even when cancellation arrives while the file is being opened."""
    opened = []

    def acquire():
        stream = opener(*args, **kwargs)
        opened.append(stream)
        return stream

    try:
        yield await run_file_io(acquire)
    finally:
        if opened:
            await run_file_io(opened[0].close)
