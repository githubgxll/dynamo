# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-level guard for a permanently terminated Dynamo Runtime."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any, TypeVar


_T = TypeVar("_T")


async def run_with_runtime_termination_guard(
    operation: Awaitable[_T],
    runtime: Any,
    shutdown_event: asyncio.Event,
    *,
    component: str,
) -> _T:
    """Run ``operation`` until it completes or the Runtime terminates.

    A Dynamo Runtime whose cancellation token has fired cannot rebuild its
    discovery clients in place.  An unexpected ``wait_shutdown`` completion
    therefore cancels the component coroutine so its normal cleanup runs, then
    raises and lets the process supervisor create a fresh Runtime.

    ``shutdown_event`` distinguishes this condition from an intentional signal
    shutdown.  Signal handling sets that event before calling
    ``runtime.shutdown()``, so the component retains its ordinary graceful
    drain behavior in that path.
    """

    operation_task = asyncio.ensure_future(operation)
    runtime_task = asyncio.ensure_future(runtime.wait_shutdown())
    try:
        done, _pending = await asyncio.wait(
            {operation_task, runtime_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if runtime_task in done and not shutdown_event.is_set():
            runtime_error: BaseException | None = None
            if not runtime_task.cancelled():
                runtime_error = runtime_task.exception()
            if not operation_task.done():
                operation_task.cancel()
            await asyncio.gather(operation_task, return_exceptions=True)
            error = RuntimeError(
                f"{component} Dynamo Runtime terminated unexpectedly; "
                "the process must restart"
            )
            if runtime_error is not None:
                raise error from runtime_error
            raise error
        return await operation_task
    except BaseException:
        if not operation_task.done():
            operation_task.cancel()
            await asyncio.gather(operation_task, return_exceptions=True)
        raise
    finally:
        if not runtime_task.done():
            runtime_task.cancel()
        await asyncio.gather(runtime_task, return_exceptions=True)
