# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import pytest


_SOURCE = (
    Path(__file__).parents[2]
    / "dingo"
    / "common"
    / "utils"
    / "runtime_termination.py"
)
_SPEC = importlib.util.spec_from_file_location("runtime_termination", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
run_with_runtime_termination_guard = _MODULE.run_with_runtime_termination_guard


class _Runtime:
    def __init__(self) -> None:
        loop = asyncio.get_running_loop()
        self.terminated: asyncio.Future[None] = loop.create_future()

    def wait_shutdown(self) -> asyncio.Future[None]:
        return self.terminated


async def test_unexpected_runtime_termination_cancels_component() -> None:
    runtime = _Runtime()
    shutdown_event = asyncio.Event()
    component_stopped = asyncio.Event()

    async def component() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            component_stopped.set()

    guarded = asyncio.create_task(
        run_with_runtime_termination_guard(
            component(), runtime, shutdown_event, component="detached Video Worker"
        )
    )
    await asyncio.sleep(0)
    runtime.terminated.set_result(None)

    with pytest.raises(RuntimeError, match="process must restart"):
        await guarded
    assert component_stopped.is_set()


async def test_signal_shutdown_keeps_graceful_component_drain() -> None:
    runtime = _Runtime()
    shutdown_event = asyncio.Event()
    component_release = asyncio.Event()

    async def component() -> str:
        await component_release.wait()
        return "drained"

    guarded = asyncio.create_task(
        run_with_runtime_termination_guard(
            component(), runtime, shutdown_event, component="detached Video Worker"
        )
    )
    await asyncio.sleep(0)
    shutdown_event.set()
    runtime.terminated.set_result(None)
    await asyncio.sleep(0)
    assert not guarded.done()
    component_release.set()
    assert await guarded == "drained"


async def test_component_completion_cancels_only_runtime_waiter() -> None:
    runtime = _Runtime()
    result = await run_with_runtime_termination_guard(
        asyncio.sleep(0, result=7),
        runtime,
        asyncio.Event(),
        component="detached Video Worker",
    )
    assert result == 7
    assert runtime.terminated.cancelled()
