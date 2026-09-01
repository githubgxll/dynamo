# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

from dingo.video_gateway.__main__ import _wait_for_shutdown


class FakeRuntime:
    def __init__(self) -> None:
        self.stopped = asyncio.Event()
        self.wait_cancelled = asyncio.Event()

    async def wait_shutdown(self) -> None:
        try:
            await self.stopped.wait()
        except asyncio.CancelledError:
            self.wait_cancelled.set()
            raise


async def test_wait_for_shutdown_detects_runtime_termination():
    runtime = FakeRuntime()
    stopped = asyncio.Event()
    waiting = asyncio.create_task(_wait_for_shutdown(runtime, stopped))
    await asyncio.sleep(0)

    runtime.stopped.set()

    assert await asyncio.wait_for(waiting, timeout=1) is True


async def test_wait_for_shutdown_accepts_signal_and_cancels_runtime_wait():
    runtime = FakeRuntime()
    stopped = asyncio.Event()
    waiting = asyncio.create_task(_wait_for_shutdown(runtime, stopped))
    await asyncio.sleep(0)

    stopped.set()

    assert await asyncio.wait_for(waiting, timeout=1) is False
    assert runtime.wait_cancelled.is_set()


async def test_wait_for_shutdown_prefers_normal_signal_when_both_are_set():
    runtime = FakeRuntime()
    stopped = asyncio.Event()
    runtime.stopped.set()
    stopped.set()

    assert await _wait_for_shutdown(runtime, stopped) is False
