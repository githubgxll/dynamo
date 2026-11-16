# SPDX-License-Identifier: Apache-2.0
import asyncio
from types import SimpleNamespace

import pytest

from dingo.video_gateway.task_runtime.scheduler import DispatchLoop
from dingo.video_gateway.task_runtime.transport import control_request


@pytest.mark.parametrize("payload", [{"_dingo_task": {}}, {"_dingo_video_task": {}}])
async def test_control_call_preserves_caller_context_and_checks_single_frame(payload):
    context = SimpleNamespace(stopped=False)

    async def direct(request, instance, ctx):
        assert request is payload and instance == 7 and ctx is context

        async def stream():
            yield {"state": "accepted"}

        return stream()

    client = SimpleNamespace(direct=direct)
    assert await control_request(client, "7", payload, context) == {"state": "accepted"}
    assert not context.stopped


@pytest.mark.parametrize("frames", [[], [{}, {}], [None], ["invalid"]])
async def test_invalid_control_framing_is_rejected(frames):
    async def direct(*args):
        async def stream():
            for frame in frames:
                yield frame

        return stream()

    with pytest.raises(ValueError):
        await control_request(SimpleNamespace(direct=direct), 7, {}, object())


async def test_control_timeout_and_error_annotations_propagate():
    release = asyncio.Event()

    async def direct(*args):
        async def stream():
            await release.wait()
            yield {}

        return stream()

    with pytest.raises(TimeoutError):
        await control_request(
            SimpleNamespace(direct=direct), 7, {}, object(), timeout_s=0.001
        )

    async def error(*args):
        async def stream():
            yield SimpleNamespace(
                is_error=lambda: True, comments=lambda: ["backend unavailable"]
            )

        return stream()

    with pytest.raises(RuntimeError, match="backend unavailable"):
        await control_request(SimpleNamespace(direct=error), 7, {}, object())


async def test_submission_during_dispatch_turn_is_not_lost_and_shutdown_is_prompt():
    stop, wakeup, second = asyncio.Event(), asyncio.Event(), asyncio.Event()
    turns = 0

    async def advance():
        nonlocal turns
        turns += 1
        if turns == 1:
            wakeup.set()  # Simulates a submission while repository I/O is pending.
        else:
            second.set()
        return False

    loop = DispatchLoop(
        stop=stop,
        wakeup=wakeup,
        advance=advance,
        interval_s=60,
        on_error=lambda: pytest.fail("unexpected scheduler error"),
    )
    task = asyncio.create_task(loop.run())
    await asyncio.wait_for(second.wait(), 1)
    stop.set()
    await asyncio.wait_for(task, 1)
    assert turns == 2


async def test_dispatch_failure_retries_and_cancellation_is_not_swallowed():
    stop, seen = asyncio.Event(), asyncio.Event()
    errors = []
    attempts = 0

    async def advance():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary store error")
        seen.set()
        return False

    loop = DispatchLoop(
        stop=stop,
        advance=advance,
        interval_s=0.001,
        error_interval_s=0.001,
        on_error=lambda: errors.append("retry"),
    )
    task = asyncio.create_task(loop.run())
    await asyncio.wait_for(seen.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert errors == ["retry"] and attempts >= 2
