# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dingo.video_gateway import dispatcher as dispatcher_module
from dingo.video_gateway.dispatcher import VideoDispatcher
from dingo.video_gateway.models import TaskStatus


def _task(*, cancelled=False):
    return SimpleNamespace(
        id="video-test",
        pool_id="pool",
        status=TaskStatus.DISPATCHING,
        owner_generation="gateway-a",
        attempt=1,
        execution_token="token",
        cancel_requested_at_ms=1 if cancelled else None,
    )


def _dispatcher(expected, reads, current=None):
    value = object.__new__(VideoDispatcher)
    value.artifacts = SimpleNamespace(read_json=AsyncMock(side_effect=reads))
    value.store = SimpleNamespace(
        get_task=AsyncMock(
            return_value=SimpleNamespace(task=current or expected, revision=1)
        )
    )
    value.telemetry = SimpleNamespace(
        increment=Mock(), record_stage_duration=Mock()
    )
    return value


@pytest.mark.asyncio
async def test_dispatch_artifact_visibility_recovers(monkeypatch):
    monkeypatch.setattr(
        dispatcher_module, "_ARTIFACT_VISIBILITY_RETRY_DELAYS_S", (0, 0)
    )
    task = _task()
    value = _dispatcher(
        task,
        [FileNotFoundError(), FileNotFoundError(), {"prompt": "ready"}],
    )

    result = await value._read_dispatch_artifact_json(
        task, "/artifacts/request.json", "request"
    )

    assert result == {"prompt": "ready"}
    assert value.artifacts.read_json.await_count == 3
    assert value.telemetry.increment.call_count == 3
    value.telemetry.record_stage_duration.assert_called_once()


@pytest.mark.asyncio
async def test_dispatch_artifact_visibility_exhaustion_is_not_worker_failure(
    monkeypatch,
):
    monkeypatch.setattr(
        dispatcher_module, "_ARTIFACT_VISIBILITY_RETRY_DELAYS_S", (0,)
    )
    task = _task()
    value = _dispatcher(task, FileNotFoundError())

    with pytest.raises(dispatcher_module._ArtifactUnavailable):
        await value._read_dispatch_artifact_json(
            task, "/artifacts/request.json", "request"
        )

    assert value.artifacts.read_json.await_count == 2
    value.telemetry.record_stage_duration.assert_not_called()


@pytest.mark.asyncio
async def test_dispatch_artifact_visibility_stops_for_cancellation(monkeypatch):
    monkeypatch.setattr(
        dispatcher_module, "_ARTIFACT_VISIBILITY_RETRY_DELAYS_S", (0,)
    )
    task = _task()
    current = _task(cancelled=True)
    value = _dispatcher(task, FileNotFoundError(), current=current)

    with pytest.raises(dispatcher_module._DispatchCancelledDuringArtifactWait):
        await value._read_dispatch_artifact_json(
            task, "/artifacts/request.json", "request"
        )

    assert value.artifacts.read_json.await_count == 1
