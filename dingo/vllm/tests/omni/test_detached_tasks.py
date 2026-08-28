# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from dingo.vllm.omni.detached_tasks import (
    DetachedOmniTaskManager,
    detached_attempt_root,
    detached_envelope,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.unit]


class _Handler:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def generate(self, request: dict[str, Any], context: Any):
        self.calls += 1
        self.started.set()
        yield {"phase": "start", "request": request, "id": context.id()}
        await self.release.wait()
        if not context.is_stopped():
            yield {"phase": "complete"}


class _LeakyAbortWaitHandler(_Handler):
    def __init__(self) -> None:
        super().__init__()
        self.waiter: asyncio.Task[bool] | None = None

    async def generate(self, request: dict[str, Any], context: Any):
        self.calls += 1
        self.waiter = context.async_killed_or_stopped()
        yield {"phase": "complete", "request": request, "id": context.id()}


class _StubbornCancelHandler(_Handler):
    async def generate(self, request: dict[str, Any], context: Any):
        self.calls += 1
        self.started.set()
        yield {"phase": "start", "request": request, "id": context.id()}
        await asyncio.Event().wait()


class _Context:
    def id(self) -> str:
        return "outer-request"

    def is_stopped(self) -> bool:
        return False


def _request(*, op: str = "submit", payload: dict[str, Any] | None = None):
    return detached_envelope(
        op=op,
        deployment_id="deployment-a",
        pool_id="fl-pool",
        task_id="01TASK",
        attempt=1,
        execution_token="a" * 32,
        payload=payload,
    )


async def _one(manager: DetachedOmniTaskManager, request: dict[str, Any]):
    return [item async for item in manager.generate(request, _Context())]


async def _terminal_status(manager: DetachedOmniTaskManager) -> dict[str, Any]:
    for _ in range(200):
        status = (await _one(manager, _request(op="status")))[0]
        if status["state"] in {"completed", "failed", "cancelled"}:
            return status
        await asyncio.sleep(0.005)
    raise AssertionError("detached task did not become terminal")


def _write_manifest(root: Path) -> None:
    task_root = root / "deployment-a" / "v1" / "pools" / "fl-pool" / "tasks" / "01TASK"
    task_root.mkdir(parents=True)
    (task_root / "_artifact.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "deployment_id": "deployment-a",
                "pool_id": "fl-pool",
                "task_id": "01TASK",
            }
        )
    )


async def test_submit_ack_is_independent_and_duplicate_is_idempotent(tmp_path: Path):
    handler = _Handler()
    manager = DetachedOmniTaskManager(handler, tmp_path, drain_timeout_s=1)
    _write_manifest(tmp_path)

    first = await _one(manager, _request(payload={"prompt": "hello"}))
    assert first[0]["accepted"] is True
    await asyncio.wait_for(handler.started.wait(), timeout=1)

    duplicate = await _one(manager, _request(payload={"prompt": "hello"}))
    assert duplicate[0]["accepted"] is False
    assert handler.calls == 1

    handler.release.set()
    status = await _terminal_status(manager)
    assert status["state"] == "completed"
    response_path = Path(status["response_path"])
    lines = [json.loads(line) for line in response_path.read_text().splitlines()]
    assert [line["phase"] for line in lines] == ["start", "complete"]
    assert status["response_bytes"] == response_path.stat().st_size
    await manager.shutdown()


async def test_cancel_marker_stops_background_request(tmp_path: Path):
    handler = _Handler()
    manager = DetachedOmniTaskManager(
        handler, tmp_path, drain_timeout_s=1, cancel_poll_interval_s=0.005
    )
    _write_manifest(tmp_path)
    await _one(manager, _request(payload={"prompt": "hello"}))
    await asyncio.wait_for(handler.started.wait(), timeout=1)

    cancelled = await _one(manager, _request(op="cancel"))
    assert cancelled[0]["state"] == "cancel_requested"
    handler.release.set()
    status = await _terminal_status(manager)
    assert status["state"] == "cancelled"
    await manager.shutdown()


async def test_cancel_forcibly_closes_a_stuck_omni_stream(tmp_path: Path):
    handler = _StubbornCancelHandler()
    manager = DetachedOmniTaskManager(
        handler,
        tmp_path,
        drain_timeout_s=1,
        cancel_poll_interval_s=0.005,
        cancel_grace_s=0.01,
    )
    _write_manifest(tmp_path)
    await _one(manager, _request(payload={"prompt": "hello"}))
    await asyncio.wait_for(handler.started.wait(), timeout=1)

    await _one(manager, _request(op="cancel"))
    status = await _terminal_status(manager)

    assert status["state"] == "cancelled"
    attempt_root = detached_attempt_root(
        tmp_path, "deployment-a", "fl-pool", "01TASK", 1, "a" * 32
    )
    assert list(attempt_root.glob("*.part-*")) == []
    await manager.shutdown()


async def test_shutdown_drains_an_accepted_task_before_returning(tmp_path: Path):
    handler = _Handler()
    manager = DetachedOmniTaskManager(handler, tmp_path, drain_timeout_s=1)
    _write_manifest(tmp_path)
    await _one(manager, _request(payload={"prompt": "hello"}))
    await asyncio.wait_for(handler.started.wait(), timeout=1)

    draining = asyncio.create_task(manager.shutdown())
    await asyncio.sleep(0.02)
    assert not draining.done()
    with pytest.raises(RuntimeError, match="draining"):
        await _one(
            manager,
            detached_envelope(
                op="submit",
                deployment_id="deployment-a",
                pool_id="fl-pool",
                task_id="01TASK",
                attempt=2,
                execution_token="b" * 32,
                payload={"prompt": "new"},
            ),
        )
    handler.release.set()
    await asyncio.wait_for(draining, timeout=1)
    status = await _terminal_status(manager)
    assert status["state"] == "completed"


async def test_normal_request_keeps_existing_handler_path(tmp_path: Path):
    handler = _Handler()
    handler.release.set()
    manager = DetachedOmniTaskManager(handler, tmp_path, drain_timeout_s=1)
    response = await _one(manager, {"prompt": "normal"})
    assert [item["phase"] for item in response] == ["start", "complete"]
    assert handler.calls == 1


async def test_completed_detached_task_resolves_context_abort_waiter(tmp_path: Path):
    handler = _LeakyAbortWaitHandler()
    manager = DetachedOmniTaskManager(handler, tmp_path, drain_timeout_s=1)
    _write_manifest(tmp_path)

    await _one(manager, _request(payload={"prompt": "hello"}))
    status = await _terminal_status(manager)

    assert status["state"] == "completed"
    assert handler.waiter is not None
    assert await asyncio.wait_for(handler.waiter, timeout=1) is True
    await manager.shutdown()


async def test_path_validation_rejects_escape_components(tmp_path: Path):
    with pytest.raises(ValueError, match="deployment_id"):
        detached_attempt_root(tmp_path, "../escape", "pool", "task", 1, "a" * 32)
    with pytest.raises(ValueError, match="execution_token"):
        detached_attempt_root(tmp_path, "deployment", "pool", "task", 1, "token")


async def test_existing_symlink_component_is_rejected(tmp_path: Path):
    outside = tmp_path.parent / "outside-detached"
    outside.mkdir(exist_ok=True)
    (tmp_path / "deployment-a").symlink_to(outside, target_is_directory=True)
    handler = _Handler()
    manager = DetachedOmniTaskManager(handler, tmp_path, drain_timeout_s=1)
    with pytest.raises(RuntimeError, match="symlink"):
        await _one(manager, _request(payload={"prompt": "hello"}))
