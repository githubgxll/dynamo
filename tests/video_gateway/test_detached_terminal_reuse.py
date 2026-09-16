"""Execution-local notification must never get ahead of the status write."""

import asyncio
import json
import threading

import pytest

from dingo.common.video_result_file import BINARY_RESULT_WRITER
from dingo.common.video_task_protocol import DetachedTaskIdentity
from dingo.vllm.omni.detached_tasks import DetachedOmniTaskManager
from tests.video_gateway.test_dispatcher import _MINIMAL_MP4

REAL_THREAD = asyncio.to_thread


@pytest.mark.parametrize("write_fails", [False, True])
async def test_wait_reuses_only_successfully_written_terminal(
    tmp_path, monkeypatch, write_fails
):
    monkeypatch.setattr(asyncio, "to_thread", REAL_THREAD)
    started, release = asyncio.Event(), asyncio.Event()
    writing, finish_write = threading.Event(), threading.Event()

    class Handler:
        async def generate(self, request, context):
            started.set()
            await release.wait()
            descriptor = await BINARY_RESULT_WRITER.get().write(_MINIMAL_MP4)
            yield {
                "status": "completed",
                "data": [{"output_format": "mp4", "artifact": descriptor}],
            }

    manager = DetachedOmniTaskManager(
        Handler(), tmp_path, binary_results=True, inline_results=True
    )
    identity = DetachedTaskIdentity("deployment", "pool", "task", 1, "a" * 32)
    task_root = manager._attempt_root(identity).parent.parent
    task_root.mkdir(parents=True)
    (task_root / "_artifact.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                deployment_id="deployment",
                pool_id="pool",
                task_id="task",
            )
        )
    )
    original_write, original_read = manager._atomic_json, manager._read_status
    reads = []

    def write(path, value):
        if value["state"] == "completed":
            writing.set()
            assert finish_write.wait(5), "test did not release status writer"
            if write_fails:
                raise OSError("injected terminal write failure")
        original_write(path, value)

    def read(path):
        reads.append(path)
        return original_read(path)

    monkeypatch.setattr(manager, "_atomic_json", write)
    try:
        assert (await manager._submit(identity, {}))["accepted"]
        await asyncio.wait_for(started.wait(), 2)
        running = manager._running[identity.key]
        monkeypatch.setattr(manager, "_read_status", read)
        first, second = (
            manager._wait_terminal(identity),
            manager._wait_terminal(identity),
        )
        assert (await anext(first))["state"] == "watching"
        assert (await anext(second))["state"] == "watching"
        result = asyncio.create_task(anext(first))
        release.set()
        assert await REAL_THREAD(writing.wait, 3)
        assert running.persisted_terminal is None
        assert not result.done()
        finish_write.set()
        terminal = await asyncio.wait_for(result, 3)
        expected = "failed" if write_fails else "completed"
        assert terminal["state"] == expected
        assert not reads, "local wait must not reread status or recheck paths"
        terminal["state"] = "tampered"
        assert (await anext(second))["state"] == expected
        await first.aclose()
        await second.aclose()
        await manager.shutdown()
        await asyncio.sleep(0)
        assert not manager._running, "no global cache of completed tasks"
        assert (await anext(manager._wait_terminal(identity)))["state"] == expected
        assert len(reads) == 1, "late reconnect must use persisted status"
        replacement = DetachedOmniTaskManager(object(), tmp_path)
        assert (await anext(replacement._wait_terminal(identity)))["state"] == expected
    finally:
        finish_write.set()
        release.set()
        await manager.shutdown()


async def test_wait_other_execution_token_cannot_use_local_execution(tmp_path):
    manager = DetachedOmniTaskManager(object(), tmp_path)
    identity = DetachedTaskIdentity("deployment", "pool", "task", 1, "b" * 32)
    assert (await anext(manager._wait_terminal(identity)))["state"] == "not_found"
