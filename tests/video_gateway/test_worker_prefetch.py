import asyncio
import json
import time

import pytest

from dingo.common.video_task_protocol import DetachedTaskIdentity
from dingo.vllm.omni.detached_tasks import DetachedOmniTaskManager


def identity(manager, name):
    value = DetachedTaskIdentity("deployment", "pool", name, 1, "a" * 32)
    root = manager._attempt_root(value).parent.parent
    root.mkdir(parents=True, exist_ok=True)
    (root / "_artifact.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                deployment_id="deployment",
                pool_id="pool",
                task_id=name,
            )
        )
    )
    return value


class Handler:
    def __init__(self):
        self.running = 0
        self.peak = 0
        self.started = []
        self.release = asyncio.Event()

    async def generate(self, payload, context):
        self.running += 1
        self.peak = max(self.peak, self.running)
        self.started.append(payload["name"])
        try:
            await self.release.wait()
            yield {"phase": "done"}
        finally:
            self.running -= 1


async def until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(wait(), 2)


@pytest.mark.parametrize("capacity", [1, 2])
async def test_prefetch_admits_one_extra_but_never_executes_above_n(tmp_path, capacity):
    handler = Handler()
    manager = DetachedOmniTaskManager(
        handler,
        tmp_path,
        execution_capacity=capacity,
        prefetch_capacity=1,
        cancel_poll_interval_s=0.01,
    )
    values = [identity(manager, str(i)) for i in range(capacity + 2)]
    try:
        for i in range(capacity):
            assert (await manager._submit(values[i], {"name": str(i)}))["accepted"]
        await until(lambda: len(handler.started) == capacity)
        assert (await manager._submit(values[capacity], {"name": str(capacity)}))[
            "accepted"
        ]
        duplicate = await manager._submit(values[capacity], {"name": str(capacity)})
        assert not duplicate["accepted"] and duplicate["state"] == "accepted"
        assert (await manager._submit(values[-1], {"name": "rejected"}))[
            "state"
        ] == "busy"
        await asyncio.sleep(0.02)
        assert handler.running == capacity and len(handler.started) == capacity
        handler.release.set()
        await until(lambda: not manager._running)
        assert handler.peak == capacity
        assert handler.started == [str(i) for i in range(capacity + 1)]
        assert (await manager._status(values[capacity]))["worker_queue_wait_s"] > 0
    finally:
        handler.release.set()
        await manager.shutdown()


@pytest.mark.parametrize("cancel_kind", ["rpc", "file"])
async def test_queued_cancel_never_enters_model_and_frees_admission(
    tmp_path, cancel_kind
):
    handler = Handler()
    manager = DetachedOmniTaskManager(
        handler, tmp_path, prefetch_capacity=1, cancel_poll_interval_s=0.01
    )
    a, b, c = [identity(manager, name) for name in ["a", "b", "c"]]
    try:
        await manager._submit(a, {"name": "a"})
        await until(lambda: handler.running == 1)
        await manager._submit(b, {"name": "b"})
        if cancel_kind == "rpc":
            await manager._cancel(b)
        else:
            manager._cancel_path(b).touch()
        await until(lambda: b.key not in manager._running)
        assert (await manager._status(b))["state"] == "cancelled"
        assert handler.started == ["a"]
        assert (await manager._submit(c, {"name": "c"}))["accepted"]
        handler.release.set()
        await until(lambda: not manager._running)
        assert handler.started == ["a", "c"]
    finally:
        handler.release.set()
        await manager.shutdown()


async def test_prefetch_deadline_expires_without_model_execution(tmp_path):
    handler = Handler()
    manager = DetachedOmniTaskManager(handler, tmp_path, prefetch_capacity=1)
    a, b = [identity(manager, name) for name in ["a", "b"]]
    try:
        await manager._submit(a, {"name": "a"})
        await until(lambda: handler.running == 1)
        await manager._submit(
            b, {"name": "b"}, deadline_at_ms=int(time.time() * 1000) + 20
        )
        await until(lambda: b.key not in manager._running)
        status = await manager._status(b)
        assert (
            status["state"] == "failed"
            and status["error"]["code"] == "worker_queue_timeout"
        )
        assert handler.started == ["a"]
    finally:
        handler.release.set()
        await manager.shutdown()


async def test_direct_admission_cannot_bypass_prefetched_execution_budget(tmp_path):
    handler = Handler()
    manager = DetachedOmniTaskManager(handler, tmp_path, prefetch_capacity=1)
    a, b = [identity(manager, name) for name in ["a", "b"]]
    try:
        await manager._submit(a, {"name": "a"})
        await until(lambda: handler.running == 1)
        await manager._submit(b, {"name": "b"})
        with pytest.raises(RuntimeError, match="capacity"):
            await anext(manager.generate({"name": "direct"}, None))
        assert handler.started == ["a"]
    finally:
        handler.release.set()
        await manager.shutdown()


@pytest.mark.parametrize("capacity", [-1, 2, True, "1"])
def test_prefetch_capacity_rejects_invalid_values(tmp_path, capacity):
    with pytest.raises(ValueError, match="prefetch"):
        DetachedOmniTaskManager(object(), tmp_path, prefetch_capacity=capacity)


async def test_shutdown_cancels_waiting_task_without_entering_model(tmp_path):
    handler = Handler()
    manager = DetachedOmniTaskManager(
        handler, tmp_path, prefetch_capacity=1, drain_timeout_s=0.02
    )
    a, b = [identity(manager, name) for name in ["a", "b"]]
    await manager._submit(a, {"name": "a"})
    await until(lambda: handler.running == 1)
    await manager._submit(b, {"name": "b"})
    await asyncio.wait_for(manager.shutdown(), 2)
    assert handler.started == ["a"]
    assert (await manager._status(b))["state"] == "cancelled"
    assert manager._execution_slots._value == 1
