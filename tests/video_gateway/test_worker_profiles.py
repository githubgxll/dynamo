# SPDX-License-Identifier: Apache-2.0
"""A non-compiler workload exercises the same durable task controller."""
import asyncio
import time
from dataclasses import asdict

import pytest

from dingo.common.task_protocol import (
    ControlRequest,
    InputDescriptor,
    TaskIdentity,
    TaskResponse,
)
from dingo.worker.contracts import WorkerProfile
from dingo.worker.runtime_io import atomic_json, read_json
from dingo.worker.task_manager import DetachedTaskManager

PROFILE = WorkerProfile(
    adapter="fixture_transform",
    task_kind="video",
    model_adapters=(("fixture_model", 1),),
    protocol_adapters=(("fixture_contract", 2),),
)


class Handler:
    def __init__(self):
        self.release = asyncio.Event()
        self.calls = 0

    async def execute(self, request, root, cancelled):
        self.calls += 1
        await self.release.wait()
        return {"result": {"transformed": True}}


def manager(tmp_path, handler, **options):
    return DetachedTaskManager(
        handler,
        tmp_path,
        profile=PROFILE,
        deployment_id="d",
        pool_id="p",
        instance_id="1",
        capacity_domain="model-one",
        storage_id="shared",
        **options,
    )


def request(m, op="submit", kind="video"):
    identity = TaskIdentity("d", kind, "p", "task-0", 1, "a" * 32, "1", m.generation)
    payload = InputDescriptor(
        "shared",
        "request.json",
        "manifest.json",
        "a" * 64,
        "route.json",
        "b" * 64,
        "c" * 64,
    )
    return ControlRequest(
        op,
        "d",
        "p",
        "1",
        m.generation,
        identity,
        payload if op == "submit" else None,
        int(time.time() * 1000) + 60000 if op == "submit" else None,
        10 if op == "wait" else None,
    ).to_dict()


async def call(m, value):
    replies = [r async for r in m.generate(value)]
    assert len(replies) == 1
    return replies[0]


async def test_other_profile_uses_same_controller_without_compiler_defaults(tmp_path):
    h = Handler()
    m = manager(tmp_path, h)
    await m.start()
    caps = m.capabilities()
    assert caps["task_kinds"] == ["video"]
    assert caps["execution_capacity"] == 1
    assert caps["protocol_adapters"] == {"fixture_contract": 2}
    assert not (tmp_path / "d" / "worker-locks").exists()
    submit = request(m)
    await call(m, submit)
    await call(m, submit)
    await asyncio.sleep(0)
    assert h.calls == 1
    h.release.set()
    assert await m.shutdown(2)
    assert TaskResponse.parse(await call(m, request(m, "status"))).state == "succeeded"


async def test_other_kind_cannot_be_sent_to_bound_profile(tmp_path):
    m = manager(tmp_path, Handler())
    await m.start()
    with pytest.raises(ValueError, match="task kind"):
        await call(m, request(m, kind="context_ir"))
    assert not m.jobs
    await m.shutdown()


async def test_unknown_native_work_waits_for_injected_idle_evidence(tmp_path):
    h = Handler()
    m = manager(tmp_path, h)
    old = TaskIdentity(
        "d", "video", "p", "task-old", 1, "b" * 32, "old", "old-generation"
    )
    root = m.attempt_root(old)
    atomic_json(
        root / "accepted.json", {"identity": asdict(old), "capacity_domain": m.domain}
    )
    atomic_json(root / "status.json", m.response(old, "running"))
    ready = asyncio.Event()

    async def idle():
        return ready.is_set()

    m.engine_idle = idle
    await m.start()
    assert not m.accepting and read_json(root / "status.json")["state"] == "running"
    ready.set()
    await asyncio.wait_for(m.recovery_task, 3)
    assert m.accepting and h.calls == 0
    status = TaskResponse.parse(read_json(root / "status.json"))
    assert status.state == "failed" and status.error_code == "worker_execution_unknown"
    await m.shutdown()


async def test_lost_status_after_acceptance_does_not_repeat_execution(tmp_path):
    h = Handler()
    m = manager(tmp_path, h)
    await m.start()
    wire = request(m)
    identity = TaskIdentity(
        **{k: wire["_dingo_task"][k] for k in TaskIdentity.__dataclass_fields__}
    )
    root = m.attempt_root(identity)
    atomic_json(
        root / "accepted.json",
        {
            "identity": asdict(identity),
            "payload": asdict(InputDescriptor.parse(wire["_dingo_task"]["payload"])),
        },
    )
    reply = TaskResponse.parse(await call(m, wire))
    assert reply.state == "accepted" and h.calls == 0
    assert TaskResponse.parse(await call(m, request(m, "status"))).state == "accepted"
    assert TaskResponse.parse(await call(m, request(m, "wait"))).state == "accepted"
    await m.shutdown()


@pytest.mark.parametrize("version", [True, 0, -1, "1"])
def test_profile_rejects_invalid_adapter_version(version):
    with pytest.raises(ValueError):
        WorkerProfile("fixture", "video", (("model", version),), (("contract", 1),))
