import asyncio
import dataclasses
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dingo.video_gateway.dispatcher import VideoDispatcher
from dingo.video_gateway.models import now_ms
from dingo.video_gateway.task_store import StoredTask, worker_key
from tests.video_gateway.test_dispatch_idle import stack
from tests.video_gateway.test_task_store import _task


def probe_stack(response):
    calls = []

    async def direct(payload, instance, context):
        calls.append(payload)

        async def stream():
            yield response

        return stream()

    dispatcher = VideoDispatcher.__new__(VideoDispatcher)
    dispatcher.context_factory = lambda *a: MagicMock()
    dispatcher.telemetry = MagicMock()
    pool = SimpleNamespace(
        config=SimpleNamespace(
            pool_id="p", scheduling=SimpleNamespace(worker_capacity=2)
        ),
        client=SimpleNamespace(direct=direct, instance_ids=lambda: [7]),
        instance_ids=[],
        capacity_cache={},
        prefetch_instances=set(),
    )
    return dispatcher, pool, calls


@pytest.mark.asyncio
async def test_capacity_cached_per_physical_registration_and_clamped():
    d, p, calls = probe_stack(
        dict(
            schema_version=1,
            capabilities=["execution_capacity_v1"],
            execution_capacity=8,
            accepting=True,
        )
    )
    await d._refresh_instances(p)
    await d._refresh_instances(p)
    assert len(calls) == 1
    assert d._worker_slots(p) == [(7, 0), (7, 1)]
    p.client.instance_ids = lambda: []
    await d._refresh_instances(p)
    assert not p.capacity_cache


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        {},
        {
            "schema_version": 1,
            "capabilities": ["execution_capacity_v1"],
            "execution_capacity": True,
            "accepting": True,
        },
    ],
)
async def test_unsupported_or_malformed_capacity_never_infers_multiple_slots(response):
    d, p, calls = probe_stack(response)
    await d._refresh_instances(p)
    await d._refresh_instances(p)
    assert len(calls) == 1 and d._worker_slots(p) == [(7, 0)]
    assert p.capacity_cache[7][1] > time.monotonic()


@pytest.mark.asyncio
async def test_single_slot_does_not_query_capabilities():
    d, p, calls = probe_stack({})
    p.config.scheduling.worker_capacity = 1
    await d._refresh_instances(p)
    assert not calls and d._worker_slots(p) == [(7, 0)]


@pytest.mark.asyncio
async def test_prefetch_capacity_is_distinct_from_execution_capacity():
    d, p, calls = probe_stack(
        dict(
            schema_version=1,
            capabilities=["execution_capacity_v1", "execution_prefetch_v1"],
            execution_capacity=1,
            prefetch_capacity=1,
            admission_capacity=2,
            accepting=True,
        )
    )
    p.config.scheduling.worker_capacity = 1
    p.config.scheduling.worker_prefetch_capacity = 1
    p.prefetch_instances = set()
    await d._refresh_instances(p)
    assert d._worker_slots(p) == [(7, 0), (7, 1)] and p.prefetch_instances == {7}


@pytest.mark.asyncio
async def test_prefetch_is_disabled_for_legacy_workers_and_mismatched_engine_n():
    for caps, n in [
        (["execution_capacity_v1"], 1),
        (["execution_capacity_v1", "execution_prefetch_v1"], 2),
    ]:
        d, p, calls = probe_stack(
            dict(
                schema_version=1,
                capabilities=caps,
                execution_capacity=n,
                prefetch_capacity=1,
                admission_capacity=n + 1,
                accepting=True,
            )
        )
        p.config.scheduling.worker_capacity = 1
        p.config.scheduling.worker_prefetch_capacity = 1
        p.prefetch_instances = set()
        await d._refresh_instances(p)
        assert d._worker_slots(p) == [(7, 0)] and not p.prefetch_instances


async def test_failed_reprobe_clears_prefetch_protocol_capability():
    d, p, _ = probe_stack({})
    p.prefetch_instances.add(7)
    p.config.scheduling.worker_prefetch_capacity = 1
    assert (await d._worker_capacity(p, 7))[0] == 1
    assert not p.prefetch_instances


@pytest.mark.asyncio
async def test_dispatch_uses_free_second_slot(make_gateway_config):
    d, p, store, client = stack(make_gateway_config)
    p.config = dataclasses.replace(
        p.config, scheduling=dataclasses.replace(p.config.scheduling, worker_capacity=2)
    )
    p.capacity_cache[7] = (2, float("inf"))
    p.lease_cache["first"] = SimpleNamespace(
        worker_key=worker_key(p.config.backend_target, 7)
    )
    task = _task("queued", pool_id=p.config.pool_id)
    task.backend_target = p.config.backend_target
    store.list_queued.return_value = [StoredTask(task, 1)]
    assert await d._dispatch_once(p)
    reserved_lease = store.reserve.call_args.args[1]
    assert reserved_lease.worker_instance_id == 7
    assert reserved_lease.worker_key == worker_key(task.backend_target, 7, 1)
    assert (await d.memory_budget.snapshot()).used_bytes == 0


@pytest.mark.asyncio
async def test_busy_retries_submission_without_changing_attempt():
    d, p, calls = probe_stack(None)
    responses = iter(
        [{"state": "busy", "accepted": False}, {"state": "accepted", "accepted": True}]
    )

    async def direct(payload, instance, context):
        calls.append(payload)

        async def stream():
            yield next(responses)

        return stream()

    p.client.direct = direct
    task = SimpleNamespace(
        worker_instance_id=7, deadline_at_ms=now_ms() + 3000, attempt=1
    )
    ack = await d._detached_submit_ack(
        p, task, {"submit": "same-token"}, MagicMock(), lambda v: v
    )
    assert ack["accepted"] and task.attempt == 1
    assert calls == [{"submit": "same-token"}] * 2


@pytest.mark.asyncio
async def test_busy_stops_at_deadline():
    d, p, calls = probe_stack({"state": "busy", "accepted": False})
    task = SimpleNamespace(worker_instance_id=7, deadline_at_ms=now_ms() - 1)
    with pytest.raises(asyncio.TimeoutError):
        await d._detached_submit_ack(p, task, {}, MagicMock(), lambda v: v)
    assert not calls
