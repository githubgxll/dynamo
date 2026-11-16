# SPDX-License-Identifier: Apache-2.0
import math
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dingo.video_gateway.context_ir.telemetry import record_transition
from dingo.video_gateway.models import TaskStatus
from dingo.video_gateway.task_runtime.transactions import UnconfirmedTransaction
from dingo.video_gateway.task_runtime.video_dispatch import (
    VideoSchedulingAdapter,
    VideoSelection,
)
from dingo.video_gateway.telemetry import GatewayTelemetry
from tests.video_gateway.test_pipeline import complete_child, setup_pipeline
from tests.video_gateway.test_seedvr2_adapter import video


def samples(telemetry, name):
    return [
        line for line in telemetry.render_prometheus() if line.startswith(name + "{")
    ]


@pytest.mark.parametrize("sr", [False, True])
async def test_public_pipeline_timings_once_and_api_agree(make_gateway_config, sr):
    service, coordinator, step, parent = await setup_pipeline(make_gateway_config)
    if not sr:
        request = dict(parent.task.normalized_request)
        request.pop("super_resolution")
        parent = await service.store.transition(
            parent.task.id,
            expected={parent.task.status},
            expected_revision=parent.revision,
            patch={"normalized_request": request},
        )
    phases = ("base", "sr") if sr else ("base",)
    for index, phase in enumerate(phases):
        child_id = parent.task.normalized_request["_pipeline"]["children"][phase]
        child = await service.store.get_task(child_id)
        root = await service.artifacts.resolve_task_root(
            child.task.deployment_id, child.task.pool_id, child_id
        )
        await complete_child(service, child_id, video(root).path, seconds=2 + index)
        child = await service.store.get_task(child_id)
        changed = await service.store.transition(
            child_id,
            expected={TaskStatus.COMPLETED},
            expected_revision=child.revision,
            patch={
                "finalize_time_s": 0.2,
                "stage_durations": {"worker_queue_wait": 0.3},
            },
        )
        # An internal child terminal cannot increment the public terminal counter.
        service.telemetry.record_transition(
            "completed",
            replace(changed.task, status=TaskStatus.FINALIZING),
            changed.task,
            gateway_generation="test",
            revision=changed.revision,
        )
        parent = await step()
        if phase == "base" and sr:
            parent = await step()
    assert parent.task.status == TaskStatus.FINALIZING
    parent = await step()
    assert parent.task.status == TaskStatus.COMPLETED
    saved = service.telemetry.render_prometheus()
    assert (
        sum(
            'stage="execution"' in line
            for line in samples(
                service.telemetry, "dingo_video_task_stage_duration_seconds_count"
            )
        )
        == 1
    )
    for _ in range(3):
        # A fresh coordinator can see the completed parent without replaying metrics.
        await coordinator.tick()
    assert saved == service.telemetry.render_prometheus()
    values = parent.task.public_dict()["metrics"]
    assert values["inference_time_s"] == (5 if sr else 2)
    assert values["worker_queue_wait_s"] == pytest.approx(0.3 * len(phases))
    assert values["finalize_time_s"] >= 0.2 * len(phases)
    assert values["pipeline"]["base_queue_wait_s"] >= 0.1
    assert values["pipeline"]["sr_queue_wait_s"] >= .1 if sr else values["pipeline"]["sr_queue_wait_s"] is None
    counts = samples(service.telemetry, "dingo_video_task_terminal_total")
    assert len(counts) == 1 and counts[0].endswith(" 1")
    for line in samples(
        service.telemetry, "dingo_video_pipeline_stage_duration_seconds_count"
    ):
        assert line.endswith(" 1")


@pytest.mark.parametrize("state", ["completed", "failed", "cancelled", "expired"])
def test_compile_transition_metrics_bounded_and_no_missing_zero(state):
    telemetry = GatewayTelemetry()
    before = SimpleNamespace(status="in_progress")
    after = SimpleNamespace(
        status=state,
        backend_id="backend",
        model="model",
        metrics={
            "queue_wait_s": 0.1,
            "compile_time_s": 1,
            "execution_time_s": 2,
            "unexpected": 42,
            "finalize_time_s": math.nan,
        },
    )
    record_transition(telemetry, before, after)
    record_transition(telemetry, after, after)
    assert samples(telemetry, "dingo_context_ir_terminal_total")[0].endswith(" 1")
    rows = samples(telemetry, "dingo_context_ir_stage_duration_seconds_count")
    assert len(rows) == (0 if state == "expired" else 3)
    assert all(row.endswith(" 1") for row in rows)


async def test_adoption_unknown_is_not_failure_and_definite_error_is(
    make_gateway_config,
):
    cfg = make_gateway_config()
    telemetry = GatewayTelemetry()
    task = SimpleNamespace(id="id", normalized_request={}, estimated_payload_bytes=1)
    reserved = SimpleNamespace(task=task)

    async def boom():
        raise RuntimeError("definite failure")

    async def unknown(*args, **kwargs):
        raise UnconfirmedTransaction(boom)

    dispatcher = SimpleNamespace(
        config=cfg,
        generation="gen",
        telemetry=telemetry,
        _draining=False,
        memory_budget=SimpleNamespace(try_acquire=AsyncMock(return_value=True)),
        store=SimpleNamespace(claim_orphaned_active=unknown),
    )
    pool = SimpleNamespace(config=cfg.pools[0], budget_waiter_id=None)
    adapter = VideoSchedulingAdapter(dispatcher, pool)
    candidate = VideoSelection(reserved, [], False, 0, adopting=True)
    with pytest.raises(UnconfirmedTransaction) as caught:
        await adapter.claim(candidate)
    assert not samples(telemetry, "dingo_video_ha_takeovers_total")
    with pytest.raises(RuntimeError):
        await caught.value.resolve()
    assert 'outcome="failed"' in samples(telemetry, "dingo_video_ha_takeovers_total")[0]


async def test_active_request_view_counts_parent_once_in_current_queue(
    make_gateway_config,
):
    from dingo.video_gateway.metric_views import request_stages

    service, _coordinator, _step, parent = await setup_pipeline(make_gateway_config)

    async def view():
        return await request_stages(
            service,
            {
                p.pool_id: await service.store.task_counts(p.pool_id)
                for p in service.config.pools
            },
        )

    values = await view()
    assert values["pipeline"] == {"gateway_queue": 1, "worker": 0}
    assert values["base"] == {"gateway_queue": 0, "worker": 0}
    child_id = parent.task.normalized_request["_pipeline"]["children"]["base"]
    child = await service.store.get_task(child_id)
    child = await service.store.transition(
        child_id, expected={child.task.status}, expected_revision=child.revision,
        patch={"status": TaskStatus.DISPATCHING})
    child = await service.store.transition(
        child_id,
        expected={child.task.status},
        expected_revision=child.revision,
        patch={"status": TaskStatus.IN_PROGRESS},
    )
    values = await view()
    assert values["pipeline"] == {"gateway_queue": 0, "worker": 1}
    assert values["base"] == {"gateway_queue": 0, "worker": 0}


async def test_missing_pipeline_timing_never_creates_zero_sample(make_gateway_config):
    from dingo.video_gateway.pipeline_timings import aggregate

    service, _coordinator, _step, parent = await setup_pipeline(make_gateway_config)
    child = await service.store.get_task(
        parent.task.normalized_request["_pipeline"]["children"]["base"]
    )
    metrics, stages = aggregate(parent.task, {"base": child.task})
    assert metrics["base_generation_s"] is None
    assert "execution" not in stages and "worker_queue" not in stages

async def test_finished_child_between_gauge_snapshots_cannot_double_count(make_gateway_config, monkeypatch):
    service, _coordinator, _step, parent = await setup_pipeline(make_gateway_config)
    child_id = parent.task.normalized_request['_pipeline']['children']['base']
    counts = {p.pool_id: await service.store.task_counts(p.pool_id) for p in service.config.pools}
    original = service.store.get_task
    async def newer(task_id):
        value = await original(task_id)
        if task_id == child_id:
            return replace(value, task=replace(value.task, status=TaskStatus.COMPLETED))
        return value
    monkeypatch.setattr(service.store, 'get_task', newer)
    from dingo.video_gateway.metric_views import request_stages
    values = await request_stages(service, counts)
    assert sum(sum(v.values()) for v in values.values()) == 1
    assert values['pipeline']['gateway_queue'] == 1

def test_downstream_unavailability_before_child_creation_is_queue_time():
    from dingo.video_gateway.pipeline_timings import aggregate
    parent=SimpleNamespace(started_at_ms=1000,queue_wait_s=.5)
    base=SimpleNamespace(created_at_ms=4000,completed_at_ms=8000,queue_wait_s=1,
        stage_durations={'worker_queue_wait':.1},inference_time_s=2,finalize_time_s=.2)
    sr=SimpleNamespace(created_at_ms=13000,queue_wait_s=2,
        stage_durations={'worker_queue_wait':.2},inference_time_s=3,finalize_time_s=.3)
    metrics, stages=aggregate(parent, {'base':base,'sr':sr})
    assert metrics['base_handoff_wait_s']==3
    assert metrics['sr_handoff_wait_s']==5
    assert metrics['base_queue_wait_s']==4
    assert metrics['sr_queue_wait_s']==7
    assert stages['queue']==11.5
    assert stages['execution']==5

@pytest.mark.parametrize('internal', [False, True])
async def test_real_early_release_finalizer_does_not_emit_internal_samples(make_gateway_config, internal):
    from dingo.common.video_result_file import BINARY_RESULT_WRITER
    from dingo.vllm.omni.detached_tasks import DetachedOmniTaskManager
    from tests.video_gateway.test_dispatcher import (
        _MINIMAL_MP4,
        _DetachedClient,
        _pool,
        _stack,
        _submit,
    )
    class Handler:
        async def generate(self, request, context):
            descriptor = await BINARY_RESULT_WRITER.get().write(_MINIMAL_MP4)
            yield {'status':'completed','data':[{'output_format':'mp4','artifact':descriptor}]}
    pool=_pool('fl-pool','public-fl','dyn://test.backend.generate')
    pool['execution_mode']='detached'
    pool['scheduling']['early_release_slot']=True
    pool['scheduling']['accept_without_workers']=True
    cfg=make_gateway_config(pools=[pool])
    manager=DetachedOmniTaskManager(Handler(),cfg.artifact_store.root,binary_results=True,inline_results=True)
    store,_artifacts,dispatcher,service=_stack(cfg,{'fl-pool':_DetachedClient(manager)})
    submitted=await _submit(service,'public-fl')
    if internal:
        before=submitted.stored
        await store.transition(before.task.id,expected={TaskStatus.QUEUED},expected_revision=before.revision,
            patch={'normalized_request':{**before.task.normalized_request,'_pipeline_child':{'parent_id':'parent','phase':'base'}}})
    await dispatcher.start()
    try:
        result=await dispatcher.wait_terminal(submitted.stored.task.id,5)
        assert result.task.status==TaskStatus.COMPLETED
        rows=samples(service.telemetry,'dingo_video_task_stage_duration_seconds_count')
        assert any('stage="finalize"' in line for line in rows) is (not internal)
        if internal:assert not rows
    finally:
        await dispatcher.stop()
        await manager.shutdown()
