# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging

from dingo.common.video_timing_schemas import MODEL_EXECUTION_KEY
from dingo.video_gateway.models import TaskStatus
from dingo.video_gateway.task_store import terminal_error
from dingo.video_gateway.telemetry import GatewayTelemetry
from tests.video_gateway.test_task_store import _task


def test_model_timings_observed_only_on_result_handoff():
    telemetry = GatewayTelemetry()
    running = _task("video-model-timing")
    running.status = TaskStatus.IN_PROGRESS
    finalizing = _task(running.id)
    finalizing.status = TaskStatus.FINALIZING
    finalizing.normalized_request[MODEL_EXECUTION_KEY] = {
        "schema": "minimax_h3.v1",
        "unit": "seconds",
        "stages": {
            "encode_prompt": 0.2,
            "diffuse": 3.0,
            "audio_decode": 0.0,
            "unknown": 9.0,
            "video_decode": float("nan"),
        },
    }
    completed = _task(running.id)
    completed.status = TaskStatus.COMPLETED
    completed.normalized_request = dict(finalizing.normalized_request)
    for before, after in (
        (running, finalizing), (finalizing, finalizing), (finalizing, completed)
    ):
        telemetry.record_transition(
            "test", before, after, gateway_generation="gateway-a", revision=42
        )
    text = "\n".join(telemetry.render_prometheus())
    labels = '{pool="pool-a",schema="minimax_h3.v1",stage="diffuse"}'
    assert f'dingo_video_model_stage_duration_seconds_count{labels} 1' in text
    assert f'dingo_video_model_stage_duration_seconds_sum{labels} 3' in text
    assert 'stage="audio_decode"' in text
    assert 'stage="video_decode"' not in text
    assert 'stage="unknown"' not in text
    assert 'task_id=' not in text


def test_missing_or_unknown_model_timings_are_not_zero_samples():
    telemetry = GatewayTelemetry()
    running = _task("video-no-timing")
    running.status = TaskStatus.IN_PROGRESS
    finalizing = _task(running.id)
    finalizing.status = TaskStatus.FINALIZING
    for value in (
        None,
        {"schema": "unknown.v1", "unit": "seconds", "stages": {"diffuse": 1.0}},
    ):
        finalizing.normalized_request[MODEL_EXECUTION_KEY] = value
        telemetry.record_transition(
            "test", running, finalizing, gateway_generation="gateway-a", revision=42
        )
    text = "\n".join(telemetry.render_prometheus())
    assert "dingo_video_model_stage_duration_seconds" not in text


def test_telemetry_renders_bounded_task_and_etcd_metrics(caplog):
    telemetry = GatewayTelemetry()
    task = _task("video-observed")
    failed = _task("video-observed")
    failed.status = TaskStatus.FAILED
    failed.error = terminal_error("worker_lease_lost", "lease lost")
    failed.queue_wait_s = 0.1
    failed.inference_time_s = 1.2
    failed.stage_durations = {"worker_queue_wait": 0.3}

    telemetry.record_submission(task.pool_id, "created", "async")
    telemetry.set_gauge(
        "dingo_video_discovery_consistent", 1, labels={"pool": task.pool_id}
    )
    telemetry.record_stage_duration(task.pool_id, "queue", 0.125)
    telemetry.record_etcd_request("kv/range", 0.01, succeeded=False)
    with caplog.at_level(logging.INFO, logger="dingo.video_gateway.audit"):
        telemetry.record_transition(
            "failed",
            task,
            failed,
            gateway_generation="gateway-a",
            revision=42,
            reason="worker_lease_lost",
        )

    text = "\n".join(telemetry.render_prometheus())
    assert (
        "dingo_video_task_submissions_total"
        in text
    )
    assert 'delivery_mode="async"' in text
    assert 'outcome="created"' in text
    assert "dingo_video_task_stage_duration_seconds_bucket" in text
    assert 'stage="queue"' in text
    assert 'dingo_video_etcd_request_errors_total{operation="kv/range"} 1' in text
    assert 'error_code="worker_lease_lost"' in text
    assert (
        'dingo_video_discovery_consistent{pool="pool-a"} 1'
        in text
    )

    event = json.loads(caplog.records[-1].message)
    assert event["log_type"] == "video_task_lifecycle"
    assert event["event"] == "failed"
    assert event["task_id"] == task.id
    assert event["gateway_generation"] == "gateway-a"
    assert event["gateway_queue_wait_s"] == 0.1
    assert event["worker_queue_wait_s"] == 0.3
    assert event["worker_inference_time_s"] == 1.2
    assert event["stage_durations"] == {"worker_queue_wait": 0.3}
    assert "prompt" not in event
