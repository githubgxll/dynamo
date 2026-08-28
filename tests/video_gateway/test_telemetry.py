# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging

from dingo.video_gateway.models import TaskStatus
from dingo.video_gateway.task_store import terminal_error
from dingo.video_gateway.telemetry import GatewayTelemetry
from tests.video_gateway.test_task_store import _task


def test_telemetry_renders_bounded_task_and_etcd_metrics(caplog):
    telemetry = GatewayTelemetry()
    task = _task("video-observed")
    failed = _task("video-observed")
    failed.status = TaskStatus.FAILED
    failed.error = terminal_error("worker_lease_lost", "lease lost")

    telemetry.record_submission(task.pool_id, "created", "async")
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

    event = json.loads(caplog.records[-1].message)
    assert event["log_type"] == "video_task_lifecycle"
    assert event["event"] == "failed"
    assert event["task_id"] == task.id
    assert event["gateway_generation"] == "gateway-a"
    assert "prompt" not in event
