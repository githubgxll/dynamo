# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dingo.common.video_timing_schemas import MODEL_EXECUTION_KEY
from dingo.video_gateway.models import TaskStatus
from dingo.video_gateway.performance import performance_record
from dingo.video_gateway.telemetry import GatewayTelemetry
from tests.video_gateway.test_task_store import _task


def test_performance_dimensions_preserve_unknowns_and_omit_input_content():
    task = _task("video-performance")
    task.normalized_request = {
        "prompt": "private prompt",
        "seconds": 8,
        "width": 1344,
        "height": 768,
        "num_inference_steps": 50,
        "generate_sound": True,
        "reference_kinds": ["image", "video", "audio"],
        "reference_metadata": [
            {"kind": "image", "width": 1000, "height": 500, "bytes": 100},
            {"kind": "video", "fps": 24, "duration_s": 3.1, "bytes": 200},
            {"kind": "audio", "duration_s": 2, "bytes": 50},
        ],
        MODEL_EXECUTION_KEY: {
            "schema": "minimax_h3.v1",
            "unit": "seconds",
            "stages": {"diffuse": 3},
        },
    }
    result = performance_record(task)
    assert result["counts"] == {"image": 1, "video": 1, "audio": 1}
    assert result["input_type"] == "image+video+audio"
    assert result["image_total_pixels"] == 500000
    assert result["video_total_seconds"] == 3.1
    assert result["video_total_frames"] is None  # Never fabricate fps * duration.
    assert result["model_execution"]["stages"] == {"diffuse": 3.0}
    assert "private prompt" not in str(result)
    assert result["timings"]["finalize_s"] is None


def test_missing_reference_metadata_does_not_report_zero_size():
    task = _task("video-old-worker")
    task.normalized_request = {"reference_kinds": ["image"]}
    result = performance_record(task)
    assert result["counts"]["image"] == 1
    assert result["image_total_pixels"] is None
    assert result["image_total_bytes"] is None
    assert result["video_total_seconds"] == 0
    assert result["model_execution"] is None


def test_performance_logging_is_optional_and_terminal_only(monkeypatch, caplog):
    import json
    import logging

    task = _task("video-performance-log")
    task.status = TaskStatus.COMPLETED
    telemetry = GatewayTelemetry()
    monkeypatch.delenv("DINGO_VIDEO_PERFORMANCE_RECORDS", raising=False)
    with caplog.at_level(logging.INFO, logger="dingo.video_gateway.audit"):
        telemetry.audit_task("completed", task, gateway_generation="test")
        assert "performance" not in json.loads(caplog.records[-1].message)
        monkeypatch.setenv("DINGO_VIDEO_PERFORMANCE_RECORDS", "1")
        telemetry.audit_task("result_handoff", task, gateway_generation="test")
        assert "performance" not in json.loads(caplog.records[-1].message)
        telemetry.audit_task("completed", task, gateway_generation="test")
        assert (
            json.loads(caplog.records[-1].message)["performance"]["schema_version"] == 1
        )
