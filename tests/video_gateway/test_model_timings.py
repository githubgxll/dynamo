# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import MappingProxyType

import pytest

from dingo.common import video_timing_schemas as schemas
from dingo.common.video_result_file import BINARY_RESULT_WRITER, normalize_inline_result
from dingo.vllm.omni.minimax_h3_timings import NATIVE_STAGE_NAMES, extract_model_execution
from tests.video_gateway.test_task_store import _task


def test_native_profiler_mapping_matches_versioned_schema():
    raw = {
        "MiniMaxH3Pipeline.encode_prompt": 1.0,
        "MiniMaxH3Pipeline.diffuse": 100.0,
        "MiniMaxH3Pipeline.decode": 2.0,
        "MiniMaxH3Pipeline._encode_visual_conditions": 3.0,
        "MiniMaxH3Pipeline._encode_reference_audio_conditions": 0.4,
        "stage_0_gen_ms": 106400,
    }
    expected = {
        "schema": "minimax_h3.v1", "unit": "seconds",
        "stages": {
            "encode_prompt": 1.0, "diffuse": 100.0, "decode": 2.0,
            "reference_visual_encode": 3.0, "reference_audio_encode": 0.4,
        },
    }
    execution = extract_model_execution(raw)
    assert execution == expected
    assert set(NATIVE_STAGE_NAMES.values()) == set(schemas.SCHEMAS[schemas.MINIMAX_H3_SCHEMA].stages)
    normalized = normalize_inline_result({
        "status": "completed",
        "data": [{"output_format": "mp4", "artifact": {
            "schema_version": 1, "filename": "worker-video-" + "a" * 32 + ".mp4",
            "bytes": 10, "sha256": "b" * 64,
        }}],
        "model_execution": execution,
        "stage_durations": {"output_total_s": 0.5},
    })
    task = _task("video-timing")
    task.normalized_request = schemas.with_model_execution({}, normalized["model_execution"])
    task.stage_durations = normalized["stage_durations"]
    restored = type(task).from_dict(json.loads(json.dumps(task.to_dict())))
    public = restored.public_dict()
    assert public["metrics"]["model_execution"] == expected
    assert "diffuse_s" not in public["metrics"]
    assert "output_total_s" not in public["metrics"]
    assert "stage_durations" not in public
    diagnostic = restored.diagnostics_dict()
    assert diagnostic["diagnostics"]["stage_durations"] == {"output_total_s": 0.5}
    assert {k: v for k, v in diagnostic.items() if k != "diagnostics"} == public


def test_invalid_or_unknown_optional_telemetry_does_not_change_task_result():
    for value in (True, -1, float("nan"), float("inf"), "1.0", 10**1000):
        assert extract_model_execution({"MiniMaxH3Pipeline.diffuse": value}) is None
    assert extract_model_execution(None) is None
    assert extract_model_execution({}) is None
    assert extract_model_execution({"OtherPipeline.diffuse": 1.0}) is None
    for schema, unit in (("unknown.v1", "seconds"), ("minimax_h3.v1", "ms")):
        assert schemas.normalize_model_execution({
            "schema": schema, "unit": unit, "stages": {"diffuse": 1},
        }) is None
    task = _task("video-unknown-schema")
    task.normalized_request[schemas.MODEL_EXECUTION_KEY] = {
        "schema": "future.v2", "unit": "seconds", "stages": {"new_stage": 1},
    }
    assert "metrics" not in task.public_dict()


def test_new_registered_model_requires_no_gateway_serializer_change(monkeypatch):
    definition = schemas.TimingSchema(
        id="example.v1", description="Test backend contract",
        stages=MappingProxyType({"synthesis": schemas.StageDefinition("Synthesis")}),
    )
    monkeypatch.setattr(schemas, "SCHEMAS", {**schemas.SCHEMAS, definition.id: definition})
    value = {"schema": definition.id, "unit": "seconds", "stages": {"synthesis": 2.0}}
    task = _task("video-other-model")
    task.normalized_request = schemas.with_model_execution({}, value)
    assert task.public_dict()["metrics"]["model_execution"] == value
    # Clearing an attempt's telemetry must keep generation parameters intact.
    task.normalized_request["seed"] = 42
    assert schemas.with_model_execution(task.normalized_request, None) == {"seed": 42}


def test_exported_schema_defines_optional_stages_and_parent_relationships():
    definition = schemas.SCHEMAS[schemas.MINIMAX_H3_SCHEMA]
    document = json.loads(json.dumps(definition.json_schema()))
    assert document["properties"]["schema"]["const"] == "minimax_h3.v1"
    stages = document["properties"]["stages"]
    assert "required" not in stages
    assert stages["properties"]["video_decode"]["x-parent-stage"] == "decode"
    assert stages["properties"]["diffuse"]["minimum"] == 0


@pytest.mark.parametrize("early_release", [False, True])
async def test_model_execution_survives_worker_handoff_and_finalization(make_gateway_config, early_release):
    from dingo.vllm.omni.detached_tasks import DetachedOmniTaskManager
    from dingo.video_gateway.models import TaskStatus
    from tests.video_gateway.test_dispatcher import _MINIMAL_MP4, _DetachedClient, _pool, _stack, _submit

    expected = {"schema": "minimax_h3.v1", "unit": "seconds", "stages": {"diffuse": 0.02}}

    class Handler:
        async def generate(self, request, context):
            descriptor = await BINARY_RESULT_WRITER.get().write(_MINIMAL_MP4)
            yield {
                "status": "completed",
                "data": [{"output_format": "mp4", "artifact": descriptor}],
                "model_execution": expected,
                "stage_durations": {"output_total_s": 0.001},
            }

    pool = _pool("fl-pool", "public-fl", "dyn://scope.backend.generate")
    pool["execution_mode"] = "detached"
    pool["scheduling"]["early_release_slot"] = early_release
    config = make_gateway_config(pools=[pool])
    manager = DetachedOmniTaskManager(Handler(), config.artifact_store.root, binary_results=True, inline_results=True)
    store, artifacts, dispatcher, service = _stack(config, {"fl-pool": _DetachedClient(manager)})
    await dispatcher.start()
    try:
        submitted = await _submit(service, "public-fl")
        terminal = await dispatcher.wait_terminal(submitted.stored.task.id, 3)
        assert terminal.task.status == TaskStatus.COMPLETED
        assert terminal.task.public_dict()["metrics"]["model_execution"] == expected
        assert terminal.task.diagnostics_dict()["diagnostics"]["stage_durations"]["output_total_s"] == 0.001
    finally:
        await dispatcher.stop()
        await manager.shutdown()
