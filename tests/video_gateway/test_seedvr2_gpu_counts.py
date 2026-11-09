# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SR admission and native telemetry must agree on the allocated rank count."""
import importlib.util
import json
from pathlib import Path

import pytest

from dingo.common.video_super_resolution import H3_SR_PROFILE_V3
from dingo.vllm.omni.seedvr2_process import SeedVR2Process


def native_support():
    root = Path(__file__).resolve().parents[2]
    path = root / "container/deps/vllm/patches/h3_seedvr2_v3/sources/vllm_omni/diffusion/models/seedvr2/long_video_support.py"
    spec = importlib.util.spec_from_file_location("sr_gpu_count_support", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("count", [4, 8])
def test_v3_shared_gpu_admission(tmp_path, monkeypatch, count):
    monkeypatch.delenv("DINGO_VIDEO_SEEDVR2_CUDA_VISIBLE_DEVICES", raising=False)
    for name in ("seedvr2_ema_3b_fp16.safetensors", "ema_vae_fp16.safetensors"):
        (tmp_path / name).touch()
    process = SeedVR2Process({
        "model": str(tmp_path), "work_root": str(tmp_path / "work"),
        "gpus": count, "engine_identity": "test", "profile": H3_SR_PROFILE_V3,
        "image_digest": "example/image@sha256:" + "a" * 64,
    })
    assert process.config["gpus"] == count


@pytest.mark.parametrize("count", [True, 0, 2, 3, 6, 16, "4"])
def test_v3_unsupported_gpu_count_rejected(tmp_path, count):
    with pytest.raises(ValueError):
        SeedVR2Process({
            "model": str(tmp_path), "work_root": str(tmp_path / "work"),
            "gpus": count, "engine_identity": "test", "profile": H3_SR_PROFILE_V3,
            "image_digest": "example/image@sha256:" + "a" * 64,
        })


@pytest.mark.parametrize("count", [4, 8])
def test_native_request_preserves_rank_count_and_requires_every_rank(tmp_path, count):
    module = native_support()
    config = module.options(
        encoding_mode="single_encode", delivery_size="1920x1080", window_frames=33,
        measurement_context=json.dumps({"expected_world_size": count}),
    )
    metrics = module.ServerMetrics(tmp_path / "job", config, 2016, 1152, 96, 33)
    assert metrics.identity["expected_world_size"] == count
    request = dict(metrics.identity, measurement_mode="basic", sr_request_id="job")
    bundle = module.validate_bundle("job", {"request.json": request, "window-map.jsonl": [{"window_index": 0}]})
    assert bundle["metrics_status"] == "METRICS_INCOMPLETE"
    assert [row["rank"] for row in bundle["missing_rank_records"]] == list(range(count))


@pytest.mark.parametrize("count", [True, 0, 2, 3, 16, "4"])
def test_native_measurement_rejects_invalid_rank_counts(count):
    with pytest.raises(ValueError, match="expected_world_size"):
        native_support().options(measurement_context=json.dumps({"expected_world_size": count}))
