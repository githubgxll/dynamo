# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image

import dingo.video_gateway.adapters.minimax_h3 as h3_module
from dingo.video_gateway.adapters.base import UploadedArtifact
from dingo.video_gateway.adapters.h3_shape import align_frame_count
from dingo.video_gateway.adapters.minimax_h3 import MiniMaxH3VideoAdapter
from dingo.video_gateway.errors import GatewayError


def _adapter(make_gateway_config, workflow="fl2va"):
    config = make_gateway_config()
    pool = config.pools[0]
    if workflow == "fl2va":
        return MiniMaxH3VideoAdapter(pool)
    raw_pool = {
        "pool_id": "ref-pool",
        "served_models": ["public-ref"],
        "backend_model": "worker-ref",
        "backend_target": "dyn://ref-scope.backend.generate",
        "adapter": {
            "name": "minimax_h3",
            "workflow": "ref2va",
            "compatibility_version": "test-wire-v1",
            "validate_media": False,
        },
        "scheduling": {"worker_capacity": 1},
    }
    return MiniMaxH3VideoAdapter(make_gateway_config(pools=[raw_pool]).pools[0])


def _base_fields(model="public-fl"):
    return {
        "model": [model],
        "prompt": ["a cinematic validation scene"],
        "seconds": ["5.0"],
        "size": ["1344x768"],
        "fps": ["24"],
        "num_inference_steps": ["50"],
        "seed": ["1101"],
    }


def _png_upload(tmp_path: Path, *, field="input_references", ordinal=0):
    path = tmp_path / f"image-{ordinal}.png"
    Image.new("RGB", (1600, 900), "red").save(path)
    payload = path.read_bytes()
    return UploadedArtifact(
        field_name=field,
        ordinal=ordinal,
        filename=path.name,
        content_type="image/png",
        path=path,
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def test_t2va_normalizes_fractional_duration_and_persists_seed(make_gateway_config):
    adapter = _adapter(make_gateway_config)
    fields = _base_fields()
    fields["seconds"] = ["5.125"]

    normalized = adapter.normalize_request(fields, [], "public-fl")

    assert normalized["task"] == "t2va"
    assert normalized["num_frames"] == round(5.125 * 24)
    assert normalized["aspect_ratio"] == "16:9"
    assert normalized["flow_shift"] == 12.0
    assert normalized["audio_flow_shift"] == 3.0
    assert normalized["seed"] == 1101
    assert normalized["seed_generated"] is False


def test_pinned_h3_frame_alignment_matches_worker_version():
    assert align_frame_count(120) == 124
    assert align_frame_count(96) == 107
    assert align_frame_count(360) == 362


def test_duration_must_have_exactly_one_source(make_gateway_config):
    adapter = _adapter(make_gateway_config)
    fields = _base_fields()
    fields["num_frames"] = ["120"]

    with pytest.raises(GatewayError) as error:
        adapter.normalize_request(fields, [], "public-fl")

    assert error.value.code == "conflicting_duration"


def test_extra_duration_and_task_are_supported(make_gateway_config):
    adapter = _adapter(make_gateway_config)
    fields = _base_fields()
    del fields["seconds"]
    fields["extra_params"] = [
        json.dumps({"duration": 5.0, "task": "t2va", "audio_flow_shift": 3.0})
    ]

    normalized = adapter.normalize_request(fields, [], "public-fl")

    assert normalized["num_frames"] == 120
    assert normalized["task"] == "t2va"


@pytest.mark.parametrize(
    ("field", "value", "expected_code"),
    [
        ("fps", "25", "invalid_fps"),
        ("size", "1000x700", "invalid_size"),
        ("aspect_ratio", "1:1", "invalid_aspect_ratio"),
        ("guidance_scale", "21", "invalid_guidance_scale"),
        ("num_inference_steps", "201", "invalid_num_inference_steps"),
    ],
)
def test_invalid_h3_parameters_have_specific_errors(
    make_gateway_config, field, value, expected_code
):
    adapter = _adapter(make_gateway_config)
    fields = _base_fields()
    fields[field] = [value]

    with pytest.raises(GatewayError) as error:
        adapter.normalize_request(fields, [], "public-fl")

    assert error.value.code == expected_code


def test_fl2va_first_image_builds_plain_data_url(tmp_path, make_gateway_config):
    adapter = _adapter(make_gateway_config)
    upload = _png_upload(tmp_path)
    normalized = adapter.normalize_request(_base_fields(), [upload], "public-fl")
    task_root = tmp_path / "task"
    (task_root / "inputs").mkdir(parents=True)
    target = task_root / "inputs" / "000.bin"
    target.write_bytes(upload.path.read_bytes())
    manifest = [
        {
            "ordinal": 0,
            "path": "inputs/000.bin",
            "bytes": target.stat().st_size,
            "content_type": "image/png",
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
    ]

    payload = adapter.build_worker_payload(normalized, manifest, task_root)

    assert payload["model"] == "worker-fl"
    assert payload["response_format"] == "b64_json"
    assert payload["output_format"] == "mp4"
    assert payload["input_reference"].startswith("data:image/png;base64,")
    assert payload["nvext"] == {
        "fps": 24,
        "num_frames": 120,
        "num_inference_steps": 50,
        "seed": 1101,
    }


def test_fl2va_tail_image_uses_versioned_envelope(tmp_path, make_gateway_config):
    adapter = _adapter(make_gateway_config)
    upload = _png_upload(tmp_path)
    fields = _base_fields()
    fields["frame_indices"] = ["[-1]"]
    normalized = adapter.normalize_request(fields, [upload], "public-fl")
    task_root = tmp_path / "task"
    (task_root / "inputs").mkdir(parents=True)
    target = task_root / "inputs" / "000.bin"
    target.write_bytes(upload.path.read_bytes())

    payload = adapter.build_worker_payload(
        normalized,
        [
            {
                "ordinal": 0,
                "path": "inputs/000.bin",
                "bytes": target.stat().st_size,
                "content_type": "image/png",
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        ],
        task_root,
    )
    envelope = json.loads(payload["input_reference"])

    assert envelope["type"] == "fl2va_keyframes_v1"
    assert envelope["frame_indices"] == [-1]
    assert len(envelope["images"]) == 1


def test_ref2va_mixed_payload_preserves_typed_order(
    tmp_path, make_gateway_config, monkeypatch
):
    adapter = _adapter(make_gateway_config, "ref2va")
    probes = {
        0: h3_module._MediaProbe("image", width=512, height=512),
        1: h3_module._MediaProbe(
            "video", width=512, height=512, fps=24.0, duration_s=3.0
        ),
        2: h3_module._MediaProbe("audio", duration_s=3.0),
    }
    monkeypatch.setattr(
        h3_module, "_probe_upload", lambda upload: probes[upload.ordinal]
    )
    uploads = []
    content_types = ["image/png", "video/mp4", "audio/wav"]
    for ordinal, content_type in enumerate(content_types):
        path = tmp_path / f"source-{ordinal}.bin"
        path.write_bytes(f"payload-{ordinal}".encode())
        uploads.append(
            UploadedArtifact(
                "input_references",
                ordinal,
                path.name,
                content_type,
                path,
                path.stat().st_size,
                str(ordinal) * 64,
            )
        )
    fields = _base_fields("public-ref")
    fields["task"] = ["ref2va"]
    normalized = adapter.normalize_request(fields, uploads, "public-ref")
    task_root = tmp_path / "task"
    (task_root / "inputs").mkdir(parents=True)
    manifest = []
    for upload in uploads:
        target = task_root / "inputs" / f"{upload.ordinal:03d}.bin"
        target.write_bytes(upload.path.read_bytes())
        manifest.append(
            {
                "ordinal": upload.ordinal,
                "path": f"inputs/{upload.ordinal:03d}.bin",
                "bytes": target.stat().st_size,
                "content_type": upload.content_type,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )

    payload = adapter.build_worker_payload(normalized, manifest, task_root)
    envelope = json.loads(payload["input_reference"])

    assert envelope["type"] == "ref2va_mixed_v1"
    assert len(envelope["images"]) == 1
    assert len(envelope["videos"]) == 1
    assert len(envelope["audios"]) == 1


def test_worker_terminal_requires_one_b64_mp4(make_gateway_config):
    adapter = _adapter(make_gateway_config)
    expected = base64.b64encode(b"\x00\x00\x00\x18ftypisompayload").decode()

    result = adapter.consume_worker_stream(
        [
            {"status": "in_progress", "progress": 0},
            {
                "status": "completed",
                "data": [{"output_format": "mp4", "b64_json": expected}],
                "inference_time_s": 1.25,
                "stage_durations": {"diffuse": 1.1, "decode": 0.15},
            },
        ]
    )

    assert result.b64_json == expected
    assert result.output_format == "mp4"
    assert result.inference_time_s == 1.25
    assert result.stage_durations == {"diffuse": 1.1, "decode": 0.15}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("true", True), ("false", False), ("1", True), ("0", False)],
)
def test_generate_sound_matches_vllm_omni_form_boolean(
    raw, expected, make_gateway_config
):
    adapter = _adapter(make_gateway_config)
    fields = _base_fields()
    fields["generate_sound"] = [raw]

    normalized = adapter.normalize_request(fields, [], "public-fl")

    assert normalized["generate_sound"] is expected


def test_generate_sound_rejects_ambiguous_boolean(make_gateway_config):
    adapter = _adapter(make_gateway_config)
    fields = _base_fields()
    fields["generate_sound"] = ["yes"]

    with pytest.raises(GatewayError) as error:
        adapter.normalize_request(fields, [], "public-fl")

    assert error.value.code == "invalid_boolean"


def test_mime_spoofing_is_rejected(tmp_path, make_gateway_config):
    adapter = _adapter(make_gateway_config)
    upload = _png_upload(tmp_path)
    spoofed = UploadedArtifact(
        upload.field_name,
        upload.ordinal,
        upload.filename,
        "image/jpeg",
        upload.path,
        upload.size,
        upload.sha256,
    )

    with pytest.raises(GatewayError) as error:
        adapter.normalize_request(_base_fields(), [spoofed], "public-fl")

    assert error.value.code == "media_type_mismatch"


def _write_h264_aac_mp4(path: Path, *, frames: int, width: int, height: int) -> None:
    import av

    with av.open(str(path), "w") as container:
        video = container.add_stream("libx264", rate=24)
        video.width = width
        video.height = height
        video.pix_fmt = "yuv420p"
        audio = container.add_stream("aac", rate=48_000)
        audio.layout = "mono"

        for index in range(frames):
            frame = av.VideoFrame(width, height, "yuv420p")
            for plane, value in zip(frame.planes, (16, 128, 128)):
                plane.update(bytes([value]) * plane.buffer_size)
            frame.pts = index
            frame.time_base = Fraction(1, 24)
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)

        samples = round(frames * 48_000 / 24)
        offset = 0
        while offset < samples:
            sample_count = min(1024, samples - offset)
            frame = av.AudioFrame(format="fltp", layout="mono", samples=sample_count)
            frame.sample_rate = 48_000
            frame.pts = offset
            frame.time_base = Fraction(1, 48_000)
            for plane in frame.planes:
                plane.update(b"\0" * plane.buffer_size)
            for packet in audio.encode(frame):
                container.mux(packet)
            offset += sample_count
        for packet in audio.encode():
            container.mux(packet)


def test_media_gate_accepts_h3_aligned_h264_aac_result(tmp_path, make_gateway_config):
    adapter = _adapter(make_gateway_config)
    adapter.options["validate_media"] = True
    result = tmp_path / "result.mp4"
    _write_h264_aac_mp4(result, frames=124, width=256, height=256)

    media = adapter.validate_artifact(
        result,
        {"width": 256, "height": 256, "num_frames": 120},
    )

    assert media["video_codec"] == "h264"
    assert media["audio_codec"] == "aac"
    assert media["frames"] == 124


def test_generate_sound_false_remuxes_without_reencoding_video(
    tmp_path, make_gateway_config
):
    import av

    adapter = _adapter(make_gateway_config)
    adapter.options["validate_media"] = True
    result = tmp_path / "result.mp4"
    _write_h264_aac_mp4(result, frames=124, width=256, height=256)

    normalized = {
        "width": 256,
        "height": 256,
        "num_frames": 120,
        "generate_sound": False,
    }
    adapter.prepare_artifact(result, normalized)
    media = adapter.validate_artifact(result, normalized)

    with av.open(str(result)) as container:
        assert len(container.streams.video) == 1
        assert len(container.streams.audio) == 0
    assert media["video_codec"] == "h264"
    assert media["audio_codec"] is None
