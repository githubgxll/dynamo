# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 FL2VA/Ref2VA request and response compatibility adapter."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import math
import os
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dingo.video_gateway.adapters.base import UploadedArtifact, WorkerVideoResult
from dingo.video_gateway.adapters.h3_shape import (
    align_frame_count,
    relative_ratio_error,
    resolve_output_shape,
)
from dingo.video_gateway.config import PoolConfig
from dingo.video_gateway.errors import GatewayError

_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}
_VIDEO_TYPES = {"video/mp4"}
_AUDIO_TYPES = {"audio/wav", "audio/x-wav"}
_ALLOWED_FIELDS = {
    "model",
    "prompt",
    "seconds",
    "size",
    "width",
    "height",
    "num_frames",
    "fps",
    "steps",
    "num_inference_steps",
    "guidance_scale",
    "seed",
    "negative_prompt",
    "response_format",
    "output_format",
    "task",
    "frame_indices",
    "flow_shift",
    "audio_flow_shift",
    "aspect_ratio",
    "short_edge",
    "num_outputs_per_prompt",
    "extra_params",
    "user",
    "generate_sound",
}
_KNOWN_UNSUPPORTED_FIELDS = {
    "start_time_seconds",
    "quality",
    "guidance_scale_2",
    "boundary_ratio",
    "true_cfg_scale",
    "sound_duration",
    "frame_interpolation",
    "lora",
    "image_reference",
    "video_reference",
    "audio_reference",
}
_MAX_PROMPT_BYTES = 32 * 1024
_MAX_IMAGE_BYTES = 30 * 1024 * 1024
_MAX_VIDEO_BYTES = 50 * 1024 * 1024
_MAX_AUDIO_BYTES = 15 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class _MediaProbe:
    kind: str
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    duration_s: float | None = None


def _one(
    fields: Mapping[str, list[str]], key: str, *, required: bool = False
) -> str | None:
    values = fields.get(key, [])
    if len(values) > 1:
        raise GatewayError(400, "duplicate_field", f"{key} may appear only once", key)
    if not values:
        if required:
            raise GatewayError(400, "missing_required_field", f"{key} is required", key)
        return None
    value = values[0].strip()
    if required and not value:
        raise GatewayError(400, "missing_required_field", f"{key} is required", key)
    return value


def _integer(value: str | None, key: str, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise GatewayError(
            400, "invalid_integer", f"{key} must be an integer", key
        ) from exc


def _number(value: str | None, key: str, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        result = float(value)
    except ValueError as exc:
        raise GatewayError(
            400, "invalid_number", f"{key} must be a number", key
        ) from exc
    if not math.isfinite(result):
        raise GatewayError(400, "invalid_number", f"{key} must be finite", key)
    return result


def _boolean_value(value: str | None, key: str, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise GatewayError(400, "invalid_boolean", f"{key} must be true or false", key)


def _extra_number(value: Any, key: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise GatewayError(400, "invalid_number", f"{key} must be a number", key)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise GatewayError(
            400, "invalid_number", f"{key} must be a number", key
        ) from exc
    if not math.isfinite(result):
        raise GatewayError(400, "invalid_number", f"{key} must be finite", key)
    return result


def _coalesce(
    direct: Any, extra: Mapping[str, Any], key: str, *, param: str | None = None
) -> Any:
    if direct is not None and key in extra:
        raise GatewayError(
            400,
            "conflicting_parameter",
            f"{param or key} was provided both directly and in extra_params",
            param or key,
        )
    return direct if direct is not None else extra.get(key)


def _parse_size(fields: Mapping[str, list[str]]) -> tuple[int, int]:
    size = _one(fields, "size")
    width = _integer(_one(fields, "width"), "width")
    height = _integer(_one(fields, "height"), "height")
    if (width is None) != (height is None):
        raise GatewayError(
            400,
            "incomplete_size",
            "width and height must be provided together",
            "size",
        )
    if size is not None and width is not None:
        raise GatewayError(
            400,
            "conflicting_size",
            "provide either size or width and height, not both",
            "size",
        )
    if size is not None:
        try:
            size_width, size_height = (int(part) for part in size.lower().split("x", 1))
        except (ValueError, TypeError) as exc:
            raise GatewayError(
                400, "invalid_size", "size must use WIDTHxHEIGHT", "size"
            ) from exc
        width, height = size_width, size_height
    elif width is None or height is None:
        raise GatewayError(
            400,
            "missing_size",
            "provide size or both width and height",
            "size",
        )
    assert width is not None and height is not None
    try:
        resolve_output_shape(width, height)
    except ValueError as exc:
        raise GatewayError(
            400,
            "invalid_size",
            str(exc),
            "size",
        ) from exc
    return width, height


def _stream_duration(container: Any, stream: Any) -> float:
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        import av

        return float(container.duration / av.time_base)
    return 0.0


def _probe_image(
    upload: UploadedArtifact, content_type: str, header: bytes
) -> _MediaProbe:
    magic_type = None
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        magic_type = "image/png"
    elif header.startswith(b"\xff\xd8\xff"):
        magic_type = "image/jpeg"
    elif header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        magic_type = "image/webp"
    if magic_type is None or magic_type != content_type:
        raise GatewayError(
            400,
            "media_type_mismatch",
            "image MIME and file signature do not match",
            upload.field_name,
        )
    if upload.size > _MAX_IMAGE_BYTES:
        raise GatewayError(
            413, "file_too_large", "image exceeds 30 MiB", upload.field_name
        )
    try:
        from PIL import Image

        with Image.open(upload.path) as image:
            image.verify()
        with Image.open(upload.path) as image:
            width, height = image.size
    except Exception as exc:
        raise GatewayError(
            400, "invalid_media", "image could not be decoded", upload.field_name
        ) from exc
    return _MediaProbe("image", width=width, height=height)


def _probe_av(upload: UploadedArtifact, kind: str) -> _MediaProbe:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError(
            "MiniMax-H3 reference validation requires the video-gateway optional extra"
        ) from exc
    try:
        with av.open(str(upload.path)) as container:
            if kind == "video":
                streams = list(container.streams.video)
                if not streams:
                    raise ValueError("missing video stream")
                stream = streams[0]
                if next(container.decode(stream), None) is None:
                    raise ValueError("video stream has no decodable frames")
                fps = float(stream.average_rate) if stream.average_rate else 0.0
                return _MediaProbe(
                    kind,
                    width=int(stream.width),
                    height=int(stream.height),
                    fps=fps,
                    duration_s=_stream_duration(container, stream),
                )
            streams = list(container.streams.audio)
            if not streams:
                raise ValueError("missing audio stream")
            if next(container.decode(streams[0]), None) is None:
                raise ValueError("audio stream has no decodable frames")
            return _MediaProbe(kind, duration_s=_stream_duration(container, streams[0]))
    except GatewayError:
        raise
    except Exception as exc:
        raise GatewayError(
            400, "invalid_media", f"{kind} could not be decoded", upload.field_name
        ) from exc


def _probe_upload(upload: UploadedArtifact) -> _MediaProbe:
    content_type = upload.content_type.lower().split(";", 1)[0].strip()
    with upload.path.open("rb") as stream:
        header = stream.read(64)
    if content_type in _IMAGE_TYPES:
        return _probe_image(upload, content_type, header)
    if content_type in _VIDEO_TYPES:
        if len(header) < 12 or header[4:8] != b"ftyp":
            raise GatewayError(
                400,
                "media_type_mismatch",
                "video MIME does not match an MP4 file",
                upload.field_name,
            )
        if upload.size > _MAX_VIDEO_BYTES:
            raise GatewayError(
                413, "file_too_large", "video exceeds 50 MiB", upload.field_name
            )
        probe = _probe_av(upload, "video")
        if (
            probe.width is None
            or probe.height is None
            or not (256 <= probe.width <= 5760 and 256 <= probe.height <= 5760)
        ):
            raise GatewayError(
                400,
                "invalid_reference_video",
                "reference video dimensions must each be between 256 and 5760",
                upload.field_name,
            )
        if probe.fps is None or not (23.976 - 0.01 <= probe.fps <= 60.0 + 0.01):
            raise GatewayError(
                400,
                "invalid_reference_video",
                "reference video FPS must be between 23.976 and 60",
                upload.field_name,
            )
        if probe.duration_s is None or not 2.0 <= probe.duration_s <= 15.0:
            raise GatewayError(
                400,
                "invalid_reference_video",
                "reference video duration must be between 2 and 15 seconds",
                upload.field_name,
            )
        return probe
    if content_type in _AUDIO_TYPES:
        if not (header.startswith(b"RIFF") and header[8:12] == b"WAVE"):
            raise GatewayError(
                400,
                "media_type_mismatch",
                "audio MIME does not match a WAV file",
                upload.field_name,
            )
        if upload.size > _MAX_AUDIO_BYTES:
            raise GatewayError(
                413, "file_too_large", "audio exceeds 15 MiB", upload.field_name
            )
        probe = _probe_av(upload, "audio")
        if probe.duration_s is None or not 2.0 <= probe.duration_s <= 15.0:
            raise GatewayError(
                400,
                "invalid_reference_audio",
                "reference audio duration must be between 2 and 15 seconds",
                upload.field_name,
            )
        return probe
    raise GatewayError(
        400,
        "unsupported_reference_type",
        f"unsupported reference content type: {upload.content_type}",
        upload.field_name,
    )


def _data_url(
    path: Path, content_type: str, *, expected_bytes: int, expected_sha256: str
) -> str:
    payload = path.read_bytes()
    if len(payload) != expected_bytes:
        raise RuntimeError("persisted reference size changed before Worker dispatch")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise RuntimeError(
            "persisted reference checksum changed before Worker dispatch"
        )
    return f"data:{content_type.split(';', 1)[0]};base64," + base64.b64encode(
        payload
    ).decode("ascii")


class MiniMaxH3VideoAdapter:
    """Translate stable external multipart requests to the current H3 wire shape."""

    def __init__(self, pool: PoolConfig) -> None:
        if pool.adapter.workflow not in {"fl2va", "ref2va"}:
            raise ValueError("minimax_h3 workflow must be 'fl2va' or 'ref2va'")
        self.pool = pool
        self.workflow = pool.adapter.workflow
        self.options = dict(pool.adapter.options)
        missing_media_dependencies = [
            name
            for name, module in (("Pillow", "PIL"), ("PyAV", "av"))
            if importlib.util.find_spec(module) is None
        ]
        if missing_media_dependencies:
            raise RuntimeError(
                "MiniMax-H3 video Gateway requires ai-dingo[video-gateway]; "
                "missing " + ", ".join(missing_media_dependencies)
            )
        supported_options = {
            "flow_shift",
            "audio_flow_shift",
            "validate_media",
            "max_encoded_reference_bytes",
        }
        unknown_options = sorted(set(self.options) - supported_options)
        if unknown_options:
            raise ValueError(
                "unknown MiniMax-H3 adapter options: " + ", ".join(unknown_options)
            )
        if not isinstance(self.options.get("validate_media", True), bool):
            raise TypeError("MiniMax-H3 validate_media must be a boolean")
        max_encoded = self.options.get("max_encoded_reference_bytes", 384 * 1024 * 1024)
        if (
            not isinstance(max_encoded, int)
            or isinstance(max_encoded, bool)
            or max_encoded <= 0
        ):
            raise ValueError(
                "MiniMax-H3 max_encoded_reference_bytes must be a positive integer"
            )
        flow_shift = float(self.options.get("flow_shift", 12.0))
        audio_flow_shift = float(self.options.get("audio_flow_shift", 3.0))
        if flow_shift != 12.0 or audio_flow_shift != 3.0:
            raise ValueError(
                "MiniMax-H3 requires flow_shift=12.0 and audio_flow_shift=3.0"
            )

    def normalize_request(
        self,
        fields: Mapping[str, list[str]],
        uploads: Sequence[UploadedArtifact],
        public_model: str,
    ) -> dict[str, Any]:
        unknown = sorted(set(fields) - _ALLOWED_FIELDS - _KNOWN_UNSUPPORTED_FIELDS)
        if unknown:
            raise GatewayError(
                400,
                "unsupported_parameter",
                f"unsupported parameters: {', '.join(unknown)}",
                unknown[0],
            )
        recognized_unsupported = sorted(set(fields) & _KNOWN_UNSUPPORTED_FIELDS)
        if recognized_unsupported:
            raise GatewayError(
                400,
                "unsupported_parameter",
                f"unsupported parameters: {', '.join(recognized_unsupported)}",
                recognized_unsupported[0],
            )
        reference_fields = {upload.field_name for upload in uploads}
        if {"input_reference", "input_references"} <= reference_fields:
            raise GatewayError(
                400,
                "conflicting_reference_fields",
                "input_reference and input_references cannot be used together",
                "input_reference",
            )
        if sum(upload.field_name == "input_reference" for upload in uploads) > 1:
            raise GatewayError(
                400,
                "duplicate_field",
                "input_reference accepts only one file; use input_references for many",
                "input_reference",
            )
        for upload in uploads:
            if upload.field_name not in {"input_reference", "input_references"}:
                raise GatewayError(
                    400,
                    "unsupported_file_field",
                    f"unsupported file field: {upload.field_name}",
                    upload.field_name,
                )

        prompt = _one(fields, "prompt", required=True)
        model = _one(fields, "model")
        assert prompt is not None
        if len(prompt.encode("utf-8")) > _MAX_PROMPT_BYTES:
            raise GatewayError(
                400,
                "prompt_too_long",
                "prompt must not exceed 32768 UTF-8 bytes",
                "prompt",
            )
        if model is not None and model != public_model:
            raise GatewayError(
                400, "model_mismatch", "model changed while parsing", "model"
            )

        extra_raw = _one(fields, "extra_params")
        extra: dict[str, Any] = {}
        if extra_raw:
            try:
                decoded = json.loads(extra_raw)
            except json.JSONDecodeError as exc:
                raise GatewayError(
                    400,
                    "invalid_extra_params",
                    "extra_params must be JSON",
                    "extra_params",
                ) from exc
            if not isinstance(decoded, dict):
                raise GatewayError(
                    400,
                    "invalid_extra_params",
                    "extra_params must be a JSON object",
                    "extra_params",
                )
            allowed_extra = {
                "task",
                "duration",
                "frame_indices",
                "flow_shift",
                "audio_flow_shift",
            }
            unknown_extra = sorted(set(decoded) - allowed_extra)
            if unknown_extra:
                raise GatewayError(
                    400,
                    "unsupported_parameter",
                    f"unsupported extra_params: {', '.join(unknown_extra)}",
                    "extra_params",
                )
            extra = decoded

        fps = _integer(_one(fields, "fps"), "fps", 24)
        if fps != 24:
            raise GatewayError(400, "invalid_fps", "MiniMax-H3 requires fps=24", "fps")

        seconds_direct = _number(_one(fields, "seconds"), "seconds")
        duration_extra = _extra_number(extra.get("duration"), "extra_params.duration")
        num_frames = _integer(_one(fields, "num_frames"), "num_frames")
        duration_sources = sum(
            value is not None for value in (seconds_direct, duration_extra, num_frames)
        )
        if duration_sources != 1:
            raise GatewayError(
                400,
                "conflicting_duration",
                "provide exactly one of seconds, num_frames, or extra_params.duration",
                "seconds",
            )
        if num_frames is None:
            seconds = seconds_direct if seconds_direct is not None else duration_extra
            assert seconds is not None
            if not 4.0 <= seconds <= 15.0:
                raise GatewayError(
                    400,
                    "invalid_duration",
                    "seconds must be between 4 and 15",
                    "seconds",
                )
            num_frames = round(seconds * 24)
        else:
            if num_frames < 96 or num_frames > 360:
                raise GatewayError(
                    400,
                    "invalid_num_frames",
                    "num_frames must represent 4 to 15 seconds at 24 fps",
                    "num_frames",
                )
            seconds = num_frames / 24.0

        width, height = _parse_size(fields)
        aspect_ratio = _one(fields, "aspect_ratio")
        try:
            resolved_shape = resolve_output_shape(
                width, height, requested_aspect_ratio=aspect_ratio
            )
        except ValueError as exc:
            raise GatewayError(
                400, "invalid_aspect_ratio", str(exc), "aspect_ratio"
            ) from exc
        short_edge = _integer(_one(fields, "short_edge"), "short_edge")
        if short_edge is not None and short_edge != 768:
            raise GatewayError(
                400,
                "unsupported_parameter",
                "short_edge must be 768 when provided",
                "short_edge",
            )
        outputs = _integer(
            _one(fields, "num_outputs_per_prompt"), "num_outputs_per_prompt", 1
        )
        if outputs != 1:
            raise GatewayError(
                400,
                "unsupported_parameter",
                "num_outputs_per_prompt must be 1",
                "num_outputs_per_prompt",
            )

        steps_primary = _one(fields, "num_inference_steps")
        steps_alias = _one(fields, "steps")
        if steps_primary is not None and steps_alias is not None:
            raise GatewayError(
                400,
                "conflicting_parameter",
                "provide only one of num_inference_steps or steps",
                "num_inference_steps",
            )
        steps_value = steps_primary if steps_primary is not None else steps_alias
        steps = _integer(steps_value, "num_inference_steps", 50)
        if steps is None or not 1 <= steps <= 200:
            raise GatewayError(
                400,
                "invalid_num_inference_steps",
                "num_inference_steps must be between 1 and 200",
                "num_inference_steps",
            )
        guidance_scale = _number(_one(fields, "guidance_scale"), "guidance_scale")
        if guidance_scale is not None and not 0.0 <= guidance_scale <= 20.0:
            raise GatewayError(
                400,
                "invalid_guidance_scale",
                "guidance_scale must be between 0 and 20",
                "guidance_scale",
            )
        seed = _integer(_one(fields, "seed"), "seed")
        seed_generated = seed is None
        if seed is None:
            seed = secrets.randbits(32)
        if not 0 <= seed <= 2**32 - 1:
            raise GatewayError(
                400, "invalid_seed", "seed must be an unsigned 32-bit integer", "seed"
            )

        response_format = _one(fields, "response_format")
        if response_format not in {None, "b64_json"}:
            raise GatewayError(
                400,
                "unsupported_response_format",
                "asynchronous video tasks support response_format=b64_json internally",
                "response_format",
            )
        output_format = _one(fields, "output_format")
        if output_format not in {None, "mp4"}:
            raise GatewayError(
                400,
                "unsupported_output_format",
                "only MP4 output is supported",
                "output_format",
            )

        probes = [_probe_upload(upload) for upload in uploads]
        kinds = [probe.kind for probe in probes]
        frame_indices_raw: Any = _coalesce(
            _one(fields, "frame_indices"), extra, "frame_indices"
        )
        if isinstance(frame_indices_raw, str):
            try:
                frame_indices_raw = json.loads(frame_indices_raw)
            except json.JSONDecodeError as exc:
                raise GatewayError(
                    400,
                    "invalid_frame_indices",
                    "frame_indices must be a JSON array",
                    "frame_indices",
                ) from exc

        expected_task = (
            "t2va" if self.workflow == "fl2va" and not uploads else self.workflow
        )
        task = _coalesce(_one(fields, "task"), extra, "task") or expected_task
        if task != expected_task:
            raise GatewayError(
                400,
                "invalid_task",
                f"pool {self.pool.pool_id} requires task={expected_task}",
                "task",
            )
        self._validate_references(kinds, probes, frame_indices_raw, width, height)

        direct_flow_shift = _number(_one(fields, "flow_shift"), "flow_shift")
        direct_audio_shift = _number(
            _one(fields, "audio_flow_shift"), "audio_flow_shift"
        )
        requested_flow_shift = _extra_number(
            _coalesce(direct_flow_shift, extra, "flow_shift"), "flow_shift"
        )
        requested_audio_shift = _extra_number(
            _coalesce(direct_audio_shift, extra, "audio_flow_shift"),
            "audio_flow_shift",
        )
        if requested_flow_shift is not None and requested_flow_shift != 12.0:
            raise GatewayError(
                400,
                "unsupported_runtime_override",
                "flow_shift must match Worker startup value 12.0",
                "flow_shift",
            )
        if requested_audio_shift is not None and requested_audio_shift != 3.0:
            raise GatewayError(
                400,
                "unsupported_runtime_override",
                "audio_flow_shift must match checkpoint value 3.0",
                "audio_flow_shift",
            )

        if self.workflow == "fl2va":
            frame_indices = frame_indices_raw
            if frame_indices is None:
                frame_indices = (
                    [] if not uploads else ([0] if len(uploads) == 1 else [0, -1])
                )
        else:
            frame_indices = None

        negative_prompt = _one(fields, "negative_prompt")
        generate_sound = _boolean_value(
            _one(fields, "generate_sound"), "generate_sound", True
        )
        if (
            negative_prompt is not None
            and len(negative_prompt.encode("utf-8")) > _MAX_PROMPT_BYTES
        ):
            raise GatewayError(
                400,
                "negative_prompt_too_long",
                "negative_prompt must not exceed 32768 UTF-8 bytes",
                "negative_prompt",
            )
        return {
            "prompt": prompt,
            "public_model": public_model,
            "backend_model": self.pool.backend_model,
            "workflow": self.workflow,
            "width": width,
            "height": height,
            "fps": 24,
            "seconds": seconds,
            "num_frames": num_frames,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "seed_generated": seed_generated,
            "negative_prompt": negative_prompt,
            "frame_indices": frame_indices,
            "reference_kinds": kinds,
            "reference_metadata": [
                {
                    "ordinal": upload.ordinal,
                    "kind": probe.kind,
                    "width": probe.width,
                    "height": probe.height,
                    "fps": probe.fps,
                    "duration_s": probe.duration_s,
                }
                for upload, probe in zip(uploads, probes)
            ],
            "aspect_ratio": resolved_shape.aspect_ratio,
            "task": expected_task,
            "user": _one(fields, "user"),
            "flow_shift": 12.0,
            "audio_flow_shift": 3.0,
            "generate_sound": generate_sound,
        }

    def _validate_references(
        self,
        kinds: list[str],
        probes: Sequence[_MediaProbe],
        frame_indices: Any,
        width: int,
        height: int,
    ) -> None:
        if self.workflow == "fl2va":
            if any(kind != "image" for kind in kinds) or len(kinds) > 2:
                raise GatewayError(
                    400,
                    "unsupported_reference_combination",
                    "FL2VA accepts zero, one, or two images",
                    "input_references",
                )
            if frame_indices is not None:
                if not isinstance(frame_indices, list):
                    raise GatewayError(
                        400,
                        "invalid_frame_indices",
                        "frame_indices must be a JSON array",
                        "frame_indices",
                    )
                allowed: list[list[int]]
                if not kinds:
                    allowed = []
                elif len(kinds) == 1:
                    allowed = [[0], [-1]]
                else:
                    allowed = [[0, -1]]
                if frame_indices not in allowed:
                    raise GatewayError(
                        400,
                        "invalid_frame_indices",
                        "frame_indices do not match the FL2VA reference count",
                        "frame_indices",
                    )
            if probes:
                dimensions = {(probe.width, probe.height) for probe in probes}
                if len(dimensions) != 1:
                    raise GatewayError(
                        400,
                        "reference_shape_mismatch",
                        "FL2VA keyframe images must have identical dimensions",
                        "input_references",
                    )
                reference_width, reference_height = next(iter(dimensions))
                assert reference_width is not None and reference_height is not None
                if (
                    relative_ratio_error(
                        width, height, reference_width / reference_height
                    )
                    > 0.02
                ):
                    raise GatewayError(
                        400,
                        "reference_aspect_ratio_mismatch",
                        "output size must match the FL2VA reference aspect ratio",
                        "size",
                    )
            return

        image_count = kinds.count("image")
        if frame_indices is not None:
            raise GatewayError(
                400,
                "unsupported_parameter",
                "frame_indices is only supported by FL2VA",
                "frame_indices",
            )
        video_count = kinds.count("video")
        audio_count = kinds.count("audio")
        if image_count > 9 or video_count > 3 or audio_count > 3 or len(kinds) > 12:
            raise GatewayError(
                400,
                "too_many_references",
                "Ref2VA reference count exceeds MiniMax-H3 limits",
                "input_references",
            )
        if image_count + video_count == 0:
            raise GatewayError(
                400,
                "missing_visual_reference",
                "Ref2VA requires at least one image or video",
                "input_references",
            )
        supported = len(kinds) == 1 and kinds[0] in {"image", "video"}
        supported = supported or (
            image_count >= 1 and video_count >= 1 and audio_count >= 1
        )
        if not supported:
            raise GatewayError(
                400,
                "unsupported_reference_combination",
                "current Ref2VA bridge supports one image, one video, or mixed image+video+audio",
                "input_references",
            )
        total_video_duration = sum(
            probe.duration_s or 0.0 for probe in probes if probe.kind == "video"
        )
        total_audio_duration = sum(
            probe.duration_s or 0.0 for probe in probes if probe.kind == "audio"
        )
        if total_video_duration > 15.0 + 1e-6 or total_audio_duration > 15.0 + 1e-6:
            raise GatewayError(
                400,
                "reference_duration_exceeded",
                "total reference video and audio duration must each not exceed 15 seconds",
                "input_references",
            )

    def build_worker_payload(
        self,
        normalized: Mapping[str, Any],
        manifest: Sequence[Mapping[str, Any]],
        task_root: Path,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": normalized["backend_model"],
            "prompt": normalized["prompt"],
            "size": f"{normalized['width']}x{normalized['height']}",
            "response_format": "b64_json",
            "output_format": "mp4",
            "nvext": {
                "fps": normalized["fps"],
                "num_frames": normalized["num_frames"],
                "num_inference_steps": normalized["num_inference_steps"],
                "seed": normalized["seed"],
            },
        }
        if normalized.get("negative_prompt") is not None:
            payload["nvext"]["negative_prompt"] = normalized["negative_prompt"]
        if normalized.get("guidance_scale") is not None:
            payload["nvext"]["guidance_scale"] = normalized["guidance_scale"]

        estimated_encoded_bytes = sum(
            ((int(entry["bytes"]) + 2) // 3) * 4 for entry in manifest
        )
        max_encoded_bytes = int(
            self.options.get("max_encoded_reference_bytes", 384 * 1024 * 1024)
        )
        if estimated_encoded_bytes > max_encoded_bytes:
            raise RuntimeError(
                "encoded Worker reference payload exceeds configured maximum"
            )

        references = []
        root = task_root.resolve()
        for entry in sorted(manifest, key=lambda item: int(item["ordinal"])):
            relative = Path(str(entry["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError("reference path escaped task root")
            cursor = root
            for component in relative.parts:
                cursor /= component
                if cursor.is_symlink():
                    raise RuntimeError("persisted reference path contains a symlink")
            path = cursor.resolve()
            if root not in path.parents or not path.is_file():
                raise RuntimeError("reference path escaped task root")
            references.append(
                (
                    str(entry["content_type"]),
                    _data_url(
                        path,
                        str(entry["content_type"]),
                        expected_bytes=int(entry["bytes"]),
                        expected_sha256=str(entry["sha256"]),
                    ),
                )
            )

        if self.workflow == "fl2va":
            indices = normalized.get("frame_indices") or []
            if len(references) == 1 and indices == [0]:
                payload["input_reference"] = references[0][1]
            elif len(references) == 1 and indices == [-1]:
                payload["input_reference"] = json.dumps(
                    {
                        "type": "fl2va_keyframes_v1",
                        "images": [references[0][1]],
                        "frame_indices": [-1],
                    },
                    separators=(",", ":"),
                )
            elif len(references) == 2:
                payload["input_reference"] = json.dumps(
                    [references[0][1], references[1][1]], separators=(",", ":")
                )
        elif len(references) == 1:
            payload["input_reference"] = references[0][1]
        elif references:
            envelope = {
                "type": "ref2va_mixed_v1",
                "images": [
                    value for mime, value in references if mime.startswith("image/")
                ],
                "videos": [
                    value for mime, value in references if mime.startswith("video/")
                ],
                "audios": [
                    value for mime, value in references if mime.startswith("audio/")
                ],
            }
            payload["input_reference"] = json.dumps(envelope, separators=(",", ":"))
        return payload

    def consume_worker_stream(self, chunks: Sequence[Any]) -> WorkerVideoResult:
        terminal: Mapping[str, Any] | None = None
        for chunk in chunks:
            if isinstance(chunk, str):
                try:
                    chunk = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
            if isinstance(chunk, Mapping):
                status = chunk.get("status")
                if status in {"completed", "failed", "cancelled"}:
                    if terminal is not None:
                        raise RuntimeError(
                            "Worker returned multiple terminal responses"
                        )
                    terminal = chunk
        if terminal is None:
            raise RuntimeError("Worker stream ended without a terminal response")
        if terminal.get("status") == "failed":
            raise RuntimeError(str(terminal.get("error") or "Worker generation failed"))
        data = terminal.get("data")
        if (
            terminal.get("status") != "completed"
            or not isinstance(data, list)
            or len(data) != 1
        ):
            raise RuntimeError(
                "Worker response must contain exactly one completed video"
            )
        first = data[0]
        if not isinstance(first, Mapping):
            raise TypeError("Worker video data has an invalid shape")
        output_format = str(first.get("output_format") or "")
        b64_json = first.get("b64_json")
        if output_format != "mp4" or not isinstance(b64_json, str) or not b64_json:
            raise RuntimeError("Worker must return output_format=mp4 with b64_json")
        inference_time_raw = terminal.get("inference_time_s")
        inference_time = (
            float(inference_time_raw) if inference_time_raw is not None else None
        )
        if inference_time is not None and (
            not math.isfinite(inference_time) or inference_time < 0
        ):
            raise RuntimeError("Worker returned an invalid inference_time_s")
        stage_durations_raw = terminal.get("stage_durations")
        stage_durations: dict[str, float] | None = None
        if stage_durations_raw is not None:
            if not isinstance(stage_durations_raw, Mapping):
                raise RuntimeError("Worker returned invalid stage_durations")
            stage_durations = {}
            for name, raw_duration in stage_durations_raw.items():
                if not isinstance(name, str) or isinstance(raw_duration, bool):
                    raise RuntimeError("Worker returned invalid stage_durations")
                try:
                    duration = float(raw_duration)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        "Worker returned invalid stage_durations"
                    ) from exc
                if not math.isfinite(duration) or duration < 0:
                    raise RuntimeError("Worker returned invalid stage_durations")
                stage_durations[name] = duration
        return WorkerVideoResult(
            b64_json=b64_json,
            output_format=output_format,
            inference_time_s=inference_time,
            stage_durations=stage_durations,
        )

    def prepare_artifact(
        self, path: Path, normalized: Mapping[str, Any]
    ) -> None:
        """Match vLLM-Omni's generate_sound=false response semantics.

        MiniMax-H3 always computes the audio branch in the currently supported
        Worker path. Remuxing only the H.264 packets avoids a second video encode
        and therefore does not change image quality.
        """

        if bool(normalized.get("generate_sound", True)):
            return
        try:
            import av
        except ImportError as exc:
            raise RuntimeError(
                "MiniMax-H3 audio stripping requires the video-gateway optional extra"
            ) from exc
        remuxed = path.with_name(f"{path.name}.video-only.mp4")
        try:
            with av.open(str(path)) as source:
                video_streams = list(source.streams.video)
                if not video_streams:
                    raise RuntimeError("Worker MP4 is missing a video stream")
                source_video = video_streams[0]
                with av.open(str(remuxed), mode="w", format="mp4") as output:
                    output_video = output.add_stream_from_template(source_video)
                    for packet in source.demux(source_video):
                        if packet.dts is None:
                            continue
                        packet.stream = output_video
                        output.mux(packet)
            with remuxed.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(remuxed, path)
        finally:
            if remuxed.exists():
                remuxed.unlink()

    def validate_artifact(
        self, path: Path, normalized: Mapping[str, Any]
    ) -> dict[str, Any]:
        with path.open("rb") as stream:
            header = stream.read(32)
        if len(header) < 12 or header[4:8] != b"ftyp":
            raise RuntimeError("Worker result is not an ISO BMFF/MP4 file")
        if not bool(self.options.get("validate_media", True)):
            return {"container": "mp4"}
        try:
            import av
        except ImportError as exc:
            raise RuntimeError(
                "MiniMax-H3 media validation requires the video-gateway optional extra"
            ) from exc
        with av.open(str(path)) as container:
            video_streams = list(container.streams.video)
            audio_streams = list(container.streams.audio)
            if not video_streams or video_streams[0].codec_context.name != "h264":
                raise RuntimeError("MP4 must contain an H.264 video stream")
            generate_sound = bool(normalized.get("generate_sound", True))
            if generate_sound:
                if not audio_streams or audio_streams[0].codec_context.name != "aac":
                    raise RuntimeError("MP4 must contain an AAC audio stream")
            elif audio_streams:
                raise RuntimeError("MP4 must not contain audio when generate_sound=false")
            video = video_streams[0]
            if (
                video.width != normalized["width"]
                or video.height != normalized["height"]
            ):
                raise RuntimeError(
                    f"MP4 dimensions {video.width}x{video.height} do not match request"
                )
            average_rate = float(video.average_rate) if video.average_rate else 0.0
            if abs(average_rate - 24.0) > 0.05:
                raise RuntimeError(
                    f"MP4 frame rate {average_rate} does not match 24 fps"
                )
            frame_count = int(video.frames or 0)
            if frame_count <= 0:
                frame_count = sum(1 for _frame in container.decode(video=0))
            expected_frames = align_frame_count(int(normalized["num_frames"]))
            if frame_count != expected_frames:
                raise RuntimeError(
                    f"MP4 frame count {frame_count} does not match expected "
                    f"MiniMax-H3 aligned count {expected_frames}"
                )
            video_duration = frame_count / average_rate
            audio_duration: float | None = None
            if generate_sound:
                audio = audio_streams[0]
                audio_duration = _stream_duration(container, audio)
                if audio_duration <= 0:
                    raise RuntimeError("MP4 AAC stream has no measurable duration")
                if abs(video_duration - audio_duration) > 0.1:
                    raise RuntimeError(
                        "MP4 audio/video duration difference exceeds 100 milliseconds"
                    )
            return {
                "container": "mp4",
                "video_codec": "h264",
                "audio_codec": "aac" if generate_sound else None,
                "width": video.width,
                "height": video.height,
                "fps": average_rate,
                "frames": frame_count,
                "video_duration_s": video_duration,
                "audio_duration_s": audio_duration,
            }
