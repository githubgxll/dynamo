# SPDX-License-Identifier: Apache-2.0
"""Uploaded-video SR-only adapter using task-scoped shared files."""
import hashlib
import json
import os
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from dingo.common.video_sr_only import SOURCE_KEY, validate_sr_only, source_aspect_ratio
from dingo.common.video_super_resolution import SR_PAYLOAD_KEY, H3_SR_PROFILE_V3
from dingo.video_gateway.adapters.h3_output_policy import plan_h3_output
from dingo.video_gateway.adapters.h3_resolution import RESOLUTION_CANVASES, restore_sr_plan
from dingo.video_gateway.adapters.minimax_h3 import MiniMaxH3VideoAdapter, _one
from dingo.video_gateway.errors import GatewayError


_FRAME_TIME = Fraction(1, 24)
_MAX_SOURCE_BYTES = 128 * 1024**2
_MIN_SOURCE_FRAMES = 107
_MAX_SOURCE_FRAMES = 362
_SOURCE_FRAME_STEP = 17
_SOURCE_ALIGNMENT = 32
_MIN_SOURCE_PIXELS = 768 * 768
_MAX_SOURCE_PIXELS = 768 * 1344


def _validate_upscale_source(source):
    width, height = source["width"], source["height"]
    if width % _SOURCE_ALIGNMENT or height % _SOURCE_ALIGNMENT:
        raise ValueError("base_video width and height must be multiples of 32")
    if not _MIN_SOURCE_PIXELS <= width * height <= _MAX_SOURCE_PIXELS:
        raise ValueError(f"base_video must have {_MIN_SOURCE_PIXELS} to {_MAX_SOURCE_PIXELS} pixels per frame")
    frames = source["frames"]
    if not _MIN_SOURCE_FRAMES <= frames <= _MAX_SOURCE_FRAMES or (frames - _MIN_SOURCE_FRAMES) % _SOURCE_FRAME_STEP:
        raise ValueError("base_video must have 107 to 362 frames in steps of 17")
    if not source["audio"]:
        raise ValueError("base_video must include an AAC audio track")


def probe_source(path, *, enforce_input_bounds=False, require_cfr=False):
    import av
    with av.open(str(path)) as container:
        videos, audios = list(container.streams.video), list(container.streams.audio)
        if len(videos) != 1 or len(audios) > 1:
            raise ValueError("source must contain one video and at most one audio stream")
        v = videos[0]
        if enforce_input_bounds and not (
            256 <= v.width <= 2048 and 256 <= v.height <= 2048
            and v.width * v.height <= 1100000
            and 0.4 <= v.width / v.height <= 2.5
        ):
            raise ValueError("source canvas is outside SR-only bounds")
        if v.codec_context.name != "h264" or v.codec_context.format.name != "yuv420p":
            raise ValueError("source must be H264/yuv420p MP4")
        if v.average_rate != 24:
            raise ValueError("source must be 24 FPS")
        if audios and audios[0].codec_context.name != "aac":
            raise ValueError("source audio must be AAC")
        frames = 0
        for frame in container.decode(video=0):
            if require_cfr and (
                frame.pts is None or frame.time_base is None
                or frame.pts * frame.time_base != frames * _FRAME_TIME
                or (frame.width, frame.height) != (v.width, v.height)
                or frame.format.name != "yuv420p"
            ):
                raise ValueError("source must have a fixed canvas and constant 24 FPS timestamps starting at zero")
            frames += 1
            if frames > _MAX_SOURCE_FRAMES:
                raise ValueError("source exceeds 362 frames")
        return dict(width=v.width, height=v.height, frames=frames, fps=24, audio=bool(audios))


def _resize_source(source, destination, width, height, max_bytes):
    import av

    with av.open(str(source)) as media, av.open(str(destination), "w", format="mp4") as output:
        video = media.streams.video[0]
        video.codec_context.thread_count = 2
        audio = next(iter(media.streams.audio), None)
        resized = output.add_stream("libx264", rate=24)
        resized.width, resized.height, resized.pix_fmt = width, height, "yuv420p"
        resized.codec_context.thread_count = 2
        resized.codec_context.sample_aspect_ratio = Fraction(1)
        resized.options = {"preset": "veryfast", "crf": "18"}
        copied_audio = output.add_stream_from_template(audio) if audio is not None else None
        bytes_written = 0

        def mux(packet):
            nonlocal bytes_written
            bytes_written += packet.size
            if bytes_written > max_bytes:
                raise GatewayError(413, "file_too_large", "prepared base_video exceeds the SR source limit", "base_video")
            output.mux(packet)

        frame_index = 0
        streams = (video, audio) if audio is not None else (video,)
        for packet in media.demux(*streams):
            if packet.stream == video:
                for frame in packet.decode():
                    frame = frame.reformat(width=width, height=height, format="yuv420p",
                                           interpolation="BICUBIC")
                    frame.pts, frame.time_base = frame_index, _FRAME_TIME
                    frame_index += 1
                    for encoded in resized.encode(frame):
                        mux(encoded)
            elif packet.dts is not None:
                packet.stream = copied_audio
                mux(packet)
        for packet in resized.encode():
            mux(packet)


class SeedVR2VideoAdapter(MiniMaxH3VideoAdapter):
    def __init__(self, pool):
        if pool.adapter.workflow != "sr_only" or pool.execution_mode != "detached":
            raise ValueError("seedvr2 requires detached sr_only workflow")
        self.pool, self.workflow = pool, "sr_only"
        self.options = dict(pool.adapter.options)
        if set(self.options) != {"storage_id"} or not isinstance(self.options["storage_id"], str) or not self.options["storage_id"]:
            raise ValueError("seedvr2 requires one explicit storage_id option")

    @property
    def max_encoded_reference_bytes(self):
        return 256 * 1024

    def estimate_encoded_reference_bytes(self, manifest):
        return 0  # Only the small descriptor crosses the request plane.

    def capabilities(self, *, max_result_bytes, max_upload_bytes=_MAX_SOURCE_BYTES):
        return {"workflow": "sr_only", "operations": ["sr_only"], "max_references": 1,
                "max_video_bytes": min(max_upload_bytes, _MAX_SOURCE_BYTES),
                "max_result_bytes": max_result_bytes,
                "video_formats": ["mp4"], "source_fps": 24, "audio_required": True,
                "min_source_frames": _MIN_SOURCE_FRAMES, "max_source_frames": _MAX_SOURCE_FRAMES,
                "source_frame_step": _SOURCE_FRAME_STEP,
                "source_dimension_alignment": _SOURCE_ALIGNMENT,
                "min_source_pixels": _MIN_SOURCE_PIXELS, "max_source_pixels": _MAX_SOURCE_PIXELS,
                "resolution_presets": {t: {r: f"{w}x{h}" for r, (w, h) in RESOLUTION_CANVASES[t].items()} for t in ("1080P", "2K")}}

    def normalize_request(self, fields, uploads, public_model, *, pipeline_sr=None, independent_upload=False):
        allowed = {"model", "operation", "resolution", "input_reference", "input_references"}
        unknown = set(fields) - allowed
        if unknown:
            raise GatewayError(400, "unsupported_parameter", "SR-only does not accept generation parameters: " + ", ".join(sorted(unknown)))
        if _one(fields, "model") != public_model or _one(fields, "operation", required=True) != "sr_only":
            raise GatewayError(400, "invalid_operation", "use operation=sr_only and the SR model")
        tier = _one(fields, "resolution", required=True)
        if tier not in ("1080P", "2K"):
            raise GatewayError(400, "invalid_resolution", "SR-only resolution must be 1080P or 2K", "resolution")
        if len(uploads) != 1 or uploads[0].field_name not in {"input_reference", "input_references"}:
            raise GatewayError(400, "invalid_source_video", "upload exactly one source MP4")
        upload = uploads[0]
        if not 0 < upload.size <= _MAX_SOURCE_BYTES:
            raise GatewayError(400, "invalid_source_video", "source exceeds 128 MiB")
        try:
            with upload.path.open('rb') as f:
                if f.read(12)[4:8] != b'ftyp':
                    raise ValueError('source must be MP4')
            source = probe_source(upload.path, enforce_input_bounds=True,
                                  require_cfr=independent_upload)
            if independent_upload:
                _validate_upscale_source(source)
            size = (source['width'], source['height'])
            if pipeline_sr and "adaptive_keyframe" in pipeline_sr:
                plan = restore_sr_plan(pipeline_sr, H3_SR_PROFILE_V3)
                if plan.adaptive_keyframe[2] != tier:
                    raise ValueError("adaptive SR tier does not match parent")
                name = plan.aspect_ratio
                w, h = plan.output_width, plan.output_height
            else:
                name = source_aspect_ratio(*size)
                w, h = RESOLUTION_CANVASES[tier][name]
                plan = plan_h3_output(w, h, sr_profile=H3_SR_PROFILE_V3)
            sr = dict(schema_version=1, profile=H3_SR_PROFILE_V3, base_width=size[0], base_height=size[1],
                      output_width=w, output_height=h, sr_width=plan.sr_width, sr_height=plan.sr_height)
            if plan.adaptive_keyframe:
                sr["adaptive_keyframe"] = list(plan.adaptive_keyframe)
                if sr != pipeline_sr:
                    raise ValueError("adaptive SR plan differs from parent")
            validate_sr_only({'operation':'sr_only', SOURCE_KEY:{'media':source}, SR_PAYLOAD_KEY:sr,
                              'size':f'{size[0]}x{size[1]}', 'nvext':{'fps':24}})
        except Exception as exc:
            raise GatewayError(400, 'invalid_source_video', str(exc)) from exc
        return {'operation':'sr_only', 'public_model':public_model, 'backend_model':self.pool.backend_model,
                'workflow':'sr_only', 'width':w, 'height':h, 'aspect_ratio':name, 'fps':24,
                'num_frames':source['frames'], 'seconds':source['frames']/24, 'generate_sound':source['audio'],
                'source_media':source, 'super_resolution':sr,
                **({'adaptive_resolution': tier} if plan.adaptive_keyframe else {}),
                'reference_metadata':[{'ordinal':upload.ordinal,'kind':'video','content_type':'video/mp4',**source}]}

    def prepare_upscale_source(self, upload, normalized, *, max_bytes):
        """Store the exact base geometry expected by the existing V3 Worker."""
        sr = normalized["super_resolution"]
        plan = plan_h3_output(sr["output_width"], sr["output_height"], sr_profile=H3_SR_PROFILE_V3)
        base = plan.base_width, plan.base_height
        source = normalized["source_media"]
        if (source["width"], source["height"]) == base:
            return upload, normalized

        limit = min(max_bytes, _MAX_SOURCE_BYTES)
        prepared = upload.path.with_name("sr-base.mp4")
        try:
            _resize_source(upload.path, prepared, *base, limit)
            size = prepared.stat().st_size
            if size > limit:
                raise GatewayError(413, "file_too_large", "prepared base_video exceeds the SR source limit", "base_video")
            expected = dict(source, width=base[0], height=base[1])
            if probe_source(prepared, require_cfr=True) != expected:
                raise RuntimeError("prepared SR source dimensions, frame count or audio changed unexpectedly")
            with prepared.open("rb") as file:
                digest = hashlib.file_digest(file, "sha256").hexdigest()
                os.fsync(file.fileno())
            os.replace(prepared, upload.path)
        finally:
            prepared.unlink(missing_ok=True)

        normalized = dict(
            normalized, source_media=expected,
            super_resolution=dict(sr, base_width=base[0], base_height=base[1]),
            reference_metadata=[dict(normalized["reference_metadata"][0], **expected)],
        )
        return replace(upload, size=size, sha256=digest), normalized

    def build_worker_payload(self, normalized, manifest, task_root):
        if len(manifest) != 1:
            raise ValueError('SR-only requires exactly one persisted input')
        item = manifest[0]
        raw = (task_root / 'input-manifest.json').read_bytes()
        source = {'schema_version':1,'storage_id':self.options['storage_id'],
                  'manifest_sha256':hashlib.sha256(raw).hexdigest(), 'media':normalized['source_media'],
                  'entry':{k:item[k] for k in ('ordinal','path','content_type','bytes','sha256')}}
        sr = normalized['super_resolution']
        if "_pipeline_child" not in normalized:
            plan = plan_h3_output(sr["output_width"], sr["output_height"], sr_profile=H3_SR_PROFILE_V3)
            if (sr["base_width"], sr["base_height"]) != (plan.base_width, plan.base_height):
                raise ValueError("saved independent SR source does not match Worker base dimensions")
        result = {'model':normalized['backend_model'], 'operation':'sr_only', SOURCE_KEY:source,
                  SR_PAYLOAD_KEY:sr,'size':f"{sr['base_width']}x{sr['base_height']}",
                  'nvext':{'fps':24,'num_frames':normalized['num_frames']}}
        validate_sr_only(result)
        return result

    def validate_artifact(self, path, normalized, **kwargs):
        media = probe_source(path)
        if any(media[k] != normalized[k] for k in ('width','height','fps')) or media['frames'] != normalized['num_frames'] or media['audio'] != normalized['generate_sound']:
            raise RuntimeError('SR result does not match source frames/audio or target geometry')
        return {'container':'mp4', **media, 'duration_s':media['frames']/24}

    def inspect_artifact_for_publication(self, path, normalized):
        return False, self.validate_artifact(path, normalized)

    def artifact_requires_processing(self, path, normalized):
        return False

    def prepare_artifact(self, path, normalized):
        return None
