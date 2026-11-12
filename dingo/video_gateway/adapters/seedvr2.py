# SPDX-License-Identifier: Apache-2.0
"""Uploaded-video SR-only adapter using task-scoped shared files."""
import hashlib
import json
from pathlib import Path

from dingo.common.video_sr_only import SOURCE_KEY, validate_sr_only
from dingo.common.video_super_resolution import SR_PAYLOAD_KEY, H3_SR_PROFILE_V3
from dingo.video_gateway.adapters.h3_output_policy import H3_SR_V3_PRESETS, plan_h3_output
from dingo.video_gateway.adapters.h3_resolution import BASE_CANVASES, RESOLUTION_CANVASES
from dingo.video_gateway.adapters.h3_shape import NAMED_ASPECT_RATIOS
from dingo.video_gateway.adapters.minimax_h3 import MiniMaxH3VideoAdapter, _one
from dingo.video_gateway.errors import GatewayError


def probe_source(path):
    import av
    with av.open(str(path)) as container:
        videos, audios = list(container.streams.video), list(container.streams.audio)
        if len(videos) != 1 or len(audios) > 1:
            raise ValueError("source must contain one video and at most one audio stream")
        v = videos[0]
        if v.codec_context.name != "h264" or v.codec_context.format.name != "yuv420p":
            raise ValueError("source must be H264/yuv420p MP4")
        if not v.average_rate or abs(float(v.average_rate) - 24) > 0.01:
            raise ValueError("source must be 24 FPS")
        if audios and audios[0].codec_context.name != "aac":
            raise ValueError("source audio must be AAC")
        frames = 0
        for frame in container.decode(video=0):
            frames += 1
            if frames > 362:
                raise ValueError("source exceeds 362 frames")
        return dict(width=v.width, height=v.height, frames=frames, fps=24, audio=bool(audios))


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

    def capabilities(self, *, max_result_bytes):
        return {"workflow": "sr_only", "operations": ["sr_only"], "max_references": 1,
                "max_video_bytes": 128 * 1024**2, "max_result_bytes": max_result_bytes,
                "video_formats": ["mp4"], "source_fps": 24, "max_source_frames": 362,
                "resolution_presets": {t: {r: f"{w}x{h}" for r, (w, h) in RESOLUTION_CANVASES[t].items()} for t in ("1080P", "2K")}}

    def normalize_request(self, fields, uploads, public_model):
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
        if not 0 < upload.size <= 128 * 1024**2:
            raise GatewayError(400, "invalid_source_video", "source exceeds 128 MiB")
        try:
            with upload.path.open('rb') as f:
                if f.read(12)[4:8] != b'ftyp':
                    raise ValueError('source must be MP4')
            source = probe_source(upload.path)
            size = (source['width'], source['height'])
            name = next((r for r, s in BASE_CANVASES.items() if s == size), None)
            if name is None:
                name = next((p[-1] for p in H3_SR_V3_PRESETS.values() if p[:2] == size), None)
            if name is None:
                ratio = size[0] / size[1]
                name = min(NAMED_ASPECT_RATIOS, key=lambda n: abs(ratio / NAMED_ASPECT_RATIOS[n] - 1))
                if abs(ratio / NAMED_ASPECT_RATIOS[name] - 1) > .03:
                    raise ValueError('source ratio does not match an admitted SR canvas')
            w, h = RESOLUTION_CANVASES[tier][name]
            plan = plan_h3_output(w, h, sr_profile=H3_SR_PROFILE_V3)
            sr = dict(schema_version=1, profile=H3_SR_PROFILE_V3, base_width=size[0], base_height=size[1],
                      output_width=w, output_height=h, sr_width=plan.sr_width, sr_height=plan.sr_height)
            validate_sr_only({'operation':'sr_only', SOURCE_KEY:{'media':source}, SR_PAYLOAD_KEY:sr,
                              'size':f'{size[0]}x{size[1]}', 'nvext':{'fps':24}})
        except Exception as exc:
            raise GatewayError(400, 'invalid_source_video', str(exc)) from exc
        return {'operation':'sr_only', 'public_model':public_model, 'backend_model':self.pool.backend_model,
                'workflow':'sr_only', 'width':w, 'height':h, 'aspect_ratio':name, 'fps':24,
                'num_frames':source['frames'], 'seconds':source['frames']/24, 'generate_sound':source['audio'],
                'source_media':source, 'super_resolution':sr,
                'reference_metadata':[{'ordinal':upload.ordinal,'kind':'video','content_type':'video/mp4',**source}]}

    def build_worker_payload(self, normalized, manifest, task_root):
        if len(manifest) != 1:
            raise ValueError('SR-only requires exactly one persisted input')
        item = manifest[0]
        raw = (task_root / 'input-manifest.json').read_bytes()
        source = {'schema_version':1,'storage_id':self.options['storage_id'],
                  'manifest_sha256':hashlib.sha256(raw).hexdigest(), 'media':normalized['source_media'],
                  'entry':{k:item[k] for k in ('ordinal','path','content_type','bytes','sha256')}}
        sr = normalized['super_resolution']
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
