# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""H3 Base -> owned SeedVR2 -> one fenced binary MP4 result.

The detached manager must hold one execution permit across this whole generator.
No intermediate H3 completion is yielded to the Gateway. The ordinary handler
is retained for non-SR requests, including its seed omission semantics.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path

from dingo.common.video_sr_timings import aggregate_sr_timings
from dingo.common.video_timing_schemas import normalize_model_execution
from dingo.common.video_result_file import (
    BINARY_RESULT_WRITER,
    BinaryResultWriter,
    validate_descriptor,
)
from dingo.common.video_worker_error import VideoWorkerFailure, failed_chunk_error, classify_worker_error
from dingo.common.video_super_resolution import H3_SR_PROFILE, H3_SR_PROFILE_V2, H3_SR_PROFILE_V3, SR_PAYLOAD_KEY
from dingo.video_gateway.adapters.h3_output_policy import plan_h3_output
from dingo.vllm.omni import seedvr2_media as media
from dingo.vllm.omni.h3_video_handler import H3VideoHandler
from dingo.vllm.omni.seedvr2_execution import (
    SeedVR2Execution,
    UnsafeRecovery,
    read_journal,
    write_journal,
)


# Allow slower rank cleanup after OOM while staying below the video supervisor's
# 35-second pause deadline. The outer RPC wait includes two seconds of overhead.



def validate_sr_request(payload: dict, *, profile=H3_SR_PROFILE) -> dict:
    sr = payload.get(SR_PAYLOAD_KEY)
    if not isinstance(sr, dict):
        raise ValueError("missing SR plan")
    plan = plan_h3_output(
        sr.get("output_width"), sr.get("output_height"), sr_profile=profile
    )
    expected = {
        "schema_version": 1,
        "profile": profile,
        "base_width": plan.base_width,
        "base_height": plan.base_height,
        "output_width": plan.output_width,
        "output_height": plan.output_height,
        "sr_width": plan.sr_width,
        "sr_height": plan.sr_height,
    }
    if (
        plan.sr_profile is None
        or sr != expected
        or any(type(sr[k]) is not int for k in expected if k != "profile")
        or payload.get("size") != f"{plan.base_width}x{plan.base_height}"
        or payload.get("nvext", {}).get("fps") != 24
    ):
        raise ValueError("invalid SR payload geometry/profile")
    return expected


class SeedVR2Handler(H3VideoHandler):
    def __init__(
        self,
        handler,
        execution: SeedVR2Execution,
        *,
        engine_identity: str,
        healthy=lambda: True,
        profile=H3_SR_PROFILE,
        measurement_identity=None,
    ):
        if not engine_identity:
            raise ValueError("SR requires a pinned engine/model identity")
        self.sr_profile = profile
        self.measurement_identity = measurement_identity
        self.handler = handler
        self.execution = execution
        self.engine_identity = engine_identity
        self.healthy = healthy
        self.quarantined = False
        self.quarantine_reason = None
        self.last_failure = None
        self.pause_started = None
        # The supervisor may additionally terminate its owned SR process.
        self._outer_quarantine = execution.quarantine
        execution.quarantine = self.quarantine
        execution.pause = self.pause
        execution.resume = self.resume

    @property
    def sr_ready(self):
        return self.video_ready

    def validate_sr_request(self, payload):
        validate_sr_request(payload, profile=self.sr_profile)
        if not self.sr_ready:
            raise RuntimeError("SR Worker is quarantined or unavailable")

    async def _base(self, payload, root, fingerprint, context):
        # Search only this task's attempts, never other tasks. Reuse is immutable
        # and requires the same complete payload and pinned engine identity.
        for candidate in sorted(root.parent.glob("*/base-checkpoint.json")):
            if candidate.parent.is_symlink():
                raise UnsafeRecovery("symlink in Base checkpoint path")
            record = read_journal(candidate)
            if not record or record.get("fingerprint") != fingerprint:
                continue
            sr_journal = read_journal(candidate.parent / "sr-journal.json")
            if sr_journal and sr_journal.get("phase") not in {
                "completed",
                "failed",
                "cancelled",
            }:
                # No blind re-POST to a different process/Worker after an
                # ambiguous previous submission. Retain Base for investigation.
                raise UnsafeRecovery("previous SR attempt has unresolved GPU work")
            artifact = validate_descriptor(record.get("artifact"))
            path = candidate.parent / "base" / artifact["filename"]
            if (
                path.parent.is_symlink()
                or await asyncio.to_thread(media.file_sha256, path)
                != artifact["sha256"]
            ):
                raise UnsafeRecovery("Base checkpoint checksum mismatch")
            return path, record.get("stages", {}), normalize_model_execution(record.get("model_execution"))
        directory = root / "base"
        directory.mkdir(exist_ok=False)
        writer = BinaryResultWriter(directory)
        token = BINARY_RESULT_WRITER.set(writer)
        terminal = None
        started = time.monotonic()
        try:
            async for chunk in self._generate_base(payload, context):
                if chunk.get("status") in {"completed", "failed", "cancelled"}:
                    if terminal is not None:
                        raise RuntimeError("H3 returned multiple terminal results")
                    terminal = chunk
            if context.is_stopped():
                raise asyncio.CancelledError
            if terminal is not None and terminal.get("status") == "failed":
                raise failed_chunk_error(terminal, stage="base_generation")
            if terminal is None or terminal.get("status") != "completed":
                raise RuntimeError("H3 Base generation did not complete")
            data = terminal.get("data")
            if not isinstance(data, list) or len(data) != 1:
                raise RuntimeError("H3 must return exactly one binary Base")
            artifact = validate_descriptor(data[0].get("artifact"))
            path = directory / artifact["filename"]
            if await asyncio.to_thread(media.file_sha256, path) != artifact["sha256"]:
                raise RuntimeError("H3 Base checksum mismatch")
            stages = {"h3_base_total_s": time.monotonic() - started}
            write_journal(
                root / "base-checkpoint.json",
                {
                    "schema_version": 1,
                    "fingerprint": fingerprint,
                    "artifact": artifact,
                    "stages": stages,
                    "model_execution": normalize_model_execution(terminal.get("model_execution")),
                },
            )
            return path, stages, normalize_model_execution(terminal.get("model_execution"))
        finally:
            BINARY_RESULT_WRITER.reset(token)

    async def generate(self, payload, context):
        if not self.sr_ready:
            raise VideoWorkerFailure("composite Worker is unavailable", {"schema_version": 1, "reason": "engine_unavailable", "stage": "worker_execution"})
        if SR_PAYLOAD_KEY not in payload:
            async for chunk in self._generate_base(payload, context):
                yield chunk
            return
        sr = validate_sr_request(payload, profile=self.sr_profile)
        writer = BINARY_RESULT_WRITER.get()
        if writer is None:
            raise RuntimeError("SR requires detached binary results")
        root = writer.root
        base_payload = {k: v for k, v in payload.items() if k != SR_PAYLOAD_KEY}
        fingerprint = hashlib.sha256(
            json.dumps(
                {"request": payload, "engine": self.engine_identity},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        started = time.monotonic()
        base, stages, model_execution = await self._base(base_payload, root, fingerprint, context)
        stages = dict(stages)
        mark = time.monotonic()
        info = await media.probe(base)
        stages["h3_base_probe_s"] = time.monotonic() - mark
        if (info["width"], info["height"]) != (sr["base_width"], sr["base_height"]):
            raise RuntimeError("H3 engine changed requested Base dimensions")
        options = None
        if self.sr_profile in {H3_SR_PROFILE_V2, H3_SR_PROFILE_V3}:
            if not self.measurement_identity:
                raise RuntimeError("SR v2 requires measurement provenance")
            options = {
                "window_frames": plan_h3_output(sr["output_width"],sr["output_height"],sr_profile=self.sr_profile).window_frames if self.sr_profile == H3_SR_PROFILE_V3 else 33 if sr["output_height"] == 1080 else 65,
                "delivery_size": f"{sr['output_width']}x{sr['output_height']}",
                "encoding_mode": "single_encode", "measurement_mode": "basic",
                "measurement_context": {**self.measurement_identity,
                    "run_id": context.id(), "batch_id": self.execution.owner,
                    "case_id": f"sr-{sr['output_height']}", "action_id": fingerprint,
                    "phase": "super_resolution"},
            }
        mark = time.monotonic()
        try:
            job = await self.execution.run(
                base,
                root / "sr-journal.json",
                fingerprint=fingerprint,
                width=sr["sr_width"],
                height=sr["sr_height"],
                frames=info["frames"],
                stopped=context.is_stopped,
                seed=7723 if options is not None else payload.get("nvext", {}).get("seed"),
                **({"options": options} if options is not None else {}),
            )
        except Exception as exc:
            raise VideoWorkerFailure(str(exc), classify_worker_error(exc, stage="super_resolution")) from exc
        stages = dict(stages, sr_service_s=time.monotonic() - mark)
        if context.is_stopped():
            raise asyncio.CancelledError
        sr_metrics = None
        if options is not None:
            manifest = await self.execution.client.manifest(job)
            expected = {"encoding_mode": "single_encode", "configured_window": options["window_frames"],
                        "selected_window": options["window_frames"], "frames": info["frames"],
                        "width": sr["output_width"], "height": sr["output_height"],
                        "final_lossy_video_encode_count": 1, "external_lossy_encode_required": False}
            if any(manifest.get(k) != v for k, v in expected.items()):
                raise RuntimeError("SR manifest does not match requested profile")
            write_journal(root / "sr-manifest.json", manifest)
            try:
                sr_metrics = await self.execution.client.metrics(job)
                write_journal(root / "sr-metrics.json", sr_metrics)
            except Exception:
                logging.getLogger(__name__).warning("SR detailed timings unavailable", exc_info=True)
        raw, final = root / "sr-raw.mp4", root / "sr-final.mp4"
        postprocess_started = time.monotonic()
        mark = time.monotonic()
        await self.execution.client.download(job, raw)
        stages["sr_download_s"] = time.monotonic() - mark
        mark = time.monotonic()
        raw_info = await media.probe(raw)
        stages["sr_raw_probe_s"] = time.monotonic() - mark
        if (raw_info["width"], raw_info["height"]) != ((sr["output_width"], sr["output_height"]) if options is not None else (sr["sr_width"], sr["sr_height"])):
            raise RuntimeError("SR engine changed requested internal dimensions")
        mark = time.monotonic()
        finalized = await media.finalize(
            raw, base, final, width=sr["output_width"], height=sr["output_height"],
            **({"require_stream_copy": True} if options is not None else {}),
            **({"expected_base_size": (sr["base_width"], sr["base_height"]),
                "base_info": info, "raw_info": raw_info}
               if self.sr_profile == H3_SR_PROFILE_V3 else {}),
        )
        stages["sr_finalize_s"] = time.monotonic() - mark
        details = finalized["postprocess_timings"]
        mode_key = (
            "remux" if finalized["postprocess_mode"] == "stream_copy" else "encode"
        )
        stages[f"sr_finalize_{mode_key}_s"] = details["mux_or_encode_s"]
        for label in ("base", "raw", "final"):
            for part in ("hash_s", "video_scan_s", "audio_scan_s"):
                key = f"{label}_{part}"
                if key in details:
                    stages[f"sr_finalize_{key}"] = details[key]
        if context.is_stopped():
            raise asyncio.CancelledError
        mark = time.monotonic()
        if await asyncio.to_thread(media.file_sha256, base) != info["sha256"]:
            raise RuntimeError("Base changed during SR")
        stages["sr_base_recheck_s"] = time.monotonic() - mark
        # The existing writer enforces the 128 MiB result limit and durable
        # publication. Gateway still fences the result by attempt/token.
        mark = time.monotonic()
        data = await asyncio.to_thread(final.read_bytes)
        stages["sr_result_read_s"] = time.monotonic() - mark
        mark = time.monotonic()
        artifact = await writer.write(data)
        stages["sr_result_persist_s"] = time.monotonic() - mark
        stages["sr_postprocess_total_s"] = time.monotonic() - postprocess_started
        yield {
            "status": "completed",
            "data": [{"output_format": "mp4", "artifact": artifact}],
            "inference_time_s": time.monotonic() - started,
            "stage_durations": stages,
            "model_execution": model_execution,
            "super_resolution_timings": aggregate_sr_timings(sr_metrics, stages),
        }
