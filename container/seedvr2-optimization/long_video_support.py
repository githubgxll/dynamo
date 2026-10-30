# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts and fail-soft metrics for the SeedVR2 long-video route."""
from __future__ import annotations

from contextlib import contextmanager
from fractions import Fraction
import json
import logging
import math
import os
from pathlib import Path
import re
import threading
import time
from uuid import uuid4

SCHEMA = "sr-timing-memory/1"
logger = logging.getLogger(__name__)
_lock = threading.Lock()
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_IDENTITY_FIELDS = {
    "run_id", "batch_id", "case_id", "action_id", "phase", "image_digest",
    "source_hashes", "model_manifest_hash", "expected_world_size",
}


class NullMetrics:
    @contextmanager
    def stage(self, *args, **kwargs):
        yield


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        pending.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def geometry(value):
    if not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]{0,5}x[1-9][0-9]{0,5}", value):
        raise ValueError("delivery_size must be WIDTHxHEIGHT")
    return tuple(map(int, value.split("x")))


def options(encoding_mode="legacy", delivery_size=None, window_frames=None,
            measurement_mode="off", measurement_context=None):
    if encoding_mode not in {"legacy", "single_encode"}:
        raise ValueError("encoding_mode must be legacy or single_encode")
    if measurement_mode not in {"off", "basic", "diagnostic"}:
        raise ValueError("measurement_mode must be off, basic, or diagnostic")
    if window_frames is not None and (isinstance(window_frames, bool) or window_frames not in (17, 33, 65)):
        raise ValueError("window_frames must be explicitly 17, 33, or 65")
    if encoding_mode == "legacy" and delivery_size is not None:
        raise ValueError("legacy does not accept delivery_size")
    delivery = geometry(delivery_size) if delivery_size is not None else None
    if encoding_mode == "single_encode" and delivery is None:
        raise ValueError("single_encode requires delivery_size")
    if delivery and (min(delivery) not in (1080, 1440) or any(x % 2 for x in delivery)):
        raise ValueError("delivery_size requires an even canvas with short edge 1080 or 1440")
    if measurement_context is None:
        context = {}
    else:
        if not isinstance(measurement_context, str) or len(measurement_context) > 16384:
            raise ValueError("measurement_context must be a small JSON object")
        try:
            context = json.loads(measurement_context)
        except (ValueError, TypeError) as error:
            raise ValueError("measurement_context must be a JSON object") from error
        if not isinstance(context, dict) or context.keys() - _IDENTITY_FIELDS:
            raise ValueError("Unsupported measurement_context keys")
        for name in ("run_id", "batch_id", "case_id", "action_id", "phase"):
            if name in context and (not isinstance(context[name], str) or not _TOKEN.fullmatch(context[name])):
                raise ValueError("Invalid measurement identity: " + name)
        if "expected_world_size" in context and context["expected_world_size"] != 8:
            raise ValueError("This experiment requires expected_world_size=8")
        for name in ("model_manifest_hash",):
            if name in context and not re.fullmatch(r"[0-9a-f]{64}", str(context[name])):
                raise ValueError("Invalid measurement hash: " + name)
        if "source_hashes" in context and (not isinstance(context["source_hashes"], dict)
                or any(not isinstance(k, str) or len(k) > 256 or not re.fullmatch(r"[0-9a-f]{64}", str(v))
                       for k, v in context["source_hashes"].items())):
            raise ValueError("Invalid source_hashes")
        if "image_digest" in context and (not isinstance(context["image_digest"], str)
                or not re.fullmatch(r"[^\s]{1,300}@sha256:[0-9a-f]{64}", context["image_digest"])):
            raise ValueError("image_digest must be a complete immutable image reference")
    if measurement_mode != "off":
        root = os.environ.get("SEEDVR2_METRICS_DIR", "")
        if not root or not Path(root).is_absolute():
            raise ValueError("Measured requests require absolute SEEDVR2_METRICS_DIR shared by API and all ranks")
    return {"encoding_mode": encoding_mode, "delivery_size": delivery,
            "window_frames": window_frames, "measurement_mode": measurement_mode,
            "measurement_context": context}


def validate_source(config, source_width, source_height, model_width, model_height):
    if config["encoding_mode"] == "single_encode":
        dw, dh = config["delivery_size"]
        if source_width <= 0 or source_height <= 0 or dw * source_height != dh * source_width:
            raise ValueError("delivery_size must preserve the exact source aspect ratio")
        if model_width * source_height != model_height * source_width:
            raise ValueError("model size must preserve the exact source aspect ratio")


def select_window(capacity, requested):
    if capacity <= 4:
        raise ValueError("SeedVR2 long video output exceeds the configured SeedVR2 clip budget")
    if requested is None:
        return capacity
    if requested > capacity:
        raise ValueError(f"Requested window_frames={requested} exceeds admitted capacity={capacity}; no automatic fallback")
    return requested


def writer_spec(config, model_width, model_height):
    size = config["delivery_size"] if config["encoding_mode"] == "single_encode" else (model_width, model_height)
    encoder_options = {"preset": "medium", "crf": "18"} if config["encoding_mode"] == "single_encode" else {
        "preset": "veryfast", "crf": "18", "tune": "zerolatency", "bf": "0"}
    return {"codec": "libx264", "width": size[0], "height": size[1], "pixel_format": "yuv420p",
            "rate": 24, "time_base": "1/24",
            "options": encoder_options,
            "tune_policy": "unspecified_libx264_default" if config["encoding_mode"] == "single_encode" else "zerolatency",
            "b_frame_policy": "unspecified_libx264_default; inspect effective_output" if config["encoding_mode"] == "single_encode" else "bf=0",
            "resize": "PyAV VideoFrame.reformat interpolation=LANCZOS after internal RGB overlap blend"}


def checked_root(request_id):
    if not isinstance(request_id, str) or not re.fullmatch(r"[0-9a-f]{32}", request_id):
        raise ValueError("Invalid native SR request id")
    value = os.environ.get("SEEDVR2_METRICS_DIR", "")
    if not value or not Path(value).is_absolute():
        raise ValueError("SEEDVR2_METRICS_DIR is not configured")
    root = Path(value) / request_id
    for path in (root, *root.parents):
        if path.is_symlink() or (path.exists() and getattr(path.lstat(), "st_file_attributes", 0) & 0x400):
            raise ValueError("Refusing symlink/reparse metrics path")
    return root


class ServerMetrics:
    def __init__(self, job, config, width, height, target, selected_window):
        self.job, self.config = Path(job), config
        self.enabled = config["measurement_mode"] != "off"
        self.failed = []
        self.windows = []
        self.activities = {}
        self.completed_counts = {}
        self.started = config.get("server_started_ns", time.perf_counter_ns())
        self.root = None
        self.identity = {**config["measurement_context"], "schema": SCHEMA,
            "sr_request_id": self.job.name, "measurement_mode": config["measurement_mode"],
            "encoding_mode": config["encoding_mode"], "configured_window": config["window_frames"] or selected_window,
            "requested_window": config["window_frames"],
            "selected_window": selected_window, "model_width": width, "model_height": height,
            "delivery_width": writer_spec(config, width, height)["width"],
            "delivery_height": writer_spec(config, width, height)["height"],
            "target_frames": target, "expected_world_size": 8,
            "clock_domain": "perf_counter_ns pid=" + str(os.getpid()), "pid": os.getpid()}
        if self.enabled:
            self.safe("initialize", self._initialize)

    def safe(self, label, function):
        try:
            return function()
        except Exception as error:
            self.failed.append({"operation": label, "error_type": type(error).__name__, "error": str(error)[:500]})
            logger.warning("SeedVR2 metrics incomplete: %s (%s)", label, type(error).__name__)
            return None

    def _initialize(self):
        self.root = checked_root(self.job.name)
        self.root.mkdir(parents=True, exist_ok=False)
        atomic_json(self.root / "request.json", self.identity)

    def row(self, phase, started, ended, **extra):
        if not self.enabled or self.root is None:
            return
        def write():
            item = {**self.identity, "stage": phase, "start_ns": started, "end_ns": ended,
                    "cpu_wall_ms": (ended - started) / 1e6, "execution_scope": "api_process", **extra}
            with _lock, (self.root / "server-stages.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(item, sort_keys=True) + "\n")
        self.safe("stage:" + phase, write)

    @contextmanager
    def stage(self, name, **extra):
        if not self.enabled:
            yield
            return
        started = time.perf_counter_ns()
        status = "completed"
        try:
            yield
        except BaseException:
            status = "failed"
            raise
        finally:
            ended = time.perf_counter_ns()
            if status == "completed":
                with _lock:
                    self.completed_counts[name] = self.completed_counts.get(name, 0) + 1
            if name in {"source_decode", "overlap_blend", "delivery_resize", "final_video_encode"}:
                with _lock:
                    activity = self.activities.setdefault(name, {"start_ns": started, "end_ns": ended,
                        "activity_cpu_wall_ms": 0.0, "call_count": 0, "failed_call_count": 0})
                    activity["end_ns"] = ended
                    activity["activity_cpu_wall_ms"] += (ended - started) / 1e6
                    activity["call_count"] += 1
                    activity["failed_call_count"] += int(status != "completed")
            else:
                self.row(name, started, ended, status=status, **extra)

    def window(self, index, start, valid):
        item = {**self.identity, "window_index": index, "model_request_id": uuid4().hex,
                "window_start": start, "input_frame_count": valid,
                "padded_frame_count": valid + (1 - valid) % 4,
                "new_frame_count": valid if start == 0 else valid - 4,
                "overlap_frames": 0 if start == 0 else 4}
        self.windows.append(item)
        if self.enabled and self.root is not None:
            def write():
                with _lock, (self.root / "window-map.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(item, sort_keys=True) + "\n")
            self.safe("window-map", write)
        return item

    def finish(self, status, error=None):
        for name, activity in self.activities.items():
            self.row(name, activity["start_ns"], activity["end_ns"],
                     activity_cpu_wall_ms=activity["activity_cpu_wall_ms"],
                     call_count=activity["call_count"], failed_call_count=activity["failed_call_count"],
                     interval_is_envelope=True,
                     scope="activity sum excludes time between calls; envelope is not additive")
        if "delivery_resize" not in self.activities:
            point = time.perf_counter_ns()
            self.row("delivery_resize", point, point, start_ns=None, end_ns=None, cpu_wall_ms=None,
                     status="not_executed_by_design", reason="legacy path or delivery equals internal geometry")
        self.row("server_sr_e2e", self.started, time.perf_counter_ns(), status=status)
        if self.enabled and self.root is not None:
            self.safe("summary", lambda: atomic_json(self.root / "summary.json", {
                **self.identity, "status": status, "error": error, "window_count": len(self.windows),
                "metrics_status": "METRICS_INCOMPLETE" if self.failed else "PENDING_RANK_COMPLETENESS_CHECK",
                "metrics_errors": self.failed, "external_controller_stages": ["base_audio_copy", "delivery_media_validate"],
                "final_lossy_video_encode_count": 1 if status == "completed" and self.config["encoding_mode"] == "single_encode" else None,
                "service_lossy_video_encode_count": 1 if status == "completed" else None,
                "external_lossy_encode_required": self.config["encoding_mode"] == "legacy",
                "window_input_lossless_encode_count": self.completed_counts.get("window_input_encode", 0),
                "window_output_lossless_encode_count": self.completed_counts.get("window_output_decode", 0)}))


def metrics_bundle(job_id):
    """Small bounded JSON API; never opens user-selected paths or tensor/video files."""
    root = checked_root(job_id)
    if not (root / "request.json").is_file():
        raise FileNotFoundError("No measurement record for this job")
    files = {}
    total = 0
    paths = [root / name for name in ("request.json", "window-map.jsonl", "server-stages.jsonl", "summary.json")]
    paths.extend(sorted((root / "windows").glob("*/*.json")))
    for path in paths:
        if not path.exists():
            continue
        relative = path.relative_to(root).as_posix()
        if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != root.parent):
            raise ValueError("Symlinked metric file refused")
        size = path.stat().st_size
        total += size
        if size > 2 * 1024 * 1024 or total > 16 * 1024 * 1024 or len(files) >= 20000:
            raise ValueError("Metrics response exceeds bounded JSON budget")
        text = path.read_text(encoding="utf-8")
        files[relative] = [json.loads(line) for line in text.splitlines() if line] if path.suffix == ".jsonl" else json.loads(text)
    windows = files.get("window-map.jsonl", [])
    missing = [{"window_index": item["window_index"], "rank": rank}
               for item in windows for rank in range(8)
               if f"windows/{item['window_index']}/rank-{rank}.json" not in files]
    settled = (root / "summary.json").is_file()
    inconsistent = []
    def number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
    def allocator(value):
        return isinstance(value, dict) and all(type(value.get(name)) is int and value[name] >= 0
            for name in ("allocated_bytes", "reserved_bytes"))
    required_phases = {"prepare_video_gpu", "vae_encode", "latent_prepare", "dit", "solver", "vae_decode", "color_uint8"}
    def measurements_valid(record, phases, postprocess=False):
        stages = record.get("stages")
        return (isinstance(stages, list) and phases.issubset({stage.get("phase") for stage in stages})
            and all(number(stage.get("cpu_wall_ms")) and number(stage.get("cuda_stream_elapsed_ms")) for stage in stages)
            and allocator(record.get("allocator_before")) and allocator(record.get("allocator_after"))
            and (allocator(record.get("allocator_process_peak_observed"))
                 and record.get("allocator_peak_scope") == "process_since_last_external_reset_not_d2h_peak"
                 if postprocess else allocator(record.get("allocator_window_peak")))
            and number(record.get("pipeline_gpu_window_stream_elapsed_ms"))
            and record.get("actual_model_request_id") is not None)
    for item in windows:
        actual_ids = set()
        for rank in range(8):
            key = f"windows/{item['window_index']}/rank-{rank}.json"
            record = files.get(key)
            if record is not None and (record.get("status") != "PASS" or record.get("sr_request_id") != job_id
                    or record.get("model_request_id") != item["model_request_id"]
                    or record.get("window_index") != item["window_index"]
                    or record.get("rank") != rank or record.get("world_size") != 8
                    or not measurements_valid(record, required_phases)):
                inconsistent.append({"file": key, "reason": "rank status, identity, stages or measurements incomplete"})
            if record is not None and isinstance(record.get("actual_model_request_id"), str):
                actual_ids.add(record["actual_model_request_id"])
        if len(actual_ids) > 1:
            inconsistent.append({"window_index": item["window_index"], "reason": "ranks disagree on actual model request"})
        postprocess = [(name, record) for name, record in files.items()
                       if name.startswith(f"windows/{item['window_index']}/postprocess-") and name.endswith(".json")]
        if not postprocess:
            inconsistent.append({"window_index": item["window_index"], "reason": "device_to_host receipt missing"})
        for name, record in postprocess:
            if (record.get("status") != "PASS" or record.get("sr_request_id") != job_id
                    or record.get("model_request_id") != item["model_request_id"]
                    or record.get("window_index") != item["window_index"]
                    or record.get("actual_model_request_id") not in actual_ids
                    or not measurements_valid(record, {"device_to_host"}, postprocess=True)):
                inconsistent.append({"file": name, "reason": "device_to_host receipt incomplete or mismatched"})
    return {"schema": SCHEMA, "sr_request_id": job_id, "files": files,
            "metrics_status": "PASS_COMPLETE" if settled and files.get("summary.json", {}).get("status") == "completed"
                and bool(windows) and not missing and not inconsistent
                and not files.get("summary.json", {}).get("metrics_errors") else "METRICS_INCOMPLETE",
            "missing_rank_records": missing,
            "inconsistent_rank_records": inconsistent,
            "scope": "API/window and rank JSON only; controller supplies NVML, original-audio copy and final validation"}
