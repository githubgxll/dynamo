# SPDX-License-Identifier: Apache-2.0
"""Source-pinned timing-only delta on the published c88 SeedVR2 helpers.

Historical basic/diagnostic schema /2 remains intact. Timing-only never calls
allocator, peak/reset or NVML functions; CUDA event boundaries retain the
existing stream scope and waits. No model, precision or sampling changes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREFIX = "vllm_omni/diffusion/models/seedvr2/"
CHANGES = tuple(PREFIX + name for name in ("sr_telemetry.py", "long_video_support.py", "long_video.py"))
TIMING_SCHEMA = "sr-timing-only/1"


def _once(source: str, before: str, after: str, count: int = 1) -> str:
    if source.count(before) != count:
        raise ValueError("Unexpected c88 SR timing source anchor: " + repr(before[:90]))
    return source.replace(before, after)


MODE_RECORD = r'''


def _timing_record(record):
    """Mode-discriminated receipts; missing memory is deliberate, never zero."""
    if record.get("measurement_mode") != "timing":
        return record
    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()
                if not key.startswith("allocator") and key != "measurement_source"}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value
    result = clean(record)
    result.update(schema=TIMING_SCHEMA, memory_observation="not_collected",
                  measurement_source=("cpu_clock" if result.get("execution_scope") == "api_cpu_postprocess"
                                      else "cuda_events_and_cpu_clock"))
    return result
'''


TIMING_VALIDATOR = r'''


def _timing_identity(record, expected, keys):
    return isinstance(record, dict) and record.get("schema") == TIMING_SCHEMA and all(
        record.get(key) == expected[key] for key in keys if key in expected)


def _timing_no_memory(value):
    if isinstance(value, dict):
        return (not any(key.startswith("allocator") for key in value)
            and value.get("measurement_source") != "torch_allocator"
            and all(_timing_no_memory(item) for item in value.values()))
    if isinstance(value, list):
        return all(_timing_no_memory(item) for item in value)
    return True


def _timing_sources(record, expected):
    actual = record.get("actual_source_hashes")
    if (not isinstance(actual, dict) or set(actual) != {"sr_telemetry.py", "pipeline_seedvr2.py", "ipc.py"}
            or any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v) for v in actual.values())):
        return False
    # Standalone production timing may omit experiment metadata. When the
    # caller supplies a source lock it remains mandatory and is never relaxed.
    return _v2_sources(record, expected) if "source_hashes" in expected else True


def _timing_stages(record, required, scope, cuda=True):
    stages = record.get("stages")
    if (not isinstance(stages, list) or len(stages) != len(required)
            or any(not isinstance(stage, dict) for stage in stages)
            or {stage.get("phase") for stage in stages} != required):
        return False
    return all(_v2_cpu_interval(stage, "cpu_wall_ms") and stage.get("failure") is None
        and stage.get("execution_scope") == scope
        and (_v2_number(stage.get("cuda_stream_elapsed_ms")) if cuda else stage.get("cuda_stream_elapsed_ms") is None)
        and stage.get("event_scope") in ({"current_cuda_stream", "passed_d2h_stream"} if cuda else {"not_applicable_cpu"})
        and _timing_no_memory(stage) for stage in stages)


def validate_timing_bundle(job_id, files):
    """Correlate real API/window/rank receipts without requiring memory data."""
    errors, missing, expected_files = [], [], set()
    def bad(where, reason):
        errors.append({"file": where, "reason": reason, "scope": "timing_only"})
    request = files.get("request.json")
    if not isinstance(request, dict):
        request = {}
    if (request.get("schema") != TIMING_SCHEMA or request.get("sr_request_id") != job_id
            or request.get("measurement_mode") != "timing" or request.get("expected_world_size") != 8):
        bad("request.json", "timing-only SP8 native request identity is required")
    windows = files.get("window-map.jsonl")
    if not isinstance(windows, list) or not windows:
        bad("window-map.jsonl", "nonempty explicit window map required")
        windows = []
    summary = files.get("summary.json")
    if (not _timing_identity(summary, request, _V2_IDENTITIES)
            or summary.get("status") != "completed" or summary.get("metrics_errors") != []
            or summary.get("window_count") != len(windows)):
        bad("summary.json", "matching completed summary required")
    server = files.get("server-stages.jsonl")
    required_server = {"api_preprocessing", "source_decode", "window_input_encode", "window_model_http",
        "window_output_decode", "delivery_resize", "final_video_encode", "media_validate", "artifact_publish", "server_sr_e2e"}
    if (not isinstance(server, list) or any(not isinstance(row, dict) for row in server)
            or not required_server.issubset({row.get("stage") for row in server if isinstance(row, dict)})):
        bad("server-stages.jsonl", "complete service stages required")
        server = []
    for row in server:
        na = row.get("stage") == "delivery_resize" and row.get("status") == "not_executed_by_design"
        if (not _timing_identity(row, request, tuple(k for k in _V2_IDENTITIES if k != "model_request_id"))
                or row.get("execution_scope") != "api_process" or not _timing_no_memory(row)
                or (not na and not _v2_number(row.get("cpu_wall_ms")))
                or (na and (row.get("cpu_wall_ms") is not None or not row.get("reason")))
                or row.get("status") in {"failed", "cancelled"}):
            bad("server-stages.jsonl", "service identity, timing or no-memory contract differs")
    previous, total_new, model_ids, bindings = None, 0, set(), {}
    for position, item in enumerate(windows):
        if not isinstance(item, dict):
            bad("window-map.jsonl", "window must be an object")
            continue
        index, label = item.get("window_index"), "windows/" + str(item.get("window_index"))
        layout = (type(index) is int and index == position and
            _timing_identity(item, request, tuple(k for k in _V2_IDENTITIES if k != "model_request_id"))
            and all(type(item.get(k)) is int and item[k] >= 0 for k in _V2_WINDOW_FIELDS)
            and item.get("selected_window") in (17, 33, 65)
            and item.get("configured_window") == item.get("selected_window")
            and all(item.get(k) == request.get(k) for k in ("model_width", "model_height", "delivery_width",
                "delivery_height", "configured_window", "selected_window")))
        if layout:
            valid, start, selected = item["input_frame_count"], item["window_start"], item["selected_window"]
            overlap = 0 if position == 0 else 4
            expected_start = 0 if previous is None else previous["window_start"] + previous["input_frame_count"] - 4
            layout = (type(request.get("target_frames")) is int and 0 < valid <= selected and start == expected_start
                and valid == min(selected, request["target_frames"] - start) and item["overlap_frames"] == overlap
                and item["new_frame_count"] == valid - overlap and item["new_frame_count"] > 0
                and item["padded_frame_count"] == valid + (1 - valid) % 4)
        model_id = item.get("model_request_id")
        if not isinstance(model_id, str) or not _TOKEN.fullmatch(model_id) or model_id in model_ids:
            layout = False
        if not layout:
            bad(label, "window geometry, identity, stride or padding differs")
        else:
            previous, total_new = item, total_new + item["new_frame_count"]
            model_ids.add(model_id)
        for name in ("window_input_encode", "window_model_http", "window_output_decode"):
            if len([row for row in server if row.get("stage") == name and row.get("window_index") == index]) != 1:
                bad(label, "one service observation per window required: " + name)
        actual_ids, seen_bindings, seen_gpu, source_sets = set(), set(), set(), set()
        for rank in range(8):
            key = f"{label}/rank-{rank}.json"
            expected_files.add(key)
            record = files.get(key)
            if record is None:
                missing.append({"window_index": index, "rank": rank})
                continue
            if not isinstance(record, dict):
                bad(key, "rank must be an object")
                continue
            actual = record.get("actual_model_request_id")
            if isinstance(actual, str) and actual:
                actual_ids.add(actual)
            if (not _timing_identity(record, item, (*_V2_IDENTITIES, *_V2_WINDOW_FIELDS))
                    or record.get("status") != "PASS" or record.get("errors") != []
                    or record.get("execution_scope") != "sr_rank" or type(record.get("rank")) is not int
                    or record.get("rank") != rank or record.get("world_size") != 8 or not _v2_process(record)
                    or record.get("video_payload_path") != "payload.video" or not actual
                    or not isinstance(record.get("gpu_uuid"), str) or not record.get("gpu_uuid")
                    or not _timing_stages(record, _V2_RANK_PHASES, "sr_rank")
                    or not _v2_number(record.get("pipeline_gpu_window_stream_elapsed_ms"))
                    or record.get("explicit_stage_waits") != 0 or record.get("explicit_window_boundary_waits") != 1
                    or record.get("memory_observation") != "not_collected" or not _timing_no_memory(record)
                    or not _timing_sources(record, item)):
                bad(key, "rank identity, seven stage timings or timing-only policy incomplete")
            if _v2_process(record) and isinstance(record.get("gpu_uuid"), str):
                binding = (record["pid"], record["process_start_time"]["value"], record["gpu_uuid"])
                if rank in bindings and bindings[rank] != binding:
                    bad(key, "rank process/GPU changed across windows")
                bindings.setdefault(rank, binding)
                seen_bindings.add(binding[:2]); seen_gpu.add(binding[2])
            if isinstance(record.get("actual_source_hashes"), dict):
                source_sets.add(json.dumps(record["actual_source_hashes"], sort_keys=True))
        if len(seen_bindings) != 8 or len(seen_gpu) != 8 or len(actual_ids) != 1 or len(source_sets) != 1:
            bad(label, "eight distinct stable workers/GPU and one actual request/source identity required")
        transfers = [name for name in files if name.startswith(label + "/transfer-rank-") and name.endswith(".json")]
        if transfers != [label + "/transfer-rank-0.json"]:
            bad(label, "one real rank0 transfer receipt required")
        for key in transfers:
            expected_files.add(key)
            record = files[key]
            if not isinstance(record, dict):
                bad(key, "transfer must be an object"); continue
            sender = files.get(label + "/rank-0.json", {})
            pack = record.get("framework_output_pack")
            if (not _timing_identity(record, item, (*_V2_IDENTITIES, *_V2_WINDOW_FIELDS))
                    or record.get("status") != "PASS" or record.get("errors") != []
                    or record.get("execution_scope") != "worker_transport" or record.get("rank") != 0
                    or record.get("world_size") != 8 or record.get("source_device_type") != "cuda"
                    or record.get("actual_model_request_id") not in actual_ids or not _v2_process(record)
                    or record.get("observed_copy_count") != 1 or record.get("matched_payload_video") is not True
                    or not _timing_stages(record, {"device_to_host_transport"}, "worker_transport")
                    or not _v2_tensor_identity(record, item) or not _v2_cpu_interval(record, "video_tensor_pack_wall_ms")
                    or any(record.get(k) != sender.get(k) for k in ("pid", "process_start_time", "gpu_uuid"))
                    or record.get("actual_source_hashes") != sender.get("actual_source_hashes")
                    or record.get("explicit_stage_waits") != 0 or record.get("explicit_window_boundary_waits") != 0
                    or record.get("memory_observation") != "not_collected" or not _timing_no_memory(record)
                    or not _timing_sources(record, item) or not isinstance(pack, dict)
                    or not _v2_cpu_interval(pack, "cpu_wall_ms")
                    or record.get("framework_output_pack_wall_ms") != pack.get("cpu_wall_ms")
                    or pack.get("cpu_start_ns", 1) > record.get("cpu_start_ns", 0)
                    or pack.get("cpu_end_ns", 0) < record.get("cpu_end_ns", 1)):
                bad(key, "actual transfer timing or source/CPU envelope differs")
        cpu_keys = [name for name in files if name.startswith(label + "/postprocess-") and name.endswith(".json")]
        if len(cpu_keys) != 1:
            bad(label, "one API CPU postprocess receipt required")
        for key in cpu_keys:
            expected_files.add(key)
            record = files[key]
            sender = files.get(label + "/rank-0.json", {})
            if (not isinstance(record, dict) or not _timing_identity(record, item, (*_V2_IDENTITIES, *_V2_WINDOW_FIELDS))
                    or record.get("status") != "PASS" or record.get("errors") != []
                    or record.get("execution_scope") != "api_cpu_postprocess" or record.get("source_device_type") != "cpu"
                    or record.get("actual_model_request_id") not in actual_ids or not _v2_process(record)
                    or not _timing_stages(record, {"cpu_video_numpy"}, "api_cpu_postprocess", cuda=False)
                    or not _v2_cpu_interval(record, "cpu_postprocess_wall_ms")
                    or record.get("memory_observation") != "not_collected" or not _timing_no_memory(record)
                    or record.get("actual_source_hashes") != sender.get("actual_source_hashes")
                    or not _timing_sources(record, item)):
                bad(key, "CPU postprocess timing or source identity differs")
    if previous is not None and (total_new != request.get("target_frames")
            or previous["window_start"] + previous["input_frame_count"] != request.get("target_frames")):
        bad("window-map.jsonl", "windows do not cover exact source frame count")
    for name in files:
        if name.startswith("windows/") and name not in expected_files:
            bad(name, "unexpected/duplicate timing receipt")
    return {"schema": TIMING_SCHEMA, "measurement_mode": "timing", "memory_observation": "not_collected",
        "sr_request_id": job_id, "files": files,
        "metrics_status": "PASS_COMPLETE" if windows and not missing and not errors else "METRICS_INCOMPLETE",
        "missing_rank_records": missing, "inconsistent_rank_records": errors,
        "scope": "API CPU stages/current-stream GPU events/actual worker transfer; no allocator or NVML observations; nested timings are not additive"}


_legacy_validate_bundle = validate_bundle


def validate_bundle(job_id, files):
    request = files.get("request.json") if isinstance(files, dict) else None
    if isinstance(request, dict) and (request.get("measurement_mode") == "timing" or request.get("schema") == TIMING_SCHEMA):
        return validate_timing_bundle(job_id, files)
    return _legacy_validate_bundle(job_id, files)
'''


def _telemetry(source: str) -> str:
    source = _once(source, 'SCHEMA = "sr-timing-memory/2"\n',
        'SCHEMA = "sr-timing-memory/2"\nTIMING_SCHEMA = "sr-timing-only/1"\n')
    source = _once(source,
        '    if not isinstance(value, dict) or value.get("schema", SCHEMA) != SCHEMA:\n',
        '    expected_schema = TIMING_SCHEMA if isinstance(value, dict) and value.get("measurement_mode") == "timing" else SCHEMA\n'
        '    if not isinstance(value, dict) or value.get("schema", expected_schema) != expected_schema:\n')
    source = _once(source, '("off", "basic", "diagnostic")', '("off", "basic", "diagnostic", "timing")')
    source = _once(source, '    return {"schema": SCHEMA, **result}', '    return {"schema": expected_schema, **result}')
    source = _once(source, '        self.reset_allocator_peak = reset_allocator_peak\n',
        '        self.collect_memory = self.context["measurement_mode"] != "timing"\n'
        '        self.reset_allocator_peak = reset_allocator_peak and self.collect_memory\n')
    source = _once(source, '    def allocator(self):\n        return ',
        '    def allocator(self):\n        if not self.collect_memory:\n            return None\n        return ')
    before = 'peak = {"allocated_bytes": int(self.cuda.max_memory_allocated(self.device)), "reserved_bytes": int(self.cuda.max_memory_reserved(self.device))}'
    source = _once(source, before, before + ' if self.collect_memory else None', count=2)
    for before, after in (
        ('atomic_json(output_directory(self.context) / filename, record)', 'atomic_json(output_directory(self.context) / filename, _timing_record(record))'),
        ('unique_json(output_directory(self.context) / f"postprocess-{os.getpid()}.json", value)', 'unique_json(output_directory(self.context) / f"postprocess-{os.getpid()}.json", _timing_record(value))'),
        ('unique_json(output_directory(context) / f"transfer-rank-{value[\'rank\']}.json", value)', 'unique_json(output_directory(context) / f"transfer-rank-{value[\'rank\']}.json", _timing_record(value))'),
        ('self.holder["observations"].append(value)', 'self.holder["observations"].append(_timing_record(value))'),
        ('unique_json(output_directory(self.context) / f"transfer-rank-{self.rank}.json", value)', 'unique_json(output_directory(self.context) / f"transfer-rank-{self.rank}.json", _timing_record(value))')):
        source = _once(source, before, after)
    return source + MODE_RECORD


def _support(source: str) -> str:
    source = _once(source, 'SCHEMA = "sr-timing-memory/2"\n',
        'SCHEMA = "sr-timing-memory/2"\nTIMING_SCHEMA = "sr-timing-only/1"\n\n'
        'def measurement_schema(mode):\n    return TIMING_SCHEMA if mode == "timing" else SCHEMA\n')
    source = _once(source, '{"off", "basic", "diagnostic"}', '{"off", "basic", "diagnostic", "timing"}')
    source = _once(source, 'measurement_mode must be off, basic, or diagnostic', 'measurement_mode must be off, basic, diagnostic, or timing')
    source = _once(source, 'self.identity = {**config["measurement_context"], "schema": SCHEMA,',
        'self.identity = {**config["measurement_context"], "schema": measurement_schema(config["measurement_mode"]),')
    source = _once(source, '        atomic_json(self.root / "request.json", self.identity)\n',
        '        atomic_json(self.root / "request.json", self.identity)\n'
        '        if self.config["measurement_mode"] == "timing":\n'
        '            self.row("api_preprocessing", self.config["api_preprocessing_start_ns"],\n'
        '                self.config["api_preprocessing_end_ns"], scope="HTTP option parsing, input storage and source validation; excludes queue and GPU")\n')
    return source + TIMING_VALIDATOR


def _long_video(source: str) -> str:
    source = _once(source, '    try:\n        optimization = _support.options(',
        '    timing_api_started_ns = time.perf_counter_ns() if measurement_mode == "timing" else None\n'
        '    try:\n        optimization = _support.options(')
    source = _once(source, '            _support.atomic_json(job / "optimization.json", optimization)\n',
        '            if measurement_mode == "timing":\n'
        '                optimization["server_started_ns"] = timing_api_started_ns\n'
        '                optimization["api_preprocessing_start_ns"] = timing_api_started_ns\n'
        '                optimization["api_preprocessing_end_ns"] = time.perf_counter_ns()\n'
        '            _support.atomic_json(job / "optimization.json", optimization)\n')
    return _once(source, '"measurement_schema": _support.SCHEMA}', '"measurement_schema": _support.measurement_schema(measurement_mode)}')


def planned_patch(relative: str, data: bytes) -> bytes:
    if relative not in CHANGES:
        raise ValueError("Unknown SR timing patch target: " + relative)
    source = data.decode("utf-8")
    function = {CHANGES[0]: _telemetry, CHANGES[1]: _support, CHANGES[2]: _long_video}[relative]
    result = function(source)
    compile(result, relative, "exec")
    return result.encode("utf-8")


def transform(relative: str, data: bytes) -> bytes:
    lock = json.loads((HERE / "sr-timing-patches.lock.json").read_text(encoding="utf-8"))
    entry = lock["changes"].get(relative)
    if not entry or hashlib.sha256(data).hexdigest() != entry["before_sha256"]:
        raise ValueError("SR timing patch input differs from c88: " + relative)
    result = planned_patch(relative, data)
    if hashlib.sha256(result).hexdigest() != entry["after_sha256"]:
        raise ValueError("SR timing patch output differs from reviewed source: " + relative)
    return result
