# SPDX-License-Identifier: Apache-2.0
"""Exact helper delta: distinguish real worker D2H from CPU API postprocessing."""
from __future__ import annotations

import hashlib

BEFORE_SHA256 = "db14071e1e8e41ba98bb40842b70e104dd580d91ceb45cb36a33fc24b038d786"

VALIDATOR = r'''

_V2_WINDOW_FIELDS = (
    "window_index", "window_start", "input_frame_count", "padded_frame_count",
    "new_frame_count", "overlap_frames", "configured_window", "selected_window",
    "model_width", "model_height", "delivery_width", "delivery_height",
)
_V2_IDENTITIES = (
    "sr_request_id", "model_request_id", "measurement_mode", "run_id", "batch_id",
    "case_id", "action_id", "phase", "image_digest", "source_hashes", "model_manifest_hash",
)
_V2_RANK_PHASES = {
    "prepare_video_gpu", "vae_encode", "latent_prepare", "dit", "solver", "vae_decode", "color_uint8",
}


def _v2_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _v2_allocator(value):
    return isinstance(value, dict) and all(type(value.get(key)) is int and value[key] >= 0
        for key in ("allocated_bytes", "reserved_bytes"))


def _v2_identity(value, expected, keys):
    return isinstance(value, dict) and value.get("schema") == SCHEMA and all(
        value.get(key) == expected[key] for key in keys if key in expected)


def _v2_process(record):
    return (type(record.get("pid")) is int and record["pid"] > 0
        and isinstance(record.get("process_start_time"), dict)
        and record["process_start_time"].get("clock_domain") == "linux_boot_ticks"
        and type(record["process_start_time"].get("value")) is int
        and record["process_start_time"]["value"] >= 0)


def _v2_sources(record, expected):
    actual = record.get("actual_source_hashes")
    critical = {"sr_telemetry.py", "pipeline_seedvr2.py", "ipc.py"}
    if not isinstance(actual, dict) or set(actual) != critical or any(
            not isinstance(name, str) or not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
            for name, value in actual.items()):
        return False
    declared = expected.get("source_hashes")
    if not isinstance(declared, dict) or not declared or any(
            not isinstance(name, str) or not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
            for name, value in declared.items()):
        return False
    for name, value in actual.items():
        matches = [candidate for key, candidate in declared.items() if key == name or key.endswith("/" + name)]
        if len(matches) != 1 or matches[0] != value:
            return False
    return True


def _v2_gpu_stages(record, phases, scope):
    stages = record.get("stages")
    if not isinstance(stages, list) or len(stages) != len(phases) or any(not isinstance(s, dict) for s in stages):
        return False
    if {s.get("phase") for s in stages} != phases:
        return False
    return all(_v2_number(s.get("cpu_wall_ms")) and _v2_number(s.get("cuda_stream_elapsed_ms"))
        and s.get("failure") is None and s.get("execution_scope") == scope
        and s.get("event_scope") in ({"current_cuda_stream"} if scope == "sr_rank" else {"current_cuda_stream", "passed_d2h_stream"})
        and _v2_allocator(s.get("allocator_before")) and _v2_allocator(s.get("allocator_after"))
        and _v2_cpu_interval(s, "cpu_wall_ms")
        for s in stages)


def _v2_cpu_interval(record, wall_field):
    start, end = record.get("cpu_start_ns"), record.get("cpu_end_ns")
    return (type(start) is int and type(end) is int and end >= start
        and _v2_number(record.get(wall_field))
        and math.isclose(record[wall_field], (end - start) / 1e6, rel_tol=1e-9, abs_tol=1e-6))


def _v2_tensor_identity(record, item):
    tensor = record.get("tensor_identity")
    if not isinstance(tensor, dict):
        return False
    shape = [1, item.get("input_frame_count"), item.get("model_height"), item.get("model_width"), 3]
    if not all(type(value) is int and value > 0 for value in shape):
        return False
    if tensor.get("shape") != shape or tensor.get("dtype") != "torch.uint8" or tensor.get("element_size") != 1:
        return False
    stride = tensor.get("stride")
    return (isinstance(stride, list) and len(stride) == 5 and all(type(v) is int and v >= 0 for v in stride)
        and type(tensor.get("numel")) is int and tensor["numel"] == math.prod(shape)
        and type(tensor.get("storage_offset")) is int and tensor["storage_offset"] >= 0
        and all(type(tensor.get(k)) is int and tensor[k] > 0
                for k in ("python_id", "data_ptr", "storage_data_ptr"))
        and tensor.get("source_device") == record.get("source_device")
        and isinstance(record.get("source_device"), str) and re.fullmatch(r"cuda:[0-9]+", record["source_device"]))


def validate_bundle(job_id, files):
    """Validate schema /2 observations. Never upgrade old evidence or infer missing timings."""
    errors, missing = [], []
    def bad(where, reason, scope="identity"):
        errors.append({"file": where, "reason": reason, "scope": scope})
    request = files.get("request.json")
    if not isinstance(request, dict) or request.get("schema") != SCHEMA or request.get("sr_request_id") != job_id:
        bad("request.json", "schema /2 and exact native request identity are required")
        request = request if isinstance(request, dict) else {}
    if request.get("expected_world_size") != 8 or request.get("measurement_mode") not in {"basic", "diagnostic"}:
        bad("request.json", "SP8 measured request contract is incomplete")
    windows = files.get("window-map.jsonl")
    if not isinstance(windows, list) or not windows:
        bad("window-map.jsonl", "nonempty explicit window map is required")
        windows = []
    summary = files.get("summary.json")
    if (not isinstance(summary, dict) or not _v2_identity(summary, request, _V2_IDENTITIES)
            or summary.get("status") != "completed" or summary.get("metrics_errors")
            or summary.get("window_count") != len(windows)):
        bad("summary.json", "completed matching request summary and exact window count are required")
    server = files.get("server-stages.jsonl")
    required_server = {"source_decode", "window_input_encode", "window_model_http", "window_output_decode",
        "delivery_resize", "final_video_encode", "media_validate", "artifact_publish", "server_sr_e2e"}
    if (not isinstance(server, list) or not server or any(not isinstance(row, dict) for row in server)
            or not required_server.issubset({row.get("stage") for row in server if isinstance(row, dict)})):
        bad("server-stages.jsonl", "complete service stage observations are required", "server_stages")
    else:
        for row in server:
            permitted_na = row.get("stage") == "delivery_resize" and row.get("status") == "not_executed_by_design"
            if (not _v2_identity(row, request, tuple(k for k in _V2_IDENTITIES if k != "model_request_id"))
                    or row.get("execution_scope") != "api_process"
                    or (not permitted_na and not _v2_number(row.get("cpu_wall_ms")))
                    or (permitted_na and (row.get("cpu_wall_ms") is not None or not row.get("reason")))
                    or row.get("status") in {"failed", "cancelled"}):
                bad("server-stages.jsonl", "service stage has mismatched identity, scope or missing timing", "server_stages")
    seen_model_ids, expected_files, stable_rank_bindings = set(), set(), {}
    previous, total_new = None, 0
    for position, item in enumerate(windows):
        if not isinstance(item, dict):
            bad("window-map.jsonl", "window entry is not an object")
            continue
        index = item.get("window_index")
        label = f"windows/{index}"
        valid_layout = (type(index) is int and index == position and
            _v2_identity(item, request, tuple(k for k in _V2_IDENTITIES if k != "model_request_id"))
            and all(type(item.get(k)) is int and item[k] >= 0 for k in _V2_WINDOW_FIELDS)
            and item.get("selected_window") in (17, 33, 65)
            and item.get("configured_window") == item.get("selected_window")
            and all(item.get(k) == request.get(k) for k in
                ("model_width", "model_height", "delivery_width", "delivery_height", "selected_window", "configured_window")))
        if valid_layout:
            valid, start, selected = item["input_frame_count"], item["window_start"], item["selected_window"]
            overlap = 0 if position == 0 else 4
            expected_start = 0 if previous is None else previous["window_start"] + previous["input_frame_count"] - 4
            valid_layout = (valid > 0 and valid <= selected and start == expected_start
                and type(request.get("target_frames")) is int
                and valid == min(selected, request["target_frames"] - start)
                and item["overlap_frames"] == overlap and item["new_frame_count"] == valid - overlap
                and item["new_frame_count"] > 0 and item["padded_frame_count"] == valid + (1 - valid) % 4)
        model_id = item.get("model_request_id")
        if not isinstance(model_id, str) or not _TOKEN.fullmatch(model_id) or model_id in seen_model_ids:
            valid_layout = False
        if not valid_layout:
            bad(label, "window identity, geometry, stride, padding or requested window is inconsistent")
        else:
            total_new += item["new_frame_count"]
            previous = item
            seen_model_ids.add(model_id)
        for phase in ("window_input_encode", "window_model_http", "window_output_decode"):
            matches = [row for row in (server if isinstance(server, list) else []) if isinstance(row, dict)
                and row.get("stage") == phase and row.get("window_index") == index]
            if len(matches) != 1:
                bad(label, "one service observation per window is required for " + phase, "server_stages")
        actual_ids, rank_bindings, rank_gpu_ids = set(), [], []
        for rank in range(8):
            key = f"{label}/rank-{rank}.json"
            expected_files.add(key)
            record = files.get(key)
            if record is None:
                missing.append({"window_index": index, "rank": rank})
                continue
            if not isinstance(record, dict):
                bad(key, "rank receipt is not an object")
                continue
            actual = record.get("actual_model_request_id")
            if isinstance(actual, str) and actual:
                actual_ids.add(actual)
            if (not _v2_identity(record, item, (*_V2_IDENTITIES, *_V2_WINDOW_FIELDS))
                    or record.get("status") != "PASS" or record.get("errors") != []
                    or record.get("execution_scope") != "sr_rank" or record.get("rank") != rank
                    or record.get("world_size") != 8 or not _v2_process(record)
                    or record.get("video_payload_path") != "payload.video"
                    or record.get("clock_domain") != "process_perf_counter_ns"
                    or not isinstance(record.get("gpu_uuid"), str) or not record["gpu_uuid"]
                    or not isinstance(actual, str) or not actual
                    or not _v2_gpu_stages(record, _V2_RANK_PHASES, "sr_rank")
                    or not all(_v2_allocator(record.get(name)) for name in
                        ("allocator_before", "allocator_after", "allocator_window_peak"))
                    or record.get("allocator_peak_scope") != "window_reset"
                    or not _v2_number(record.get("pipeline_gpu_window_stream_elapsed_ms"))
                    or not _v2_sources(record, item)
                    or (item.get("measurement_mode") == "basic" and record.get("explicit_stage_waits") != 0)):
                bad(key, "rank schema, correlated identities, process, seven GPU stages or allocator evidence is incomplete", "rank_pipeline")
            if _v2_process(record):
                rank_bindings.append((record["pid"], record["process_start_time"]["value"]))
                if isinstance(record.get("gpu_uuid"), str) and record["gpu_uuid"]:
                    binding = (record["pid"], record["process_start_time"]["value"], record["gpu_uuid"])
                    if rank in stable_rank_bindings and stable_rank_bindings[rank] != binding:
                        bad(key, "rank process start/PID/GPU binding changed across windows", "rank_pipeline")
                    else:
                        stable_rank_bindings.setdefault(rank, binding)
            if isinstance(record.get("gpu_uuid"), str):
                rank_gpu_ids.append(record["gpu_uuid"])
        if len(set(rank_bindings)) != 8 or len(set(rank_gpu_ids)) != 8:
            bad(label, "SP8 requires eight distinct SR process and GPU bindings", "rank_pipeline")
        if len(actual_ids) != 1:
            bad(label, "eight ranks must agree on one actual model request identity")
        # The transfer sender and CPU consumer are distinct observations. A
        # CPU .numpy() receipt can never stand in for the actual worker copy.
        transfer_keys = [name for name in files if name.startswith(label + "/transfer-rank-") and name.endswith(".json")]
        if transfer_keys != [f"{label}/transfer-rank-0.json"]:
            bad(label, "exactly one actual worker rank0 video transfer receipt is required", "actual_d2h")
            bad(label, "unique worker framework pack observation is missing or duplicated", "framework_output_pack")
        for key in transfer_keys:
            expected_files.add(key)
            record = files[key]
            if (not isinstance(record, dict) or not _v2_identity(record, item, (*_V2_IDENTITIES, *_V2_WINDOW_FIELDS))
                    or record.get("status") != "PASS" or record.get("errors") != []
                    or record.get("execution_scope") != "worker_transport" or record.get("rank") != 0
                    or record.get("world_size") != 8 or record.get("source_device_type") != "cuda"
                    or record.get("video_payload_path") != "payload.video"
                    or record.get("clock_domain") != "process_perf_counter_ns"
                    or record.get("observed_copy_count") != 1 or record.get("matched_payload_video") is not True
                    or record.get("actual_model_request_id") not in actual_ids or not _v2_process(record)
                    or record.get("transport_mode") not in {"async_side_stream", "sync_current_stream"}
                    or not _v2_number(record.get("video_tensor_pack_wall_ms"))
                    or not _v2_gpu_stages(record, {"device_to_host_transport"}, "worker_transport")
                    or not _v2_allocator(record.get("allocator_before")) or not _v2_allocator(record.get("allocator_after"))
                    or record.get("allocator_window_peak") is not None
                    or not _v2_allocator(record.get("allocator_process_peak_observed"))
                    or record.get("allocator_peak_scope") != "process_since_last_external_reset_not_D2H_peak"
                    or record.get("cuda_status") != "applicable" or record.get("allocator_status") != "applicable"
                    or not _v2_tensor_identity(record, item)
                    or record.get("explicit_stage_waits") != 0 or record.get("explicit_window_boundary_waits") != 0
                    or record.get("copy_kind") not in {"pinned_cpu", "registered_shm", "synchronous_cpu"}
                    or record.get("stages", [{}])[0].get("event_scope") != (
                        "passed_d2h_stream" if record.get("transport_mode") == "async_side_stream" else "current_cuda_stream")
                    or record.get("stages", [{}])[0].get("cpu_start_ns", -1) < record.get("cpu_start_ns", 0)
                    or record.get("stages", [{}])[0].get("cpu_end_ns", 1) > record.get("cpu_end_ns", 0)
                    or not _v2_sources(record, item)):
                bad(key, "actual worker CUDA transfer receipt is missing, mismatched or incomplete", "actual_d2h")
            if isinstance(record, dict):
                sender = files.get(f"{label}/rank-0.json", {})
                if any(record.get(k) != sender.get(k) for k in ("pid", "process_start_time", "gpu_uuid")):
                    bad(key, "transfer sender does not match the actual rank0 process/GPU", "actual_d2h")
                pack = record.get("framework_output_pack")
                if (not isinstance(pack, dict) or pack.get("phase") != "framework_output_pack"
                        or pack.get("execution_scope") != "worker_transport" or pack.get("event_scope") != "cpu_wall_envelope"
                        or not _v2_cpu_interval(pack, "cpu_wall_ms")
                        or record.get("framework_output_pack_wall_ms") != pack.get("cpu_wall_ms")
                        or not _v2_cpu_interval(record, "video_tensor_pack_wall_ms")
                        or pack.get("cpu_start_ns", 1) > record.get("cpu_start_ns", 0)
                        or pack.get("cpu_end_ns", 0) < record.get("cpu_end_ns", 1)):
                    bad(key, "framework output pack CPU envelope is missing or does not enclose tensor packing", "framework_output_pack")
        cpu_keys = [name for name in files if name.startswith(label + "/postprocess-") and name.endswith(".json")]
        if len(cpu_keys) != 1:
            bad(label, "exactly one correlated API CPU postprocess receipt is required", "api_cpu_postprocess")
        for key in cpu_keys:
            expected_files.add(key)
            record = files[key]
            stages = record.get("stages") if isinstance(record, dict) else None
            upstream = files.get(f"{label}/transfer-rank-0.json")
            if (not isinstance(record, dict) or not _v2_identity(record, item, (*_V2_IDENTITIES, *_V2_WINDOW_FIELDS))
                    or record.get("status") != "PASS" or record.get("errors") != []
                    or record.get("execution_scope") != "api_cpu_postprocess" or record.get("source_device_type") != "cpu"
                    or record.get("clock_domain") != "process_perf_counter_ns"
                    or record.get("video_payload_path") != "payload.video" or record.get("source_device") != "cpu"
                    or record.get("local_d2h") != "not_executed_here"
                    or record.get("upstream_transfer_receipt") != "transfer-rank-0.json"
                    or not isinstance(upstream, dict) or not _v2_identity(upstream, item, (*_V2_IDENTITIES, *_V2_WINDOW_FIELDS))
                    or upstream.get("actual_model_request_id") != record.get("actual_model_request_id")
                    or upstream.get("video_payload_path") != "payload.video"
                    or record.get("actual_model_request_id") not in actual_ids or not _v2_process(record)
                    or record.get("cuda_status") != "not_applicable" or record.get("allocator_status") != "not_applicable"
                    or any(name not in record for name in
                        ("rank", "world_size", "gpu_uuid", "allocator_before", "allocator_after", "allocator_window_peak",
                         "allocator_process_peak_observed", "cuda_stream_elapsed_ms"))
                    or record.get("cuda_stream_elapsed_ms") is not None
                    or any(record.get(name) is not None for name in
                        ("rank", "world_size", "gpu_uuid", "allocator_before", "allocator_after", "allocator_window_peak",
                         "allocator_process_peak_observed", "pipeline_gpu_window_stream_elapsed_ms"))
                    or not isinstance(stages, list) or len(stages) != 1
                    or not isinstance(stages[0], dict) or stages[0].get("phase") != "cpu_video_numpy"
                    or stages[0].get("execution_scope") != "api_cpu_postprocess"
                    or not _v2_number(stages[0].get("cpu_wall_ms"))
                    or stages[0].get("cuda_stream_elapsed_ms") is not None
                    or stages[0].get("event_scope") != "not_applicable_cpu" or stages[0].get("failure") is not None
                    or stages[0].get("cuda_status") != "not_applicable"
                    or stages[0].get("allocator_status") != "not_applicable"
                    or any(name not in stages[0] for name in ("allocator_before", "allocator_after", "cuda_stream_elapsed_ms"))
                    or any(stages[0].get(name) is not None for name in ("allocator_before", "allocator_after"))
                    or not _v2_cpu_interval(record, "cpu_postprocess_wall_ms")
                    or not _v2_cpu_interval(stages[0], "cpu_wall_ms")
                    or stages[0].get("cpu_start_ns", -1) < record.get("cpu_start_ns", 0)
                    or stages[0].get("cpu_end_ns", 1) > record.get("cpu_end_ns", 0)
                    or not _v2_sources(record, item)):
                bad(key, "CPU postprocess requires genuine CPU wall time and explicit CUDA/allocator not-applicable fields", "api_cpu_postprocess")
    if previous is not None and (total_new != request.get("target_frames")
            or previous["window_start"] + previous["input_frame_count"] != request.get("target_frames")):
        bad("window-map.jsonl", "windows do not cover the exact source frame count")
    extras = [name for name in files if name.startswith("windows/") and name not in expected_files]
    for name in extras:
        bad(name, "unexpected or duplicate window receipt")
    scope_names = ("rank_pipeline", "actual_d2h", "framework_output_pack", "api_cpu_postprocess", "server_stages")
    scope_flags = {name: "PASS_COMPLETE" if not any(error["scope"] in {"identity", name} for error in errors)
        and (name != "rank_pipeline" or not missing) else "METRICS_INCOMPLETE" for name in scope_names}
    return {"schema": SCHEMA, "sr_request_id": job_id, "files": files,
        "metrics_status": "PASS_COMPLETE" if windows and not missing and not errors else "METRICS_INCOMPLETE",
        "missing_rank_records": missing, "inconsistent_rank_records": errors,
        "scope_completeness": scope_flags,
        "scope": "SP8 GPU stages, actual worker CUDA-to-host transfer, separate API CPU postprocess; external NVML/controller stages remain separate"}
'''


def transform(data):
    if hashlib.sha256(data).hexdigest() != BEFORE_SHA256:
        raise ValueError("SeedVR2 metrics repair support parent bytes differ")
    source = data.decode("utf-8")
    marker = '    windows = files.get("window-map.jsonl", [])\n'
    if source.count(marker) != 1 or source.count('SCHEMA = "sr-timing-memory/1"') != 1:
        raise ValueError("SeedVR2 support source anchors differ")
    source = source.replace('SCHEMA = "sr-timing-memory/1"', 'SCHEMA = "sr-timing-memory/2"', 1)
    source = source[:source.index(marker)] + '    return validate_bundle(job_id, files)\n' + VALIDATOR
    compile(source, "long_video_support.py", "exec")
    return source.encode("utf-8")
