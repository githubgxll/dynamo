"""Bounded video-only SR telemetry; missing measurements never become zeroes."""

from __future__ import annotations

import math
from collections.abc import Mapping

SR_TIMINGS_KEY = "_super_resolution_timings_v1"
STAGES = (
    "service", "source_decode", "window_input_encode", "window_model_wait",
    "window_output_decode", "overlap_blend", "delivery_crop",
    "delivery_transform", "final_video_encode", "audio_mux", "media_validate",
    "artifact_publish", "result_download", "result_finalize", "postprocess",
)


def _number(value):
    if type(value) not in (int, float):
        return None
    try:
        numeric = float(value)
    except (OverflowError, ValueError):
        return None
    return numeric if math.isfinite(numeric) and numeric >= 0 else None


def normalize_sr_timings(value):
    if (not isinstance(value, Mapping)
            or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1 or value.get("unit") != "seconds"):
        return None
    raw = value.get("stages")
    if not isinstance(raw, Mapping) or len(raw) > 32:
        return None
    stages = {name: number for name in STAGES
              if (number := _number(raw.get(name))) is not None}
    return {"schema_version": 1, "unit": "seconds", "stages": stages} if stages else None


def with_sr_timings(request, value):
    result = dict(request)
    result.pop(SR_TIMINGS_KEY, None)
    timing = normalize_sr_timings(value)
    if timing is not None:
        result[SR_TIMINGS_KEY] = timing
    return result


def aggregate_sr_timings(metrics, worker_stages):
    """Aggregate one request, not per-window/rank histogram observations.

    HTTP intervals use one API process clock. Activity totals deliberately avoid
    first-to-last envelopes, which include time waiting for other pipeline work.
    """
    stages = {}
    for name, source in (("service", "sr_service_s"),
                         ("result_download", "sr_download_s"),
                         ("result_finalize", "sr_finalize_s"),
                         ("postprocess", "sr_postprocess_total_s")):
        value = _number(worker_stages.get(source))
        if value is not None:
            stages[name] = value
    if isinstance(metrics, Mapping) and metrics.get("schema") == "sr-timing-memory/2" and metrics.get("metrics_status") == "PASS_COMPLETE":
        files = metrics.get("files")
        if isinstance(files, Mapping):
            _aggregate_native(files, stages)
    return normalize_sr_timings({"schema_version": 1, "unit": "seconds", "stages": stages})


def _aggregate_native(files, stages):
    request, summary = files.get("request.json"), files.get("summary.json")
    rows = files.get("server-stages.jsonl")
    if not isinstance(request, Mapping) or not isinstance(summary, Mapping) or not isinstance(rows, list) or len(rows) > 10000:
        return
    count = summary.get("window_count")
    if type(count) is not int or not 1 <= count <= 512 or summary.get("status") != "completed":
        return
    request_id = request.get("sr_request_id")
    if not isinstance(request_id, str) or not request_id or summary.get("sr_request_id") != request_id:
        return
    groups = {}
    for row in rows:
        if not isinstance(row, Mapping):
            return
        name = row.get("stage")
        if isinstance(name, str):
            groups.setdefault(name, []).append(row)

    def valid(row):
        return (row.get("sr_request_id") == request_id
                and row.get("status") in (None, "completed")
                and row.get("failed_call_count", 0) == 0)

    for target, source in (("source_decode", "source_decode"),
                           ("overlap_blend", "overlap_blend"),
                           ("delivery_crop", "delivery_crop"),
                           ("delivery_transform", "delivery_resize"),
                           ("final_video_encode", "final_video_encode"),
                           ("audio_mux", "audio_mux"),
                           ("media_validate", "media_validate"),
                           ("artifact_publish", "artifact_publish")):
        matches = groups.get(source, [])
        if len(matches) != 1 or not valid(matches[0]):
            continue
        row = matches[0]
        key = "activity_cpu_wall_ms" if source in {"source_decode", "overlap_blend", "delivery_crop", "delivery_resize", "final_video_encode"} else "cpu_wall_ms"
        value = _number(row.get(key))
        if value is not None:
            stages[target] = value / 1000

    for target, source in (("window_input_encode", "window_input_encode"),
                           ("window_output_decode", "window_output_decode"),
                           ("window_model_wait", "window_model_http")):
        matches = groups.get(source, [])
        if (len(matches) != count or any(not valid(row) for row in matches)
                or any(type(row.get("window_index")) is not int for row in matches)
                or {row["window_index"] for row in matches} != set(range(count))):
            continue
        if target != "window_model_wait":
            values = [_number(row.get("cpu_wall_ms")) for row in matches]
            if all(value is not None for value in values) and math.isfinite(sum(values)):
                stages[target] = sum(values) / 1000
            continue
        clocks = [row.get("clock_domain") for row in matches]
        if not all(isinstance(clock, str) and clock for clock in clocks) or len(set(clocks)) != 1:
            continue
        intervals = []
        for row in matches:
            start, end = row.get("start_ns"), row.get("end_ns")
            if type(start) is not int or type(end) is not int or start < 0 or end < start or end > 2**63 - 1:
                break
            intervals.append((start, end))
        else:
            intervals.sort()
            total, left, right = 0, *intervals[0]
            for start, end in intervals[1:]:
                if start > right:
                    total += right - left
                    left, right = start, end
                else:
                    right = max(right, end)
            seconds = _number((total + right - left) / 1e9)
            if seconds is not None:
                stages[target] = seconds
