# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-level performance dimensions, extracted without file or store I/O."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from dingo.common.video_timing_schemas import (
    MODEL_EXECUTION_KEY,
    normalize_model_execution,
)
from dingo.video_gateway.models import VideoTask


def _number(value: Any) -> float | int | None:
    if type(value) in (int, float):
        try:
            if math.isfinite(value) and value >= 0:
                return value
        except OverflowError:
            pass
    return None


def performance_record(task: VideoTask) -> dict[str, Any]:
    """Versioned summary; unavailable dimensions remain null, not invented zeroes."""
    request = task.normalized_request
    raw = request.get("reference_metadata")
    references = []
    if isinstance(raw, list):
        for item in raw[:32]:
            if not isinstance(item, Mapping):
                continue
            references.append(
                {
                    "kind": item.get("kind")
                    if item.get("kind") in {"image", "video", "audio"}
                    else "unknown",
                    **{
                        key: _number(item.get(key))
                        for key in (
                            "ordinal",
                            "width",
                            "height",
                            "fps",
                            "duration_s",
                            "bytes",
                            "frame_count",
                            "used_frame_count",
                        )
                    },
                }
            )
    kinds = request.get("reference_kinds")
    if not isinstance(kinds, list):
        kinds = [item["kind"] for item in references]
    counts = {kind: kinds.count(kind) for kind in ("image", "video", "audio")}

    def total(kind: str, key: str):
        items = [r for r in references if r["kind"] == kind]
        values = [r.get(key) for r in items]
        if len(items) != counts[kind] or any(v is None for v in values):
            return None
        return sum(values)

    images = [r for r in references if r["kind"] == "image"]
    pixels = [
        r["width"] * r["height"]
        if r["width"] is not None and r["height"] is not None
        else None
        for r in images
    ]
    image_pixels = (
        sum(pixels)
        if len(images) == counts["image"] and all(v is not None for v in pixels)
        else None
    )
    stages = task.stage_durations or {}
    return {
        "schema_version": 1,
        "configuration_revision": task.configuration_revision,
        "generation": {
            key: request.get(key)
            for key in (
                "seconds",
                "width",
                "height",
                "fps",
                "num_frames",
                "num_inference_steps",
                "generate_sound",
            )
        },
        "input_type": "+".join(kind for kind, count in counts.items() if count)
        or "text",
        "reference_count": len(kinds),
        "counts": counts,
        "image_total_pixels": image_pixels,
        "image_total_bytes": total("image", "bytes"),
        "video_total_seconds": total("video", "duration_s"),
        "video_total_bytes": total("video", "bytes"),
        "video_total_frames": total("video", "frame_count"),
        "audio_total_seconds": total("audio", "duration_s"),
        "audio_total_bytes": total("audio", "bytes"),
        "references": references,
        "timings": {
            "end_to_end_s": max(0, (task.completed_at_ms - task.created_at_ms) / 1000)
            if task.completed_at_ms is not None
            else None,
            "queue_s": _number(task.queue_wait_s),
            "worker_queue_s": _number(stages.get("worker_queue_wait")),
            "execution_s": _number(task.inference_time_s),
            "finalize_s": _number(task.finalize_time_s),
        },
        "model_execution": normalize_model_execution(request.get(MODEL_EXECUTION_KEY)),
    }
