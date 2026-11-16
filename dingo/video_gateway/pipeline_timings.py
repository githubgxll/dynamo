# SPDX-License-Identifier: Apache-2.0
"""Public request timing from measured pipeline stages; never invent zeroes."""
import math


def measured(value):
    if type(value) not in (int, float):
        return None
    try:
        numeric = float(value)
    except (OverflowError, ValueError):
        return None
    return numeric if math.isfinite(numeric) and numeric >= 0 else None


def complete_sum(values):
    values = [measured(v) for v in values]
    return measured(sum(values)) if values and all(v is not None for v in values) else None


def aggregate(parent, children):
    metrics = {"parent_queue_wait_s": parent.queue_wait_s}
    stages = {}
    for phase, child in children.items():
        for stage, value in {
            "queue": child.queue_wait_s,
            "worker_queue": (child.stage_durations or {}).get("worker_queue_wait"),
            "execution": child.inference_time_s,
            "finalize": child.finalize_time_s,
        }.items():
            metrics[f"{phase}_{stage}_s"] = measured(value)
    # Orchestration may wait for a healthy downstream Worker before a child
    # exists. Keep that interval instead of counting only its later queue row.
    for phase, child in children.items():
        ready = parent.started_at_ms if phase == "base" else children["base"].completed_at_ms
        ready = measured(ready)
        created = measured(child.created_at_ms)
        handoff = max(0.0, (created - ready) / 1000) if ready is not None and created is not None else None
        metrics[f"{phase}_handoff_wait_s"] = handoff
        metrics[f"{phase}_dispatch_queue_s"] = metrics[f"{phase}_queue_s"]
        metrics[f"{phase}_queue_s"] = complete_sum([handoff, metrics[f"{phase}_queue_s"]])
    # Preserve existing API field names.
    metrics.update(
        base_generation_s=metrics.get("base_execution_s"),
        base_queue_wait_s=metrics.get("base_queue_s"),
        super_resolution_s=metrics.get("sr_execution_s"),
        sr_queue_wait_s=metrics.get("sr_queue_s"),
    )
    for stage in ("queue", "worker_queue", "execution"):
        values = [metrics[f"{phase}_{stage}_s"] for phase in children]
        if stage == "queue":
            values.insert(0, parent.queue_wait_s)
        value = complete_sum(values)
        if value is not None:
            stages[stage] = value
    # Compatibility diagnostic keys, distinct from model stage timings.
    stages.update({"base_generation": metrics["base_generation_s"]})
    if "sr" in children:
        stages["super_resolution"] = metrics["super_resolution_s"]
    return metrics, {k: v for k, v in stages.items() if v is not None}
