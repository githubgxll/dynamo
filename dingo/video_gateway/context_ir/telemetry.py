# SPDX-License-Identifier: Apache-2.0
"""Compile observations only after a confirmed business-state transition."""
from dingo.video_gateway.pipeline_timings import measured

BUCKETS = (0.01, 0.1, 1, 5, 10, 30, 60, 120, 300, 600, 900)
STAGES = (
    "prepare_time_s",
    "asset_upload_time_s",
    "compile_time_s",
    "intent_resolution_s",
    "finalize_time_s",
    "queue_wait_s",
    "execution_time_s",
)


def record_transition(telemetry, before, after):
    if before.status == after.status or after.status not in {
        "completed",
        "failed",
        "cancelled",
        "expired",
    }:
        return
    labels = {"backend": after.backend_id, "model": after.model}
    telemetry.increment(
        "dingo_context_ir_terminal_total", labels={**labels, "state": after.status}
    )
    # Expiry is a retention event, never another compilation duration sample.
    if after.status == "expired":
        return
    for stage in STAGES:
        value = measured(after.metrics.get(stage))
        if value is not None:
            telemetry.observe(
                "dingo_context_ir_stage_duration_seconds",
                value,
                labels={**labels, "stage": stage},
                buckets=BUCKETS,
            )
