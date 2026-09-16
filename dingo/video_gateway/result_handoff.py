"""Small, durable result references; never embed media bytes in task records."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from dingo.common.video_result_file import validate_descriptor
from dingo.video_gateway.models import TaskStatus, VideoTask, now_ms

HANDOFF_KEY = "_result_handoff_v1"


def make_handoff(
    task: VideoTask, descriptor: Mapping[str, Any], *, timeout_s: float
) -> dict[str, Any]:
    if task.attempt < 1 or not task.execution_token or not task.owner_generation:
        raise ValueError("result handoff requires a fenced execution")
    if (
        isinstance(timeout_s, bool)
        or not math.isfinite(timeout_s)
        or not 0 < timeout_s <= 86400
    ):
        raise ValueError("finalization timeout must be finite and within (0, 86400]")
    timestamp = now_ms()
    return {
        "schema_version": 1,
        "task_id": task.id,
        "attempt": task.attempt,
        "execution_token": task.execution_token,
        "artifact": validate_descriptor(dict(descriptor)),
        "created_at_ms": timestamp,
        "deadline_at_ms": timestamp + int(timeout_s * 1000),
        "failures": 0,
    }


def read_handoff(task: VideoTask) -> dict[str, Any] | None:
    raw = task.normalized_request.get(HANDOFF_KEY)
    if raw is None:
        return None
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("invalid result handoff schema")
    if (raw.get("task_id"), raw.get("attempt"), raw.get("execution_token")) != (
        task.id,
        task.attempt,
        task.execution_token,
    ):
        raise ValueError("result handoff execution identity mismatch")
    if (
        not task.execution_token
        or type(raw.get("deadline_at_ms")) is not int
        or type(raw.get("created_at_ms")) is not int
        or raw["deadline_at_ms"] < raw["created_at_ms"]
        or type(raw.get("failures")) is not int
        or raw["failures"] < 0
    ):
        raise ValueError("incomplete result handoff")
    validate_descriptor(dict(raw["artifact"]))
    return dict(raw)


def validate_handoff_transition(before: VideoTask, after: VideoTask) -> None:
    if before.status != TaskStatus.IN_PROGRESS or after.status != TaskStatus.FINALIZING:
        raise ValueError("execution handoff requires in_progress -> finalizing")
    if before.cancel_requested_at_ms is not None:
        raise ValueError("cannot hand off a cancelled execution")
    if (
        before.owner_generation,
        before.attempt,
        before.execution_token,
        before.worker_key,
    ) != (
        after.owner_generation,
        after.attempt,
        after.execution_token,
        after.worker_key,
    ):
        raise ValueError("handoff cannot change execution identity")
    if read_handoff(after) is None:
        raise ValueError("durable result reference missing")
