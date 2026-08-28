# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent task and lease records for asynchronous video generation."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


def now_ms() -> int:
    return int(time.time() * 1000)


class TaskStatus(str, Enum):
    QUEUED = "queued"
    DISPATCHING = "dispatching"
    IN_PROGRESS = "in_progress"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


TERMINAL_STATUSES = {
    TaskStatus.COMPLETED,
    TaskStatus.FAILED,
    TaskStatus.CANCELLED,
    TaskStatus.EXPIRED,
}
ACTIVE_STATUSES = {
    TaskStatus.DISPATCHING,
    TaskStatus.IN_PROGRESS,
    TaskStatus.FINALIZING,
}
ALLOWED_TRANSITIONS = {
    TaskStatus.QUEUED: {
        TaskStatus.DISPATCHING,
        TaskStatus.CANCELLED,
        TaskStatus.FAILED,
    },
    TaskStatus.DISPATCHING: {
        TaskStatus.IN_PROGRESS,
        TaskStatus.CANCELLED,
        TaskStatus.FAILED,
    },
    TaskStatus.IN_PROGRESS: {
        TaskStatus.FINALIZING,
        TaskStatus.CANCELLED,
        TaskStatus.FAILED,
    },
    TaskStatus.FINALIZING: {
        TaskStatus.COMPLETED,
        TaskStatus.CANCELLED,
        TaskStatus.FAILED,
    },
    TaskStatus.COMPLETED: {TaskStatus.FAILED, TaskStatus.EXPIRED},
    TaskStatus.FAILED: {TaskStatus.EXPIRED},
    TaskStatus.CANCELLED: {TaskStatus.EXPIRED},
    TaskStatus.EXPIRED: set(),
}


@dataclass(slots=True)
class TaskError:
    code: str
    message: str
    retryable: bool = False


@dataclass(slots=True)
class VideoTask:
    schema_version: int
    id: str
    deployment_id: str
    pool_id: str
    model: str
    backend_model: str
    backend_target: str
    configuration_revision: str
    delivery_mode: str
    status: TaskStatus
    request_digest: str
    request_path: str
    input_manifest_path: str
    created_at_ms: int
    queued_at_ms: int
    expires_at_ms: int
    created_seq: int = 0
    principal_hash: str | None = None
    idempotency_hash: str | None = None
    attempt: int = 0
    worker_instance_id: int | str | None = None
    worker_key: str | None = None
    worker_lease_id: int | None = None
    owner_generation: str | None = None
    execution_token: str | None = None
    assigned_at_ms: int | None = None
    started_at_ms: int | None = None
    deadline_at_ms: int | None = None
    cancel_requested_at_ms: int | None = None
    completed_at_ms: int | None = None
    expired_at_ms: int | None = None
    artifact_deleted_at_ms: int | None = None
    result_path: str | None = None
    result_bytes: int | None = None
    result_sha256: str | None = None
    inference_time_s: float | None = None
    finalize_time_s: float | None = None
    queue_wait_s: float | None = None
    stage_durations: dict[str, float] | None = None
    error: TaskError | None = None
    estimated_payload_bytes: int = 0
    normalized_request: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> VideoTask:
        data = dict(value)
        data["status"] = TaskStatus(data["status"])
        error = data.get("error")
        if isinstance(error, Mapping):
            data["error"] = TaskError(**error)
        return cls(**data)

    def public_dict(self) -> dict[str, Any]:
        public_status = self.status
        if public_status in {TaskStatus.DISPATCHING, TaskStatus.FINALIZING}:
            public_status = TaskStatus.IN_PROGRESS
        result: dict[str, Any] = {
            "id": self.id,
            "object": "video",
            "model": self.model,
            "status": public_status.value,
            "progress": 100 if self.status == TaskStatus.COMPLETED else 0,
            "created_at": self.created_at_ms // 1000,
            "expires_at": self.expires_at_ms // 1000,
        }
        width = self.normalized_request.get("width")
        height = self.normalized_request.get("height")
        if width is not None and height is not None:
            result["size"] = f"{width}x{height}"
        if self.normalized_request.get("seconds") is not None:
            result["seconds"] = self.normalized_request["seconds"]
        if self.completed_at_ms is not None:
            result["completed_at"] = self.completed_at_ms // 1000
        if self.status == TaskStatus.COMPLETED:
            result.update(
                {
                    "media_type": "video/mp4",
                    "file_name": f"{self.id}.mp4",
                    "bytes": self.result_bytes,
                    "sha256": self.result_sha256,
                }
            )
        if any(
            value is not None
            for value in (
                self.queue_wait_s,
                self.inference_time_s,
                self.finalize_time_s,
            )
        ):
            result["metrics"] = {
                "queue_wait_s": self.queue_wait_s,
                "inference_time_s": self.inference_time_s,
                "finalize_time_s": self.finalize_time_s,
            }
        if self.completed_at_ms is not None:
            # vLLM-Omni defines inference_time_s as end-to-end server time,
            # including time spent queued. Keep the Worker-reported formatter
            # time in metrics while exposing the native-compatible value here.
            result["inference_time_s"] = max(
                0.0, (self.completed_at_ms - self.created_at_ms) / 1000.0
            )
        durations = dict(self.stage_durations or {})
        if self.queue_wait_s is not None:
            durations.setdefault("queue_wait", self.queue_wait_s)
        if self.finalize_time_s is not None:
            durations.setdefault("finalize", self.finalize_time_s)
        if durations:
            result["stage_durations"] = durations
        if self.error is not None:
            result["error"] = asdict(self.error)
        return result


@dataclass(slots=True)
class StoredTask:
    task: VideoTask
    revision: int


@dataclass(slots=True)
class WorkerLease:
    pool_id: str
    worker_key: str
    worker_instance_id: int | str
    backend_target: str
    task_id: str
    owner_generation: str
    state: str
    heartbeat_at_ms: int
    execution_token: str | None = None
    owner_expires_at_ms: int | None = None
    reuse_after_ms: int | None = None
    etcd_lease_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> WorkerLease:
        return cls(**dict(value))
