"""Compiler task metadata. Prompt and media bytes never enter this record."""
from dataclasses import asdict, dataclass, field
from typing import Any
from dingo.video_gateway.task_runtime.transactions import encode

TERMINAL = {"completed", "failed", "cancelled", "expired"}


@dataclass
class ContextIRTask:
    id: str
    deployment_id: str
    principal_hash: str
    model: str
    target_pool_id: str
    backend_id: str
    backend_epoch: str
    route_revision: str
    request_digest: str
    idempotency_hash: str | None
    request_path: str
    input_manifest_path: str
    route_snapshot_path: str
    reservation_id: str
    created_at_ms: int
    queue_deadline_ms: int
    input_bytes: int
    reserved_bytes: int
    schema_version: int = 1
    model_adapter_id: str = "minimax_h3"
    model_adapter_version: int = 1
    protocol_adapter_id: str = "open_h3_ir"
    protocol_adapter_version: int = 1
    created_seq: int = 0
    status: str = "queued"
    phase: str = "queued"
    attempt: int = 0
    slot_id: int | None = None
    owner_id: str | None = None
    owner_generation: str | None = None
    started_at_ms: int | None = None
    execution_deadline_ms: int | None = None
    cancel_requested_at_ms: int | None = None
    terminal_at_ms: int | None = None
    expires_at_ms: int | None = None
    expired_at_ms: int | None = None
    inputs_cleaned_at_ms: int | None = None
    upstream_call_intent_at_ms: int | None = None
    upstream_response_received_at_ms: int | None = None
    result_path: str | None = None
    result_sha256: str | None = None
    result_bytes: int | None = None
    error: dict | None = None
    metrics: dict = field(default_factory=dict)
    cleanup_pending: bool = False

    def to_dict(self):
        result = asdict(self)
        if len(encode(result).encode()) > 32768:
            raise ValueError("ContextIR task metadata exceeds 32 KiB")
        return result

    @classmethod
    def from_dict(cls, value):
        return cls(**value)

    def public_dict(self, result=None):
        value = {"id": self.id, "object": "context_ir", "model": self.model,
                 "status": self.status, "phase": self.phase,
                 "created_at": self.created_at_ms // 1000,
                 "completed_at": self.terminal_at_ms // 1000 if self.terminal_at_ms else None,
                 "expires_at": self.expires_at_ms // 1000 if self.expires_at_ms else None,
                 "metrics": self.metrics}
        if self.status == "queued":
            value["queue_deadline"] = self.queue_deadline_ms // 1000
        if self.error:
            value["error"] = self.error
        if self.status == "completed" and result is not None:
            value.update(result)
        return value


@dataclass(frozen=True)
class StoredContextIRTask:
    task: ContextIRTask
    revision: int


def failure(code, message, retryable=False):
    return {"code": code, "message": message, "retryable": retryable}
