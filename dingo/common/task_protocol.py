# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict, dependency-light v2 control protocol. Video v1 is unchanged.

Parsing validates descriptors, not file contents. The receiver must separately
verify its deployment/instance generation, storage root, hashes and capacity.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from dingo.common.video_task_protocol import IDENTIFIER, TASK_ID, TOKEN

ENVELOPE_KEY = "_dingo_task"
SCHEMA_VERSION = 2
MAX_CONTROL_BYTES = 2 * 1024 * 1024
MAX_RESULT_BYTES = 512 * 1024
MAX_WAIT_MS = 30_000
TASK_KINDS = frozenset({"video", "context_ir"})
TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled"})
TASK_STATES = TERMINAL_STATES | {"accepted", "preparing", "running"}
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def identifier(value: Any, name: str, pattern=IDENTIFIER) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"invalid {name}")
    return value


def integer(value: Any, name: str, *, minimum=1, maximum=None) -> int:
    if (
        type(value) is not int
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"invalid {name}")
    return value


def relative_path(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 4096
        or "\\" in value
        or ":" in value
        or any(ord(c) < 32 for c in value)
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("invalid relative artifact path")
    return value


def resolve_artifact(root: Path, path: str) -> Path:
    """Resolve an existing descriptor; callers still verify hash/size on read.

    This is not a substitute for safe opening when an untrusted process can change
    the filesystem concurrently. Artifact directories must have trusted writers."""
    relative_path(path)
    root = root.resolve(strict=True)
    resolved = (root / path).resolve(strict=True)
    if root not in resolved.parents or not resolved.is_file():
        raise ValueError("artifact escaped root or is not a regular file")
    return resolved


def _object(value: Any, required: set[str], optional=()) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError("expected an object")  # noqa: TRY004 - wire validation error
    if set(value) - required - set(optional) or required - set(value):
        raise ValueError("missing or unknown protocol fields")
    return value


def _bounded(value: Mapping) -> None:
    try:
        size = len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode())
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("invalid JSON control message") from exc
    if size > MAX_CONTROL_BYTES:
        raise ValueError("control message exceeds size limit")


@dataclass(frozen=True)
class InputDescriptor:
    storage_id: str
    request_path: str
    manifest_path: str
    manifest_sha256: str
    route_path: str
    route_sha256: str
    request_digest: str

    def __post_init__(self):
        identifier(self.storage_id, "storage_id")
        for name in ("request_path", "manifest_path", "route_path"):
            relative_path(getattr(self, name))
        for name in ("manifest_sha256", "route_sha256", "request_digest"):
            identifier(getattr(self, name), name, SHA256)

    @classmethod
    def parse(cls, value):
        return cls(**_object(value, {f.name for f in fields(cls)}))


@dataclass(frozen=True)
class ResultDescriptor:
    storage_id: str
    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self):
        identifier(self.storage_id, "storage_id")
        relative_path(self.path)
        identifier(self.sha256, "sha256", SHA256)
        integer(self.size_bytes, "size_bytes", minimum=0, maximum=MAX_RESULT_BYTES)

    @classmethod
    def parse(cls, value):
        return cls(**_object(value, {f.name for f in fields(cls)}))


@dataclass(frozen=True)
class TaskIdentity:
    deployment_id: str
    task_kind: str
    pool_id: str
    task_id: str
    attempt: int
    execution_token: str
    instance_id: str
    generation: str

    def __post_init__(self):
        for name in ("deployment_id", "pool_id", "instance_id", "generation"):
            identifier(getattr(self, name), name)
        identifier(self.task_id, "task_id", TASK_ID)
        identifier(self.execution_token, "execution_token", TOKEN)
        integer(self.attempt, "attempt")
        if not isinstance(self.task_kind, str) or self.task_kind not in TASK_KINDS:
            raise ValueError("unsupported task_kind")


IDENTITY_FIELDS = {f.name for f in fields(TaskIdentity)}
INSTANCE_FIELDS = {"deployment_id", "pool_id", "instance_id", "generation"}


@dataclass(frozen=True)
class ControlRequest:
    op: str
    deployment_id: str
    pool_id: str
    instance_id: str
    generation: str | None = None
    identity: TaskIdentity | None = None
    payload: InputDescriptor | None = None
    deadline_at_ms: int | None = None
    wait_timeout_ms: int | None = None

    def to_dict(self) -> dict:
        value = {"schema_version": SCHEMA_VERSION, "op": self.op}
        value.update(
            {
                k: getattr(self, k)
                for k in INSTANCE_FIELDS
                if getattr(self, k) is not None
            }
        )
        if self.identity is not None:
            value.update(asdict(self.identity))
        if self.payload is not None:
            value["payload"] = asdict(self.payload)
        for k in ("deadline_at_ms", "wait_timeout_ms"):
            if getattr(self, k) is not None:
                value[k] = getattr(self, k)
        return {ENVELOPE_KEY: value}


def parse_request(message: Mapping) -> ControlRequest:
    _object(message, {ENVELOPE_KEY})
    _bounded(message)
    value = message[ENVELOPE_KEY]
    if not isinstance(value, Mapping):
        raise ValueError("invalid envelope")  # noqa: TRY004 - wire data
    if type(value.get("schema_version")) is not int or value["schema_version"] != 2:
        raise ValueError("unsupported task protocol version")
    op = value.get("op")
    required = {"schema_version", "op"}
    if op in ("capabilities", "drain"):
        # Discovery provides the runtime instance ID, not its process generation.
        # Bootstrap capabilities without a generation; drain always fences it.
        required_instance = (
            INSTANCE_FIELDS - {"generation"}
            if op == "capabilities"
            else INSTANCE_FIELDS
        )
        _object(
            value,
            required | required_instance,
            {"generation"} if op == "capabilities" else (),
        )
        for k in INSTANCE_FIELDS & value.keys():
            identifier(value[k], k)
        return ControlRequest(
            op=op, **{k: value[k] for k in INSTANCE_FIELDS & value.keys()}
        )
    extra = {
        "submit": {"payload", "deadline_at_ms"},
        "status": set(),
        "wait": {"wait_timeout_ms"},
        "cancel": set(),
    }
    if not isinstance(op, str) or op not in extra:
        raise ValueError("unsupported control operation")
    _object(value, required | IDENTITY_FIELDS | extra[op])
    identity = TaskIdentity(**{k: value[k] for k in IDENTITY_FIELDS})
    payload = InputDescriptor.parse(value["payload"]) if op == "submit" else None
    deadline = (
        integer(value["deadline_at_ms"], "deadline_at_ms") if op == "submit" else None
    )
    wait = (
        integer(value["wait_timeout_ms"], "wait_timeout_ms", maximum=MAX_WAIT_MS)
        if op == "wait"
        else None
    )
    return ControlRequest(
        op,
        identity.deployment_id,
        identity.pool_id,
        identity.instance_id,
        identity.generation,
        identity,
        payload,
        deadline,
        wait,
    )


@dataclass(frozen=True)
class TaskResponse:
    identity: TaskIdentity
    state: str
    result: ResultDescriptor | None = None
    error_code: str | None = None

    @classmethod
    def parse(cls, value: Mapping):
        _object(
            value, {"schema_version", "identity", "state"}, {"result", "error_code"}
        )
        _bounded(value)
        if type(value["schema_version"]) is not int or value["schema_version"] != 2:
            raise ValueError("unsupported response version")
        identity = TaskIdentity(**_object(value["identity"], IDENTITY_FIELDS))
        state = value["state"]
        if not isinstance(state, str) or state not in TASK_STATES | {
            "not_accepted",
            "not_found",
        }:
            raise ValueError("invalid task state")
        result = ResultDescriptor.parse(value["result"]) if "result" in value else None
        code = (
            identifier(value["error_code"], "error_code")
            if "error_code" in value
            else None
        )
        if (state == "succeeded") != (result is not None):
            raise ValueError("only succeeded responses require a result descriptor")
        if (state in {"failed", "not_accepted"}) != (code is not None):
            raise ValueError("failed/rejected responses require a bounded error code")
        return cls(identity, state, result, code)


@dataclass(frozen=True)
class WorkerCapabilities:
    instance_id: str
    generation: str
    capacity_domain: str
    storage_id: str
    protocol_versions: tuple[int, ...]
    task_kinds: tuple[str, ...]
    execution_capacity: int
    prefetch_capacity: int
    model_adapters: tuple[tuple[str, int], ...]
    protocol_adapters: tuple[tuple[str, int], ...]
    queue_allowed: bool
    accepting: bool
    draining: bool

    @classmethod
    def parse(cls, value: Mapping):
        _object(value, {f.name for f in fields(cls)})
        _bounded(value)
        parsed = dict(value)
        for k in ("instance_id", "generation", "capacity_domain", "storage_id"):
            identifier(value[k], k)
        for k in ("queue_allowed", "accepting", "draining"):
            if type(value[k]) is not bool:
                raise ValueError(f"invalid {k}")
        if value["draining"] and (value["accepting"] or value["queue_allowed"]):
            raise ValueError("draining worker cannot accept new work")
        if value["accepting"] and not value["queue_allowed"]:
            raise ValueError("accepting worker must allow queue admission")
        for k in ("protocol_versions", "task_kinds"):
            items = value[k]
            if not isinstance(items, list) or not items:
                raise ValueError(f"invalid {k}")
            for item in items:
                if k == "protocol_versions":
                    integer(item, k)
                elif not isinstance(item, str) or item not in TASK_KINDS:
                    raise ValueError("unsupported task kind")
            if len(set(items)) != len(items):
                raise ValueError(f"duplicate {k}")
            parsed[k] = tuple(items)
        integer(value["execution_capacity"], "execution_capacity")
        integer(value["prefetch_capacity"], "prefetch_capacity", minimum=0)
        if "context_ir" in value["task_kinds"] and value["prefetch_capacity"] != 0:
            raise ValueError("ContextIR does not support prefetch")
        for k in ("model_adapters", "protocol_adapters"):
            if not isinstance(value[k], dict) or not value[k]:
                raise ValueError(f"invalid {k}")
            for name, version in value[k].items():
                identifier(name, k)
                integer(version, k)
            parsed[k] = tuple(sorted(value[k].items()))
        return cls(**parsed)
