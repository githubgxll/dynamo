# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable CAS contracts; no scheduler or business-record codec lives here."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from dingo.common.task_protocol import (
    TASK_KINDS,
    ResultDescriptor,
    TaskIdentity,
    identifier,
    integer,
)
from dingo.common.video_task_protocol import TASK_ID


@dataclass(frozen=True)
class TaskRef:
    deployment_id: str
    task_kind: str
    task_id: str

    def __post_init__(self):
        identifier(self.deployment_id, "deployment_id")
        identifier(self.task_id, "task_id", TASK_ID)
        if not isinstance(self.task_kind, str) or self.task_kind not in TASK_KINDS:
            raise ValueError("invalid task_kind")


@dataclass(frozen=True)
class WorkerRef:
    pool_id: str
    instance_id: str
    generation: str
    capacity_domain: str

    def __post_init__(self):
        for name in ("pool_id", "instance_id", "generation", "capacity_domain"):
            identifier(getattr(self, name), name)


@dataclass(frozen=True)
class AttemptFence:
    task: TaskRef
    worker: WorkerRef
    attempt: int
    execution_token: str

    def __post_init__(self):
        self.identity()  # Same validation as the wire identity.

    def identity(self) -> TaskIdentity:
        return TaskIdentity(
            self.task.deployment_id,
            self.task.task_kind,
            self.worker.pool_id,
            self.task.task_id,
            self.attempt,
            self.execution_token,
            self.worker.instance_id,
            self.worker.generation,
        )


@dataclass(frozen=True)
class KeySnapshot:
    """An observed existing key; absence is expressed explicitly by the builder."""

    key: str
    revision: int
    value: bytes

    def __post_init__(self):
        if not isinstance(self.key, str) or not self.key.startswith("/"):
            raise ValueError("snapshot requires an absolute etcd key")
        integer(self.revision, "revision")
        if not isinstance(self.value, bytes):
            raise TypeError("snapshot value must be immutable bytes")


@dataclass(frozen=True)
class OwnerSnapshot:
    generation: str
    record: KeySnapshot

    def __post_init__(self):
        identifier(self.generation, "owner generation")
        if self.record.value != self.generation.encode():
            raise ValueError("owner generation differs from lease record")


@dataclass(frozen=True)
class TaskSnapshot:
    ref: TaskRef
    record: KeySnapshot


@dataclass(frozen=True)
class WorkerSnapshot:
    ref: WorkerRef
    record: KeySnapshot
    capacity: int

    def __post_init__(self):
        integer(self.capacity, "capacity")


@dataclass(frozen=True)
class AttemptSnapshot:
    fence: AttemptFence
    record: KeySnapshot
    task: TaskSnapshot
    slot: KeySnapshot
    active: KeySnapshot
    owner: OwnerSnapshot

    def __post_init__(self):
        if self.task.ref != self.fence.task:
            raise ValueError("attempt belongs to another task")


class CasStatus(str, Enum):
    APPLIED = "applied"
    CONFLICT = "conflict"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CasResult:
    status: CasStatus
    revision: int | None = None

    def __post_init__(self):
        if not isinstance(self.status, CasStatus):
            raise TypeError("invalid CAS status")
        if self.status is CasStatus.UNKNOWN:
            if self.revision is not None:
                raise ValueError("unknown CAS cannot have a confirmed revision")
        else:
            integer(self.revision, "revision", minimum=0)


class ClaimStatus(str, Enum):
    CLAIMED = "claimed"
    CONFLICT = "conflict"
    NO_CAPACITY = "no_capacity"
    INELIGIBLE = "ineligible"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ClaimResult:
    status: ClaimStatus
    attempt: AttemptSnapshot | None = None

    def __post_init__(self):
        if not isinstance(self.status, ClaimStatus):
            raise TypeError("invalid claim status")
        if (self.status is ClaimStatus.CLAIMED) != (self.attempt is not None):
            raise ValueError("only a confirmed claim may return an attempt")


@dataclass(frozen=True)
class ValidatedOutcome:
    """Created by the result adapter after physical completion is confirmed.

    Repository adapters must preserve a public cancelled state, and merge usage
    receipts instead of replacing an older incomplete aggregate with a new one."""

    state: str
    result: ResultDescriptor | None
    usage_json: bytes

    def __post_init__(self):
        if not isinstance(self.state, str) or self.state not in {
            "succeeded",
            "failed",
            "cancelled",
        }:
            raise ValueError("outcome must describe physical completion")
        if (self.state == "succeeded") != (self.result is not None):
            raise ValueError("success requires a result descriptor")
        if not isinstance(self.usage_json, bytes):
            raise TypeError("usage snapshot must be immutable bytes")


class TaskRepository(Protocol):
    async def load(self, ref: TaskRef) -> TaskSnapshot | None:
        ...

    async def claim(
        self,
        task: TaskSnapshot,
        worker: WorkerSnapshot,
        slot: int,
        owner: OwnerSnapshot,
    ) -> ClaimResult:
        ...

    async def adopt(self, attempt: AttemptSnapshot, owner: OwnerSnapshot) -> CasResult:
        ...

    async def mark_dispatched(self, attempt: AttemptSnapshot) -> CasResult:
        ...

    async def request_cancel(self, task: TaskSnapshot) -> CasResult:
        ...

    async def settle(
        self, attempt: AttemptSnapshot, outcome: ValidatedOutcome
    ) -> CasResult:
        ...


@dataclass(frozen=True)
class RuntimeKeys:
    """New control keys only; existing video and HTTP business keys stay put."""

    deployment_id: str

    def __post_init__(self):
        identifier(self.deployment_id, "deployment_id")

    @property
    def root(self) -> str:
        return f"/dingo/task-runtime/v2/{self.deployment_id}"

    def _task(self, ref: TaskRef) -> str:
        if ref.deployment_id != self.deployment_id:
            raise ValueError("cross-deployment task")
        return f"{ref.task_kind}/{ref.task_id}"

    def active(self, ref: TaskRef) -> str:
        return f"{self.root}/active/{self._task(ref)}"

    def attempt(self, fence: AttemptFence) -> str:
        return (
            f"{self.root}/attempts/{self._task(fence.task)}/"
            f"{fence.attempt}-{fence.execution_token}"
        )

    def slot(self, worker: WorkerRef, slot_id: int) -> str:
        integer(slot_id, "slot_id", minimum=0)
        # No model, task kind or logical pool: aliases share physical capacity.
        return (
            f"{self.root}/slots/{worker.capacity_domain}/{worker.instance_id}/"
            f"{worker.generation}/{slot_id}"
        )
