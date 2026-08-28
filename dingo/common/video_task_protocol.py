# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light private protocol shared by Video Gateway and Omni Worker."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ENVELOPE_KEY = "_dingo_video_task"
SCHEMA_VERSION = 1
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
TOKEN = re.compile(r"^[0-9a-f]{32,64}$")


def detached_attempt_root(
    artifact_root: str | Path,
    deployment_id: str,
    pool_id: str,
    task_id: str,
    attempt: int,
    execution_token: str,
) -> Path:
    if not IDENTIFIER.fullmatch(deployment_id):
        raise ValueError("invalid detached deployment_id")
    if not IDENTIFIER.fullmatch(pool_id):
        raise ValueError("invalid detached pool_id")
    if not TASK_ID.fullmatch(task_id):
        raise ValueError("invalid detached task_id")
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
        raise ValueError("invalid detached attempt")
    if not TOKEN.fullmatch(execution_token):
        raise ValueError("invalid detached execution_token")
    root = Path(artifact_root).expanduser().resolve()
    result = (
        root
        / deployment_id
        / "v1"
        / "pools"
        / pool_id
        / "tasks"
        / task_id
        / "attempts"
        / f"{attempt}-{execution_token}"
    ).absolute()
    if root not in result.parents:
        raise ValueError("detached task path escaped configured root")
    return result


@dataclass(frozen=True, slots=True)
class DetachedTaskIdentity:
    deployment_id: str
    pool_id: str
    task_id: str
    attempt: int
    execution_token: str

    @classmethod
    def from_envelope(cls, value: Mapping[str, Any]) -> "DetachedTaskIdentity":
        allowed = {
            "schema_version",
            "op",
            "deployment_id",
            "pool_id",
            "task_id",
            "attempt",
            "execution_token",
            "payload",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown detached task fields: {sorted(unknown)}")
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported detached task schema_version")
        identity = cls(
            deployment_id=str(value.get("deployment_id", "")),
            pool_id=str(value.get("pool_id", "")),
            task_id=str(value.get("task_id", "")),
            attempt=value.get("attempt"),
            execution_token=str(value.get("execution_token", "")),
        )
        detached_attempt_root(
            "/tmp/dingo-detached-validation",
            identity.deployment_id,
            identity.pool_id,
            identity.task_id,
            identity.attempt,
            identity.execution_token,
        )
        return identity

    @property
    def key(self) -> tuple[str, str, str, int, str]:
        return (
            self.deployment_id,
            self.pool_id,
            self.task_id,
            self.attempt,
            self.execution_token,
        )


def detached_envelope(
    *,
    op: str,
    deployment_id: str,
    pool_id: str,
    task_id: str,
    attempt: int,
    execution_token: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "op": op,
        "deployment_id": deployment_id,
        "pool_id": pool_id,
        "task_id": task_id,
        "attempt": attempt,
        "execution_token": execution_token,
    }
    if payload is not None:
        value["payload"] = payload
    DetachedTaskIdentity.from_envelope(value)
    return {ENVELOPE_KEY: value}
