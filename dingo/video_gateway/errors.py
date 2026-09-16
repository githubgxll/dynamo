# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Error types shared by the asynchronous video gateway."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(slots=True)
class GatewayError(Exception):
    """An expected API error with a stable HTTP and machine-readable shape."""

    status: int
    code: str
    message: str
    param: str | None = None
    error_type: str = "invalid_request_error"
    headers: Mapping[str, str] | None = None

    def __str__(self) -> str:
        return self.message

    def as_response(self) -> dict:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


class StoreConflict(RuntimeError):
    """Raised when an optimistic Task Store transaction loses a race."""


class StoreUnavailable(RuntimeError):
    """Raised when the configured Task Store cannot be reached safely."""


class HandoffReservationLost(StoreConflict):
    """Result handoff definitively no longer owns its execution reservation."""


class ResultTooLarge(RuntimeError):
    """Raised when a Worker result exceeds the configured artifact policy."""


class WorkerExecutionFailed(RuntimeError):
    """A Worker reported failure; not automatically retryable."""


class WorkerUnavailable(WorkerExecutionFailed):
    """Positive evidence of a lost Worker or unavailable execution engine."""


def worker_execution_error(error: object) -> WorkerExecutionFailed:
    """Conservative compatibility classifier for this deployed Worker protocol.

    Worker errors currently have no infrastructure-specific structured code.
    Match only complete, known engine-unavailable messages, never substrings,
    arbitrary RuntimeError, or the generic worker_failed code by itself.
    Unknown, parameter, media and model errors are deliberately not retried.
    """
    message = error
    if isinstance(error, dict):
        if error.get("code") != "worker_failed":
            return WorkerExecutionFailed(
                str(error.get("message") or "Worker generation failed")
            )
        message = error.get("message")
    if isinstance(message, str) and message in {
        "Executor shut down",
        "Stage-0 has no live replica",
    }:
        return WorkerUnavailable(message)
    return WorkerExecutionFailed(str(message or "Worker generation failed"))
