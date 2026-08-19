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
