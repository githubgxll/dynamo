# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from typing import Any


def request_cache_salt(request: Mapping[str, Any]) -> str | None:
    """Return the first non-empty cache salt, preferring resolved routing hints."""
    extra_args = request.get("extra_args") or {}
    for source in (
        request.get("routing"),
        extra_args.get("nvext") if isinstance(extra_args, Mapping) else None,
        request.get("nvext"),
        request,
    ):
        if isinstance(source, Mapping):
            salt = source.get("cache_salt")
            if salt:
                return salt
    return None


def request_cache_retention_ttl_s(request: Mapping[str, Any]) -> int | None:
    """Return the admission-resolved prompt-cache retention TTL in seconds.

    The frontend sets ``cache_retention_ttl_s`` (whole seconds; 0 = no
    protection) as a top-level request field during admission. Absent or
    ``None`` means no protection was requested or admitted; transport then
    forwards nothing and the engine runs unprotected.
    """
    value = request.get("cache_retention_ttl_s")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value
