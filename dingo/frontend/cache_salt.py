# SPDX-License-Identifier: Apache-2.0
"""Cache namespace transport shared by the Python chat processors."""

from typing import Any


def resolve_cache_salt(request: dict[str, Any]) -> str | None:
    """Match Rust request_cache_salt after HTTP header/NvExt policy.

    The HTTP boundary has already placed x-tenant-id in nvext.cache_salt.
    Body salts are not trimmed. Invalid canonical types are rejected, while
    non-string legacy values are ignored, as in the Rust request schema.
    """
    nvext = request.get("nvext") or {}
    salt = nvext.get("cache_salt")
    if salt is not None and not isinstance(salt, str):
        raise ValueError("nvext.cache_salt must be a string or null")
    if salt:
        return salt
    legacy = request.get("cache_salt")
    return legacy if isinstance(legacy, str) and legacy else None


def project_cache_salt(preproc: dict[str, Any], salt: str | None) -> None:
    """Keep router namespace and worker salt consistent without mutating input."""
    if salt is None:
        return
    preproc["routing"] = {**(preproc.get("routing") or {}), "cache_salt": salt}
    extra_args = dict(preproc.get("extra_args") or {})
    extra_args["nvext"] = {**(extra_args.get("nvext") or {}), "cache_salt": salt}
    preproc["extra_args"] = extra_args
