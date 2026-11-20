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


def request_cache_retention_ttl_s(request: dict[str, Any]) -> int | None:
    """Read the admission-resolved prompt-cache TTL from the request.

    Rust admission validated the customer's ``prompt_cache_options`` /
    ``cache_control`` markers and wrote whole seconds onto the server-owned
    ``cache_retention_ttl_s`` request field. Only that trusted value is
    returned: this helper never defaults or invents a TTL, and never reads
    the client-writable ``chat_template_args`` (whose homonymous key, if a
    client forges one, is stripped at the Rust HTTP boundary).
    """
    value = request.get("cache_retention_ttl_s")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def project_cache_retention_ttl_s(preproc: dict[str, Any], request: dict[str, Any]) -> None:
    """Persist the admission-resolved TTL onto the worker payload.

    Companion to [`project_cache_salt`]: the typed top-level
    ``cache_retention_ttl_s`` field is what the SGLang request handlers read
    when forwarding the retention hint to the engine. An absent/invalid
    admission value leaves the payload untouched so transport forwards
    nothing and the engine runs best-effort.
    """
    seconds = request_cache_retention_ttl_s(request)
    if seconds is not None:
        preproc["cache_retention_ttl_s"] = seconds
