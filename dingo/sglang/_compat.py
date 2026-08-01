# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Compatibility shim for SGLang internal APIs.

SGLang is pre-1.0 and routinely moves, renames, or introduces APIs between
releases. This module is the single place where we handle those differences
so the rest of the component can import from here without version-specific
try/except blocks.

Policy: support current SGLang release + 1 version back (N and N-1). Each
fallback branch must document which version it covers and when it can be
removed. When the old version falls outside the support window, delete the
fallback and any associated polyfills.

Runtime data-contract notes (not code-level shims):

* ``meta_info["routed_experts"]`` is a base64 UTF-8 string from sglang
  >= 0.5.11. Pass through; do not re-encode.
"""

import inspect
import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Top-level sglang exports: Engine, ServerArgs
#
# Some SGLang dev builds (including 0.5.x snapshots) do not re-export these
# from sglang/__init__.py, while Dynamo historically uses `import sglang as sgl`
# followed by `sgl.Engine(...)` throughout this backend.
# ---------------------------------------------------------------------------
def ensure_sglang_top_level_exports() -> None:
    """Restore top-level SGLang exports omitted by some install flavors."""
    import sglang as sgl

    if not hasattr(sgl, "Engine"):
        from sglang.srt.entrypoints.engine import Engine

        sgl.Engine = Engine

    if not hasattr(sgl, "ServerArgs"):
        from sglang.srt.server_args import ServerArgs

        sgl.ServerArgs = ServerArgs


ensure_sglang_top_level_exports()


@lru_cache(maxsize=32)
def _get_async_generate_supported_kwarg_names(
    async_generate: Any,
) -> frozenset[str] | None:
    """Return supported async_generate keyword names, or None for **kwargs."""
    try:
        signature = inspect.signature(async_generate)
    except (TypeError, ValueError):
        logger.debug(
            "Could not inspect SGLang Engine.async_generate signature; "
            "dropping optional compatibility kwargs"
        )
        return frozenset()

    names: set[str] = set()
    for name, param in signature.parameters.items():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return None
        if param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            names.add(name)

    return frozenset(names)


def filter_supported_async_generate_kwargs(
    engine: Any, kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Return only async_generate kwargs accepted by this SGLang engine.

    SGLang occasionally adds optional Engine.async_generate kwargs before every
    supported install flavor has them. Keep the compatibility boundary narrow:
    callers decide which kwargs are optional, and this helper only drops those
    optional kwargs when the installed engine cannot accept them.
    """
    async_generate = engine.async_generate
    signature_source = getattr(async_generate, "__func__", async_generate)

    try:
        supported_kwarg_names = _get_async_generate_supported_kwarg_names(
            signature_source
        )
    except TypeError:
        supported_kwarg_names = _get_async_generate_supported_kwarg_names.__wrapped__(
            signature_source
        )

    if supported_kwarg_names is None:
        return kwargs

    return {key: value for key, value in kwargs.items() if key in supported_kwarg_names}


def enable_disjoint_streaming_output(server_args: Any) -> None:
    """Enable SGLang's disjoint streaming output.

    Diffusion workers pass a ``SimpleNamespace`` stub that does not carry the
    field, so this is a no-op when the attribute is absent.

    Compatibility notes:
    * sglang >= 0.5.16 makes ``ServerArgs`` fields read-only after materialization
      and routes runtime mutations through ``ServerArgs.override(source, **fields)``.
      Calling plain ``server_args.<field> = value`` raises ``AttributeError``.
    * Older sglang (< 0.5.16) has no such protection; plain assignment works.
    * Diffusion workers pass a ``SimpleNamespace`` without ``override``; for those
      we fall back to direct setattr (no protection in place).
    """
    if not hasattr(server_args, "incremental_streaming_output"):
        return

    override_fn = getattr(server_args, "override", None)
    if callable(override_fn):
        # sglang >= 0.5.16 legal mutation path
        try:
            override_fn("dingo.enable_disjoint_streaming_output", incremental_streaming_output=True)
            return
        except Exception as e:  # pragma: no cover - defensive against signature drift
            logger.warning("server_args.override(...) failed (%s); falling back to setattr", e)

    # Legacy sglang (< 0.5.16) or duck-typed stub: bare assignment
    try:
        server_args.incremental_streaming_output = True
    except AttributeError:
        # Last resort: bypass the read-only guard via object.__setattr__.
        # This is identical in effect to a successful setattr on a mutable
        # object and only runs when the read-only protection is in place but
        # ``override`` was unusable (e.g. signature changed).
        object.__setattr__(server_args, "incremental_streaming_output", True)


__all__ = [
    "enable_disjoint_streaming_output",
    "ensure_sglang_top_level_exports",
    "filter_supported_async_generate_kwargs",
]
