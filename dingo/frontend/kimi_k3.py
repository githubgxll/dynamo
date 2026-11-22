#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Kimi-K3 request normalization helpers for the SGLang frontend processor.

Pure helpers shared by ``sglang_prepost.preprocess_chat_request``. They
implement the Kimi-K3 serving contract already shipped by SGLang's
``KimiK3ChatEncodingAdapter`` (serving_kimi_k3.py) without importing the
HTTP serving stack: the native HTTP layer has already folded ``thinking``
into ``chat_template_args`` by the time Python sees the request, but older
deployments pass the original body through, so both shapes are resolved
here. Every function is side-effect free on the request object; nothing
rewrites ``messages`` in place or hoists dynamic tools to the top level.

Effort levels are the K3 encoder's own space: low / medium / high / max.
The OpenAI tier mapping matches SGLang's serving_kimi_k3 tables.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from dingo.frontend.utils import PreprocessError

# --- tool-name contract (serving_kimi_k3.py::_KIMI_K3_TOOL_NAME_RE) ------
# Letter or '_' first, then letters, digits, '_' or '-'; at most 256 chars.
KIMI_K3_TOOL_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,255}")

# Thinking payload types accepted by the wire contract.
KIMI_K3_THINKING_TYPES = (None, "enabled", "disabled", "adaptive")

# K3 effort levels; "medium" is a K3.1 addition.
KIMI_K3_LEVELS = ("low", "medium", "high", "max")

# OpenAI reasoning-effort tiers mapped onto the nearest K3 level, exactly as
# SGLang's serving_kimi_k3.py tables do; K3.1 additionally keeps "medium".
KIMI_K3_LEVEL_BY_TIER = {
    "minimal": "low",
    "low": "low",
    "medium": "high",
    "high": "high",
    "xhigh": "max",
    "ultra": "max",
    "max": "max",
}
KIMI_K31_LEVEL_BY_TIER = {**KIMI_K3_LEVEL_BY_TIER, "medium": "medium"}


def kimi_k3_level_by_tier(model_tier: str = "k31") -> dict[str, str]:
    """The effort-tier mapping for the given model tier (``k3`` or ``k31``)."""
    return KIMI_K31_LEVEL_BY_TIER if model_tier == "k31" else KIMI_K3_LEVEL_BY_TIER


def _levels_text(level_by_tier: dict[str, str]) -> str:
    return ", ".join(repr(level) for level in dict.fromkeys(level_by_tier.values()))


# --- normalized request ----------------------------------------------------


@dataclass
class KimiK3Normalized:
    """One canonical answer for the Kimi-K3 request controls.

    ``effective_tools`` is the union of the raw global ``tools`` and every
    legal dynamic declaration, in declaration order (global first). It feeds
    the required/named empty checks, the grammar, and the output name
    validation. The messages are NOT rewritten: the raw declarations keep
    rendering at their original position through the encoder.
    """

    effective_tools: list[dict[str, Any]] = field(default_factory=list)
    # Global tools in their original dict form -- template input, never
    # re-serialized through a schema model so field presence stays explicit.
    raw_global_tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    tool_choice_explicit: bool = False
    response_format: dict[str, Any] | None = None
    # Resolved effort in the encoder's level space; None means "no explicit
    # effort" (the encoder's own default applies).
    thinking_effort: str | None = None
    thinking_enabled: bool | None = None


# --- dynamic-tool validation -----------------------------------------------


def _validate_tool_entry(
    tool: Any, where: str, *, require_type: bool = True
) -> tuple[str, dict[str, Any]]:
    """Validate one tool entry (wrapped or bare function-schema form).

    ``where`` is the error prefix, e.g. ``messages[0].tools[2]``. Returns
    ``(name, wrapped)`` where ``wrapped`` is the OpenAI form
    ``{"type": "function", "function": {...}}`` for the effective-tools
    projection. The raw entry is never mutated: bare declarations stay bare
    in ``messages`` so the encoder renders exactly what the client sent.

    ``type`` must be present and ``"function"`` on the wire form; a missing
    ``type`` is not defaulted here because the K3 contract requires the
    explicit field. The bare form (``{"name": ..., "parameters": ...}``, no
    ``type`` key) is validated in place, matching the native
    ``normalize_dynamic_system_tool`` tolerance.
    """
    if not isinstance(tool, dict):
        raise PreprocessError(f"{where} must be a JSON object")

    if "function" in tool:
        if require_type and tool.get("type") != "function":
            raise PreprocessError(f"{where}: tool 'type' must be 'function'")
        function = tool["function"]
        if not isinstance(function, dict):
            raise PreprocessError(f"{where}: tool 'function' must be a JSON object")
        wrapped = tool
    elif "name" in tool and "type" not in tool:
        # Bare function-schema form preserved verbatim by the protocol
        # crate; validate it as its own function object and wrap only the
        # projection copy.
        function = tool
        wrapped = {"type": "function", "function": dict(tool)}
    elif "type" in tool:
        if tool["type"] != "function":
            raise PreprocessError(f"{where}: tool 'type' must be 'function'")
        raise PreprocessError(
            f"{where}: a tool with type=\"function\" needs a function object"
        )
    else:
        raise PreprocessError(
            f"{where}: tool needs a 'type' and 'function' object, or the bare "
            "function-schema form with a 'name'"
        )

    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise PreprocessError(f"{where}: tool needs a non-empty string name")
    if KIMI_K3_TOOL_NAME_RE.fullmatch(name) is None:
        raise PreprocessError(
            f"{where}: invalid tool name {name!r}: it must start with a letter "
            "or '_', contain only letters, digits, '_' or '-', and be at "
            "most 256 characters."
        )
    return name, wrapped


def validate_kimi_k3_dynamic_tools(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate every ``messages[i].tools`` declaration; return them in order.

    Rules (KVV dynamic_tools suite / serving_kimi_k3 contract):
      * only ``system``/``developer`` messages may declare tools;
      * a declaring message must not also carry non-empty ``content``;
      * each entry needs an explicit ``type="function"`` (or the bare
        function-schema form), a ``function``/``name`` string and a legal
        256-char name; the tools value must be a JSON array of objects;
      * names must be unique across the global list and all declarations.
    Raises :class:`PreprocessError` on the first violation.
    """
    dynamic: list[dict[str, Any]] = []
    names: dict[str, str] = {}
    for index, message in enumerate(messages):
        if "tools" not in message:
            continue
        role = message.get("role")
        where = f"messages[{index}]"
        if role not in ("system", "developer"):
            raise PreprocessError(
                f"{where}: only a system or developer message can declare "
                f"tools, got role {role!r}."
            )
        content = message.get("content")
        if isinstance(content, str) and content:
            raise PreprocessError(
                f"{where}: a message that declares tools must have no content."
            )
        if isinstance(content, list) and content:
            raise PreprocessError(
                f"{where}: a message that declares tools must have no content."
            )
        tools = message["tools"]
        if not isinstance(tools, list):
            raise PreprocessError(f"{where}: tools must be an array")
        for tool_index, tool in enumerate(tools):
            name, wrapped = _validate_tool_entry(
                tool, f"{where}.tools[{tool_index}]"
            )
            dynamic.append(wrapped)
            if name in names:
                raise PreprocessError(
                    f"{where}.tools[{tool_index}]: duplicate tool name {name!r} "
                    f"(already declared in {names[name]})"
                )
            names[name] = where
    return dynamic


# --- effort resolution ------------------------------------------------------


def _thinking_payload(request: dict[str, Any]) -> dict[str, Any] | None:
    thinking = request.get("thinking")
    return thinking if isinstance(thinking, dict) else None


def _ctk(request: dict[str, Any]) -> dict[str, Any]:
    ctk = request.get("chat_template_kwargs") or request.get("chat_template_args")
    return ctk if isinstance(ctk, dict) else {}


def resolve_kimi_k3_thinking_effort(
    request: dict[str, Any], level_by_tier: dict[str, str]
) -> tuple[str | None, bool | None]:
    """Resolve thinking + effort once, for every reader to share.

    Precedence (KVV test_thinking_effort contract):
      1. explicit ``thinking.type="disabled"`` wins over everything: effort
         cleared, thinking off;
      2. native-normalized requests already carry the decision in CTK
         (``thinking`` bool / ``thinking_effort``): the CTK grade is the
         single decision and no root field may override it;
      3. an explicit ``thinking.effort`` beats the top-level
         ``reasoning_effort`` (thinking.effort=low + reasoning_effort=max
         must behave like low);
      4. the top-level ``reasoning_effort`` applies when effort is absent
         (mapped through the tier table);
      5. anything else stays None: the encoder's deployment default applies.

    ``reasoning_effort="none"`` means "thinking off", matching the adapter.
    Returns ``(thinking_effort, thinking_enabled)`` where ``enabled`` is None
    when no explicit control was set at all.

    Raises :class:`PreprocessError` on an invalid level or thinking type.
    """
    ctk = _ctk(request)
    levels = tuple(level_by_tier.values())

    # (1) The disabled switch outranks every grade, folded or raw.
    payload = _thinking_payload(request)
    if payload is not None:
        thinking_type = payload.get("type")
        if thinking_type not in KIMI_K3_THINKING_TYPES:
            raise PreprocessError(
                f"invalid thinking: {payload!r}; expected an object whose type, "
                "if set, is 'enabled', 'disabled' or 'adaptive'."
            )
        if thinking_type == "disabled":
            return None, False

    # (2) A CTK grade is a decision an earlier normalization made (native
    # fold or an operator's explicit projection); it is authoritative over
    # the raw root fields below.
    ctk_effort = ctk.get("thinking_effort")
    if ctk_effort is not None:
        if ctk_effort not in levels:
            raise PreprocessError(
                f"chat_template_kwargs.thinking_effort must be one of "
                f"{_levels_text(level_by_tier)}; got {ctk_effort!r}."
            )
        enabled = ctk.get("thinking")
        return ctk_effort, enabled if isinstance(enabled, bool) else None

    if payload is not None and payload.get("effort") is not None:
        # (3) Explicit nested effort beats the top-level field.
        effort = payload["effort"]
        if effort not in levels:
            raise PreprocessError(
                f"thinking.effort must be one of {_levels_text(level_by_tier)}; "
                f"got {effort!r}."
            )
        thinking_type = payload.get("type")
        return effort, True if thinking_type in (None, "enabled") else None


    reasoning_effort = request.get("reasoning_effort")
    if reasoning_effort is None or reasoning_effort == "none":
        return None, (False if reasoning_effort == "none" else None)
    if not isinstance(reasoning_effort, str):
        raise PreprocessError(
            f"reasoning_effort must be a string tier; got {reasoning_effort!r}."
        )
    if reasoning_effort in level_by_tier:
        return level_by_tier[reasoning_effort], None
    raise PreprocessError(
        f"Kimi K3 supports reasoning_effort {_levels_text(level_by_tier)} "
        f"(the other OpenAI tiers map onto them); got {reasoning_effort!r}."
    )


# --- tool_choice / response_format projection ------------------------------


def _is_named_tool_choice(tool_choice: Any) -> bool:
    return (
        isinstance(tool_choice, dict)
        and tool_choice.get("type") == "function"
        and isinstance(tool_choice.get("function"), dict)
        and bool(tool_choice["function"].get("name"))
    )


def kimi_k3_passes_tool_choice(
    normalized: KimiK3Normalized,
) -> dict[str, Any] | None:
    """The encoder renders ``tool_choice`` control messages only when the
    client explicitly constrained the call (serving_kimi_k3.py parity):

      * ``required`` with a non-empty effective tool set;
      * ``none`` with any effective tools, or an explicit ``none`` even
        without tools (explicit none must render "MUST NOT call tools",
        while the implicit default none of a bare request must not).
    """
    choice = normalized.tool_choice
    has_tools = bool(normalized.effective_tools)
    if choice == "required":
        return "required" if has_tools else None
    if choice == "none":
        return "none" if (has_tools or normalized.tool_choice_explicit) else None
    return None


# --- one-call normalization -------------------------------------------------


def normalize_kimi_k3_request(
    request: dict[str, Any],
    *,
    level_by_tier: dict[str, str] | None = None,
) -> KimiK3Normalized:
    """Validate and normalize a Kimi-K3 chat request into one answer.

    Reads the request as the Python preprocessor receives it -- either the
    original wire body (direct engine path) or the native-normalized body
    where ``thinking`` was folded into ``chat_template_args``. Raises
    :class:`PreprocessError` (user-facing 400) on contract violations; never
    mutates ``request`` or its ``messages``.
    """
    level_by_tier = (
        level_by_tier if level_by_tier is not None else KIMI_K31_LEVEL_BY_TIER
    )
    messages = request.get("messages") or []
    for message_index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part_index, part in enumerate(content):
            if isinstance(part, dict) and part.get("type") == "video_url":
                raise PreprocessError(
                    f"messages[{message_index}].content[{part_index}]: "
                    "Kimi-K3 supports image input only; video_url is not supported."
                )

    dynamic = validate_kimi_k3_dynamic_tools(messages)
    raw_global = request.get("tools")
    if raw_global is not None and not isinstance(raw_global, list):
        raise PreprocessError("tools must be an array when present")
    global_wrapped: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_where: dict[str, str] = {}
    for tool_index, tool in enumerate(raw_global or []):
        name, wrapped = _validate_tool_entry(tool, f"tools[{tool_index}]")
        if name in seen:
            raise PreprocessError(
                f"tools[{tool_index}]: duplicate tool name {name!r} "
                f"(already declared in {seen_where[name]})"
            )
        seen.add(name)
        seen_where[name] = f"tools[{tool_index}]"
        global_wrapped.append(wrapped)
    for tool in dynamic:
        name = tool["function"]["name"]
        if name in seen:
            raise PreprocessError(
                f"duplicate tool name {name!r} between the global tools and a "
                "dynamic declaration"
            )
        seen.add(name)
    effective = global_wrapped + dynamic

    tool_choice = request.get("tool_choice", "auto")
    tool_choice_explicit = "tool_choice" in request
    if tool_choice == "required" and not effective:
        raise PreprocessError('tool_choice is "required" but tools is empty')
    if _is_named_tool_choice(tool_choice):
        chosen = tool_choice["function"]["name"]
        if chosen not in seen:
            raise PreprocessError(
                f"tool_choice names function {chosen!r}, but it is not present "
                f"in tools (available: {sorted(seen) or 'none'})"
            )

    ctk = _ctk(request)
    for key in ("conversation", "tools"):
        if key in ctk:
            raise PreprocessError(
                f"chat_template_kwargs cannot set {key!r}; use the request "
                "fields instead."
            )
    if "tool_choice" in ctk and tool_choice_explicit:
        raise PreprocessError(
            "chat_template_kwargs cannot set 'tool_choice' together with the "
            "request-level tool_choice; use one of the two."
        )
    if "response_format" in ctk and request.get("response_format") is not None:
        raise PreprocessError(
            "chat_template_kwargs cannot set 'response_format' together with "
            "the request-level response_format; use one of the two."
        )

    response_format = request.get("response_format")
    if response_format is not None and not isinstance(response_format, dict):
        raise PreprocessError("response_format must be a JSON object")

    thinking_effort, thinking_enabled = resolve_kimi_k3_thinking_effort(
        request, level_by_tier
    )

    return KimiK3Normalized(
        effective_tools=effective,
        raw_global_tools=list(raw_global) if raw_global else None,
        tool_choice=tool_choice,
        tool_choice_explicit=tool_choice_explicit,
        response_format=response_format,
        thinking_effort=thinking_effort,
        thinking_enabled=thinking_enabled,
    )


def kimi_k3_template_kwargs(normalized: KimiK3Normalized) -> dict[str, Any]:
    """Encoder-facing template kwargs for the Kimi-K3 rendering branch.

    Every key is set only when an explicit control exists, so a request that
    set nothing renders exactly as it did before -- and an explicit ``none``
    still renders its "MUST NOT call tools" declaration because the decision
    came from the client, not from an empty default.
    """
    kwargs: dict[str, Any] = {}
    if normalized.thinking_enabled is not None:
        kwargs["thinking"] = normalized.thinking_enabled
    if normalized.thinking_effort is not None:
        kwargs["thinking_effort"] = normalized.thinking_effort
    tool_choice = kimi_k3_passes_tool_choice(normalized)
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    if normalized.response_format is not None:
        kwargs["response_format"] = normalized.response_format
    return kwargs
