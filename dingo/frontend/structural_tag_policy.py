# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any, Literal

logger = logging.getLogger(__name__)

ToolChoiceKind = Literal["none", "auto", "required", "named", "other"]


def runtime_structural_tag_options(
    runtime_config: Any,
) -> tuple[str, str, str]:
    """Read structural-tag policy from a model-card runtime config."""
    if not isinstance(runtime_config, dict):
        return "off", "auto", "auto"
    # New cards project legacy fields for old readers. Prefer the canonical
    # setting here, including its explicit disabled value, when present.
    if "structural_tag" in runtime_config:
        config = runtime_config["structural_tag"]
        if not isinstance(config, dict):
            return "off", "auto", "auto"
        return "on", config.get("scope", "auto"), config.get("schema", "auto")
    return (
        runtime_config.get("structural_tag_mode", "off"),
        runtime_config.get("structural_tag_scope", "auto"),
        runtime_config.get("structural_tag_schema", "auto"),
    )


# Published by the vLLM worker (dynamo.vllm.main.publish_vllm_structural_tag_reasoning_policy)
# and read by the Rust preprocessor (lib/llm/src/preprocessor/structural_tag.rs).
TOOL_CALL_STRUCTURAL_TAG_EXCLUDES_REASONING_RUNTIME_KEY = (
    "tool_call_structural_tag_excludes_reasoning"
)


def runtime_structural_tag_excludes_reasoning(runtime_config: Any) -> bool:
    """Whether tool-call structural tags must leave reasoning to the backend.

    Mirrors ``resolve_reasoning_boundary`` in the Rust preprocessor: the
    canonical ``structural_tag.reasoning_boundary`` setting (``auto`` by
    default) decides, and ``auto`` follows the vLLM worker's published
    ``tool_call_structural_tag_excludes_reasoning`` flag. Missing or invalid
    worker metadata keeps the compatibility behavior (the tag models reasoning).
    """
    if not isinstance(runtime_config, dict):
        return False
    runtime_data = runtime_config.get("runtime_data")
    backend_excludes = (
        isinstance(runtime_data, dict)
        and runtime_data.get(TOOL_CALL_STRUCTURAL_TAG_EXCLUDES_REASONING_RUNTIME_KEY)
        is True
    )
    config = runtime_config.get("structural_tag")
    boundary = (
        config.get("reasoning_boundary", "auto") if isinstance(config, dict) else "auto"
    )
    if boundary == "backend":
        return True
    if boundary == "structural_tag" and backend_excludes:
        # Rust rejects this pairing at startup; follow the backend, which
        # consumes the reasoning block before the grammar sees any token.
        logger.warning(
            "structural_tag.reasoning_boundary=structural_tag conflicts with the "
            "backend's reasoning-aware guided-decoding policy; using 'backend'"
        )
    return backend_excludes


def should_attempt_structural_tag(
    *,
    mode: str,
    scope: str,
    tool_choice_kind: ToolChoiceKind,
    has_tools: bool,
    any_explicit_strict: bool,
    parallel_tool_calls_explicitly_false: bool,
) -> bool:
    """Return whether frontend preprocessing should request a structural tag."""
    if mode != "on" or not has_tools or tool_choice_kind == "none":
        return False
    if tool_choice_kind in {"required", "named"}:
        return True
    if tool_choice_kind != "auto":
        return False
    if scope == "always":
        return True
    return any_explicit_strict or parallel_tool_calls_explicitly_false


def effective_tool_strict(request_strict: bool | None, schema_mode: str) -> bool:
    """Resolve request strictness for grammar construction.

    In ``auto`` schema mode, omitted ``strict`` is schema-enforced and only an
    explicit ``strict: false`` opts out. ``strict`` schema mode overrides that
    request-level opt-out.
    """
    return schema_mode == "strict" or request_strict is not False


def _has_anchored_negative_lookahead(schema: Any) -> bool:
    """Inspect schema nodes, not examples/defaults or arbitrary property names."""
    if not isinstance(schema, dict):
        return False
    pattern = schema.get("pattern")
    if isinstance(pattern, str) and pattern.startswith("^(?!"):
        return True
    for key in (
        "properties",
        "patternProperties",
        "$defs",
        "definitions",
        "dependentSchemas",
    ):
        children = schema.get(key)
        if isinstance(children, dict) and any(
            _has_anchored_negative_lookahead(child) for child in children.values()
        ):
            return True
    for key in (
        "items",
        "additionalItems",
        "additionalProperties",
        "contains",
        "propertyNames",
        "not",
        "if",
        "then",
        "else",
        "unevaluatedProperties",
        "unevaluatedItems",
        "allOf",
        "anyOf",
        "oneOf",
        "prefixItems",
    ):
        child = schema.get(key)
        if isinstance(child, list):
            if any(_has_anchored_negative_lookahead(item) for item in child):
                return True
        elif _has_anchored_negative_lookahead(child):
            return True
    return False


def glm53_lookahead_guidance_tools(
    request: dict[str, Any], *, parser: str | None, schema_mode: str
) -> list[dict[str, Any]]:
    """October 2026 GLM-5.3 compatibility, for the grammar copy only.

    xgrammar cannot compile the reported anchored negative lookahead. For auto
    tool choice with omitted strictness, use the existing non-strict argument
    mode while retaining native function-envelope guidance. Never change prompt
    schemas or explicit strict/forced-choice policy. Remove this exception once
    the supported grammar backend compiles this pattern in regression tests.
    """
    tools = request.get("tools") or []
    model = request.get("model")
    if (
        not isinstance(model, str)
        or model.rsplit("/", 1)[-1].lower().replace("-", "").replace("_", "")
        != "glm5.3"
        or parser != "glm47"
        or schema_mode != "auto"
        or request.get("tool_choice", "auto") != "auto"
    ):
        return tools
    result = []
    for tool in tools:
        function = tool.get("function", {})
        if function.get("strict") is None and _has_anchored_negative_lookahead(
            function.get("parameters")
        ):
            result.append({**tool, "function": {**function, "strict": False}})
        else:
            result.append(tool)
    return result
