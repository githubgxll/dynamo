#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Isolated unit tests for the Kimi-K3 request normalization helpers.

These cover only the helper contract boundaries that build 6 failed on and
that no existing frontend test exercises: the dynamic-tool legality rules,
the thinking.effort precedence, the explicit-``none`` vs default rendering
distinction, raw template-input explicitness, and the chat_template_kwargs
conflict rules. They are pure-Python: no tokenizer, no sglang imports, no
HTTP.

The 54 dynamic_tools / 2 thinking_effort KVV cases must still be re-run
end-to-end per case; passing here proves the helper logic only.
"""

import pytest

from dingo.frontend.kimi_k3 import (
    KIMI_K3_LEVEL_BY_TIER,
    KIMI_K31_LEVEL_BY_TIER,
    KimiK3Normalized,
    kimi_k3_passes_tool_choice,
    kimi_k3_template_kwargs,
    normalize_kimi_k3_request,
    resolve_kimi_k3_thinking_effort,
    validate_kimi_k3_dynamic_tools,
)
from dingo.frontend.utils import PreprocessError

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
]

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


def _fn(name: str) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "description": "A tool.", "parameters": {}},
    }


# --- dynamic-tool legality (KVV dynamic_tools negatives) -------------------


@pytest.mark.parametrize(
    ("messages", "reason"),
    [
        # role constraints
        (
            [{"role": "user", "content": "hi", "tools": [WEATHER_TOOL]}],
            "user role",
        ),
        (
            [{"role": "assistant", "content": "hi", "tools": [WEATHER_TOOL]}],
            "assistant role",
        ),
        # content/tools mutual exclusion
        (
            [{"role": "system", "content": "not empty", "tools": [WEATHER_TOOL]}],
            "string content",
        ),
        (
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "x"}],
                    "tools": [WEATHER_TOOL],
                }
            ],
            "parts content",
        ),
        # tool shape
        ([{"role": "system", "content": "", "tools": {"type": "function"}}], "not array"),
        ([{"role": "system", "content": "", "tools": [None]}], "non-object entry"),
        (
            [
                {
                    "role": "system",
                    "content": "",
                    "tools": [{"type": "bogus", "function": {"name": "x"}}],
                }
            ],
            "bogus type",
        ),
        (
            [{"role": "system", "content": "", "tools": [{"function": {"name": "x"}}]}],
            "missing type",
        ),
        (
            [
                {
                    "role": "system",
                    "content": "",
                    "tools": [{"type": "function", "name": "x"}],
                }
            ],
            "missing function object",
        ),
        (
            [
                {
                    "role": "system",
                    "content": "",
                    "tools": [{"type": "function", "function": {"description": "d"}}],
                }
            ],
            "missing name",
        ),
        # name syntax
        (
            [
                {
                    "role": "system",
                    "content": "",
                    "tools": [_fn("1bad_name")],
                }
            ],
            "leading digit",
        ),
        (
            [{"role": "system", "content": "", "tools": [_fn("bad@name")]}],
            "special char",
        ),
        ([{"role": "system", "content": "", "tools": [_fn("")]}], "empty name"),
        (
            [{"role": "system", "content": "", "tools": [_fn("a" * 257)]}],
            "name over 256",
        ),
        # duplicates
        (
            [{"role": "system", "content": "", "tools": [_fn("dup"), _fn("dup")]}],
            "same-message duplicate",
        ),
        (
            [
                {"role": "system", "content": "", "tools": [_fn("dup")]},
                {"role": "system", "content": "", "tools": [_fn("dup")]},
            ],
            "cross-message duplicate",
        ),
    ],
)
def test_dynamic_tool_contract_violations_reject_before_generation(
    messages, reason
):
    with pytest.raises(PreprocessError):
        validate_kimi_k3_dynamic_tools(messages)
    # And the whole-request entry must raise too, before any rendering.
    with pytest.raises(PreprocessError):
        normalize_kimi_k3_request(
            {"model": "moonshotai/Kimi-K3.1", "messages": messages}
        )


def test_global_and_dynamic_duplicate_name_rejected():
    request = {
        "model": "moonshotai/Kimi-K3.1",
        "tools": [_fn("dup")],
        "messages": [{"role": "system", "content": "", "tools": [_fn("dup")]}],
    }
    with pytest.raises(PreprocessError, match="duplicate tool name"):
        normalize_kimi_k3_request(request)


def test_global_list_duplicate_name_rejected():
    with pytest.raises(PreprocessError, match="duplicate tool name"):
        normalize_kimi_k3_request(
            {
                "model": "moonshotai/Kimi-K3.1",
                "tools": [_fn("dup"), _fn("dup")],
                "messages": [{"role": "user", "content": "hi"}],
            }
        )


def test_bare_function_schema_form_wrapped_only_in_projection():
    """The native protocol crate preserves bare ``{name, parameters}``
    declarations verbatim. The projection wraps them into the OpenAI form so
    the parser/grammar path sees one shape, while ``messages`` keeps the
    client's raw entry for the encoder to render as sent."""
    bare = {"name": "bare_tool", "parameters": {}}
    messages = [{"role": "developer", "content": "", "tools": [dict(bare)]}]
    normalized = normalize_kimi_k3_request(
        {"model": "moonshotai/Kimi-K3.1", "messages": messages}
    )
    assert normalized.effective_tools == [
        {"type": "function", "function": bare}
    ]
    assert messages[0]["tools"] == [bare]


def test_legal_declarations_build_effective_set_without_rewriting_messages():
    messages = [
        {"role": "system", "content": "", "tools": [WEATHER_TOOL]},
        {"role": "user", "content": "weather?"},
        {"role": "system", "content": "", "tools": [_fn("get_stock")]},
    ]
    snapshot = [dict(message) for message in messages]
    request = {
        "model": "moonshotai/Kimi-K3.1",
        "tools": [_fn("global_tool")],
        "messages": messages,
        "tool_choice": "required",
    }
    normalized = normalize_kimi_k3_request(request)

    assert [t["function"]["name"] for t in normalized.effective_tools] == [
        "global_tool",
        "get_weather",
        "get_stock",
    ]
    # messages and request stay untouched: the dynamic declarations keep
    # their position and the global tools are not hoisted anywhere.
    assert messages == snapshot
    assert "tools" not in messages[0] or messages[0]["tools"] == [WEATHER_TOOL]
    assert normalized.raw_global_tools == [_fn("global_tool")]
    assert request["tools"] == [_fn("global_tool")]


def test_256_char_name_boundary_accepted_and_257_rejected():
    ok = _fn("t" + "a" * 255)  # exactly 256
    request = {
        "model": "moonshotai/Kimi-K3.1",
        "messages": [{"role": "system", "content": "", "tools": [ok]}],
        "tool_choice": "required",
    }
    normalized = normalize_kimi_k3_request(request)
    assert normalized.effective_tools[0]["function"]["name"] == "t" + "a" * 255

    with pytest.raises(PreprocessError, match="256"):
        normalize_kimi_k3_request(
            {
                "model": "moonshotai/Kimi-K3.1",
                "messages": [{"role": "system", "content": "", "tools": [_fn("a" * 257)]}],
            }
        )


def test_required_with_only_dynamic_tools_passes_empty_check():
    request = {
        "model": "moonshotai/Kimi-K3.1",
        "messages": [{"role": "system", "content": "", "tools": [WEATHER_TOOL]}],
        "tool_choice": "required",
    }
    normalized = normalize_kimi_k3_request(request)
    assert normalized.tool_choice == "required"
    assert kimi_k3_passes_tool_choice(normalized) == "required"


def test_required_without_any_tools_rejected():
    with pytest.raises(PreprocessError, match="required"):
        normalize_kimi_k3_request(
            {
                "model": "moonshotai/Kimi-K3.1",
                "messages": [{"role": "user", "content": "hi"}],
                "tool_choice": "required",
            }
        )


def test_named_choice_checks_effective_set_including_dynamic():
    request = {
        "model": "moonshotai/Kimi-K3.1",
        "messages": [{"role": "system", "content": "", "tools": [WEATHER_TOOL]}],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
    }
    normalized = normalize_kimi_k3_request(request)  # dynamic name resolves
    with pytest.raises(PreprocessError, match="not present"):
        normalize_kimi_k3_request(
            {
                "model": "moonshotai/Kimi-K3.1",
                "messages": request["messages"],
                "tool_choice": {"type": "function", "function": {"name": "nope"}},
            }
        )


def test_raw_global_tools_keep_field_explicitness():
    tool = {"type": "function", "function": {"name": "t", "parameters": None}}
    normalized = normalize_kimi_k3_request(
        {
            "model": "moonshotai/Kimi-K3.1",
            "tools": [tool],
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    # No strict: False / default injection -- the raw dict comes back as sent.
    assert normalized.raw_global_tools == [tool]
    assert "strict" not in normalized.raw_global_tools[0]["function"]


# --- thinking.effort precedence (KVV effort conflicts) ---------------------


def test_ctk_folded_effort_is_authoritative():
    # The native fold resolves everything once into the CTK; a stale root
    # reasoning_effort must not win over it.
    request = {
        "model": "moonshotai/Kimi-K3.1",
        "reasoning_effort": "max",
        "chat_template_kwargs": {"thinking": True, "thinking_effort": "low"},
    }
    effort, enabled = resolve_kimi_k3_thinking_effort(request, KIMI_K31_LEVEL_BY_TIER)
    assert (effort, enabled) == ("low", True)


def test_explicit_disabled_beats_effort_and_reasoning_effort():
    request = {
        "model": "moonshotai/Kimi-K3.1",
        "thinking": {"type": "disabled", "keep": "all", "effort": "high"},
        "reasoning_effort": "max",
    }
    effort, enabled = resolve_kimi_k3_thinking_effort(request, KIMI_K31_LEVEL_BY_TIER)
    assert (effort, enabled) == (None, False)
    kwargs = kimi_k3_template_kwargs(normalize_kimi_k3_request(request))
    assert kwargs.get("thinking") is False and "thinking_effort" not in kwargs


def test_explicit_disabled_beats_folded_ctk_effort():
    # The client's explicit disabled switch outranks even the folded grade.
    request = {
        "model": "moonshotai/Kimi-K3.1",
        "thinking": {"type": "disabled"},
        "chat_template_kwargs": {"thinking": True, "thinking_effort": "low"},
    }
    effort, enabled = resolve_kimi_k3_thinking_effort(request, KIMI_K31_LEVEL_BY_TIER)
    assert (effort, enabled) == (None, False)


def test_thinking_effort_beats_top_level_reasoning_effort():
    # KVV: thinking.effort=low + reasoning_effort=max must behave like low.
    request = {
        "model": "moonshotai/Kimi-K3.1",
        "thinking": {"type": "enabled", "keep": "all", "effort": "low"},
        "reasoning_effort": "max",
    }
    effort, enabled = resolve_kimi_k3_thinking_effort(request, KIMI_K31_LEVEL_BY_TIER)
    assert (effort, enabled) == ("low", True)


def test_reasoning_effort_maps_when_effort_absent():
    request = {
        "model": "moonshotai/Kimi-K3.1",
        "thinking": {"type": "enabled", "keep": "all"},
        "reasoning_effort": "max",
    }
    effort, _ = resolve_kimi_k3_thinking_effort(request, KIMI_K31_LEVEL_BY_TIER)
    assert effort == "max"


def test_reasoning_effort_tier_mapping_per_model():
    base = {"model": "x"}
    assert resolve_kimi_k3_thinking_effort(
        {**base, "reasoning_effort": "medium"}, KIMI_K3_LEVEL_BY_TIER
    ) == ("high", None)
    assert resolve_kimi_k3_thinking_effort(
        {**base, "reasoning_effort": "medium"}, KIMI_K31_LEVEL_BY_TIER
    ) == ("medium", None)
    assert resolve_kimi_k3_thinking_effort(
        {**base, "reasoning_effort": "minimal"}, KIMI_K31_LEVEL_BY_TIER
    ) == ("low", None)
    with pytest.raises(PreprocessError, match="reasoning_effort"):
        resolve_kimi_k3_thinking_effort(
            {**base, "reasoning_effort": "bogus"}, KIMI_K31_LEVEL_BY_TIER
        )
    with pytest.raises(PreprocessError, match="thinking_effort"):
        resolve_kimi_k3_thinking_effort(
            {**base, "chat_template_kwargs": {"thinking_effort": "bogus"}},
            KIMI_K31_LEVEL_BY_TIER,
        )
    with pytest.raises(PreprocessError, match="thinking.effort"):
        resolve_kimi_k3_thinking_effort(
            {**base, "thinking": {"type": "enabled", "effort": "bogus"}},
            KIMI_K31_LEVEL_BY_TIER,
        )


def test_invalid_thinking_type_rejected_not_substituted():
    with pytest.raises(PreprocessError, match="invalid thinking"):
        resolve_kimi_k3_thinking_effort(
            {"model": "x", "thinking": {"type": "bogus"}}, KIMI_K31_LEVEL_BY_TIER
        )


def test_no_controls_leave_encoder_default():
    effort, enabled = resolve_kimi_k3_thinking_effort(
        {"model": "x", "messages": []}, KIMI_K31_LEVEL_BY_TIER
    )
    assert (effort, enabled) == (None, None)
    assert kimi_k3_template_kwargs(
        KimiK3Normalized()
    ) == {}


def test_reasoning_effort_none_disables_but_default_none_does_not():
    effort, enabled = resolve_kimi_k3_thinking_effort(
        {"model": "x", "reasoning_effort": "none"}, KIMI_K31_LEVEL_BY_TIER
    )
    assert (effort, enabled) == (None, False)
    effort_default, enabled_default = resolve_kimi_k3_thinking_effort(
        {"model": "x"}, KIMI_K31_LEVEL_BY_TIER
    )
    assert (effort_default, enabled_default) == (None, None)


# --- explicit "none" vs default none, and prompt-declaration distinction ----


def test_explicit_none_renders_control_even_without_tools():
    normalized = normalize_kimi_k3_request(
        {
            "model": "moonshotai/Kimi-K3.1",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": "none",
        }
    )
    assert normalized.tool_choice_explicit is True
    assert kimi_k3_passes_tool_choice(normalized) == "none"


def test_implicit_default_none_is_not_projected():
    normalized = normalize_kimi_k3_request(
        {"model": "moonshotai/Kimi-K3.1", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert normalized.tool_choice_explicit is False
    assert kimi_k3_passes_tool_choice(normalized) is None


def test_explicit_none_with_tools_renders_control_and_keeps_declarations():
    # tool_choice=none forbids calling but must NOT drop the tool
    # declarations from the prompt inputs.
    normalized = normalize_kimi_k3_request(
        {
            "model": "moonshotai/Kimi-K3.1",
            "tools": [WEATHER_TOOL],
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": "none",
        }
    )
    kwargs = kimi_k3_template_kwargs(normalized)
    assert kwargs["tool_choice"] == "none"
    assert normalized.raw_global_tools == [WEATHER_TOOL]
    assert normalized.effective_tools == [WEATHER_TOOL]


def test_required_without_effective_tools_projects_no_control():
    normalized = KimiK3Normalized(
        effective_tools=[], tool_choice="required", tool_choice_explicit=True
    )
    assert kimi_k3_passes_tool_choice(normalized) is None


# --- chat_template_kwargs conflicts ----------------------------------------


@pytest.mark.parametrize(
    "ctk",
    [
        {"conversation": [{"role": "user", "content": "x"}]},
        {"tools": [WEATHER_TOOL]},
    ],
)
def test_reserved_ctk_keys_rejected(ctk):
    with pytest.raises(PreprocessError, match="chat_template_kwargs"):
        normalize_kimi_k3_request(
            {
                "model": "moonshotai/Kimi-K3.1",
                "messages": [{"role": "user", "content": "hi"}],
                "chat_template_kwargs": ctk,
            }
        )


def test_ctk_tool_choice_conflicts_with_request_field():
    with pytest.raises(PreprocessError, match="tool_choice"):
        normalize_kimi_k3_request(
            {
                "model": "moonshotai/Kimi-K3.1",
                "messages": [{"role": "user", "content": "hi"}],
                "tool_choice": "auto",
                "chat_template_kwargs": {"tool_choice": "required"},
            }
        )


def test_ctk_tool_choice_alone_is_not_conflict():
    # An operator projection without a request-level field is the folded
    # decision; it must pass.
    normalized = normalize_kimi_k3_request(
        {
            "model": "moonshotai/Kimi-K3.1",
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"tool_choice": "required"},
        }
    )
    assert normalized.tool_choice == "auto"  # request field untouched


def test_ctk_response_format_conflict_rejected():
    with pytest.raises(PreprocessError, match="response_format"):
        normalize_kimi_k3_request(
            {
                "model": "moonshotai/Kimi-K3.1",
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {"type": "json_object"},
                "chat_template_kwargs": {"response_format": {"type": "text"}},
            }
        )


def test_response_format_projected_into_template_kwargs():
    response_format = {"type": "json_object"}
    normalized = normalize_kimi_k3_request(
        {
            "model": "moonshotai/Kimi-K3.1",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": response_format,
        }
    )
    assert kimi_k3_template_kwargs(normalized)["response_format"] == response_format
