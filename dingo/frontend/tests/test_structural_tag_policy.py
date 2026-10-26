# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from copy import deepcopy

from dingo.frontend.structural_tag_policy import (
    effective_tool_strict,
    glm53_lookahead_guidance_tools,
    runtime_structural_tag_options,
    should_attempt_structural_tag,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.parametrize(
    ("mode", "scope", "choice", "strict", "single", "expected"),
    [
        ("off", "always", "auto", False, False, False),
        ("on", "always", "auto", False, False, True),
        ("on", "auto", "auto", True, False, True),
        ("on", "auto", "auto", False, True, True),
        ("on", "auto", "auto", False, False, False),
        ("on", "auto", "required", False, False, True),
        ("on", "auto", "named", False, False, True),
        ("on", "always", "none", True, True, False),
    ],
)
def test_structural_tag_activation_policy(
    mode, scope, choice, strict, single, expected
):
    assert (
        should_attempt_structural_tag(
            mode=mode,
            scope=scope,
            tool_choice_kind=choice,
            has_tools=True,
            any_explicit_strict=strict,
            parallel_tool_calls_explicitly_false=single,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("request_strict", "schema_mode", "expected"),
    [
        (None, "auto", True),
        (True, "auto", True),
        (False, "auto", False),
        (False, "strict", True),
    ],
)
def test_effective_tool_strict(request_strict, schema_mode, expected):
    assert effective_tool_strict(request_strict, schema_mode) is expected


def test_runtime_structural_tag_options_are_conservative_when_absent():
    assert runtime_structural_tag_options(None) == ("off", "auto", "auto")
    assert runtime_structural_tag_options({}) == ("off", "auto", "auto")


def test_runtime_structural_tag_options_read_model_card_policy():
    assert runtime_structural_tag_options(
        {
            "structural_tag_mode": "on",
            "structural_tag_scope": "always",
            "structural_tag_schema": "strict",
        }
    ) == ("on", "always", "strict")


@pytest.mark.parametrize(
    ("strict", "mode", "expected"),
    [
        (None, "auto", False),
        (True, "auto", True),
        (None, "strict", None),
    ],
)
def test_glm53_report_lookahead_only_relaxes_implicit_strictness(
    strict, mode, expected
):
    request = {
        "model": "glm-5.3",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "demo",
                    "strict": strict,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "pattern": "^(?!x)y"}
                        },
                    },
                },
            }
        ],
    }
    before = deepcopy(request)
    result = glm53_lookahead_guidance_tools(request, parser="glm47", schema_mode=mode)
    assert result[0]["function"]["strict"] is expected
    assert (
        result[0]["function"]["parameters"]
        == before["tools"][0]["function"]["parameters"]
    )
    assert request == before
    request["tool_choice"] = "required"
    assert (
        glm53_lookahead_guidance_tools(request, parser="glm47", schema_mode=mode)
        == request["tools"]
    )
