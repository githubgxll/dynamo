# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only release registry/parser integration; never downloads a tokenizer.

Run with DYN_GLM_TOKENIZER_PATH=/models/GLM-5.3-Flash pytest -q <this file>.
Requires the deployed vLLM release dependencies, but no engine or GPU allocation.
"""

import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from transformers import AutoTokenizer
from vllm.reasoning import ReasoningParserManager
from vllm.sampling_params import SamplingParams
from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser

from dingo.frontend.prepost import StreamingPostProcessor, preprocess_chat_request

pytestmark = [pytest.mark.unit, pytest.mark.vllm, pytest.mark.pre_merge]


@pytest.fixture(scope="module")
def glm_tokenizer():
    path = os.environ.get("DYN_GLM_TOKENIZER_PATH")
    if not path:
        pytest.skip("set DYN_GLM_TOKENIZER_PATH to a local GLM tokenizer directory")
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def run_glm_roundtrip(
    tokenizer, *, named=False, stream=True, reasoning=False, strict=None
):
    """Callable without pytest/plugins for isolated deployed-image verification."""
    function = {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    }
    if strict is not None:
        function["strict"] = strict
    request = {
        "model": "glm-test",
        "messages": [{"role": "user", "content": "Weather in Hangzhou?"}],
        "tools": [{"type": "function", "function": function}],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
        if named
        else "required",
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": reasoning},
    }
    reasoning_class = ReasoningParserManager.get_reasoning_parser("glm45")
    result = asyncio.run(
        preprocess_chat_request(
            request,
            tokenizer=tokenizer,
            renderer=SimpleNamespace(
                render_messages_async=AsyncMock(
                    return_value=(
                        None,
                        {
                            "prompt_token_ids": tokenizer.encode(
                                "<think>", add_special_tokens=False
                            )
                        },
                    )
                )
            ),
            tool_parser_class=Glm47MoeModelToolParser,
            reasoning_parser_class=reasoning_class,
        )
    )
    assert set(result.guided_decoding) == {"structural_tag"}
    tag = result.guided_decoding["structural_tag"]
    assert "<tool_call>" in tag
    assert "get_weather" in tag
    assert result.request_for_sampling.structured_outputs.json is None
    assert result.request_for_sampling.response_format is None
    assert result.request_for_sampling.skip_special_tokens is False

    native = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Hangzhou</arg_value></tool_call>"
    extracted = result.tool_parser.extract_tool_calls(
        native, result.request_for_sampling
    )
    assert extracted.tools_called
    assert extracted.tool_calls[0].function.name == "get_weather"
    assert json.loads(extracted.tool_calls[0].function.arguments) == {
        "city": "Hangzhou"
    }

    # Use a fresh parser after the direct extraction; parsers retain stream state.
    post = StreamingPostProcessor(
        tokenizer=tokenizer,
        request_for_sampling=result.request_for_sampling,
        sampling_params=SamplingParams(skip_special_tokens=False),
        prompt_token_ids=result.prompt_token_ids,
        tool_parser=Glm47MoeModelToolParser(
            tokenizer, result.request_for_sampling.tools
        ),
        reasoning_parser_class=reasoning_class,
        chat_template_kwargs=result.chat_template_kwargs,
        stream_response=stream,
    )
    if reasoning:
        # Deliberately combine reasoning-end and tool-start in one chunk.
        chunks = [
            "Checking weather",
            "</think><tool_call>get_weather",
            native.split("get_weather", 1)[1],
        ]
    else:
        # Special markers are atomic tokenizer IDs, not arbitrary character
        # fragments. Place markers in separate chunks and split argument text.
        chunks = [
            "<tool_call>",
            "get_weather",
            "<arg_key>",
            "city",
            "</arg_key>",
            "<arg_value>",
            "Hang",
            "zhou",
            "</arg_value>",
            "</tool_call>",
        ]
    choices = []
    for index, text in enumerate(chunks):
        choice = post.process_output(
            SimpleNamespace(
                index=0,
                text=text,
                token_ids=tokenizer.encode(text, add_special_tokens=False),
                finish_reason="stop" if index == len(chunks) - 1 else None,
                logprobs=None,
            )
        )
        if choice is not None:
            choices.append(choice)
    assert choices[-1]["finish_reason"] == "tool_calls"
    calls = {}
    for choice in choices:
        assert not choice["delta"].get("content")
        for call in choice["delta"].get("tool_calls", []):
            merged = calls.setdefault(call["index"], {"name": "", "arguments": ""})
            function_delta = call.get("function", {})
            merged["name"] += function_delta.get("name") or ""
            merged["arguments"] += function_delta.get("arguments") or ""
    assert len(calls) == 1
    assert calls[0]["name"] == "get_weather"
    assert json.loads(calls[0]["arguments"]) == {"city": "Hangzhou"}
    if reasoning:
        assert "Checking weather" in "".join(
            choice["delta"].get("reasoning_content", "") for choice in choices
        )
    return tag


@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("reasoning", [False, True])
@pytest.mark.parametrize("strict", [None, False, True])
def test_real_glm_registry_parser_roundtrip(
    glm_tokenizer, named, stream, reasoning, strict
):
    run_glm_roundtrip(
        glm_tokenizer, named=named, stream=stream, reasoning=reasoning, strict=strict
    )
