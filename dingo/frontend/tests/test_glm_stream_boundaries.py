# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real SGLang parsers with synthetic GLM token streams."""

import json

import pytest

from dingo.frontend.sglang_prepost import (
    SglangStreamingPostProcessor,
    convert_tools,
    create_parsers,
)


class ByteTokenizer:
    markers = {"<think>": 256, "</think>": 257, "<tool_call>": 258}
    unk_token_id = -1

    def convert_tokens_to_ids(self, text):
        return self.markers.get(text, -1)

    def decode(self, ids, **kwargs):
        special = {v: k.encode() for k, v in self.markers.items()}
        return b"".join(special[i] if i in special else bytes([i]) for i in ids).decode(
            "utf-8", errors="replace"
        )


def make_post():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]
    tool, reason = create_parsers(
        {"tools": tools},
        tool_call_parser_name="glm47",
        reasoning_parser_name="glm45",
        force_reasoning=True,
    )
    return SglangStreamingPostProcessor(
        tokenizer=ByteTokenizer(),
        tool_call_parser=tool,
        reasoning_parser=reason,
        sglang_tools=convert_tools(tools),
        tool_call_parser_name="glm47",
        reasoning_parser_name="glm45",
    )


def run(post, ids, batch):
    results = []
    for start in range(0, len(ids), batch):
        item = post.process_output(
            {
                "token_ids": ids[start : start + batch],
                "finish_reason": "stop" if start + batch >= len(ids) else None,
            }
        )
        if item:
            results.append(item)
    return results


@pytest.mark.parametrize("batch", [1, 2, 7, 10000])
@pytest.mark.parametrize("reasoning", ["", "字面量 </think> 不结束思考。"])
def test_literal_tags_and_utf8_before_tool_call(batch, reasoning):
    argument = "中文 <think></think>"
    ids = (
        list(reasoning.encode())
        + [257, 258]
        + list(
            (
                "lookup<arg_key>query</arg_key><arg_value>"
                + argument
                + "</arg_value></tool_call>"
            ).encode()
        )
    )
    results = run(make_post(), ids, batch)
    assert (
        "".join(r["delta"].get("reasoning_content", "") for r in results) == reasoning
    )
    assert not any(r["delta"].get("content") for r in results)
    calls = {}
    for r in results:
        for call in r["delta"].get("tool_calls", []):
            out = calls.setdefault(call["index"], {"name": "", "arguments": ""})
            for key in out:
                out[key] += call.get("function", {}).get(key, "")
    assert len(calls) == 1
    assert calls[0]["name"] == "lookup"
    assert json.loads(calls[0]["arguments"]) == {"query": argument}
    assert results[-1]["finish_reason"] == "tool_calls"


@pytest.mark.parametrize("batch", [1, 2, 7, 10000])
def test_undeclared_call_is_an_error_not_empty_success(batch):
    ids = [257, 258] + list(
        b"undeclared<arg_key>query</arg_key><arg_value>example</arg_value></tool_call>"
    )
    with pytest.raises(ValueError, match="not declared in request.tools"):
        run(make_post(), ids, batch)
