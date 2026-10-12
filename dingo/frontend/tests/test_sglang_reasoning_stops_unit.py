# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only regression tests using SGLang's actual GLM reasoning parser."""

import asyncio
import json

import pytest
from _routed_engine_fakes import FakeRoutedItem
from sglang.srt.parser.reasoning_parser import ReasoningParser

from dingo.frontend.sglang_prepost import SglangStreamingPostProcessor
from dingo.frontend.sglang_processor import (
    DEFAULT_SGLANG_MAX_TOKENS,
    SglangProcessor,
    _build_dynamo_preproc,
    _request_stop_strings,
)

pytestmark = [pytest.mark.unit, pytest.mark.sglang, pytest.mark.gpu_0]


class TextTokenizer:
    def decode(self, token_ids, *, skip_special_tokens):
        return "".join(chr(token_id) for token_id in token_ids)


def make_post(stops, **kwargs):
    return SglangStreamingPostProcessor(
        tokenizer=TextTokenizer(),
        tool_call_parser=None,
        reasoning_parser=ReasoningParser(
            model_type="glm45", stream_reasoning=True, force_reasoning=True
        ),
        stop_strings=set(stops),
        **kwargs,
    )


def feed(post, text, finish=None):
    return post.process_output(
        {"token_ids": list(map(ord, text)), "finish_reason": finish}
    )


@pytest.mark.parametrize(
    "limits,expected",
    [
        ({}, DEFAULT_SGLANG_MAX_TOKENS),
        ({"max_tokens": 256}, 256),
        ({"max_tokens": 256, "max_completion_tokens": 512}, 512),
        ({"max_tokens": 256, "max_completion_tokens": 0}, 0),
        (
            {"max_tokens": None, "max_completion_tokens": None},
            DEFAULT_SGLANG_MAX_TOKENS,
        ),
    ],
)
def test_glm_negative_penalty_is_neutral_with_finite_limit(limits, expected):
    result = _build_dynamo_preproc(
        {"frequency_penalty": -2, **limits}, [1], "glm-5.3", None
    )
    assert result["sampling_options"]["frequency_penalty"] == 0.0
    assert result["stop_conditions"]["max_tokens"] == expected


@pytest.mark.parametrize("model", ["glm-5.3", "GLM-5.3", "zai-org/GLM-5.3"])
@pytest.mark.parametrize("penalty", [-2, -0.5, 0, 0.5, 2])
def test_glm_frequency_penalty_compatibility(model, penalty):
    request = {"model": model, "frequency_penalty": penalty}
    result = _build_dynamo_preproc(request, [1], model, None, require_reasoning=True)
    assert result["sampling_options"]["frequency_penalty"] == max(0, penalty)
    assert result["model"] == model
    assert result["require_reasoning"] is True
    assert request == {"model": model, "frequency_penalty": penalty}


@pytest.mark.parametrize(
    "model", ["test-model", "glm-5.2", "glm-5.3-flash", "my-glm-5.3"]
)
def test_other_models_keep_negative_frequency_penalty(model):
    result = _build_dynamo_preproc(
        {"frequency_penalty": -2}, [1], model, None, reasoning_parser=object()
    )
    assert result["sampling_options"]["frequency_penalty"] == -2


def test_message_field_does_not_become_sampling_option():
    result = _build_dynamo_preproc(
        {"messages": [{"role": "user", "content": "hi", "frequency_penalty": -2}]},
        [1],
        "glm-5.3",
        None,
    )
    assert result["sampling_options"]["frequency_penalty"] == 0.0


def test_only_string_stops_move_to_reasoning_frontend():
    request = {"stop": ["\n"], "stop_token_ids": [2]}
    result = _build_dynamo_preproc(
        request, [1], "glm-5.3", [3], reasoning_parser=object()
    )
    assert result["stop_conditions"]["stop"] == []
    assert result["stop_conditions"]["stop_token_ids"] == [2]
    assert result["eos_token_ids"] == [3]
    assert request["stop"] == ["\n"]
    plain = _build_dynamo_preproc(request, [1], "glm-5.3", [3])
    assert plain["stop_conditions"]["stop"] == ["\n"]


@pytest.mark.parametrize("include", [False, True])
def test_reasoning_newline_does_not_stop_answer_newline_does(include):
    request = {"stop": "\n", "include_stop_str_in_output": include}
    post = make_post(
        _request_stop_strings(request, match_visible=True),
        include_stop_str_in_output=include,
    )
    thought = feed(post, "thinking\nmore thinking\n")
    assert thought["delta"]["reasoning_content"] == "thinking\nmore thinking\n"
    assert thought["finish_reason"] is None
    answer = feed(post, "</think>ALPHA\nBETA")
    assert answer["delta"]["content"] == ("ALPHA\n" if include else "ALPHA")
    assert answer["finish_reason"] == "stop"
    assert post.matched_stop == "\n"
    assert feed(post, "must not appear") is None


@pytest.mark.parametrize("split", range(1, 5))
def test_stop_across_chunks(split):
    post = make_post(["<END>"])
    feed(post, "thinking\n</think>")
    first = feed(post, "hello<END>"[: 5 + split])
    second = feed(post, "<END>"[split:] + "ignored")
    assert first["delta"]["content"] == "hello"
    assert first["finish_reason"] is None
    assert second["finish_reason"] == "stop"
    assert not second["delta"].get("content")


def test_partial_stop_flushed_on_length_and_reasoning_only_length():
    post = make_post(["<END>"])
    feed(post, "thinking</think>")
    feed(post, "hello<EN")
    end = feed(post, "", "length")
    assert end["delta"]["content"] == "<EN"
    assert end["finish_reason"] == "length"
    post = make_post(["\n"])
    end = feed(post, "thinking\n" * 20, "length")
    assert end["finish_reason"] == "length"
    assert "content" not in end["delta"]
    assert end["delta"]["reasoning_content"].count("\n") == 20


def test_first_match_and_min_tokens():
    post = make_post(["END", "STOP"])
    end = feed(post, "thinking</think>answerSTOPdiscardEND")
    assert end["delta"]["content"] == "answer"
    assert post.matched_stop == "STOP"
    post = make_post(["\n"], min_tokens=150)
    assert feed(post, "thinking</think>a\n")["finish_reason"] is None
    end = feed(post, "x" * 160 + "\nignored")
    assert end["finish_reason"] == "stop"


@pytest.mark.parametrize("interval", [1, 20])
@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("native_stream", [False, True])
def test_local_stop_closes_generation_and_emits_terminal_usage(
    interval, continuous, native_stream
):
    class Engine:
        closed = False
        consumed = 0

        async def generate(self, request, **kwargs):
            texts = ["thinking\n", "</think>ALPHA\nBETAignored", "never read"]

            # Native AsyncResponseStream has no aclose method: dropping its
            # last reference closes the receiver and stops child generation.
            engine = self

            class NativeStream:
                def __aiter__(self):
                    return self

                async def __anext__(self):
                    if engine.consumed == len(texts):
                        raise StopAsyncIteration
                    text = texts[engine.consumed]
                    engine.consumed += 1
                    return FakeRoutedItem({"token_ids": list(map(ord, text))})

                def __del__(self):
                    engine.closed = True

            if native_stream:
                return NativeStream()

            async def stream():
                try:
                    for text in texts:
                        self.consumed += 1
                        yield FakeRoutedItem({"token_ids": list(map(ord, text))})
                finally:
                    self.closed = True

            return stream()

    engine = Engine()
    processor = SglangProcessor(
        tokenizer=TextTokenizer(),
        routed_engine=engine,
        tool_call_parser_name=None,
        reasoning_parser_name="glm45",
        eos_token_ids=None,
        stream_interval=interval,
    )

    async def collect():
        items = []
        async for item in processor._generate_and_stream(
            "test",
            {
                "model": "glm-5.3",
                "nvext": {"extra_fields": ["stop_reason"]},
                "stream_options": {"continuous_usage_stats": continuous},
            },
            {},
            [1, 2],
            make_post(["\n"]),
        ):
            items.append(item)
            choices = item.get("data", {}).get("choices", [])
            if choices and choices[0]["finish_reason"]:
                assert engine.closed
        return items

    items = asyncio.run(collect())
    assert engine.consumed == 2
    terminal = items[-1]["data"]
    assert terminal["choices"][0]["finish_reason"] == "stop"
    assert terminal["choices"][0]["delta"]["content"] == "ALPHA"
    generated = len("thinking\n</think>ALPHA\nBETAignored")
    assert terminal["usage"]["completion_tokens"] == generated
    assert terminal["usage"]["total_tokens"] == generated + 2
    assert terminal["nvext"]["stop_reason"] == "\n"
    assert json.loads(items[-1]["comment"][0])["output_tokens"] == generated
