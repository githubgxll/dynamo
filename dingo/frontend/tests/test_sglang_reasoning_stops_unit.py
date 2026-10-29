# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only regression tests using SGLang's actual GLM reasoning parser."""

import asyncio
import json
import re
from types import SimpleNamespace

import pytest
from _routed_engine_fakes import FakeRoutedEngine, FakeRoutedItem
from sglang.srt.parser.reasoning_parser import ReasoningParser

import dingo.frontend.sglang_processor as processor_module
from dingo.frontend.sglang_prepost import SglangStreamingPostProcessor
from dingo.frontend.sglang_processor import (
    DEFAULT_SGLANG_MAX_TOKENS,
    SglangProcessor,
    _build_dynamo_preproc,
    _infer_reasoning_parser_name,
    _request_stop_strings,
)

pytestmark = [pytest.mark.unit, pytest.mark.sglang, pytest.mark.gpu_0]


@pytest.mark.parametrize(
    "model_type,expected",
    [
        ("glm_moe_dsa", "glm45"),
        ("glm5_next", "glm45"),
        ("qwen3", None),
        ("qwen3_moe", None),
        ("qwen3_next", None),
        ("unknown", None),
    ],
)
def test_infer_reasoning_parser_requires_known_model_and_thinking_template(
    tmp_path, model_type, expected
):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    assert (
        _infer_reasoning_parser_name(str(tmp_path), "<|assistant|><think>") == expected
    )
    assert _infer_reasoning_parser_name(str(tmp_path), "plain assistant") is None


def test_infer_reasoning_parser_missing_config(tmp_path):
    assert _infer_reasoning_parser_name(str(tmp_path), "<|assistant|><think>") is None


@pytest.mark.parametrize(
    "model_type,expected,source",
    [
        ("glm5_next", "glm45", "inferred"),
        ("glm_moe_dsa", "glm45", "inferred"),
        ("glm5_next", "glm45", "frontend"),
        ("glm5_next", "glm45", "worker"),
        ("qwen3", "qwen3", "frontend"),
        ("qwen3", "qwen3", "worker"),
    ],
)
def test_factory_selects_parser_and_preserves_worker_stop_during_reasoning(
    tmp_path, monkeypatch, model_type, expected, source
):
    # Exercise discovery -> factory -> parser -> preprocessing/streaming.
    # No model weights or tokenizer downloads are needed.
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    tokenizer = TextTokenizer()
    tokenizer.chat_template = "<|assistant|><think>"
    monkeypatch.setattr(processor_module, "_load_tokenizer", lambda *_: tokenizer)
    monkeypatch.setattr(
        processor_module, "_resolve_model_eos_token_ids", lambda **_: []
    )
    monkeypatch.setattr(
        processor_module, "PythonAsyncEngine", lambda generator, _: generator
    )
    configured = "qwen3"
    selected = expected if source == "inferred" else configured
    factory = processor_module.SglangEngineFactory(
        SimpleNamespace(
            trust_remote_code=False,
            glm_history_arguments_models=[],
            preprocess_workers=0,
            exclude_tools_when_tool_choice_none=True,
        ),
        reasoning_parser_name=configured if source == "frontend" else None,
    )
    mdc = SimpleNamespace(
        model_type=lambda: SimpleNamespace(supports_chat=lambda: True),
        local_dir=lambda: str(tmp_path),
        name=lambda: "custom-serving-alias",
        runtime_config=lambda: (
            {"reasoning_parser": configured} if source == "worker" else {}
        ),
    )
    engine = FakeRoutedEngine(
        [
            {
                "token_ids": list(map(ord, "thinking\nmore thinking")),
            },
        ]
    )

    async def run():
        generator = await factory.chat_engine_factory(None, mdc, engine)
        processor = generator.__self__
        assert processor.reasoning_parser_name == selected
        request = {"model": "custom-serving-alias", "stop": ["\n"]}
        parser = ReasoningParser(
            model_type=processor.reasoning_parser_name,
            stream_reasoning=True,
            force_reasoning=True,
        )
        preproc = _build_dynamo_preproc(
            request, [1], request["model"], None, reasoning_parser=parser
        )
        assert preproc["stop_conditions"]["stop"] == ["\n"]
        post = SglangStreamingPostProcessor(
            tokenizer=tokenizer,
            tool_call_parser=None,
            reasoning_parser=parser,
            stop_strings={"\n"},
        )
        return [
            x
            async for x in processor._generate_and_stream(
                "test", request, preproc, [1], post
            )
        ]

    items = asyncio.run(run())
    choices = [c for item in items for c in item.get("data", {}).get("choices", [])]
    assert (
        "".join(c["delta"].get("reasoning_content", "") for c in choices)
        == "thinking"
    )
    assert not any(c["delta"].get("content") for c in choices)
    assert choices[-1]["finish_reason"] == "stop"


class TextTokenizer:
    def decode(self, token_ids, *, skip_special_tokens):
        return "".join(chr(token_id) for token_id in token_ids)


class ControlTokenizer:
    markers = {"<think>": 256, "</think>": 257, "<tool_call>": 258}
    unk_token_id = -1

    def convert_tokens_to_ids(self, text):
        return self.markers.get(text, -1)

    def encode(self, text):
        ids = []
        for part in re.split(r"(<think>|</think>|<tool_call>)", text):
            ids.extend([self.markers[part]] if part in self.markers else part.encode())
        return ids

    def decode(self, token_ids, *, skip_special_tokens):
        markers = {v: k.encode() for k, v in self.markers.items()}
        return b"".join(
            markers[token] if token in markers else bytes([token])
            for token in token_ids
        ).decode("utf-8", errors="replace")


def make_post(stops, *, token_aware=False, **kwargs):
    return SglangStreamingPostProcessor(
        tokenizer=ControlTokenizer() if token_aware else TextTokenizer(),
        tool_call_parser=None,
        reasoning_parser=ReasoningParser(
            model_type="glm45", stream_reasoning=True, force_reasoning=True
        ),
        reasoning_parser_name="glm45" if token_aware else None,
        stop_strings=set(stops),
        **kwargs,
    )


def feed(post, text, finish=None):
    ids = (
        post.tokenizer.encode(text)
        if isinstance(post.tokenizer, ControlTokenizer)
        else list(map(ord, text))
    )
    return post.process_output(
        {"token_ids": ids, "finish_reason": finish}
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


@pytest.mark.parametrize("reasoning_parser", [None, object()])
@pytest.mark.parametrize("stop", ["\n", ["\n", "END"]])
@pytest.mark.parametrize("include", [False, True])
def test_string_and_token_stops_remain_worker_owned(reasoning_parser, stop, include):
    request = {
        "stop": stop,
        "stop_token_ids": [2],
        "include_stop_str_in_output": include,
        "min_tokens": 150,
    }
    result = _build_dynamo_preproc(
        request, [1], "glm-5.3", [3], reasoning_parser=reasoning_parser
    )
    assert result["stop_conditions"]["stop"] == (
        [stop] if isinstance(stop, str) else stop
    )
    assert result["stop_conditions"]["stop_token_ids"] == [2]
    assert result["eos_token_ids"] == [3]
    assert result["sampling_options"]["include_stop_str_in_output"] is include
    assert result["stop_conditions"]["min_tokens"] == 150
    assert request["stop"] == stop


@pytest.mark.parametrize("token_aware", [False, True])
@pytest.mark.parametrize("include", [False, True])
def test_stop_finishes_during_reasoning_without_worker_matching(include, token_aware):
    request = {"stop": "\n", "include_stop_str_in_output": include}
    post = make_post(
        _request_stop_strings(request, match_visible=True),
        token_aware=token_aware,
        include_stop_str_in_output=include,
    )
    thought = feed(post, "thinking\nmore thinking</think>answer")
    assert thought["delta"]["reasoning_content"] == (
        "thinking\n" if include else "thinking"
    )
    assert not thought["delta"].get("content")
    assert thought["finish_reason"] == "stop"
    assert post.locally_finished
    assert post.local_stop_reason == "\n"
    assert feed(post, "ignored") is None


@pytest.mark.parametrize("token_aware", [False, True])
def test_min_tokens_keeps_early_stops_and_matches_later_stop(token_aware):
    post = make_post(["\n"], token_aware=token_aware, min_tokens=100)
    thought = feed(post, "thinking\n</think>")
    assert thought["delta"]["reasoning_content"] == "thinking\n"
    answer = feed(post, "answer\n")
    assert answer["delta"]["content"] == "answer\n"
    assert answer["finish_reason"] is None
    assert not post.has_pending_stop_text
    end = feed(post, "x" * 100 + "\nignored")
    assert end["delta"]["content"] == "x" * 100
    assert end["finish_reason"] == "stop"


@pytest.mark.parametrize("token_aware", [False, True])
@pytest.mark.parametrize("split", range(1, 5))
@pytest.mark.parametrize("in_reasoning", [False, True])
def test_stop_across_chunks(token_aware, split, in_reasoning):
    post = make_post(["<END>"], token_aware=token_aware)
    if not in_reasoning:
        feed(post, "thinking</think>")
    first = feed(post, "hello<END>"[: 5 + split])
    second = feed(post, "<END>"[split:] + "ignored")
    field = "reasoning_content" if in_reasoning else "content"
    assert first["delta"][field] == "hello"
    assert first["finish_reason"] is None
    assert second["finish_reason"] == "stop"
    assert not second["delta"].get(field)


@pytest.mark.parametrize("token_aware", [False, True])
@pytest.mark.parametrize("include", [False, True])
def test_engine_reported_answer_stop_is_trimmed(token_aware, include):
    post = make_post(
        ["END"], token_aware=token_aware, include_stop_str_in_output=include
    )
    feed(post, "thinking</think>")
    ids = list(map(ord, "answerEND"))
    choice = post.process_output(
        {"token_ids": ids, "finish_reason": "stop", "stop_reason": "END"}
    )
    assert choice["delta"]["content"] == ("answerEND" if include else "answer")
    assert choice["finish_reason"] == "stop"


@pytest.mark.parametrize("token_aware", [False, True])
def test_partial_stop_flushed_on_length(token_aware):
    post = make_post(["<END>"], token_aware=token_aware)
    feed(post, "thinking</think>")
    assert feed(post, "hello<EN")["delta"]["content"] == "hello"
    end = feed(post, "", "length")
    assert end["delta"]["content"] == "<EN"
    assert end["finish_reason"] == "length"


def test_glm_pending_prefix_keeps_original_channel_across_control_token():
    post = make_post(["x</think>END"], token_aware=True)
    assert feed(post, "thoughtx")["delta"]["reasoning_content"] == "thought"
    assert feed(post, "</think>") is None
    choice = feed(post, "other", "length")
    assert choice["delta"]["reasoning_content"] == "x"
    assert choice["delta"]["content"] == "other"
    assert choice["finish_reason"] == "length"


def test_glm_stop_can_span_reasoning_control_token():
    post = make_post(["x</think>END"], token_aware=True)
    assert feed(post, "thoughtx")["delta"]["reasoning_content"] == "thought"
    assert feed(post, "</think>") is None
    end = feed(post, "ENDignored")
    assert end["finish_reason"] == "stop"
    assert not end["delta"].get("content")
    assert not end["delta"].get("reasoning_content")


def test_glm_literal_think_tags_do_not_change_channel_with_stops_enabled():
    post = make_post(["END"], token_aware=True)
    choice = post.process_output(
        {"token_ids": list(b"literal </think> still reasoningENDignored")}
    )
    assert choice["delta"]["reasoning_content"] == "literal </think> still reasoning"
    assert not choice["delta"].get("content")
    assert choice["finish_reason"] == "stop"


def test_glm_utf8_stop_split_across_tokens():
    post = make_post(["停止"], token_aware=True)
    text = "思考停止忽略".encode()
    choices = []
    for token in text:
        choice = post.process_output({"token_ids": [token]})
        if choice:
            choices.append(choice)
    assert "".join(c["delta"].get("reasoning_content", "") for c in choices) == "思考"
    assert choices[-1]["finish_reason"] == "stop"


@pytest.mark.parametrize("interval", [1, 20])
@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("native_stream", [False, True])
@pytest.mark.parametrize("token_aware", [None, False, True])
def test_local_stop_closes_generation_and_emits_terminal_usage(
    interval, continuous, native_stream, token_aware
):
    class Engine:
        closed = False
        consumed = 0

        async def generate(self, request, **kwargs):
            texts = ["ALPHA\nBETAignored", "never read"]

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
        reasoning_parser_name=None,
        eos_token_ids=None,
        stream_interval=interval,
    )

    post = (
        SglangStreamingPostProcessor(
            tokenizer=TextTokenizer(),
            tool_call_parser=None,
            reasoning_parser=None,
            stop_strings={"\n"},
        )
        if token_aware is None
        else make_post(["\n"], token_aware=token_aware)
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
            post,
        ):
            items.append(item)
            choices = item.get("data", {}).get("choices", [])
            if choices and choices[0]["finish_reason"]:
                assert engine.closed
        return items

    items = asyncio.run(collect())
    assert engine.consumed == 1
    terminal = items[-1]["data"]
    assert terminal["choices"][0]["finish_reason"] == "stop"
    field = "content" if token_aware is None else "reasoning_content"
    assert terminal["choices"][0]["delta"][field] == "ALPHA"
    generated = len("ALPHA\nBETAignored")
    assert terminal["usage"]["completion_tokens"] == generated
    assert terminal["usage"]["total_tokens"] == generated + 2
    assert terminal["nvext"]["stop_reason"] == "\n"
    assert terminal["llm_metrics"]["output_tokens"] == generated
