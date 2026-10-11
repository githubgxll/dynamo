#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Metrics-focused SGLang processor tests with lightweight SGLang stubs."""

import asyncio
import importlib
import json
import sys
import types
from typing import Any

import pytest
from _routed_engine_fakes import FakeRoutedEngine

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

_MISSING = object()


@pytest.fixture
def module_stubs():
    original_modules = {}

    def remember(name: str):
        if name not in original_modules:
            original_modules[name] = sys.modules.get(name, _MISSING)

    def install_module(name: str, **attrs):
        remember(name)
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module
        return module

    def remove_module(name: str):
        remember(name)
        sys.modules.pop(name, None)

    yield install_module, remove_module

    for name, module in original_modules.items():
        if module is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _install_module(install_module, name: str, **attrs):
    return install_module(name, **attrs)


def _install_sglang_stubs(install_module):
    class _Function:
        pass

    class _Tool:
        pass

    class _ToolChoice:
        pass

    class _ToolChoiceFuncName:
        pass

    class _ToolCallItem:
        pass

    class _FunctionCallParser:
        pass

    class _JsonArrayParser:
        pass

    class _ReasoningParser:
        pass

    _install_module(install_module, "sglang")
    _install_module(install_module, "sglang.srt")
    _install_module(install_module, "sglang.srt.entrypoints")
    _install_module(install_module, "sglang.srt.entrypoints.openai")
    _install_module(
        install_module,
        "sglang.srt.entrypoints.openai.protocol",
        Function=_Function,
        Tool=_Tool,
        ToolChoice=_ToolChoice,
        ToolChoiceFuncName=_ToolChoiceFuncName,
    )
    _install_module(
        install_module, "sglang.srt.environ", ToolStrictLevel=object, envs=object()
    )
    _install_module(install_module, "sglang.srt.function_call")
    _install_module(
        install_module,
        "sglang.srt.function_call.core_types",
        ToolCallItem=_ToolCallItem,
    )
    _install_module(
        install_module,
        "sglang.srt.function_call.function_call_parser",
        FunctionCallParser=_FunctionCallParser,
    )
    _install_module(
        install_module,
        "sglang.srt.function_call.json_array_parser",
        JsonArrayParser=_JsonArrayParser,
    )
    _install_module(
        install_module,
        "sglang.srt.function_call.utils",
        get_json_schema_constraint=lambda *args, **kwargs: None,
    )
    _install_module(install_module, "sglang.srt.parser")
    _install_module(
        install_module,
        "sglang.srt.parser.jinja_template_utils",
        detect_jinja_template_content_format=lambda *args, **kwargs: "string",
        process_content_for_template_format=lambda content, *_args, **_kwargs: content,
    )
    _install_module(
        install_module,
        "sglang.srt.parser.reasoning_parser",
        ReasoningParser=_ReasoningParser,
    )
    _install_module(install_module, "sglang.srt.utils")
    _install_module(
        install_module,
        "sglang.srt.utils.hf_transformers_utils",
        get_tokenizer=lambda *args, **kwargs: None,
        get_config=lambda *args, **kwargs: None,
        get_generation_config=lambda *args, **kwargs: None,
    )


class _PostProcessor:
    def pop_reasoning_token_count(self) -> int:
        return 1

    def process_output(self, mapped_response):
        return {
            "index": 0,
            "delta": {"content": "x"},
            "finish_reason": mapped_response["finish_reason"],
        }


def _load_processor_module(module_stubs):
    install_module, remove_module = module_stubs
    _install_sglang_stubs(install_module)
    _install_module(install_module, "dynamo._internal", ModelDeploymentCard=object)
    _install_module(
        install_module,
        "dingo.frontend.frontend_args",
        FrontendConfig=object,
    )
    _install_module(
        install_module,
        "dynamo.llm",
        ModelCardInstanceId=object,
        PythonAsyncEngine=object,
        RoutedEngine=object,
    )
    _install_module(
        install_module,
        "dynamo.llm.exceptions",
        InvalidArgument=type("InvalidArgument", (Exception,), {}),
        Unknown=type("Unknown", (Exception,), {}),
    )
    remove_module("dingo.frontend.sglang_prepost")
    remove_module("dingo.frontend.sglang_processor")
    return importlib.import_module("dingo.frontend.sglang_processor")


def test_stream_emits_llm_metrics_annotation(module_stubs):
    module = _load_processor_module(module_stubs)
    completion_usage = {
        "prompt_tokens": 10,
        "completion_tokens": 3,
        "total_tokens": 13,
        "prompt_tokens_details": {"cached_tokens": 4},
    }
    processor = module.SglangProcessor(
        tokenizer=None,
        routed_engine=FakeRoutedEngine(
            items=[
                {
                    "token_ids": [101, 102, 103],
                    "finish_reason": "stop",
                    "completion_usage": completion_usage,
                }
            ]
        ),
        tool_call_parser_name=None,
        reasoning_parser_name=None,
        eos_token_ids=None,
    )

    async def collect():
        return [
            item
            async for item in processor._generate_and_stream(
                "req-metrics",
                {"model": "test-model"},
                {},
                list(range(10)),
                _PostProcessor(),
            )
        ]

    items = asyncio.run(collect())
    metric_items = [item for item in items if item.get("event") == "llm_metrics"]

    assert len(metric_items) == 1
    envelope = metric_items[0]
    assert envelope["_dynamo_annotated"] is True
    assert envelope["data"]["usage"] == completion_usage
    metrics = json.loads(envelope["comment"][0])
    assert metrics == {
        "input_tokens": 10,
        "output_tokens": 3,
        "chunk_tokens": 3,
        "cached_tokens": 4,
    }


@pytest.mark.parametrize("finish_reason", ["tool_calls", "length"])
@pytest.mark.parametrize("tool_count", [1, 2])
@pytest.mark.parametrize("stream_interval", [1, 20])
@pytest.mark.parametrize("continuous", [False, True])
def test_terminal_tool_payload_precedes_finish(
    module_stubs: Any,
    finish_reason: str,
    tool_count: int,
    stream_interval: int,
    continuous: bool,
) -> None:
    module = _load_processor_module(module_stubs)
    tool_calls = [
        {
            "index": i,
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"location":"上海"}'},
        }
        for i in range(tool_count)
    ]
    choice = {
        "index": 0,
        "delta": {
            "role": "assistant",
            "tool_calls": tool_calls,
            "content": "查询天气",
            "reasoning_content": "调用工具",
        },
        "finish_reason": finish_reason,
        "logprobs": None,
    }

    class Post(_PostProcessor):
        def process_output(self, mapped_response: dict[str, Any]) -> dict[str, Any]:
            return choice

    usage = {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
    processor = module.SglangProcessor(
        tokenizer=None,
        routed_engine=FakeRoutedEngine(
            items=[
                {
                    "token_ids": [101, 102, 103],
                    "finish_reason": "length" if finish_reason == "length" else "stop",
                    "completion_usage": usage,
                    "stop_reason": 154829,
                    "engine_data": {"cached_tokens_details": {"device": 4, "host": 0}},
                }
            ]
        ),
        tool_call_parser_name=None,
        reasoning_parser_name=None,
        eos_token_ids=None,
    )
    processor.stream_interval = stream_interval

    async def collect() -> list[dict[str, Any]]:
        return [
            item
            async for item in processor._generate_and_stream(
                "req-tools",
                {
                    "model": "glm-5.3",
                    "stream": True,
                    "stream_options": {
                        "include_usage": True,
                        "continuous_usage_stats": continuous,
                    },
                    "nvext": {"extra_fields": ["stop_reason"]},
                },
                {},
                list(range(10)),
                Post(),
            )
        ]

    items = asyncio.run(collect())
    assert len(items) == 2
    data, terminal = [item["data"] for item in items]
    assert data["choices"] == [{**choice, "finish_reason": None}]
    assert terminal["choices"] == [{**choice, "delta": {}}]
    if continuous:
        assert data["usage"] == terminal["usage"]
        assert data["nvext"] == terminal["nvext"]
    else:
        assert "usage" not in data and "nvext" not in data
    assert terminal["usage"] == {
        **usage,
        "completion_tokens_details": {"reasoning_tokens": 1},
    }
    assert terminal["nvext"]["stop_reason"] == 154829
    assert (
        terminal["nvext"][module._INTERNAL_SGLEXT_KEY]["cached_tokens_details"][
            "device"
        ]
        == 4
    )
    for field in ("id", "model", "created", "object"):
        assert data[field] == terminal[field]
    assert choice["finish_reason"] == finish_reason
    assert choice["delta"]["tool_calls"] == tool_calls
    metrics = [
        json.loads(item["comment"][0])
        for item in items
        if item.get("event") == "llm_metrics"
    ]
    assert len(metrics) == 1
    assert metrics[0]["chunk_tokens"] == metrics[0]["output_tokens"] == 3
    assert items[0]["event"] == "llm_metrics"


class _ByteTokenizer:
    def decode(self, ids, **kwargs):
        return bytes(ids).decode("utf-8", errors="replace")


def _usage_chunk(ids, count, *, finish=None, cached=4):
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": count,
        "total_tokens": 10 + count,
        "prompt_tokens_details": (
            {"cached_tokens": cached} if cached is not None else None
        ),
    }
    return {
        "token_ids": ids,
        "finish_reason": finish,
        "completion_usage": usage,
    }


def _usage_processor(module, chunks, *, interval=1):
    processor = module.SglangProcessor(
        tokenizer=_ByteTokenizer(),
        routed_engine=FakeRoutedEngine(items=chunks),
        tool_call_parser_name=None,
        reasoning_parser_name=None,
        eos_token_ids=None,
    )
    processor.stream_interval = interval
    post = module.SglangStreamingPostProcessor(
        tokenizer=processor.tokenizer,
        tool_call_parser=None,
        reasoning_parser=None,
    )
    return processor, post


def _usage_request(options=None, *, stream=True):
    request = {"model": "test-model", "stream": stream}
    if options is not None:
        request["stream_options"] = options
    return request


_CONTINUOUS_USAGE = {"include_usage": True, "continuous_usage_stats": True}


@pytest.mark.parametrize(
    "options,expected_counts",
    [
        (None, [4]),
        ({"include_usage": True}, [4]),
        ({"continuous_usage_stats": True}, [4]),
        (_CONTINUOUS_USAGE, [1, 2, 3, 4]),
    ],
)
def test_usage_survives_buffered_utf8_without_changing_text(
    module_stubs, options, expected_counts
):
    module = _load_processor_module(module_stubs)
    chunks = [
        _usage_chunk([byte], index, finish="stop" if index == 4 else None)
        for index, byte in enumerate("你!".encode("utf-8"), start=1)
    ]
    chunks[0]["engine_data"] = {
        "cached_tokens_details": {"device": 2, "host": 2}
    }
    processor, post = _usage_processor(module, chunks)

    async def collect():
        return [
            item
            async for item in processor._generate_and_stream(
                "req-buffered", _usage_request(options), {}, list(range(10)), post
            )
        ]

    events = asyncio.run(collect())
    data = [event["data"] for event in events if "data" in event]
    assert "".join(
        choice["delta"].get("content", "")
        for item in data
        for choice in item["choices"]
    ) == "你!"
    usages = [item["usage"] for item in data if "usage" in item]
    assert [usage["completion_tokens"] for usage in usages] == expected_counts
    assert [usage["total_tokens"] for usage in usages] == [
        10 + count for count in expected_counts
    ]
    assert all(usage["prompt_tokens_details"]["cached_tokens"] == 4 for usage in usages)
    if options == _CONTINUOUS_USAGE:
        assert data[0]["choices"] == data[1]["choices"] == []
        assert data[0]["nvext"][module._INTERNAL_SGLEXT_KEY] == {
            "cached_tokens_details": {"device": 2, "host": 2}
        }
    assert data[-1]["choices"][0]["finish_reason"] == "stop"


def test_running_usage_preserves_batching_and_engine_terminal_count(module_stubs):
    module = _load_processor_module(module_stubs)
    chunks = [
        _usage_chunk([byte], index)
        for index, byte in enumerate(b"ABCDE", start=1)
    ]
    # The engine counted a hidden stop token; do not replace its authoritative
    # completion count with the number of visible token IDs.
    chunks.append(_usage_chunk([], 6, finish="stop"))
    processor, post = _usage_processor(module, chunks, interval=3)

    async def collect():
        return [
            item
            async for item in processor._generate_and_stream(
                "req-batched",
                _usage_request(_CONTINUOUS_USAGE),
                {},
                list(range(10)),
                post,
            )
        ]

    events = asyncio.run(collect())
    data = [event["data"] for event in events if "data" in event]
    assert [
        item["choices"][0]["delta"].get("content", "") for item in data
    ] == ["A", "BCD", "E"]
    assert [item["usage"]["completion_tokens"] for item in data] == [1, 4, 6]
    assert [item["usage"]["total_tokens"] for item in data] == [11, 14, 16]
    # Metrics describe token delivery, not a sum of cumulative usage snapshots.
    assert sum(json.loads(event["comment"][0])["chunk_tokens"] for event in events) == 5


@pytest.mark.parametrize("cached", [0, 4, None])
def test_first_content_usage_remains_available_without_draining_stream(
    module_stubs, cached
):
    module = _load_processor_module(module_stubs)
    chunks = [
        _usage_chunk([65], 1, cached=cached),
        _usage_chunk([66], 2, finish="stop", cached=cached),
    ]
    processor, post = _usage_processor(module, chunks)

    async def disconnect():
        stream = processor._generate_and_stream(
            "req-disconnect",
            _usage_request(_CONTINUOUS_USAGE),
            {},
            list(range(10)),
            post,
        )
        first = await anext(stream)
        await stream.aclose()
        return first["data"]

    first = asyncio.run(disconnect())
    assert first["choices"][0]["delta"]["content"] == "A"
    assert first["choices"][0]["finish_reason"] is None
    assert first["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 1,
        "total_tokens": 11,
        "prompt_tokens_details": (
            {"cached_tokens": cached} if cached is not None else None
        ),
    }


def test_empty_token_report_emits_usage_without_waiting_for_finish(module_stubs):
    module = _load_processor_module(module_stubs)
    processor, post = _usage_processor(
        module,
        [_usage_chunk([], 1), _usage_chunk([65], 2, finish="stop")],
        interval=8,
    )

    async def first_event():
        stream = processor._generate_and_stream(
            "req-no-visible-token",
            _usage_request(_CONTINUOUS_USAGE),
            {},
            list(range(10)),
            post,
        )
        first = await anext(stream)
        await stream.aclose()
        return first

    event = asyncio.run(first_event())
    assert event["data"]["choices"] == []
    assert event["data"]["usage"]["completion_tokens"] == 1
    assert json.loads(event["comment"][0])["chunk_tokens"] == 0
