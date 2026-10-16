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
def test_terminal_tool_payload_precedes_finish(
    module_stubs: Any, finish_reason: str, tool_count: int, stream_interval: int
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
                {"model": "glm-5.3", "nvext": {"extra_fields": ["stop_reason"]}},
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
