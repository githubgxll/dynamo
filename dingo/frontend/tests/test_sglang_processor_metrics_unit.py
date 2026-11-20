#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""SGLang processor metrics and import tests with lightweight backend stubs."""

import asyncio
import builtins
import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from _routed_engine_fakes import FakeRoutedEngine

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
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


def _install_sglang_stubs(install_module, tokenizer_module):
    """Install lightweight backend modules with the selected tokenizer layout."""

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

    class _GptOssDetector:
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
    _install_module(
        install_module, "xgrammar", StructuralTag=object, StructuralTagItem=object
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
        "sglang.srt.parser.conversation",
        chat_template_exists=lambda *_args, **_kwargs: False,
    )
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
        GptOssDetector=_GptOssDetector,
    )
    _install_module(install_module, "sglang.srt.utils")
    _install_module(
        install_module,
        tokenizer_module,
        get_tokenizer=lambda *args, **kwargs: None,
        get_config=lambda *args, **kwargs: types.SimpleNamespace(
            eos_token_id=[154820, 154827]
        ),
        get_generation_config=lambda *args, **kwargs: types.SimpleNamespace(
            eos_token_id=[154820, 154829]
        ),
    )


class _PostProcessor:
    locally_finished = False

    has_pending_stop_text = False

    local_stop_reason = None

    def process_output(self, mapped_response):
        return {
            "index": 0,
            "delta": {"content": "x"},
            "finish_reason": mapped_response["finish_reason"],
        }

    def pop_reasoning_token_count(self) -> int:
        return 1


def _load_processor_module(module_stubs, tokenizer_module="sglang.srt.hf_transformers"):
    """Import a fresh processor against the selected SGLang module layout."""
    install_module, remove_module = module_stubs
    http_error = type("HttpError", (Exception,), {})
    for name in (
        "sglang.srt.hf_transformers",
        "sglang.srt.utils.hf_transformers",
    ):
        remove_module(name)
    _install_sglang_stubs(install_module, tokenizer_module)
    # Importing common.utils initializes native bindings absent from this fixture.
    _install_module(
        install_module,
        "dingo.common.utils",
        __path__=[str(Path(__file__).resolve().parents[2] / "common" / "utils")],
    )
    _install_module(
        install_module,
        "dingo.common.utils.input_params",
        resolve_thinking_token_budget=lambda *_args, **_kwargs: None,
    )
    _install_module(install_module, "dingo.common.multimodal")
    _install_module(
        install_module,
        "dingo.common.multimodal.cache_uuid",
        reject_unsupported_multimodal_uuids=lambda *_args, **_kwargs: None,
    )
    _install_module(
        install_module,
        "dingo.frontend.sglang_prepost",
        ReasoningParser=object,
        SglangStreamingPostProcessor=object,
        ToolCallParserType=object,
        _client_wants_separate_reasoning=lambda *_args, **_kwargs: False,
        _get_history_tool_calls_count=lambda *_args, **_kwargs: 0,
        _guided_output_requires_reasoning=lambda *_args, **_kwargs: False,
        convert_tools=lambda *_args, **_kwargs: None,
        create_parsers=lambda *_args, **_kwargs: (None, None),
        detect_force_reasoning_from_template=lambda *_args, **_kwargs: False,
        preprocess_chat_request=lambda *_args, **_kwargs: None,
        resolve_skip_special_tokens=lambda *_args, **_kwargs: True,
    )
    _install_module(
        install_module,
        "dingo.frontend.frontend_args",
        FrontendConfig=object,
    )
    _install_module(
        install_module,
        "dynamo._internal",
        ModelDeploymentCard=object,
    )
    _install_module(
        install_module,
        "dynamo.llm",
        ModelCardInstanceId=object,
        HttpError=http_error,
        ModelType=object,
        PythonAsyncEngine=object,
        RoutedEngine=object,
    )
    _install_module(
        install_module,
        "dynamo.llm.exceptions",
        HttpError=http_error,
        InvalidArgument=type("InvalidArgument", (Exception,), {}),
        Unknown=type("Unknown", (Exception,), {}),
        WorkerUnavailable=type("WorkerUnavailable", (Exception,), {}),
    )
    remove_module("dingo.frontend.sglang_prepost")
    remove_module("dingo.frontend.sglang_processor")
    return importlib.import_module("dingo.frontend.sglang_processor")


def test_output_limit_default_is_glm53_only(module_stubs):
    module = _load_processor_module(module_stubs)
    for model, default in [
        ("glm-5.3", 65536),
        ("zai-org/GLM-5.3", 65536),
        ("glm5.3", 65536),
        ("GLM_5.3", 65536),
        ("kimi-k2.5", None),
        ("moonshotai/Kimi-K2-Instruct", None),
        ("glm-5.2", None),
        ("glm-5.3-flash", None),
        ("glm53", None),
    ]:
        for limits, expected in [
            ({}, default),
            ({"max_tokens": None, "max_completion_tokens": None}, default),
            ({"max_tokens": 256}, 256),
            ({"max_tokens": 90000}, 90000),
            ({"max_tokens": 256, "max_completion_tokens": 512}, 512),
            ({"max_tokens": 256, "max_completion_tokens": 0}, 0),
        ]:
            request = {"model": model, **limits}
            original = request.copy()
            result = module._build_dynamo_preproc(request, [1], model, None)
            assert result["stop_conditions"]["max_tokens"] == expected, (model, limits)
            assert request == original


def test_glm53_context_budget_keeps_other_models_unchanged():
    from dingo.frontend.glm53 import model_context_length, output_budget

    for model in ("glm-5.3", "org/GLM_5.3", "glm5.3"):
        assert output_budget(model, 131072, 16, 131072) == 131056
        assert output_budget(model, 65536, 16, 131072) == 65536
        with pytest.raises(ValueError):
            output_budget(model, 131073, 16, 131072)
        with pytest.raises(ValueError):
            output_budget(model, 16, 131072, 131072)
    for model in ("kimi-k2.5", "glm-5.2", "glm-5.3-flash", "glm53"):
        assert output_budget(model, 131073, 16, 131072) == 131073
    assert (
        model_context_length(
            {"context_length": 8192}, {"architectural_max_context_length": 131072}
        )
        == 8192
    )
    assert (
        model_context_length(
            {"context_length": 0}, {"architectural_max_context_length": 131072}
        )
        is None
    )
    assert (
        model_context_length({}, {"architectural_max_context_length": 131072}) == 131072
    )


@pytest.mark.parametrize("glm53", [False, True])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("terminal_eos", [False, True])
def test_glm53_hides_eos_before_string_stop(module_stubs, glm53, batched, terminal_eos):
    module = _load_processor_module(module_stubs)

    class Tokenizer:
        def decode(self, ids, **kwargs):
            return "".join("<|user|>" if i == 777 else chr(i) for i in ids)

    post = module.SglangStreamingPostProcessor(
        tokenizer=Tokenizer(),
        tool_call_parser=None,
        reasoning_parser=None,
        eos_token_ids=[777],
        stop_strings={"e"},
        skip_special_tokens=False,
        glm53_profile=glm53,
    )
    chunks = [[79, 75, 777]] if batched else [[79], [75], [777]]
    results = []
    for index, ids in enumerate(chunks):
        result = post.process_output(
            {
                "token_ids": ids,
                "finish_reason": "stop"
                if terminal_eos and index == len(chunks) - 1
                else None,
                "stop_reason": "e",
            }
        )
        if result:
            results.append(result)
    result = post.process_output({"token_ids": [], "finish_reason": "stop"})
    if result:
        results.append(result)
    text = "".join(r["delta"].get("content", "") for r in results)
    assert text == ("OK" if glm53 else "OK<|us")


def test_glm53_budget_is_sent_to_routed_engine(module_stubs):
    module = _load_processor_module(module_stubs)
    module.preprocess_chat_request = lambda *args, **kwargs: types.SimpleNamespace(
        prompt_token_ids=list(range(10)),
        guided_decoding=None,
        tool_call_parser=None,
        reasoning_parser=None,
        named_zero_arg_tool=None,
        response_format_guided_active=False,
        tool_guided_active=False,
        require_reasoning=False,
        force_reasoning=False,
    )
    for model, expected in [("glm-5.3", 131062), ("kimi-k2.5", 131072)]:
        engine = FakeRoutedEngine(items=[])
        processor = module.SglangProcessor(
            None, engine, None, None, None, context_length=131072
        )
        processor._build_postprocessor = lambda *args, **kwargs: _PostProcessor()

        async def collect():
            return [
                item
                async for item in processor._generator_inner(
                    {"model": model, "max_tokens": 131072}
                )
            ]

        asyncio.run(collect())
        assert engine.requests[0]["stop_conditions"]["max_tokens"] == expected


def test_stream_embeds_typed_llm_metrics(module_stubs):
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
    assert len(items) == 1
    envelope = items[0]
    assert envelope["_dynamo_annotated"] is True
    assert envelope["data"]["usage"] == completion_usage
    assert "event" not in envelope
    assert "comment" not in envelope
    # Zero counts are omitted (text-only request), mirroring the Rust skip-zero behavior.
    assert envelope["data"]["llm_metrics"] == {
        "input_tokens": 10,
        "output_tokens": 3,
        "chunk_tokens": 3,
        "cached_tokens": 4,
    }


def test_stream_emits_multimodal_counts(module_stubs):
    # Mixed-media request: assert the SGLang extraction branch reports nonzero
    # counts (2 images + 1 video), mirroring the vLLM processor test so the
    # extract_mm_urls wiring cannot silently regress to zero.
    module = _load_processor_module(module_stubs)
    processor = module.SglangProcessor(
        tokenizer=None,
        routed_engine=FakeRoutedEngine(
            items=[
                {
                    "token_ids": [101, 102, 103],
                    "finish_reason": "stop",
                    "completion_usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 3,
                        "total_tokens": 13,
                    },
                }
            ]
        ),
        tool_call_parser_name=None,
        reasoning_parser_name=None,
        eos_token_ids=None,
    )

    request = {
        "model": "test-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "compare"},
                    {"type": "image_url", "image_url": {"url": "http://x/a.png"}},
                    {"type": "image_url", "image_url": {"url": "http://x/b.png"}},
                    {"type": "video_url", "video_url": {"url": "http://x/c.mp4"}},
                ],
            }
        ],
    }

    async def collect():
        return [
            item
            async for item in processor._generate_and_stream(
                "req-mm-counts",
                request,
                {},
                list(range(10)),
                _PostProcessor(),
            )
        ]

    items = asyncio.run(collect())
    assert len(items) == 1
    metrics = items[0]["data"]["llm_metrics"]
    assert metrics["image_count"] == 2
    assert metrics["video_count"] == 1
    # audio has zero parts, so the key is omitted from the emitted metrics.
    assert metrics.get("audio_count") is None


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
        item["data"]["llm_metrics"]
        for item in items
        if "llm_metrics" in item.get("data", {})
    ]
    assert len(metrics) == 1
    assert metrics[0]["chunk_tokens"] == metrics[0]["output_tokens"] == 3
    assert "llm_metrics" in items[0]["data"]
    assert "llm_metrics" not in items[1]["data"]


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


@pytest.mark.parametrize("interval", [1, 20])
@pytest.mark.parametrize("streaming", [False, True])
def test_text_deltas_and_terminal_are_emitted_once(module_stubs, interval, streaming):
    module = _load_processor_module(module_stubs)
    processor, post = _usage_processor(
        module,
        [
            {"token_ids": [65]},
            {"token_ids": [66]},
            # Changes logprob shape while a batch is pending (interval=20).
            {"token_ids": [67], "log_probs": [-0.1]},
            {"token_ids": [], "finish_reason": "stop"},
        ],
        interval=interval,
    )

    async def collect():
        return [
            item["data"]
            async for item in processor._generate_and_stream(
                "once",
                {"model": "glm-5.3", "stream": streaming, "logprobs": True},
                {},
                [],
                post,
            )
        ]

    chunks = asyncio.run(collect())
    choices = [chunk["choices"][0] for chunk in chunks]
    assert "".join(choice["delta"].get("content", "") for choice in choices) == "ABC"
    assert sum(choice["finish_reason"] is not None for choice in choices) == 1
    assert sum(chunk["llm_metrics"]["chunk_tokens"] for chunk in chunks) == 3
    assert processor.routed_engine.stream_released


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
    chunks[0]["engine_data"] = {"cached_tokens_details": {"device": 2, "host": 2}}
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
    assert (
        "".join(
            choice["delta"].get("content", "")
            for item in data
            for choice in item["choices"]
        )
        == "你!"
    )
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
        _usage_chunk([byte], index) for index, byte in enumerate(b"ABCDE", start=1)
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
    assert [item["choices"][0]["delta"].get("content", "") for item in data] == [
        "A",
        "BCD",
        "E",
    ]
    assert [item["usage"]["completion_tokens"] for item in data] == [1, 4, 6]
    assert [item["usage"]["total_tokens"] for item in data] == [11, 14, 16]
    # Metrics describe token delivery, not a sum of cumulative usage snapshots.
    assert sum(event["data"]["llm_metrics"]["chunk_tokens"] for event in events) == 5


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
    assert event["data"]["llm_metrics"]["chunk_tokens"] == 0


@pytest.mark.parametrize("stream_interval", [1, 2])
@pytest.mark.parametrize(
    "stream_options,intermediate_usage",
    [
        (None, False),
        ({"include_usage": True, "continuous_usage_stats": False}, False),
        ({"include_usage": False, "continuous_usage_stats": True}, False),
        ({"include_usage": True, "continuous_usage_stats": True}, True),
    ],
)
def test_sparse_usage_survives_flushes(
    module_stubs, stream_interval, stream_options, intermediate_usage
):
    module = _load_processor_module(module_stubs)
    processor = module.SglangProcessor(
        tokenizer=None,
        routed_engine=FakeRoutedEngine(
            items=[
                {
                    "token_ids": [101],
                    "completion_usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 1,
                        "total_tokens": 4,
                        "prompt_tokens_details": {"cached_tokens": 0},
                    },
                },
                {
                    "token_ids": [102],
                    "completion_usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5,
                        "prompt_tokens_details": {"cached_tokens": 2},
                    },
                },
                {"token_ids": [103]},
                {"token_ids": [], "finish_reason": "stop"},
            ]
        ),
        tool_call_parser_name=None,
        reasoning_parser_name=None,
        eos_token_ids=None,
    )
    processor.stream_interval = stream_interval

    async def collect():
        return [
            item
            async for item in processor._generate_and_stream(
                "req-sparse",
                {
                    "model": "test-model",
                    "stream": True,
                    "stream_options": stream_options,
                },
                {},
                [1, 2, 3],
                _PostProcessor(),
            )
        ]

    items = asyncio.run(collect())
    expected_counts = [1, 2, 3, 3] if stream_interval == 1 else [1, 3, 3]
    assert len(items) == len(expected_counts)
    for index, (item, count) in enumerate(zip(items, expected_counts)):
        expected_cache = 0 if index == 0 else 2
        assert item["data"]["llm_metrics"]["cached_tokens"] == expected_cache
        assert ("usage" in item["data"]) == (
            intermediate_usage
            or (
                index == len(items) - 1
                and bool((stream_options or {}).get("include_usage"))
            )
        )
        if "usage" in item["data"]:
            usage = item["data"]["usage"]
            assert usage["completion_tokens"] == count
            assert usage["total_tokens"] == 3 + count
            assert usage["prompt_tokens_details"] == {"cached_tokens": expected_cache}


@pytest.mark.parametrize(
    "tokenizer_module",
    [
        "sglang.srt.hf_transformers",
        "sglang.srt.utils.hf_transformers",
    ],
)
def test_tokenizer_import_layouts(module_stubs, tokenizer_module):
    """Resolve the tokenizer from current sources and the pinned releases."""
    module = _load_processor_module(module_stubs, tokenizer_module)
    assert module.get_tokenizer is sys.modules[tokenizer_module].get_tokenizer
    assert module.get_config is sys.modules[tokenizer_module].get_config
    assert (
        module.get_generation_config
        is sys.modules[tokenizer_module].get_generation_config
    )


@pytest.mark.parametrize(
    "tokenizer_module",
    ["sglang.srt.hf_transformers", "sglang.srt.utils.hf_transformers"],
)
def test_model_eos_sources_are_loaded(module_stubs, tokenizer_module):
    module = _load_processor_module(module_stubs, tokenizer_module)
    # Use the imported functions, rather than monkeypatching missing globals:
    # the old NameError was swallowed and silently retained only tokenizer EOS.
    assert module._resolve_model_eos_token_ids(
        tokenizer=types.SimpleNamespace(eos_token_id=154820),
        source_path="local-model",
        trust_remote_code=False,
    ) == [154820, 154827, 154829]


@pytest.mark.parametrize(
    "tokenizer_module",
    ["sglang.srt.hf_transformers", "sglang.srt.utils.hf_transformers"],
)
def test_tokenizer_import_preserves_dependency_error(
    module_stubs, monkeypatch, tokenizer_module
):
    """Preserve missing dependencies instead of treating them as a layout move."""
    original_import = builtins.__import__
    error = ModuleNotFoundError(
        "No module named 'tokenizer_dependency'", name="tokenizer_dependency"
    )

    def import_with_missing_dependency(name, *args, **kwargs):
        """Fail inside the selected tokenizer module while importing others normally."""
        if name == tokenizer_module:
            raise error
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_with_missing_dependency)
    with pytest.raises(ModuleNotFoundError) as caught:
        _load_processor_module(module_stubs, tokenizer_module)
    assert caught.value is error
