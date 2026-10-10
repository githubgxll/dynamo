#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Unit tests for vLLM processor components.

Tests for the tool-stripping behaviour of _prepare_request when
tool_choice='none' and the exclude_tools_when_tool_choice_none flag.
"""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from _routed_engine_fakes import FakeRoutedEngine as _FakeRoutedEngine
from transformers import AutoTokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import StructuredOutputsParams
from vllm.tool_parsers.qwen3_engine_tool_parser import Qwen3EngineToolParser

import dingo.frontend.prepost as prepost_module
from dingo.frontend.prepost import _prepare_request

# Needs vllm packages (gpu_1 container), but does not allocate GPU VRAM.
pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_1,
    pytest.mark.xpu_1,
    pytest.mark.pre_merge,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.timeout(180),  # 0-GiB unit tests, floor 180s
]

MODEL = "Qwen/Qwen3-0.6B"

TOOL_REQUEST = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Hello"}],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ],
}


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)


# ---------------------------------------------------------------------------
# _prepare_request: tool_choice=none tool-stripping
# ---------------------------------------------------------------------------


class TestPrepareRequestToolStripping:  # FRONTEND.1 + FRONTEND.3 — tool stripping when tool_choice=none on chat-template input
    """Test that _prepare_request strips/keeps tools based on the flag."""

    def test_tool_choice_none_strips_tools_from_template(self, tokenizer):
        """When exclude flag is on and tool_choice=none, tools are excluded from template kwargs."""
        _, _, _, _, chat_params = _prepare_request(
            {**TOOL_REQUEST, "tool_choice": "none"},
            tokenizer=tokenizer,
            tool_parser_class=None,
            exclude_tools_when_tool_choice_none=True,
        )
        assert chat_params.chat_template_kwargs["tools"] is None, (
            "tool_choice=none with exclude flag should strip tools from template"
        )

    def test_tool_choice_none_keeps_tools_when_flag_off(self, tokenizer):
        """When exclude flag is off, tool_choice=none still includes tools in template kwargs."""
        _, _, _, _, chat_params = _prepare_request(
            {**TOOL_REQUEST, "tool_choice": "none"},
            tokenizer=tokenizer,
            tool_parser_class=None,
            exclude_tools_when_tool_choice_none=False,
        )
        tools = chat_params.chat_template_kwargs["tools"]
        assert tools is not None and len(tools) == 1, (
            "tool_choice=none with flag off should keep tools in template"
        )

    def test_tool_choice_auto_keeps_tools(self, tokenizer):
        """tool_choice=auto should always include tools regardless of flag."""
        _, _, _, _, chat_params = _prepare_request(
            {**TOOL_REQUEST, "tool_choice": "auto"},
            tokenizer=tokenizer,
            tool_parser_class=None,
            exclude_tools_when_tool_choice_none=True,
        )
        tools = chat_params.chat_template_kwargs["tools"]
        assert tools is not None and len(tools) == 1, (
            "tool_choice=auto should keep tools in template"
        )

    def test_tool_choice_required_keeps_tools(self, tokenizer):
        """tool_choice=required should always include tools regardless of flag."""
        _, _, _, _, chat_params = _prepare_request(
            {**TOOL_REQUEST, "tool_choice": "required"},
            tokenizer=tokenizer,
            tool_parser_class=None,
            exclude_tools_when_tool_choice_none=True,
        )
        tools = chat_params.chat_template_kwargs["tools"]
        assert tools is not None and len(tools) == 1, (
            "tool_choice=required should keep tools in template"
        )

    def test_no_tools_in_request(self, tokenizer):
        """Request without tools should produce None tools in template kwargs."""
        _, _, _, _, chat_params = _prepare_request(
            {"model": MODEL, "messages": [{"role": "user", "content": "Hello"}]},
            tokenizer=tokenizer,
            tool_parser_class=None,
            exclude_tools_when_tool_choice_none=True,
        )
        assert chat_params.chat_template_kwargs["tools"] is None, (
            "No tools in request should produce None tools in template"
        )


_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"label": {"type": "string", "enum": ["yes", "no"]}},
                "required": ["label"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}
_RESPONSE_FORMATS = [
    pytest.param({"type": "json_object"}, {"type": "object"}, id="json-object"),
    pytest.param(
        {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "strict": True,
                "schema": _RESPONSE_SCHEMA,
            },
        },
        _RESPONSE_SCHEMA,
        id="json-schema",
    ),
]


def _structured_request(**overrides):
    return deepcopy(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hello"}],
            "response_format": {"type": "json_object"},
            **overrides,
        }
    )


def _fake_renderer():
    return SimpleNamespace(
        render_messages_async=AsyncMock(
            return_value=(None, {"prompt_token_ids": [1, 2, 3]})
        )
    )


async def _preprocess_structured(request, **kwargs):
    kwargs.setdefault("tool_parser_class", None)
    return await prepost_module.preprocess_chat_request(
        request, tokenizer=object(), renderer=_fake_renderer(), **kwargs
    )


@pytest.mark.parametrize("skip_validation", [False, True])
@pytest.mark.parametrize("request_kind", ["raw", "constructed"])
@pytest.mark.parametrize("tools_payload", [{}, {"tools": None}, {"tools": []}])
class TestAutoToolChoiceWithoutTools:
    def test_normalizes_without_mutating_input(
        self, monkeypatch, skip_validation, request_kind, tools_payload
    ):
        monkeypatch.setattr(prepost_module, "SKIP_REQUEST_VALIDATION", skip_validation)
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hello"}],
            **tools_payload,
        }
        expected = prepost_module._validate_chat_completion_request(payload)
        request = {**payload, "tool_choice": "auto"}
        if request_kind == "constructed":
            request = ChatCompletionRequest.model_construct(**request)
        original = deepcopy(dict(request))

        result = prepost_module._validate_chat_completion_request(request)

        assert result.tool_choice == expected.tool_choice
        assert result.tools == expected.tools
        assert result.messages == original["messages"]
        assert dict(request) == original

    @pytest.mark.asyncio
    async def test_streaming_title_schema_survives_nested_revalidation(
        self, monkeypatch, skip_validation, request_kind, tools_payload
    ):
        monkeypatch.setattr(prepost_module, "SKIP_REQUEST_VALIDATION", skip_validation)
        schema = {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
            "additionalProperties": False,
        }
        request = _structured_request(
            messages=[{"role": "user", "content": "Generate a session title."}],
            stream=True,
            tool_choice="auto",
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "title", "strict": True, "schema": schema},
            },
            **tools_payload,
        )
        if request_kind == "constructed":
            request = ChatCompletionRequest.model_construct(**request)
        original = deepcopy(dict(request))
        renderer = _fake_renderer()

        result = await prepost_module.preprocess_chat_request(
            request, tokenizer=object(), renderer=renderer, tool_parser_class=None
        )

        assert result.guided_decoding == {"json": schema}
        assert result.request_for_sampling.stream is True
        assert result.request_for_sampling.tool_choice != "auto"
        assert not isinstance(result.request_for_sampling.response_format, dict)
        assert result.tool_parser is None
        assert result.prompt_token_ids == [1, 2, 3]
        assert (
            renderer.render_messages_async.call_args.args[1].chat_template_kwargs[
                "tools"
            ]
            is None
        )
        assert dict(request) == original


@pytest.mark.parametrize("skip_validation", [False, True])
@pytest.mark.parametrize(
    "tool_choice",
    [
        "auto",
        "required",
        "none",
        {"type": "function", "function": {"name": "get_weather"}},
    ],
)
def test_validation_preserves_tool_choice_with_tools(
    monkeypatch, skip_validation, tool_choice
):
    monkeypatch.setattr(prepost_module, "SKIP_REQUEST_VALIDATION", skip_validation)
    request = deepcopy({**TOOL_REQUEST, "tool_choice": tool_choice})
    original = deepcopy(request)
    result = prepost_module._validate_chat_completion_request(request)
    choice = result.tool_choice
    if hasattr(choice, "model_dump"):
        choice = choice.model_dump()
    assert choice == tool_choice
    assert result.tools[0].function.name == "get_weather"
    assert request == original


@pytest.mark.parametrize("tools_payload", [{}, {"tools": None}, {"tools": []}])
@pytest.mark.parametrize(
    "tool_choice",
    ["required", {"type": "function", "function": {"name": "get_weather"}}],
)
def test_validation_rejects_forced_tool_choice_without_tools(
    monkeypatch, tools_payload, tool_choice
):
    monkeypatch.setattr(prepost_module, "SKIP_REQUEST_VALIDATION", False)
    request = _structured_request(tool_choice=tool_choice, **tools_payload)
    with pytest.raises((ValueError, VLLMValidationError)):
        prepost_module._validate_chat_completion_request(request)


@pytest.mark.parametrize("skip_validation", [False, True])
@pytest.mark.parametrize("tools_payload", [{}, {"tools": None}, {"tools": []}])
def test_validation_preserves_explicit_none_without_tools(
    monkeypatch, skip_validation, tools_payload
):
    monkeypatch.setattr(prepost_module, "SKIP_REQUEST_VALIDATION", skip_validation)
    result = prepost_module._validate_chat_completion_request(
        _structured_request(tool_choice="none", **tools_payload)
    )
    assert result.tool_choice == "none"


def test_skip_validation_preserves_trusted_required_without_tools(monkeypatch):
    monkeypatch.setattr(prepost_module, "SKIP_REQUEST_VALIDATION", True)
    result = prepost_module._validate_chat_completion_request(
        {"model": MODEL, "messages": [], "tool_choice": "required", "tools": []}
    )
    assert result.tool_choice == "required"


class _GrammarToolParser:
    """Adapted from main's focused adjust_request grammar-parser fake."""

    def __init__(self, tokenizer, tools):
        pass

    def adjust_request(self, request):
        request.structured_outputs = StructuredOutputsParams(
            grammar='root ::= "<tool_call>"'
        )
        return request


class _ResponseFormatConsumingToolParser(_GrammarToolParser):
    def adjust_request(self, request):
        request.response_format = None
        return super().adjust_request(request)


class _PassthroughReasoningParser:
    def __init__(self, tokenizer, *, chat_template_kwargs):
        self.chat_template_kwargs = chat_template_kwargs

    def adjust_request(self, request):
        request.skip_special_tokens = False
        return request

    def is_reasoning_end(self, prompt_token_ids):
        return False


class _RewritingReasoningParser(_PassthroughReasoningParser):
    def adjust_request(self, request):
        super().adjust_request(request)
        request.structured_outputs = StructuredOutputsParams(
            structural_tag='{"format": "reasoning"}'
        )
        return request


@pytest.mark.asyncio
class TestResponseFormatGuidance:
    @pytest.mark.parametrize("skip_validation", [False, True])
    @pytest.mark.parametrize("request_kind", ["raw", "validated", "constructed"])
    @pytest.mark.parametrize(("response_format", "schema"), _RESPONSE_FORMATS)
    async def test_response_format_extracts_inner_schema(
        self, monkeypatch, skip_validation, request_kind, response_format, schema
    ):
        monkeypatch.setattr(prepost_module, "SKIP_REQUEST_VALIDATION", skip_validation)
        request = _structured_request(response_format=response_format)
        if request_kind == "validated":
            request = ChatCompletionRequest.model_validate(request)
        elif request_kind == "constructed":
            request = ChatCompletionRequest.model_construct(**request)
        result = await _preprocess_structured(request)
        assert result.guided_decoding == {"json": schema}
        assert result.prompt_token_ids == [1, 2, 3]
        assert not isinstance(result.request_for_sampling.response_format, dict)

    @pytest.mark.parametrize("skip_validation", [False, True])
    @pytest.mark.parametrize("response_format", [None, {"type": "text"}])
    async def test_unconstrained_response_has_no_guidance(
        self, monkeypatch, skip_validation, response_format
    ):
        monkeypatch.setattr(prepost_module, "SKIP_REQUEST_VALIDATION", skip_validation)
        request = _structured_request(response_format=response_format)
        if response_format is None:
            request.pop("response_format")
        assert (await _preprocess_structured(request)).guided_decoding is None

    @pytest.mark.parametrize("skip_validation", [False, True])
    async def test_raw_structured_outputs_are_validated(
        self, monkeypatch, skip_validation
    ):
        monkeypatch.setattr(prepost_module, "SKIP_REQUEST_VALIDATION", skip_validation)
        result = await _preprocess_structured(
            _structured_request(
                response_format=None,
                structured_outputs={"regex": "yes|no", "whitespace_pattern": " *"},
            )
        )
        assert result.guided_decoding == {"regex": "yes|no", "whitespace_pattern": " *"}
        assert isinstance(
            result.request_for_sampling.structured_outputs, StructuredOutputsParams
        )

    @pytest.mark.parametrize(
        "tool_choice",
        [
            "auto",
            "none",
            "required",
            {"type": "function", "function": {"name": "get_weather"}},
        ],
    )
    @pytest.mark.parametrize(("response_format", "schema"), _RESPONSE_FORMATS)
    @pytest.mark.parametrize(
        "tool_parser_class", [_GrammarToolParser, _ResponseFormatConsumingToolParser]
    )
    async def test_tool_choice_constraint_precedence(
        self, tool_choice, response_format, schema, tool_parser_class
    ):
        result = await _preprocess_structured(
            _structured_request(
                tools=TOOL_REQUEST["tools"],
                tool_choice=tool_choice,
                response_format=response_format,
            ),
            tool_parser_class=tool_parser_class,
        )
        expected = (
            {"json": schema}
            if tool_choice in ("auto", "none")
            else {"grammar": 'root ::= "<tool_call>"'}
        )
        assert result.guided_decoding == expected

    @pytest.mark.parametrize("with_tools", [False, True])
    @pytest.mark.parametrize("rewrite", [False, True])
    async def test_reasoning_rewrite_wins_only_when_changed(self, with_tools, rewrite):
        overrides = (
            {"tools": TOOL_REQUEST["tools"], "tool_choice": "auto"}
            if with_tools
            else {}
        )
        result = await _preprocess_structured(
            _structured_request(**overrides),
            tool_parser_class=_ResponseFormatConsumingToolParser
            if with_tools
            else None,
            reasoning_parser_class=_RewritingReasoningParser
            if rewrite
            else _PassthroughReasoningParser,
        )
        expected = (
            {"structural_tag": '{"format": "reasoning"}'}
            if rewrite
            else {"json": {"type": "object"}}
        )
        assert result.guided_decoding == expected
        assert result.request_for_sampling.skip_special_tokens is False

    async def test_in_place_reasoning_schema_rewrite_is_forwarded(self):
        class InPlaceReasoningParser(_PassthroughReasoningParser):
            def adjust_request(self, request):
                super().adjust_request(request)
                request.structured_outputs.json["properties"]["answer"]["type"] = (
                    "integer"
                )
                return request

        result = await _preprocess_structured(
            _structured_request(
                response_format=None,
                structured_outputs={
                    "json": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                    }
                },
            ),
            reasoning_parser_class=InPlaceReasoningParser,
        )
        assert result.guided_decoding == {
            "json": {"type": "object", "properties": {"answer": {"type": "integer"}}}
        }

    @pytest.mark.parametrize(
        "kwargs_key", ["chat_template_kwargs", "chat_template_args"]
    )
    async def test_disabled_thinking_skips_reasoning_adjustment(self, kwargs_key):
        class ParserMustNotBeBuilt:
            def __init__(self, *args, **kwargs):
                raise AssertionError(
                    "disabled thinking must not construct a reasoning parser"
                )

        result = await _preprocess_structured(
            _structured_request(**{kwargs_key: {"enable_thinking": False}}),
            reasoning_parser_class=ParserMustNotBeBuilt,
        )
        assert result.guided_decoding == {"json": {"type": "object"}}
        assert result.request_for_sampling.skip_special_tokens is True


_NATIVE_TAG = json.dumps(
    {
        "type": "structural_tag",
        "format": {
            "type": "const_string",
            "value": "<tool_call>get_weather</tool_call>",
        },
    }
)
_FORCED_CHOICES = [
    "required",
    {"type": "function", "function": {"name": "get_weather"}},
]


class _NativeGLMToolParser:
    supports_required_and_named = False
    structural_tag_model = "glm_4_7"

    def __init__(self, tokenizer, tools):
        self.tools = tools
        self.adjusted = False

    def adjust_request(self, request):
        # Release adjust_request would otherwise install generic JSON here.
        assert request.structured_outputs.structural_tag == _NATIVE_TAG
        assert request.structured_outputs.json is None
        assert request.structured_outputs.regex is None
        assert request.response_format is None
        request.skip_special_tokens = False
        self.adjusted = True
        return request


@pytest.fixture
def native_registry(monkeypatch):
    from vllm.tool_parsers import structural_tag_registry

    registry = Mock(return_value=_NATIVE_TAG)
    monkeypatch.setattr(structural_tag_registry, "get_model_structural_tag", registry)
    return registry


@pytest.mark.asyncio
class TestNativeGLMGuidance:
    @pytest.mark.parametrize("skip_validation", [False, True])
    @pytest.mark.parametrize("request_kind", ["raw", "validated", "constructed"])
    @pytest.mark.parametrize("tool_choice", _FORCED_CHOICES)
    async def test_native_tag_installed_before_adjust(
        self, monkeypatch, native_registry, skip_validation, request_kind, tool_choice
    ):
        monkeypatch.setattr(prepost_module, "SKIP_REQUEST_VALIDATION", skip_validation)
        request = _structured_request(
            tools=TOOL_REQUEST["tools"], tool_choice=tool_choice
        )
        if request_kind == "validated":
            request = ChatCompletionRequest.model_validate(request)
        elif request_kind == "constructed":
            request = ChatCompletionRequest.model_construct(**request)
        result = await _preprocess_structured(
            request, tool_parser_class=_NativeGLMToolParser
        )
        assert result.guided_decoding == {"structural_tag": _NATIVE_TAG}
        assert result.tool_parser.adjusted
        assert result.request_for_sampling.skip_special_tokens is False
        native_registry.assert_called_once()
        # Registry must receive normalized OpenAI models, not raw nested dicts.
        call = native_registry.call_args
        model = call.kwargs.get("model", call.args[0] if call.args else None)
        tools = call.kwargs.get("tools", call.args[1] if len(call.args) > 1 else None)
        choice = call.kwargs.get(
            "tool_choice", call.args[2] if len(call.args) > 2 else None
        )
        assert model == "glm_4_7"
        assert tools[0].function.name == "get_weather"
        assert tools[0].function.strict is None
        assert choice == "required" or choice.function.name == "get_weather"
        assert call.kwargs["reasoning"] is False

    @pytest.mark.parametrize("tool_choice", _FORCED_CHOICES)
    @pytest.mark.parametrize(
        "existing", [{"grammar": 'root ::= "yes"'}, {"structural_tag": _NATIVE_TAG}]
    )
    async def test_forced_tag_replaces_conflicting_and_identical_guidance(
        self, native_registry, tool_choice, existing
    ):
        result = await _preprocess_structured(
            _structured_request(
                tools=TOOL_REQUEST["tools"],
                tool_choice=tool_choice,
                structured_outputs=existing,
            ),
            tool_parser_class=_NativeGLMToolParser,
            reasoning_parser_class=_PassthroughReasoningParser,
        )
        assert result.guided_decoding == {"structural_tag": _NATIVE_TAG}
        assert result.request_for_sampling.response_format is None
        assert result.request_for_sampling.structured_outputs.json is None
        assert result.request_for_sampling.structured_outputs.regex is None
        assert result.request_for_sampling.skip_special_tokens is False

    @pytest.mark.parametrize(
        "failure", ["none", "error", "missing-tools", "unknown-name"]
    )
    async def test_failure_precedes_renderer(self, native_registry, failure):
        request = _structured_request(
            tools=TOOL_REQUEST["tools"], tool_choice="required"
        )
        if failure == "none":
            native_registry.return_value = None
        elif failure == "error":
            native_registry.side_effect = RuntimeError("registry failed")
        elif failure == "missing-tools":
            request["tools"] = []
        else:
            request["tool_choice"] = {
                "type": "function",
                "function": {"name": "unknown"},
            }
        renderer = _fake_renderer()
        expected_error = (
            (ValueError, VLLMValidationError)
            if failure in ("missing-tools", "unknown-name")
            else ValueError
        )
        with pytest.raises(expected_error):
            await prepost_module.preprocess_chat_request(
                request,
                tokenizer=object(),
                renderer=renderer,
                tool_parser_class=_NativeGLMToolParser,
            )
        renderer.render_messages_async.assert_not_awaited()

    async def test_unavailable_registry_fails_before_renderer(self, monkeypatch):
        import sys

        monkeypatch.setitem(
            sys.modules, "vllm.tool_parsers.structural_tag_registry", None
        )
        renderer = _fake_renderer()
        with pytest.raises(ValueError):
            await prepost_module.preprocess_chat_request(
                _structured_request(
                    tools=TOOL_REQUEST["tools"], tool_choice="required"
                ),
                tokenizer=object(),
                renderer=renderer,
                tool_parser_class=_NativeGLMToolParser,
            )
        renderer.render_messages_async.assert_not_awaited()

    async def test_reasoning_cannot_remove_forced_constraint(self, native_registry):
        renderer = _fake_renderer()
        with pytest.raises(ValueError):
            await prepost_module.preprocess_chat_request(
                _structured_request(
                    tools=TOOL_REQUEST["tools"], tool_choice="required"
                ),
                tokenizer=object(),
                renderer=renderer,
                tool_parser_class=_NativeGLMToolParser,
                reasoning_parser_class=_RewritingReasoningParser,
            )
        renderer.render_messages_async.assert_not_awaited()

    @pytest.mark.parametrize("tool_choice", ["auto", "none"])
    async def test_optional_choices_do_not_call_registry(
        self, native_registry, tool_choice
    ):
        class OptionalGLMParser(_GrammarToolParser):
            supports_required_and_named = False
            structural_tag_model = "glm_4_7"

        result = await _preprocess_structured(
            _structured_request(tools=TOOL_REQUEST["tools"], tool_choice=tool_choice),
            tool_parser_class=OptionalGLMParser,
        )
        assert result.guided_decoding == {"json": {"type": "object"}}
        native_registry.assert_not_called()

    @pytest.mark.parametrize(
        "supports,model", [(True, "glm_4_7"), (False, "qwen3_coder"), (False, "hy_v4")]
    )
    async def test_other_parser_capabilities_unchanged(
        self, native_registry, supports, model
    ):
        class OtherParser(_GrammarToolParser):
            supports_required_and_named = supports
            structural_tag_model = model

        result = await _preprocess_structured(
            _structured_request(tools=TOOL_REQUEST["tools"], tool_choice="required"),
            tool_parser_class=OtherParser,
        )
        assert result.guided_decoding == {"grammar": 'root ::= "<tool_call>"'}
        native_registry.assert_not_called()


class TestReasoningParserMetadata:
    def test_no_reasoning_parser_returns_none(self):
        from dingo.frontend.vllm_processor import _build_reasoning_parser_metadata

        assert _build_reasoning_parser_metadata(
            None,
            object(),
            {},
            SimpleNamespace(include_reasoning=True),
            [1, 2, 3],
        ) == (None, None)

    def test_include_reasoning_false_marks_reasoning_ended(self):
        from dingo.frontend.vllm_processor import _build_reasoning_parser_metadata

        class ParserShouldNotBeBuilt:
            def __init__(self, *args, **kwargs):
                raise AssertionError("parser should not be constructed")

        reasoning_ended, parser_kwargs = _build_reasoning_parser_metadata(
            ParserShouldNotBeBuilt,
            object(),
            {"reasoning_effort": "low"},
            SimpleNamespace(include_reasoning=False),
            [1, 2, 3],
        )

        assert reasoning_ended is True
        assert parser_kwargs == {"chat_template_kwargs": {"reasoning_effort": "low"}}

    def test_parser_receives_chat_template_kwargs(self):
        from dingo.frontend.vllm_processor import _build_reasoning_parser_metadata

        class FakeReasoningParser:
            def __init__(self, tokenizer, *, chat_template_kwargs):
                self.tokenizer = tokenizer
                self.chat_template_kwargs = chat_template_kwargs

            def is_reasoning_end(self, prompt_token_ids):
                return prompt_token_ids == [9, 9]

        tokenizer = object()
        reasoning_ended, parser_kwargs = _build_reasoning_parser_metadata(
            FakeReasoningParser,
            tokenizer,
            {"reasoning_effort": "high"},
            SimpleNamespace(include_reasoning=True),
            [9, 9],
        )

        assert reasoning_ended is True
        assert parser_kwargs == {"chat_template_kwargs": {"reasoning_effort": "high"}}

    def test_kv_router_copies_reasoning_metadata_to_extra_args(self):
        from dingo.frontend.vllm_processor import _inject_routing_metadata

        kv_kwargs = {"extra_args": {"mm_hashes": [123]}}
        _inject_routing_metadata(
            {
                "reasoning_ended": False,
                "reasoning_parser_kwargs": {
                    "chat_template_kwargs": {"reasoning_effort": "high"}
                },
            },
            kv_kwargs,
        )

        assert kv_kwargs["extra_args"] == {
            "mm_hashes": [123],
            "reasoning_ended": False,
            "reasoning_parser_kwargs": {
                "chat_template_kwargs": {"reasoning_effort": "high"}
            },
        }


class _FakeOutputProcessor:
    def __init__(self):
        self.request_states = {}
        self.added_requests = []
        self.aborted_requests = []

    def add_request(self, preproc, *args, **kwargs):
        self.added_requests.append((preproc, args, kwargs))
        self.request_states[preproc.request_id] = object()

    def process_outputs(self, outputs):
        return SimpleNamespace(
            reqs_to_abort=[],
            request_outputs=[SimpleNamespace(outputs=[SimpleNamespace(index=0)])],
        )

    def abort_requests(self, request_ids, internal=False):
        self.aborted_requests.append((request_ids, internal))
        for request_id in request_ids:
            self.request_states.pop(request_id, None)


class _FakePostProcessor:
    def process_output(self, output):
        return {
            "index": output.index,
            "delta": {"content": "x"},
            "finish_reason": None,
        }


@pytest.fixture
def vllm_processor_module(monkeypatch):
    import dingo.frontend.vllm_processor as module

    class FakeEngineCoreOutput:
        __struct_fields__ = ()

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setattr(module, "EngineCoreOutput", FakeEngineCoreOutput)
    monkeypatch.setattr(module._nvtx, "start_range", lambda *args, **kwargs: object())
    monkeypatch.setattr(module._nvtx, "end_range", lambda rng: None)
    return module


def _make_processor(module, routed_engine):
    processor = module.VllmProcessor.__new__(module.VllmProcessor)
    processor.routed_engine = routed_engine
    processor.output_processor = _FakeOutputProcessor()
    return processor


def _base_preproc():
    return {
        "model": MODEL,
        "token_ids": [1, 2, 3],
        "stop_conditions": {"max_tokens": 4},
        "sampling_options": {"temperature": 0.0},
        "output_options": {},
        "eos_token_ids": [],
        "annotations": [],
        "routing": None,
    }


async def _run_generate(processor, preproc, *, mm_routing_info=None, context=None):
    vllm_preproc = SimpleNamespace(
        sampling_params=SimpleNamespace(n=1),
        request_id="vllm-request",
        external_req_id=None,
    )
    post_processors = {0: _FakePostProcessor()}

    return [
        item
        async for item in processor._generate_and_stream(
            "request-id",
            {"model": MODEL},
            preproc,
            preproc["token_ids"],
            vllm_preproc,
            post_processors,
            mm_routing_info=mm_routing_info,
            context=context,
        )
    ]


class TestRoutedEnginePath:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(("response_format", "schema"), _RESPONSE_FORMATS)
    @pytest.mark.parametrize("native_tools", [False, True])
    async def test_generator_routes_response_format_and_reasoning_metadata(
        self,
        vllm_processor_module,
        monkeypatch,
        response_format,
        schema,
        native_registry,
        native_tools,
    ):
        module = vllm_processor_module
        routed_engine = _FakeRoutedEngine()
        processor = _make_processor(module, routed_engine)
        processor.tokenizer = SimpleNamespace(eos_token_id=2)
        processor.tool_parser_class = _NativeGLMToolParser if native_tools else None
        processor.reasoning_parser_class = _PassthroughReasoningParser
        processor.exclude_tools_when_tool_choice_none = True
        processor.enable_auto_tool_choice = False
        renderer = _fake_renderer()
        renderer.process_for_engine_async = AsyncMock(return_value={})

        def process_inputs(request_id, engine_inputs, sampling_params, tasks):
            return SimpleNamespace(
                request_id=request_id,
                external_req_id=None,
                sampling_params=sampling_params,
                mm_features=None,
            )

        processor.input_processor = SimpleNamespace(
            renderer=renderer,
            generation_config_fields={},
            process_inputs=process_inputs,
        )
        processor._prepare_mm_routing = AsyncMock(return_value=(None, [], False))
        monkeypatch.setattr(
            module.InputProcessor, "assign_request_id", lambda request: None
        )
        monkeypatch.setattr(
            module, "StreamingPostProcessor", lambda **kwargs: _FakePostProcessor()
        )
        request = _structured_request(
            response_format=response_format,
            max_tokens=17,
            temperature=0.25,
            top_p=0.8,
            seed=42,
            reasoning_effort="high",
            chat_template_kwargs={"enable_thinking": True},
        )

        if native_tools:
            request.update(
                tools=deepcopy(TOOL_REQUEST["tools"]), tool_choice="required"
            )

        _ = [chunk async for chunk in processor._generator_inner(request)]

        assert len(routed_engine.requests) == 1
        payload = routed_engine.requests[0]
        expected = {"structural_tag": _NATIVE_TAG} if native_tools else {"json": schema}
        assert payload["sampling_options"]["guided_decoding"] == expected
        assert payload["sampling_options"]["temperature"] == 0.25
        assert payload["sampling_options"]["top_p"] == 0.8
        assert payload["sampling_options"]["seed"] == 42
        assert payload["stop_conditions"]["max_tokens"] == 17
        assert payload["output_options"]["skip_special_tokens"] is False
        assert payload["token_ids"] == [1, 2, 3]
        assert payload["extra_args"]["reasoning_ended"] is False
        assert payload["extra_args"]["reasoning_parser_kwargs"] == {
            "chat_template_kwargs": {
                "enable_thinking": True,
                "reasoning_effort": "high",
            }
        }

    @pytest.mark.asyncio
    async def test_routed_engine_gets_extra_args_metadata(self, vllm_processor_module):
        routed_engine = _FakeRoutedEngine()
        processor = _make_processor(vllm_processor_module, routed_engine)
        preproc = _base_preproc()
        preproc["extra_args"] = {"mm_hashes": [123]}
        preproc["reasoning_ended"] = False
        preproc["reasoning_parser_kwargs"] = {
            "chat_template_kwargs": {"reasoning_effort": "high"}
        }
        preproc["mm_processor_kwargs"] = {"use_audio_in_video": True}

        await _run_generate(processor, preproc)

        assert routed_engine.requests[0]["extra_args"] == {
            "mm_hashes": [123],
            "reasoning_ended": False,
            "reasoning_parser_kwargs": {
                "chat_template_kwargs": {"reasoning_effort": "high"}
            },
            "mm_processor_kwargs": {"use_audio_in_video": True},
        }

    @pytest.mark.asyncio
    async def test_routed_stream_produces_openai_chunks(self, vllm_processor_module):
        routed_engine = _FakeRoutedEngine(
            [{"token_ids": [101], "index": 0, "finish_reason": None}]
        )
        processor = _make_processor(vllm_processor_module, routed_engine)

        chunks = await _run_generate(processor, _base_preproc())

        # One annotated envelope per iteration carries both data and the
        # llm_metrics annotation; observer strips the annotation before SSE.
        assert len(chunks) == 1
        envelope = chunks[0]

        assert envelope["_dynamo_annotated"] is True
        assert envelope["data"] == {
            "id": "request-id",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "x"},
                    "finish_reason": None,
                }
            ],
            "created": envelope["data"]["created"],
            "model": MODEL,
            "object": "chat.completion.chunk",
        }

        assert envelope["event"] == "llm_metrics"
        assert len(envelope["comment"]) == 1
        assert json.loads(envelope["comment"][0]) == {
            "input_tokens": 3,
            "output_tokens": 1,
            "chunk_tokens": 1,
        }


OBJECT_TYPED_TOOL_REQUEST = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "set my profile"}],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "set_profile",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "profile": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "age": {"type": "integer"},
                            },
                        }
                    },
                    "required": ["profile"],
                },
            },
        }
    ],
    "tool_choice": "auto",
}


# ---------------------------------------------------------------------------
# _prepare_request: schema-aware tool-parser end-to-end regression
# ---------------------------------------------------------------------------


class TestSchemaAwareToolParser:
    """Schema-aware parsers (e.g. qwen3_coder) need ``tools`` at construction
    to coerce object/array-typed parameter values from raw text into JSON;
    without them, the value comes through as a string-in-a-string inside the
    final ``arguments`` JSON.
    """

    def test_qwen3_coder_coerces_object_typed_arg(self, tokenizer):
        """qwen3_coder must coerce object-typed parameter values into nested
        objects, not leave them as JSON-encoded strings inside ``arguments``.
        """
        model_output = (
            "<tool_call><function=set_profile>\n"
            "<parameter=profile>\n"
            '{"name": "Alice", "age": 30}\n'
            "</parameter>\n"
            "</function></tool_call>"
        )

        request_for_sampling, parser, _, _, _ = _prepare_request(
            OBJECT_TYPED_TOOL_REQUEST,
            tokenizer=tokenizer,
            tool_parser_class=Qwen3EngineToolParser,
        )
        assert parser is not None, "Expected _prepare_request to construct the parser"

        result = parser.extract_tool_calls(model_output, request_for_sampling)

        assert result.tools_called, f"Expected tools_called=True; got {result!r}"
        assert len(result.tool_calls) == 1
        args = json.loads(result.tool_calls[0].function.arguments)
        assert isinstance(args["profile"], dict), (
            f"Schema-aware parser should coerce object-typed arg to dict; "
            f"got {type(args['profile']).__name__}: {args['profile']!r}"
        )
        assert args["profile"] == {"name": "Alice", "age": 30}


# ---------------------------------------------------------------------------
# _prepare_request: chat_template_kwargs forwarding
# ---------------------------------------------------------------------------


@pytest.mark.core
class TestChatTemplateKwargsForwarding:
    """chat_template_kwargs from the request are forwarded to ChatParams.

    Uses Qwen3 which supports enable_thinking: False to suppress <think> blocks.
    """

    @staticmethod
    def _messages():
        return [{"role": "user", "content": "Hello"}]

    def _prepare(self, request, tokenizer):
        """Return (chat_params, messages) from _prepare_request."""
        _, _, _, messages, chat_params = _prepare_request(
            request,
            tokenizer=tokenizer,
            tool_parser_class=None,
        )
        return chat_params, messages

    def _render(self, tokenizer, chat_params) -> str:
        """Render prompt text using the chat_params template kwargs."""
        kwargs = {**chat_params.chat_template_kwargs, "tokenize": False}
        return tokenizer.apply_chat_template(self._messages(), **kwargs)

    def test_qwen3_enable_thinking_true_no_closed_think_block(self, tokenizer):
        """enable_thinking=True leaves reasoning open (model generates <think> itself)."""
        chat_params, _ = self._prepare(
            {
                "model": MODEL,
                "messages": self._messages(),
                "chat_template_kwargs": {"enable_thinking": True},
            },
            tokenizer,
        )
        prompt = self._render(tokenizer, chat_params)
        assert "</think>" not in prompt

    def test_qwen3_thinking_flag_changes_tokens(self, tokenizer):
        """enable_thinking=True vs False produces different rendered prompts."""
        think_params, _ = self._prepare(
            {
                "model": MODEL,
                "messages": self._messages(),
                "chat_template_kwargs": {"enable_thinking": True},
            },
            tokenizer,
        )
        no_think_params, _ = self._prepare(
            {
                "model": MODEL,
                "messages": self._messages(),
                "chat_template_kwargs": {"enable_thinking": False},
            },
            tokenizer,
        )
        assert self._render(tokenizer, think_params) != self._render(
            tokenizer, no_think_params
        )

    def test_reasoning_effort_forwarded_to_template_kwargs(self, tokenizer):
        """reasoning_effort is always present in chat_params.chat_template_kwargs."""
        chat_params, _ = self._prepare(
            {
                "model": MODEL,
                "messages": self._messages(),
                "reasoning_effort": "low",
            },
            tokenizer,
        )
        assert chat_params.chat_template_kwargs.get("reasoning_effort") == "low"


@pytest.mark.parametrize(
    ("runtime_config", "expected"),
    [
        ({"context_length": 1048576}, 1048576),
        ({}, None),
        ({"context_length": None}, None),
        ({"context_length": 0}, None),
        ({"context_length": -1}, None),
        ({"context_length": "1048576"}, None),
        ({"context_length": True}, None),
        (None, None),
    ],
)
def test_runtime_config_context_length(vllm_processor_module, runtime_config, expected):
    mdc = SimpleNamespace(runtime_config=lambda: runtime_config)

    assert vllm_processor_module._runtime_config_context_length(mdc) == expected
