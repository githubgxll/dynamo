#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Tests for tool call parsing in SglangStreamingPostProcessor.

Covers the interaction between SGLang's FunctionCallParser, ReasoningParser,
and our post-processor's incremental tool-call streaming (id+name first,
then argument fragments), including the finish-time parse_non_stream
fallback/merge for the chunking-sensitivity issue in
BaseFormatDetector.parse_streaming_increment.
"""

import json
from typing import Any

import pytest
from sglang.srt.entrypoints.openai.protocol import Function as SglangFunction
from sglang.srt.entrypoints.openai.protocol import Tool as SglangTool
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.json_array_parser import JsonArrayParser
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.utils.hf_transformers_utils import get_tokenizer

from dingo.frontend.sglang_prepost import SglangStreamingPostProcessor

# Needs sglang packages (gpu_1 container), but does not allocate GPU VRAM.
pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
    pytest.mark.profiled_vram_gib(0),
]

MODEL = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer(MODEL)


TOOLS = [
    SglangTool(
        type="function",
        function=SglangFunction(
            name="search_gutenberg_books",
            description="Search for books in the Project Gutenberg library",
            parameters={
                "type": "object",
                "properties": {
                    "search_terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of search terms to find books",
                    }
                },
                "required": ["search_terms"],
            },
        ),
    ),
    SglangTool(
        type="function",
        function=SglangFunction(
            name="get_weather",
            description="Get weather for a city",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        ),
    ),
]


def _run_postprocessor(tokenizer, full_text, batch_size, *, use_reasoning=True):
    """Tokenize text, feed through post-processor in batches, return all choices."""
    tcp = FunctionCallParser(tools=TOOLS, tool_call_parser="hermes")
    rp = (
        ReasoningParser(model_type="qwen3", stream_reasoning=True)
        if use_reasoning
        else None
    )

    post = SglangStreamingPostProcessor(
        tokenizer=tokenizer,
        tool_call_parser=tcp,
        reasoning_parser=rp,
    )

    token_ids = tokenizer.encode(full_text)
    results = []
    for i in range(0, len(token_ids), batch_size):
        batch = token_ids[i : i + batch_size]
        is_last = i + batch_size >= len(token_ids)
        choice = post.process_output(
            {"token_ids": batch, "finish_reason": "stop" if is_last else None}
        )
        if choice:
            results.append(choice)
    return results


def _merge_tool_call_entries(entries):
    """Merge OpenAI streaming tool_call delta entries into complete calls.

    Mirrors the client-side reassembly contract: an entry carrying ``id``
    starts the call (name + empty arguments), entries without ``id``
    append argument fragments, and a finish-time recovered call arrives
    as a single complete entry.
    """
    calls: dict[int, dict] = {}
    order: list[int] = []
    for e in entries:
        idx = e["index"]
        if idx not in calls:
            calls[idx] = {
                "index": idx,
                "id": None,
                "type": "function",
                "function": {"name": None, "arguments": ""},
            }
            order.append(idx)
        fn = e.get("function", {})
        if e.get("id"):
            calls[idx]["id"] = e["id"]
        if fn.get("name"):
            calls[idx]["function"]["name"] = fn["name"]
        calls[idx]["function"]["arguments"] += fn.get("arguments", "")
    return [calls[i] for i in order]


def _extract_tool_calls(results):
    """Reassemble complete tool calls from incremental deltas across choices."""
    entries = []
    for r in results:
        entries.extend(r.get("delta", {}).get("tool_calls") or [])
    return _merge_tool_call_entries(entries)


# ---------------------------------------------------------------------------
# Single tool call
# ---------------------------------------------------------------------------


class TestSingleToolCall:  # FRONTEND.4 — single tool-call output assembly
    """Single tool call with reasoning, various batch sizes."""

    TEXT = (
        "<think>\nLet me search for books.\n</think>\n\n"
        '<tool_call>\n{"name": "search_gutenberg_books", '
        '"arguments": {"search_terms": ["James Joyce"]}}\n</tool_call>'
    )

    def test_large_batches(self, tokenizer):
        """stream_interval=20 scenario -- complete JSON in one chunk."""
        tc = _extract_tool_calls(_run_postprocessor(tokenizer, self.TEXT, 20))
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "search_gutenberg_books"
        args = json.loads(tc[0]["function"]["arguments"])
        assert args == {"search_terms": ["James Joyce"]}

    def test_small_batches(self, tokenizer):
        """Token-by-token-ish scenario -- streaming deltas work directly."""
        tc = _extract_tool_calls(_run_postprocessor(tokenizer, self.TEXT, 3))
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "search_gutenberg_books"
        args = json.loads(tc[0]["function"]["arguments"])
        assert args == {"search_terms": ["James Joyce"]}

    def test_medium_batches(self, tokenizer):
        """Intermediate batch size."""
        tc = _extract_tool_calls(_run_postprocessor(tokenizer, self.TEXT, 10))
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "search_gutenberg_books"
        args = json.loads(tc[0]["function"]["arguments"])
        assert args == {"search_terms": ["James Joyce"]}

    def test_tool_call_has_id_and_type(self, tokenizer):
        """Each tool call must have id and type fields."""
        tc = _extract_tool_calls(_run_postprocessor(tokenizer, self.TEXT, 20))
        assert tc[0]["id"].startswith("call_")
        assert tc[0]["type"] == "function"
        assert tc[0]["index"] == 0


class TestKimiToolCallIds:  # FRONTEND.4 — Kimi-specific tool-call ID format on output
    def test_kimi_uses_history_adjusted_ids(self):
        class DummyTokenizer:
            def decode(self, token_ids, skip_special_tokens=True):
                return "".join(chr(x) for x in token_ids)

        class DummyToolCall:
            def __init__(self, tool_index, name, parameters):
                self.tool_index = tool_index
                self.name = name
                self.parameters = parameters

        class DummyParser:
            tool_call_parser = "kimi_k2"
            detector = type("Detector", (), {"_buffer": ""})()

            def parse_stream_chunk(self, text):
                return "", [
                    DummyToolCall(0, "get_weather", '{"city":"Paris"}'),
                    DummyToolCall(
                        1, "search_gutenberg_books", '{"search_terms":["Joyce"]}'
                    ),
                ]

        post = SglangStreamingPostProcessor(
            tokenizer=DummyTokenizer(),
            tool_call_parser=DummyParser(),
            reasoning_parser=None,
            history_tool_calls_count=3,
            tool_call_parser_name="kimi_k2",
        )

        choice = post.process_output(
            {
                "token_ids": [ord("x")],
                "finish_reason": "stop",
            }
        )

        tc = _merge_tool_call_entries(choice["delta"]["tool_calls"])
        assert [item["id"] for item in tc] == [
            "functions.get_weather:3",
            "functions.search_gutenberg_books:4",
        ]

    def test_kimi_reparse_uses_sequential_index_not_tool_index(self):
        """kimi_k2 IDs after re-parse use the output position, not tool_index.

        ``FunctionCallParser.parse_non_stream`` can return
        ``ToolCallItem.tool_index`` values that reflect the tool-definition
        position rather than the call's sequential position.  IDs must
        align with the emitted ``index`` field, so they are built from
        the post-processor's ``seq_idx``.
        """

        class DummyTokenizer:
            def decode(self, token_ids, skip_special_tokens=True):
                return "".join(chr(x) for x in token_ids)

        class DummyToolCall:
            def __init__(self, tool_index, name, parameters):
                self.tool_index = tool_index
                self.name = name
                self.parameters = parameters

        class DummyParser:
            tool_call_parser = "kimi_k2"
            detector = type("Detector", (), {"_buffer": ""})()

            def parse_stream_chunk(self, text):
                # Streaming misses both calls — forces the re-parse path.
                return "", []

            def has_tool_call(self, text):
                return True

            def parse_non_stream(self, text):
                # Non-sequential tool_index values, as parse_non_stream
                # sometimes returns tool-definition positions.
                return "", [
                    DummyToolCall(5, "get_weather", '{"city":"Paris"}'),
                    DummyToolCall(2, "search_gutenberg_books", '{"q":"Joyce"}'),
                ]

        post = SglangStreamingPostProcessor(
            tokenizer=DummyTokenizer(),
            tool_call_parser=DummyParser(),
            reasoning_parser=None,
            history_tool_calls_count=3,
            tool_call_parser_name="kimi_k2",
        )

        choice = post.process_output(
            {
                "token_ids": [ord("x")],
                "finish_reason": "stop",
            }
        )

        tc = _merge_tool_call_entries(choice["delta"]["tool_calls"])
        # IDs must use seq_idx (0, 1) + history (3), not tool_index (5, 2).
        assert [item["id"] for item in tc] == [
            "functions.get_weather:3",
            "functions.search_gutenberg_books:4",
        ]
        assert [item["index"] for item in tc] == [0, 1]


# ---------------------------------------------------------------------------
# No reasoning parser
# ---------------------------------------------------------------------------


class TestNoReasoningParser:  # FRONTEND.2 — graceful behavior when no reasoning parser configured
    """Tool calls without reasoning parser active."""

    TEXT = (
        '<tool_call>\n{"name": "get_weather", '
        '"arguments": {"city": "Paris"}}\n</tool_call>'
    )

    def test_large_batches(self, tokenizer):
        tc = _extract_tool_calls(
            _run_postprocessor(tokenizer, self.TEXT, 15, use_reasoning=False)
        )
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "get_weather"
        args = json.loads(tc[0]["function"]["arguments"])
        assert args == {"city": "Paris"}

    def test_small_batches(self, tokenizer):
        tc = _extract_tool_calls(
            _run_postprocessor(tokenizer, self.TEXT, 3, use_reasoning=False)
        )
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "get_weather"
        args = json.loads(tc[0]["function"]["arguments"])
        assert args == {"city": "Paris"}


# ---------------------------------------------------------------------------
# Multiple tool calls
# ---------------------------------------------------------------------------


class TestMultipleToolCalls:  # FRONTEND.4 — parallel/multiple tool-call assembly
    """Two tool calls in a single response."""

    TEXT = (
        "<think>\nI'll search and check weather.\n</think>\n\n"
        '<tool_call>\n{"name": "search_gutenberg_books", '
        '"arguments": {"search_terms": ["Joyce"]}}\n</tool_call>\n'
        '<tool_call>\n{"name": "get_weather", '
        '"arguments": {"city": "London"}}\n</tool_call>'
    )

    def test_both_tools_present(self, tokenizer):
        tc = _extract_tool_calls(_run_postprocessor(tokenizer, self.TEXT, 10))
        assert len(tc) == 2
        names = {t["function"]["name"] for t in tc}
        assert names == {"search_gutenberg_books", "get_weather"}

    def test_arguments_correct(self, tokenizer):
        tc = _extract_tool_calls(_run_postprocessor(tokenizer, self.TEXT, 10))
        by_name = {t["function"]["name"]: t for t in tc}
        assert json.loads(
            by_name["search_gutenberg_books"]["function"]["arguments"]
        ) == {"search_terms": ["Joyce"]}
        assert json.loads(by_name["get_weather"]["function"]["arguments"]) == {
            "city": "London"
        }

    def test_distinct_ids(self, tokenizer):
        tc = _extract_tool_calls(_run_postprocessor(tokenizer, self.TEXT, 10))
        ids = [t["id"] for t in tc]
        assert len(set(ids)) == len(ids), "Tool call IDs must be unique"


# ---------------------------------------------------------------------------
# Content alongside tool calls
# ---------------------------------------------------------------------------


class TestContentWithToolCalls:  # FRONTEND.4 — text content interleaved with tool calls
    """Reasoning content and regular content are preserved alongside tool calls."""

    TEXT = (
        "<think>\nThinking about it.\n</think>\n\n"
        '<tool_call>\n{"name": "get_weather", '
        '"arguments": {"city": "NYC"}}\n</tool_call>'
    )

    def test_reasoning_content_present(self, tokenizer):
        results = _run_postprocessor(tokenizer, self.TEXT, 20)
        reasoning = ""
        for r in results:
            rc = r.get("delta", {}).get("reasoning_content", "")
            reasoning += rc
        assert "Thinking about it" in reasoning

    def test_content_is_whitespace_only(self, tokenizer):
        """Content between </think> and <tool_call> should be whitespace only."""
        results = _run_postprocessor(tokenizer, self.TEXT, 20)
        content = ""
        for r in results:
            c = r.get("delta", {}).get("content", "")
            content += c
        assert content.strip() == ""


# ---------------------------------------------------------------------------
# No tool calls (plain text)
# ---------------------------------------------------------------------------


class TestNoToolCalls:  # FRONTEND.4 — text-only response (no tool calls)
    """When no tool call markup is present, no tool_calls should appear."""

    TEXT = "<think>\nJust thinking.\n</think>\n\nHello, world!"

    def test_no_tool_calls_emitted(self, tokenizer):
        tc = _extract_tool_calls(_run_postprocessor(tokenizer, self.TEXT, 10))
        assert tc == []

    def test_content_preserved(self, tokenizer):
        results = _run_postprocessor(tokenizer, self.TEXT, 10)
        content = ""
        for r in results:
            c = r.get("delta", {}).get("content", "")
            content += c
        assert "Hello, world!" in content


# ---------------------------------------------------------------------------
# Single-chunk tool calls (finish-time re-parse fallback)
# ---------------------------------------------------------------------------


class TestSingleChunkFallback:  # FRONTEND.4 — non-streaming fallback assembly
    """When all tool call tokens + finish arrive in one batch, the streaming
    parser only processes one event.  The finish-time re-parse must recover
    arguments and any additional tool calls."""

    TEXT = (
        "<think>\nLet me search for books.\n</think>\n\n"
        '<tool_call>\n{"name": "search_gutenberg_books", '
        '"arguments": {"search_terms": ["James Joyce"]}}\n</tool_call>'
    )

    def test_all_tokens_plus_finish_in_one_batch(self, tokenizer):
        """Entire response + finish in a single process_output call."""
        tcp = FunctionCallParser(tools=TOOLS, tool_call_parser="hermes")
        rp = ReasoningParser(model_type="qwen3", stream_reasoning=True)
        post = SglangStreamingPostProcessor(
            tokenizer=tokenizer,
            tool_call_parser=tcp,
            reasoning_parser=rp,
        )
        token_ids = tokenizer.encode(self.TEXT)
        # Feed ALL tokens at once with finish_reason
        choice = post.process_output({"token_ids": token_ids, "finish_reason": "stop"})
        assert choice is not None
        tc = _merge_tool_call_entries(choice.get("delta", {}).get("tool_calls", []))
        assert len(tc) == 1, f"Expected 1 tool call, got {len(tc)}"
        assert tc[0]["function"]["name"] == "search_gutenberg_books"
        args = json.loads(tc[0]["function"]["arguments"])
        assert args == {"search_terms": ["James Joyce"]}

    def test_multiple_tools_single_chunk(self, tokenizer):
        """Multiple tool calls in one chunk -- re-parse must find all."""
        text = (
            "<think>\nI'll search and check weather.\n</think>\n\n"
            '<tool_call>\n{"name": "search_gutenberg_books", '
            '"arguments": {"search_terms": ["Joyce"]}}\n</tool_call>\n'
            '<tool_call>\n{"name": "get_weather", '
            '"arguments": {"city": "London"}}\n</tool_call>'
        )
        tcp = FunctionCallParser(tools=TOOLS, tool_call_parser="hermes")
        rp = ReasoningParser(model_type="qwen3", stream_reasoning=True)
        post = SglangStreamingPostProcessor(
            tokenizer=tokenizer,
            tool_call_parser=tcp,
            reasoning_parser=rp,
        )
        token_ids = tokenizer.encode(text)
        choice = post.process_output({"token_ids": token_ids, "finish_reason": "stop"})
        assert choice is not None
        tc = _merge_tool_call_entries(choice.get("delta", {}).get("tool_calls", []))
        assert len(tc) == 2, f"Expected 2 tool calls, got {len(tc)}"
        names = {t["function"]["name"] for t in tc}
        assert names == {"search_gutenberg_books", "get_weather"}
        for t in tc:
            args = json.loads(t["function"]["arguments"])
            assert args, f"Arguments should not be empty for {t['function']['name']}"

    def test_finish_reason_rewritten_to_tool_calls(self, tokenizer):
        """finish_reason should be 'tool_calls' when re-parse finds calls."""
        tcp = FunctionCallParser(tools=TOOLS, tool_call_parser="hermes")
        post = SglangStreamingPostProcessor(
            tokenizer=tokenizer,
            tool_call_parser=tcp,
            reasoning_parser=None,
        )
        text = (
            '<tool_call>\n{"name": "get_weather", '
            '"arguments": {"city": "NYC"}}\n</tool_call>'
        )
        token_ids = tokenizer.encode(text)
        choice = post.process_output({"token_ids": token_ids, "finish_reason": "stop"})
        assert choice is not None
        assert choice["finish_reason"] == "tool_calls"


class TestMalformedToolCalls:  # FRONTEND.4 — malformed model output → graceful degradation
    """Contract under incremental streaming:

    - Without tools-list confirmation, a name detected without any argument
      fragment is never emitted and does not rewrite the finish_reason.
    - An unknown tool name is suppressed mid-stream and purged at
      finish; nothing reaches the client.
    - A known name with malformed (non-JSON) arguments IS streamed
      optimistically — matching native SGLang behaviour — and the
      finish-time re-parse attempts authoritative recovery.
    """

    class DummyTokenizer:
        def decode(self, token_ids, skip_special_tokens=True):
            return "".join(chr(x) for x in token_ids)

    class DummyToolCall:
        def __init__(self, tool_index, name, parameters):
            self.tool_index = tool_index
            self.name = name
            self.parameters = parameters

    def test_name_without_arguments_is_never_emitted(self):
        dummy_tokenizer = self.DummyTokenizer()
        dummy_tc = self.DummyToolCall

        class DummyParser:
            def parse_stream_chunk(self, text):
                # Name event only — no argument fragment ever arrives.
                return "", [dummy_tc(0, "get_weather", None)]

            def has_tool_call(self, text):
                return "<tool_call>" in text

            def parse_non_stream(self, text):
                return "", []

        post = SglangStreamingPostProcessor(
            tokenizer=dummy_tokenizer,
            tool_call_parser=DummyParser(),
            reasoning_parser=None,
        )

        malformed = (
            '<tool_call>\n{"name": "get_weather", '
            '"arguments": {"city": "Paris"}\n</tool_call>'
        )
        choice = post.process_output(
            {
                "token_ids": [ord(c) for c in malformed],
                "finish_reason": "stop",
            }
        )

        assert choice is not None
        assert choice["finish_reason"] == "stop"
        assert choice.get("delta", {}).get("tool_calls", []) == []

    def test_unknown_tool_name_is_never_streamed(self):
        dummy_tokenizer = self.DummyTokenizer()
        dummy_tc = self.DummyToolCall

        class DummyParser:
            def parse_stream_chunk(self, text):
                return "", [dummy_tc(0, "evil_tool", '{"x": 1}')]

            def has_tool_call(self, text):
                return True

            def parse_non_stream(self, text):
                return "", []

        post = SglangStreamingPostProcessor(
            tokenizer=dummy_tokenizer,
            tool_call_parser=DummyParser(),
            reasoning_parser=None,
            sglang_tools=TOOLS,
        )

        text = '<tool_call>\n{"name": "evil_tool", "arguments": {"x": 1}}\n</tool_call>'
        choice = post.process_output(
            {
                "token_ids": [ord(c) for c in text],
                "finish_reason": "stop",
            }
        )

        assert choice is not None
        assert choice["finish_reason"] == "stop"
        assert choice.get("delta", {}).get("tool_calls", []) == []

    def test_malformed_arguments_stream_optimistically(self):
        dummy_tokenizer = self.DummyTokenizer()
        dummy_tc = self.DummyToolCall

        class DummyParser:
            def parse_stream_chunk(self, text):
                # Known name with malformed (unrecoverable) arguments.
                return "", [dummy_tc(0, "get_weather", '{"city": "Paris"')]

            def has_tool_call(self, text):
                return True

            def parse_non_stream(self, text):
                return "", []

        post = SglangStreamingPostProcessor(
            tokenizer=dummy_tokenizer,
            tool_call_parser=DummyParser(),
            reasoning_parser=None,
            sglang_tools=TOOLS,
        )

        malformed = (
            '<tool_call>\n{"name": "get_weather", '
            '"arguments": {"city": "Paris"}\n</tool_call>'
        )
        choice = post.process_output(
            {
                "token_ids": [ord(c) for c in malformed],
                "finish_reason": "stop",
            }
        )

        # Native-compatible tradeoff: the fragments went out before the
        # JSON could be validated; the call counts as emitted, so the
        # finish_reason is rewritten to tool_calls.
        assert choice is not None
        assert choice["finish_reason"] == "tool_calls"
        tc = _merge_tool_call_entries(choice.get("delta", {}).get("tool_calls", []))
        assert tc[0]["function"]["name"] == "get_weather"
        assert tc[0]["function"]["arguments"] == '{"city": "Paris"'


# ---------------------------------------------------------------------------
# JsonArrayParser path (tool_choice="required" / named function)
# ---------------------------------------------------------------------------


class TestJsonArrayParserReparse:  # FRONTEND.4 — JSON-array parser reparse path
    """Exercise the JsonArrayParser branch of the finish-time re-parse.

    Under ``tool_choice="required"`` or a named function, guided decoding
    constrains the model to emit a raw JSON array and
    SglangStreamingPostProcessor is constructed with a JsonArrayParser
    instead of a FunctionCallParser. The re-parse path uses
    ``has_tool_call`` on the parser as a cheap gate and
    ``_parse_json_array_buffer`` for recovery — this class locks in that
    API surface so a SGLang upgrade can't silently break it.
    """

    def test_single_call_reparse(self, tokenizer):
        """Full JSON array arriving in one chunk triggers the re-parse."""
        text = '[{"name": "get_weather", "parameters": {"city": "NYC"}}]'
        post = SglangStreamingPostProcessor(
            tokenizer=tokenizer,
            tool_call_parser=JsonArrayParser(),
            reasoning_parser=None,
            sglang_tools=TOOLS,
        )
        token_ids = tokenizer.encode(text)
        choice = post.process_output({"token_ids": token_ids, "finish_reason": "stop"})
        assert choice is not None
        tc = _merge_tool_call_entries(choice.get("delta", {}).get("tool_calls", []))
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "get_weather"
        assert json.loads(tc[0]["function"]["arguments"]) == {"city": "NYC"}
        assert choice["finish_reason"] == "tool_calls"

    def test_multiple_calls_reparse(self, tokenizer):
        """Multiple calls in one chunk; re-parse must recover all."""
        text = (
            '[{"name": "search_gutenberg_books", '
            '"parameters": {"search_terms": ["Joyce"]}}, '
            '{"name": "get_weather", "parameters": {"city": "London"}}]'
        )
        post = SglangStreamingPostProcessor(
            tokenizer=tokenizer,
            tool_call_parser=JsonArrayParser(),
            reasoning_parser=None,
            sglang_tools=TOOLS,
        )
        token_ids = tokenizer.encode(text)
        choice = post.process_output({"token_ids": token_ids, "finish_reason": "stop"})
        assert choice is not None
        tc = _merge_tool_call_entries(choice.get("delta", {}).get("tool_calls", []))
        assert len(tc) == 2
        names = {t["function"]["name"] for t in tc}
        assert names == {"search_gutenberg_books", "get_weather"}

    def test_plain_text_skips_reparse(self, tokenizer):
        """Plain text with no JSON markers must not crash the re-parse path.

        Locks in that the ``has_tool_call`` gate on JsonArrayParser returns
        False for text without '[' or '{', so ``_parse_json_array_buffer``
        and the secondary FunctionCallParser fallback are never reached.
        """
        post = SglangStreamingPostProcessor(
            tokenizer=tokenizer,
            tool_call_parser=JsonArrayParser(),
            reasoning_parser=None,
            sglang_tools=TOOLS,
        )
        token_ids = tokenizer.encode("Hello, world!")
        choice = post.process_output({"token_ids": token_ids, "finish_reason": "stop"})
        # No tool calls, plain content preserved, no crash.
        tc = _merge_tool_call_entries(
            (choice or {}).get("delta", {}).get("tool_calls", [])
        )
        assert tc == []


# ---------------------------------------------------------------------------
# Incremental tool-call streaming (TTFT)
# ---------------------------------------------------------------------------


class TestIncrementalToolStreaming:  # FRONTEND.4 — tool_call deltas stream before finish
    """TTFT contract: clients must see tool_call deltas as the parser
    detects them instead of everything arriving with the finish chunk."""

    TEXT = (
        '<tool_call>\n{"name": "get_weather", '
        '"arguments": {"city": "Paris"}}\n</tool_call>'
    )

    MULTI_TEXT = (
        '<tool_call>\n{"name": "search_gutenberg_books", '
        '"arguments": {"search_terms": ["Joyce"]}}\n</tool_call>\n'
        '<tool_call>\n{"name": "get_weather", '
        '"arguments": {"city": "London"}}\n</tool_call>'
    )

    @staticmethod
    def _all_entries(results):
        return [
            e for r in results for e in (r.get("delta", {}).get("tool_calls") or [])
        ]

    def test_tool_deltas_arrive_before_finish(self, tokenizer):
        results = _run_postprocessor(tokenizer, self.TEXT, 3, use_reasoning=False)
        assert results[-1]["finish_reason"] is not None
        first_tool_idx = next(
            i for i, r in enumerate(results) if r.get("delta", {}).get("tool_calls")
        )
        assert first_tool_idx < len(results) - 1, (
            "tool_call deltas must stream before the finish chunk "
            "(buffering them is what inflated client-side TTFT)"
        )

    def test_first_entry_carries_id_and_name(self, tokenizer):
        results = _run_postprocessor(tokenizer, self.TEXT, 3, use_reasoning=False)
        entries = self._all_entries(results)
        assert entries, "expected streamed tool_call entries"
        first = entries[0]
        assert first["id"].startswith("call_")
        assert first["type"] == "function"
        assert first["function"]["name"] == "get_weather"
        assert first["function"]["arguments"] == ""

    def test_reassembled_arguments_complete(self, tokenizer):
        tc = _extract_tool_calls(
            _run_postprocessor(tokenizer, self.TEXT, 3, use_reasoning=False)
        )
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "get_weather"
        assert json.loads(tc[0]["function"]["arguments"]) == {"city": "Paris"}

    def test_multiple_calls_stream_with_distinct_indices(self, tokenizer):
        results = _run_postprocessor(tokenizer, self.MULTI_TEXT, 5, use_reasoning=False)
        entries = self._all_entries(results)
        assert entries, "expected streamed tool_call entries"
        tc = _merge_tool_call_entries(entries)
        assert {t["function"]["name"] for t in tc} == {
            "search_gutenberg_books",
            "get_weather",
        }
        assert len({t["index"] for t in tc}) == 2
        ids = [t["id"] for t in tc]
        assert len(set(ids)) == len(ids), "tool call ids must be unique"

    def test_finish_reason_rewritten_to_tool_calls(self, tokenizer):
        results = _run_postprocessor(tokenizer, self.TEXT, 3, use_reasoning=False)
        assert results[-1]["finish_reason"] == "tool_calls"

    def test_pure_tool_call_first_choice_is_not_delayed(self, tokenizer):
        """Pure tool-call responses (no reasoning preface) must produce a
        tool_call-bearing choice well before the final one — this is the
        scenario where buffered emission made TTFT equal to the whole
        generation time."""
        results = _run_postprocessor(tokenizer, self.TEXT, 3, use_reasoning=False)
        first_tool_idx = next(
            i for i, r in enumerate(results) if r.get("delta", {}).get("tool_calls")
        )
        # The name completes after roughly the first third of the tokens;
        # allow generous slack but require strict precedence over finish.
        assert first_tool_idx <= len(results) // 2


class TestToolStreamingRecoveryRegression:
    """Exercise terminal recovery and early identity without model downloads."""

    class Tokenizer:
        def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
            return "".join(chr(token) for token in token_ids)

    class Parser:
        def __init__(
            self, events: list[list[ToolCallItem]], recovered: list[ToolCallItem]
        ) -> None:
            self.events = iter(events)
            self.recovered = recovered

        def parse_stream_chunk(self, text: str) -> tuple[str, list[ToolCallItem]]:
            return "", next(self.events)

        def has_tool_call(self, text: str) -> bool:
            return True

        def parse_non_stream(self, text: str) -> tuple[str, list[ToolCallItem]]:
            return "", self.recovered

    @staticmethod
    def call(index: int, name: str | None, arguments: str) -> ToolCallItem:
        return ToolCallItem(tool_index=index, name=name, parameters=arguments)

    def make_post(
        self,
        events: list[list[ToolCallItem]],
        recovered: list[ToolCallItem],
        *,
        confirm: bool = True,
    ) -> SglangStreamingPostProcessor:
        return SglangStreamingPostProcessor(
            tokenizer=self.Tokenizer(),
            tool_call_parser=self.Parser(events, recovered),
            reasoning_parser=None,
            sglang_tools=TOOLS if confirm else None,
        )

    @staticmethod
    def feed(
        post: SglangStreamingPostProcessor, text: str = "x", *, finish: bool = False
    ) -> dict[str, Any] | None:
        return post.process_output(
            {
                "token_ids": [ord(c) for c in text],
                "finish_reason": "stop" if finish else None,
            }
        )

    @pytest.mark.parametrize("confirm", [False, True])
    @pytest.mark.parametrize("all_at_finish", [False, True])
    def test_terminal_recovery_preserves_current_deltas(
        self, confirm: bool, all_at_finish: bool
    ) -> None:
        call = self.call
        final_args = '{"city":"Paris"}'
        tail = final_args if all_at_finish else '"Paris"}'
        events = [] if all_at_finish else [[call(0, "get_weather", '{"city":')]]
        events.append(
            [
                call(0, "get_weather" if all_at_finish else None, tail),
                call(1, "search_gutenberg_books", ""),
            ]
        )
        post = self.make_post(
            events,
            [
                call(0, "get_weather", final_args),
                call(1, "search_gutenberg_books", '{"search_terms":["Joyce"]}'),
            ],
            confirm=confirm,
        )
        choices = [] if all_at_finish else [self.feed(post)]
        choices.append(self.feed(post, finish=True))
        merged = _extract_tool_calls([c for c in choices if c])
        assert len(merged) == 2
        by_name = {c["function"]["name"]: c for c in merged}
        assert by_name["get_weather"]["function"]["arguments"] == final_args
        assert json.loads(
            by_name["search_gutenberg_books"]["function"]["arguments"]
        ) == {"search_terms": ["Joyce"]}
        assert len({c["id"] for c in merged}) == 2
        assert choices[-1]["finish_reason"] == "tool_calls"

    @pytest.mark.parametrize("same_name", [False, True])
    @pytest.mark.parametrize("empty_finish", [False, True])
    def test_completely_missed_call_is_recovered(
        self, same_name: bool, empty_finish: bool
    ) -> None:
        first_args = '{"city":"Paris"}'
        second_name = "get_weather" if same_name else "search_gutenberg_books"
        second_args = '{"city":"Rome"}' if same_name else '{"search_terms":["Joyce"]}'
        post = self.make_post(
            [[self.call(0, "get_weather", first_args)], []],
            [
                self.call(0, "get_weather", first_args),
                # Non-stream indices can be tool-definition indices, including
                # the same index for two calls of the same tool.
                self.call(0 if same_name else 1, second_name, second_args),
            ],
        )
        first = self.feed(post)
        assert first is not None
        assert set(post._tool_call_names.values()) == {"get_weather"}
        assert post._tool_call_args == {0: [first_args]}
        final = self.feed(post, text="" if empty_finish else "x", finish=True)
        assert final is not None
        entries = first["delta"]["tool_calls"] + final["delta"]["tool_calls"]
        merged = _merge_tool_call_entries(entries)
        assert [c["index"] for c in merged] == [0, 1]
        assert [c["function"]["name"] for c in merged] == ["get_weather", second_name]
        assert [c["function"]["arguments"] for c in merged] == [first_args, second_args]
        assert len({c["id"] for c in merged}) == 2
        assert sum(bool(e.get("id")) for e in entries) == 2
        assert all(e["index"] == 1 for e in final["delta"]["tool_calls"])
        assert final["finish_reason"] == "tool_calls"

    def test_plain_text_without_tool_markers_skips_reparse(self) -> None:
        class PlainTextParser(self.Parser):
            def has_tool_call(self, text: str) -> bool:
                return False

            def parse_non_stream(self, text: str) -> tuple[str, list[ToolCallItem]]:
                pytest.fail("Plain text should not require tool-call recovery")

        post = self.make_post([], [])
        post.tool_call_parser = PlainTextParser([[]], [])
        final = self.feed(post, "hello", finish=True)
        assert final is not None
        assert final["finish_reason"] == "stop"
        assert not final["delta"].get("tool_calls")

    def test_confirmed_name_precedes_arguments_and_keeps_identity(self) -> None:
        call = self.call
        post = self.make_post(
            [[call(0, "get_weather", "")], [], [call(0, None, '{"city":"Paris"}')]],
            [],
        )
        first = self.feed(post)
        assert first["finish_reason"] is None
        identity = first["delta"]["tool_calls"][0]
        assert identity["function"] == {"name": "get_weather", "arguments": ""}
        assert self.feed(post) is None
        final = self.feed(post, finish=True)
        entries = first["delta"]["tool_calls"] + final["delta"]["tool_calls"]
        assert sum(bool(e.get("id")) for e in entries) == 1
        merged = _merge_tool_call_entries(entries)
        assert merged[0]["id"] == identity["id"]
        assert merged[0]["function"]["arguments"] == '{"city":"Paris"}'

    def test_name_only_call_gets_recovered_arguments_on_same_index(self) -> None:
        call = self.call
        post = self.make_post(
            [[call(0, "get_weather", "")]],
            [call(0, "get_weather", '{"city":"Paris"}')],
        )
        first = self.feed(post)
        final = self.feed(post, text="", finish=True)
        merged = _extract_tool_calls([first, final])
        assert len(merged) == 1
        assert merged[0]["index"] == 0
        assert merged[0]["function"]["arguments"] == '{"city":"Paris"}'
        assert "id" not in final["delta"]["tool_calls"][0]

    def test_unrecoverable_name_is_not_retracted_or_given_fake_arguments(self) -> None:
        post = self.make_post([[self.call(0, "get_weather", "")]], [])
        first = self.feed(post)
        final = self.feed(post, text="", finish=True)
        assert first["delta"]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert final["finish_reason"] == "tool_calls"
        assert not final["delta"].get("tool_calls")
        assert _extract_tool_calls([first, final])[0]["function"]["arguments"] == ""

    def test_unknown_name_only_is_suppressed(self) -> None:
        post = self.make_post([[self.call(0, "unknown_tool", "")]], [])
        assert self.feed(post) is None
        final = self.feed(post, text="", finish=True)
        assert final["finish_reason"] == "stop"
        assert not final["delta"].get("tool_calls")

    def test_glm47_union_schema_emits_identity_before_tool_closes(self) -> None:
        tools = [
            SglangTool(
                type="function",
                function=SglangFunction(
                    name="lookup",
                    parameters={
                        "oneOf": [
                            {
                                "type": "object",
                                "properties": {"value": {"type": "string"}},
                            },
                            {
                                "type": "object",
                                "properties": {"value": {"type": "integer"}},
                            },
                        ]
                    },
                ),
            )
        ]
        post = SglangStreamingPostProcessor(
            tokenizer=self.Tokenizer(),
            tool_call_parser=FunctionCallParser(tools=tools, tool_call_parser="glm47"),
            reasoning_parser=None,
            sglang_tools=tools,
        )
        chunks = [
            "<tool_call>lookup<arg_key>value</arg_key><arg_value>",
            "long value part 1",
            "long value part 2",
            "</arg_value></tool_call>",
        ]
        first = self.feed(post, chunks[0])
        assert first is not None
        assert first["finish_reason"] is None
        assert first["delta"]["tool_calls"][0]["function"]["name"] == "lookup"
        choices = [first]
        for i, chunk in enumerate(chunks[1:], 1):
            choice = self.feed(post, chunk, finish=i == len(chunks) - 1)
            if choice:
                choices.append(choice)
        merged = _extract_tool_calls(choices)
        assert len(merged) == 1
        assert json.loads(merged[0]["function"]["arguments"]) == {
            "value": "long value part 1long value part 2"
        }
