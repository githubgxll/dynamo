#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Unit tests for Kimi-K3 template-token provenance.

Pure Python: the segment grammar is exercised with hand-built segments that
mirror the checkpoint renderer's exact output shapes (see
encoding_k3.build_chat_segments), and the fused-body byte mapping runs
against a stub tokenizer backend whose token bytes reproduce the text.
No tiktoken, no sglang, no checkpoint files.
"""

import sys
from dataclasses import dataclass

import pytest

from dingo.frontend.kimi_k3_provenance import (
    ProvenanceUnavailable,
    _FusedTemplate,
    _fused_template_spans,
    classify_kimi_k3_segments,
    render_kimi_k3_prompt_with_provenance,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@dataclass(frozen=True)
class Seg:
    """Mirror of the checkpoint EncodeSegment for test construction."""

    text: str
    allow_special: bool = False


OPEN = Seg("<|open|>", True)
CLOSE = Seg("<|close|>", True)
SEP = Seg("<|sep|>", True)
EOM = Seg("<|end_of_msg|>", True)


def attrs(*pairs):
    """Attribute segments exactly as the renderer's _attr emits them."""
    out = []
    for key, value in pairs:
        out.append(Seg(f" {key}"))
        out.append(Seg('="'))
        if value != "":
            out.append(Seg(str(value)))
        out.append(Seg('"'))
    return out


def open_tag(tag, *pairs):
    return [OPEN, Seg(tag), *attrs(*pairs), SEP]


def close_tag(tag):
    return [CLOSE, Seg(tag), SEP]


def internal_message(type_, body):
    return (
        open_tag("message", ("role", "system"), ("type", type_))
        + [Seg(body)]
        + close_tag("message")
        + [EOM]
    )


def user_message(content, name=None):
    pairs = [("role", "user")]
    if name is not None:
        pairs.append(("name", name))
    return open_tag("message", *pairs) + [Seg(content)] + close_tag("message") + [EOM]


GENERATION = open_tag("message", ("role", "assistant")) + [OPEN, Seg("think"), SEP]

THINKING_EFFORT_BODY = (
    "`thinking_effort` guides on how much to think in your thinking "
    "channel (not including the response channel), supported values "
    "include `low`, `medium`, `high`, and `max`.\n"
    "Now the system is invoked with `thinking_effort=max`."
)

TOOL_DECLARE_PREFIX = (
    "# Tools\n"
    "Here are the available tools, described in JSONSchema.\n\n"
    "```json\n"
)
TOOL_DECLARE_SUFFIX = "\n```"


def template_text(segments, classifications=None):
    """Decode the template ranges of a synthetic single-char-per-token stream."""
    if classifications is None:
        classifications = classify_kimi_k3_segments(segments)
    out = []
    for seg, cls in zip(segments, classifications):
        if cls == "template":
            out.append(seg.text)
        elif isinstance(cls, _FusedTemplate):
            data = seg.text.encode("utf-8")
            out.append(data[: len(cls.prefix)].decode("utf-8", "ignore"))
            out.append(data[-len(cls.suffix) :].decode("utf-8", "ignore"))
    return "".join(out)


# --------------------------------------------------------------------------
# classify_kimi_k3_segments
# --------------------------------------------------------------------------


class TestClassification:
    def test_default_thinking_request_marks_only_template_prefix(self):
        segments = internal_message("thinking-effort", THINKING_EFFORT_BODY) + (
            user_message("Copper quartz cargo wheel") + GENERATION
        )
        cls = classify_kimi_k3_segments(segments)
        assert cls[: len(internal_message("thinking-effort", "x"))] == [
            "template"
        ] * len(internal_message("thinking-effort", "x"))
        text = template_text(segments, cls)
        # The whole internal message is template; customer content is not.
        assert "thinking_effort=max" in text
        assert "Copper quartz cargo wheel" not in text

    def test_customer_text_identical_to_internal_body_is_not_template(self):
        segments = (
            internal_message("thinking-effort", THINKING_EFFORT_BODY)
            + open_tag("message", ("role", "system"))
            + [Seg(THINKING_EFFORT_BODY)]
            + close_tag("message")
            + [EOM]
            + user_message("hi")
            + GENERATION
        )
        cls = classify_kimi_k3_segments(segments)
        # Exactly one body occurrence is template: the internal one.
        body_flags = [c for s, c in zip(segments, cls) if s.text == THINKING_EFFORT_BODY]
        assert body_flags == ["template", "customer"]

    def test_unknown_control_token_fails_closed(self):
        with pytest.raises(ProvenanceUnavailable):
            classify_kimi_k3_segments([Seg("<|media_pad|>", True)])

    def test_customer_role_and_name_attributes(self):
        segments = (
            open_tag("message", ("role", "user"), ("name", "alice"))
            + [Seg("hi")]
            + close_tag("message")
            + [EOM]
        )
        cls = classify_kimi_k3_segments(segments)
        flags = [(s.text, c) for s, c in zip(segments, cls)]
        # The role *value* segment is template; the name *value* is customer.
        role_value_index = segments.index(Seg("user"))
        assert flags[role_value_index] == ("user", "template")
        assert ("alice", "customer") in flags
        assert ("hi", "customer") in flags

    def test_unknown_attribute_fails_closed(self):
        with pytest.raises(ProvenanceUnavailable):
            classify_kimi_k3_segments(
                [OPEN, Seg("message"), Seg(" mystery"), Seg('="'), Seg("x"), Seg('"'), SEP]
            )

    def test_unknown_role_fails_closed(self):
        with pytest.raises(ProvenanceUnavailable):
            classify_kimi_k3_segments(
                open_tag("message", ("role", "wizard")) + [Seg("hi")] + close_tag("message")
            )

    def test_unknown_internal_type_fails_closed(self):
        with pytest.raises(ProvenanceUnavailable):
            classify_kimi_k3_segments(open_tag("message", ("role", "system"), ("type", "warp")))

    def test_mismatched_close_tag_fails_closed(self):
        with pytest.raises(ProvenanceUnavailable):
            classify_kimi_k3_segments(
                open_tag("message", ("role", "user")) + [CLOSE, Seg("tools"), SEP]
            )

    def test_body_outside_any_tag_fails_closed(self):
        with pytest.raises(ProvenanceUnavailable):
            classify_kimi_k3_segments([Seg("stray text")])

    def test_second_body_in_internal_message_fails_closed(self):
        with pytest.raises(ProvenanceUnavailable):
            classify_kimi_k3_segments(
                internal_message("thinking-effort", THINKING_EFFORT_BODY)
                + [Seg("extra body")]
                + close_tag("message")
            )

    def test_unknown_thinking_effort_body_fails_closed(self):
        with pytest.raises(ProvenanceUnavailable):
            classify_kimi_k3_segments(
                internal_message("thinking-effort", "modified checkpoint text")
            )


    def test_tool_message_grammar(self):
        segments = (
            open_tag("message", ("role", "tool"), ("tool", "get_weather"), ("index", 1))
            + [Seg("sunny 22C")]
            + close_tag("message")
            + [EOM]
        )
        cls = classify_kimi_k3_segments(segments)
        flags = [(s.text, c) for s, c in zip(segments, cls)]
        assert ("tool", "template") in flags  # role value
        assert ("get_weather", "customer") in flags
        assert ("1", "template") in flags  # index value
        assert ("sunny 22C", "customer") in flags

    def test_assistant_tool_call_grammar(self):
        segments = (
            open_tag("message", ("role", "assistant"))
            + open_tag("think")
            + close_tag("think")
            + open_tag("response")
            + close_tag("response")
            + open_tag("tools")
            + open_tag("call", ("tool", "get_weather"), ("index", 1))
            + open_tag("argument", ("key", "city"), ("type", "string"))
            + [Seg("Paris")]
            + close_tag("argument")
            + close_tag("call")
            + close_tag("tools")
            + close_tag("message")
            + [EOM]
        )
        cls = classify_kimi_k3_segments(segments)
        flags = [(s.text, c) for s, c in zip(segments, cls)]
        assert ("string", "customer") in flags  # argument type derived from value
        assert ("Paris", "customer") in flags
        assert ("city", "customer") in flags
        assert ("get_weather", "customer") in flags
        assert all(c == "template" for t, c in flags if t in ("<|open|>", "<|close|>", "<|sep|>"))

    def test_fused_tool_declare_body(self):
        body = TOOL_DECLARE_PREFIX + '{"x":1}' + TOOL_DECLARE_SUFFIX
        cls = classify_kimi_k3_segments(internal_message("tool-declare", body))
        fused = [c for c in cls if isinstance(c, _FusedTemplate)]
        assert len(fused) == 1
        assert fused[0].prefix == TOOL_DECLARE_PREFIX.encode("utf-8")
        assert fused[0].suffix == TOOL_DECLARE_SUFFIX.encode("utf-8")

    def test_fused_dynamic_tool_declare_body(self):
        body = (
            "## New Tools Available\n"
            "The system dynamically extends the toolset via lazy-loading.\n"
            "You have access to all existing and extended tools.\n"
            "Here are the specs for the extended tools.\n\n"
            "```json\n"
        ) + '{"y":2}' + "\n```"
        cls = classify_kimi_k3_segments(internal_message("tool-declare", body))
        assert any(isinstance(c, _FusedTemplate) for c in cls)

    def test_malformed_tool_declare_body_fails_closed(self):
        with pytest.raises(ProvenanceUnavailable):
            classify_kimi_k3_segments(
                internal_message("tool-declare", "# Tools\nbut wrong wrapper")
            )

    def test_json_object_announcement_fully_template(self):
        body = (
            "The system is invoked with `response_format=json_object`.\n"
            "Your response must be raw JSON data without markdown code "
            "blocks (```json) or any additional formatting."
        )
        cls = classify_kimi_k3_segments(internal_message("response-format", body))
        assert all(c == "template" for c in cls)


# --------------------------------------------------------------------------
# Fused-body byte mapping
# --------------------------------------------------------------------------


class StubBackend:
    """Token-byte backend: tokens are bytes of the text split at N bytes each."""

    def __init__(self, text, token_size):
        data = text.encode("utf-8")
        self.chunks = [
            data[i : i + token_size] for i in range(0, len(data), token_size)
        ]

    def decode_single_token_bytes(self, token_id):
        return self.chunks[token_id]


def test_fused_template_spans_clean_boundary():
    prefix, suffix = b"AAAA", b"ZZZZ"
    # Token bytes exactly tile the text: AA|AA|cust|ZZ|ZZ.
    chunks = [b"AA", b"AA", b"cust", b"ZZ", b"ZZ"]
    backend = type(
        "B",
        (),
        {"decode_single_token_bytes": staticmethod(lambda i: chunks[i])},
    )()
    text = (prefix + b"cust" + suffix).decode()
    fused = _FusedTemplate(prefix, suffix)
    spans = _fused_template_spans(
        fused, text, list(range(len(chunks))), backend.decode_single_token_bytes
    )
    assert spans == ((0, 2), (3, 5))


def test_fused_template_spanning_boundary_token_is_customer():
    prefix, suffix = b"AAAA", b"ZZZZ"
    # Chunks tile "AAAA"+"cust"+"ZZZZ" exactly; token b"Ac" spans the
    # server/customer byte boundary (last prefix byte + first payload byte).
    chunks = [b"AA", b"A", b"Ac", b"ust", b"ZZ", b"ZZ"]
    assert b"".join(chunks) == prefix + b"cust" + suffix
    backend = type(
        "B",
        (),
        {"decode_single_token_bytes": staticmethod(lambda i: chunks[i])},
    )()
    text = (prefix + b"cust" + suffix).decode()
    fused = _FusedTemplate(prefix, suffix)
    spans = _fused_template_spans(
        fused, text, list(range(len(chunks))), backend.decode_single_token_bytes
    )
    # Prefix tokens 0-1 template; the spanning token 2 is customer;
    # suffix tokens 4-5 template.
    assert spans == ((0, 2), (4, 6))


def test_fused_template_byte_roundtrip_mismatch_fails_closed():
    backend = type("B", (), {"decode_single_token_bytes": staticmethod(lambda i: [b"XX"][i])})()
    text = "AAAAcustZZZZ"
    fused = _FusedTemplate(b"AAAA", b"ZZZZ")
    with pytest.raises(ProvenanceUnavailable):
        _fused_template_spans(fused, text, [0], backend.decode_single_token_bytes)


def test_fused_template_missing_prefix_fails_closed():
    backend = type("B", (), {"decode_single_token_bytes": staticmethod(lambda i: [b"QQQQ", b"cust", b"ZZZZ"][i])})()
    text = "QQQQcustZZZZ"
    fused = _FusedTemplate(b"AAAA", b"ZZZZ")
    with pytest.raises(ProvenanceUnavailable):
        _fused_template_spans(fused, text, [0, 1, 2], backend.decode_single_token_bytes)


# --------------------------------------------------------------------------
# render_kimi_k3_prompt_with_provenance (fail-closed entry points)
# --------------------------------------------------------------------------


class _FakeModule:
    """Module-like holder exposing the checkpoint renderer entry points."""

    def __init__(self, segments):
        self._segments = segments
        self.build_chat_segments = lambda *a, **k: list(self._segments)
        self.is_batched_conversation = (
            lambda conversation: isinstance(conversation, list)
            and bool(conversation)
            and isinstance(conversation[0], list)
        )


def _fresh_cache(monkeypatch):
    import dingo.frontend.kimi_k3_provenance as prov

    monkeypatch.setattr(prov, "_HELPER_CACHE", {})
    monkeypatch.setattr(prov, "_COMPAT_CACHE", {})
    return prov


def _make_tokenizer_with_module(monkeypatch, segments):
    """Tokenizer + checkpoint-module stand-in: one token per prompt byte."""
    prov = _fresh_cache(monkeypatch)
    module = _FakeModule(segments)
    import types

    class Tok:
        def _encode_text_piece(self, text, allow_special_tokens=True):
            return list(range(len(text.encode("utf-8"))))

        def apply_chat_template(self, conversation, **kwargs):
            ids = []
            for s in module.build_chat_segments(conversation):
                ids.extend(range(len(s.text.encode("utf-8"))))
            return ids

    tokmod = types.ModuleType("fake_kimi_checkpoint")
    tokmod.build_chat_segments = module.build_chat_segments
    tokmod.is_batched_conversation = module.is_batched_conversation
    Tok.__module__ = tokmod.__name__
    sys.modules[tokmod.__name__] = tokmod

    class _BytesBackend:
        def __init__(self, segments):
            self.stream = b"".join(s.text.encode() for s in segments)

        def decode_single_token_bytes(self, token_id):
            return self.stream[token_id : token_id + 1]

    tok = Tok()
    tok.model = _BytesBackend(segments)
    return prov, tok


class TestRenderEntryPoint:
    def _default_segments(self):
        return internal_message("thinking-effort", THINKING_EFFORT_BODY) + (
            user_message("hi") + GENERATION
        )

    def test_render_returns_ids_and_ranges(self, monkeypatch):
        prov, tok = _make_tokenizer_with_module(
            monkeypatch, self._default_segments()
        )
        out = prov.render_kimi_k3_prompt_with_provenance(
            tok, [{"role": "user", "content": "hi"}],
            {"add_generation_prompt": True, "tokenize": True},
        )
        assert out is not None
        ids, ranges = out
        # ids are the per-segment concatenation (local 0..len-1 per segment).
        expected = []
        for s in self._default_segments():
            expected.extend(range(len(s.text.encode("utf-8"))))
        assert ids == expected
        assert ranges, "template ranges must be non-empty"
        # ranges are ordered, disjoint, within bounds
        last_end = 0
        for a, b in ranges:
            assert 0 <= a < b <= len(ids)
            assert a >= last_end
            last_end = b

    def test_multimodal_image_prompts_return_none(self, monkeypatch):
        prov, tok = _make_tokenizer_with_module(
            monkeypatch, self._default_segments()
        )
        out = prov.render_kimi_k3_prompt_with_provenance(
            tok, [{"role": "user", "content": "hi"}],
            {"add_generation_prompt": True, "tokenize": True, "image_prompts": ["<|media_pad|>"]},
        )
        assert out is None

    def test_truncation_kwargs_return_none(self, monkeypatch):
        prov, tok = _make_tokenizer_with_module(
            monkeypatch, self._default_segments()
        )
        out = prov.render_kimi_k3_prompt_with_provenance(
            tok, [{"role": "user", "content": "hi"}],
            {"add_generation_prompt": True, "tokenize": True, "truncation": True, "max_length": 5},
        )
        assert out is None

    def test_tokenize_false_returns_none(self, monkeypatch):
        prov, tok = _make_tokenizer_with_module(
            monkeypatch, self._default_segments()
        )
        out = prov.render_kimi_k3_prompt_with_provenance(
            tok, [{"role": "user", "content": "hi"}],
            {"add_generation_prompt": True, "tokenize": False},
        )
        assert out is None

    def test_batched_conversation_returns_none(self, monkeypatch):
        prov, tok = _make_tokenizer_with_module(
            monkeypatch, self._default_segments()
        )
        out = prov.render_kimi_k3_prompt_with_provenance(
            tok, [[{"role": "user", "content": "hi"}]],
            {"add_generation_prompt": True, "tokenize": True},
        )
        assert out is None

    def test_unknown_control_token_returns_none(self, monkeypatch):
        segments = [Seg("<|media_pad|>", True)]
        prov, tok = _make_tokenizer_with_module(monkeypatch, segments)
        out = prov.render_kimi_k3_prompt_with_provenance(
            tok, [{"role": "user", "content": "hi"}],
            {"add_generation_prompt": True, "tokenize": True},
        )
        assert out is None

    def test_non_checkpoint_tokenizer_returns_none(self, monkeypatch):
        prov = _fresh_cache(monkeypatch)

        class PlainTokenizer:
            pass

        out = prov.render_kimi_k3_prompt_with_provenance(
            PlainTokenizer(), [{"role": "user", "content": "hi"}],
            {"add_generation_prompt": True, "tokenize": True},
        )
        assert out is None
