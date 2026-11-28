#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Kimi-K3/K3.1 template-token provenance.

The K3(.1) checkpoint renders chat prompts with its own Python encoder:
``encoding_k3.build_chat_segments`` turns the conversation into a flat list
of ``EncodeSegment`` objects -- XTML structural markers carried as special
tokens plus ordinary BPE text for customer content -- and
``TikTokenTokenizer._encode_chat_segments`` tokenizes the segments one at a
time, concatenating the results into ``prompt_token_ids``.

Because tokenization is segment-local, the physical token-ID coordinates of
every template-owned segment are known exactly *while* the prompt is being
tokenized. This module performs that same single pass and emits **template
token ranges**: half-open ``[start, end)`` ranges into ``prompt_token_ids``
covering exactly the tokens contributed by the server-side template. There
is no second tokenization, no text re-encoding, and no byte-offset search
over customer text.

Template-owned tokens
---------------------
- All XTML structural markers: ``<|open|>``/``<|close|>``/``<|sep|>``,
  ``<|end_of_msg|>``, tag names, and attribute scaffolding (`` role``,
  ``="``, ``"``).
- Structural attribute values: ``role`` (one of the four renderer roles) and
  ``index`` (renderer-assigned 1-based positions), and ``type`` where the
  renderer writes a constant (the internal ``message`` types, and the
  always-``object`` type of ``json`` tags). ``name``/``tool``/``key`` values
  and ``argument``-tag ``type`` values come from customer data and stay
  customer-owned.
- The full bodies of the internal system messages whose text the server
  owns end to end: ``thinking-effort`` (all four effort levels, matched as
  exact strings) and the ``tool-choice`` / ``response_format=json_object``
  announcements.
- The static wrapper of the two *fused* bodies -- the ``tool-declare``
  announcements (global and dynamic) and the
  ``response_format=json_schema`` announcement -- where the checkpoint
  concatenates server text with a customer JSON payload into one segment:
  an exact server-owned byte prefix and byte suffix (the checkpoint's own
  strings, ``"\\n```"`` closers). The bytes are mapped onto the *actual*
  tokenization via the tokenizer backend's token bytes
  (``decode_single_token_bytes``): a token lies fully inside the prefix or
  suffix bytes iff it is template-owned; a token spanning the boundary
  between server and customer bytes is customer-owned. The text is never
  split or re-tokenized.
- The generation prompt that opens the assistant turn.

Identical customer text is never confused with the template: internal
bodies are recognized by structural position (a ``role="system"`` message
that carries a ``type`` attribute -- customer system messages never do)
plus exact-string matching, not by global text search.

Unavailability contract
-----------------------
Any situation where attribution cannot be guaranteed makes the whole
measurement unavailable (``render_kimi_k3_prompt_with_provenance``
returns ``None``; callers fall back to the stock
``tokenizer.apply_chat_template`` and report no ranges, which consumers
must treat as "missing", never zero):

- unknown segment grammar: control tokens other than the four XTML
  markers (e.g. media prompts standing in for customer images), unknown
  tag names or attribute keys, unexpected attribute values, more than one
  body inside an internal message, or text where the renderer never
  emits it;
- byte mismatch: a fused body whose decoded token bytes do not reproduce
  its text, or whose exact server prefix/suffix is absent;
- multimodal input (``image_prompts`` non-empty, or any media placeholder
  segment): image tokens expand inside the engine, so physical ID
  coordinates computed here would not describe the served sequence;
- a checkpoint revision whose strings drift from the ones embedded here;
- call shapes this module does not mirror (batched conversations,
  ``tokenize=False``, or the ``padding``/``truncation``/``max_length``/
  ``return_tensors``/``return_dict`` options Dingo never passes);
- a failed compatibility probe: before provenance is ever reported, a
  one-time per-tokenizer-class probe verifies byte-for-byte ID equality
  with ``tokenizer.apply_chat_template`` on representative conversations.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ProvenanceUnavailable",
    "classify_kimi_k3_segments",
    "render_kimi_k3_prompt_with_provenance",
]

logger = logging.getLogger(__name__)


class ProvenanceUnavailable(Exception):
    """Template attribution cannot be guaranteed for this request."""


# --------------------------------------------------------------------------
# Checkpoint renderer grammar (encoding_k3.py)
# --------------------------------------------------------------------------

_OPEN = "<|open|>"
_CLOSE = "<|close|>"
_SEP = "<|sep|>"
_END_OF_MSG = "<|end_of_msg|>"

_CONTROL_TOKENS = frozenset({_OPEN, _CLOSE, _SEP, _END_OF_MSG})

_ROLES = frozenset({"user", "system", "tool", "assistant"})

_TAG_ATTR_KEYS: dict[str, frozenset[str]] = {
    "message": frozenset({"role", "type", "name", "tool", "index"}),
    "call": frozenset({"tool", "index"}),
    "argument": frozenset({"key", "type"}),
    "json": frozenset({"type"}),
    "think": frozenset(),
    "response": frozenset(),
    "tools": frozenset(),
}

# ``apply_chat_template`` named parameters that change the return shape or
# truncate the ids; Dingo never passes them, and this module refuses to
# mirror them.
_BAIL_KWARGS = frozenset(
    {"padding", "truncation", "max_length", "return_tensors", "return_dict"}
)

# Internal system messages the renderer injects on its own.
_INTERNAL_MESSAGE_TYPES = frozenset(
    {"tool-declare", "thinking-effort", "tool-choice", "response-format"}
)

# Exact body of the thinking-effort announcement, per effort level. The
# checkpoint validates the level against this set before rendering.
_TE_PREFIX = (
    "`thinking_effort` guides on how much to think in your thinking "
    "channel (not including the response channel), supported values "
    "include `low`, `medium`, `high`, and `max`.\n"
    "Now the system is invoked with `thinking_effort="
)
_EFFORT_LEVELS = ("low", "medium", "high", "max")
_THINKING_EFFORT_BODIES = frozenset(
    _TE_PREFIX + level + "`." for level in _EFFORT_LEVELS
)

_TOOL_CHOICE_BODIES = frozenset(
    (
        "The system is invoked with `tool_choice=required`.\n"
        "You MUST call tools in the next message.",
        "The system is invoked with `tool_choice=none`.\n"
        "You MUST NOT call any tools in the next message.",
    )
)

_JSON_OBJECT_BODY = (
    "The system is invoked with `response_format=json_object`.\n"
    "Your response must be raw JSON data without markdown code "
    "blocks (```json) or any additional formatting."
)

# Fused bodies: an exact server-owned byte prefix and byte suffix around a
# customer JSON payload, straight from the checkpoint's renderer.
_JSON_SCHEMA_PREFIX = (
    "The system is invoked with `response_format=json_schema`.\n"
    "Your response must be raw JSON data without markdown code "
    "blocks (```json) or any additional formatting.\n"
    "The JSON data must match the following schema:\n"
    "```json\n"
)
_TOOL_DECLARE_PREFIX = (
    "# Tools\n"
    "Here are the available tools, described in JSONSchema.\n\n"
    "```json\n"
)
_TOOL_DECLARE_DYNAMIC_PREFIX = (
    "## New Tools Available\n"
    "The system dynamically extends the toolset via lazy-loading.\n"
    "You have access to all existing and extended tools.\n"
    "Here are the specs for the extended tools.\n\n"
    "```json\n"
)
_CODE_SUFFIX = "\n```"

_ATTR_KEY_PREFIX = " "
_ATTR_OPEN = '="'
_ATTR_CLOSE = '"'


# Per-segment attribution outcomes.
_TEMPLATE = "template"
_CUSTOMER = "customer"


@dataclass(frozen=True)
class _FusedTemplate:
    """Server-owned exact byte prefix/suffix around a customer payload."""

    prefix: bytes
    suffix: bytes


def _attr_value_is_template(tag: str, key: str, value: str) -> str:
    """Classify an attribute *value* segment; raise on unknown grammar.

    ``role``/``index`` are renderer constants or renderer-assigned positions;
    ``type`` is a renderer constant on ``message`` (internal types) and
    ``json`` (always ``object``) tags, but customer-derived on ``argument``
    tags. ``name``/``tool``/``key`` are customer data.
    """
    if key == "role":
        if value not in _ROLES:
            raise ProvenanceUnavailable(f"unknown message role {value!r}")
        return _TEMPLATE
    if key == "index":
        if not value.isdigit():
            raise ProvenanceUnavailable(f"non-positional index {value!r}")
        return _TEMPLATE
    if key == "type":
        if tag == "message":
            if value not in _INTERNAL_MESSAGE_TYPES:
                raise ProvenanceUnavailable(
                    f"unknown internal message type {value!r}"
                )
            return _TEMPLATE
        if tag == "json":
            if value != "object":
                raise ProvenanceUnavailable(f"unknown json type {value!r}")
            return _TEMPLATE
        return _CUSTOMER  # argument tag: derived from the customer's value
    return _CUSTOMER  # name / tool / key: customer data


def _classify_internal_body(message_type: str, body: str) -> str | _FusedTemplate:
    """Classify the single body segment of an internal system message."""
    if message_type == "thinking-effort":
        if body in _THINKING_EFFORT_BODIES:
            return _TEMPLATE
        raise ProvenanceUnavailable("unknown thinking-effort body")
    if message_type == "tool-choice":
        if body in _TOOL_CHOICE_BODIES:
            return _TEMPLATE
        raise ProvenanceUnavailable("unknown tool-choice body")
    if message_type == "response-format":
        if body == _JSON_OBJECT_BODY:
            return _TEMPLATE
        if (
            body.startswith(_JSON_SCHEMA_PREFIX)
            and body.endswith(_CODE_SUFFIX)
            and len(body) > len(_JSON_SCHEMA_PREFIX) + len(_CODE_SUFFIX)
        ):
            return _FusedTemplate(
                _JSON_SCHEMA_PREFIX.encode("utf-8"), _CODE_SUFFIX.encode("utf-8")
            )
        raise ProvenanceUnavailable("unknown response-format body")
    if message_type == "tool-declare":
        for prefix in (_TOOL_DECLARE_PREFIX, _TOOL_DECLARE_DYNAMIC_PREFIX):
            if (
                body.startswith(prefix)
                and body.endswith(_CODE_SUFFIX)
                and len(body) > len(prefix) + len(_CODE_SUFFIX)
            ):
                return _FusedTemplate(
                    prefix.encode("utf-8"), _CODE_SUFFIX.encode("utf-8")
                )
        raise ProvenanceUnavailable("unknown tool-declare body")
    raise ProvenanceUnavailable(f"unknown internal message type {message_type!r}")


# --------------------------------------------------------------------------
# Segment classification
# --------------------------------------------------------------------------


def classify_kimi_k3_segments(segments: list[Any]) -> list[str | _FusedTemplate]:
    """Classify each rendered segment as template- or customer-owned.

    Walks the flat segment list produced by the checkpoint's
    ``build_chat_segments`` with a state machine driven only by segment
    order, ``allow_special`` flags, and the tag/attribute grammar of the
    renderer. Raises :class:`ProvenanceUnavailable` on any segment the
    grammar does not explain exactly; never mutates ``segments`` and never
    matches body text globally, so customer text identical to template text
    cannot be misattributed.
    """
    out: list[str | _FusedTemplate] = []
    # Stack of [tag, attrs, internal_body_seen] for currently open tags;
    # body segments are classified against the innermost open tag. The
    # generation prompt intentionally leaves its tags open.
    stack: list[list[Any]] = []
    pending_tag = ""
    pending_attrs: dict[str, str] = {}
    phase = "body"  # "body" | "tag_name" | "closing_name" | "attrs"
    attr_key: str | None = None
    attr_pos = 0  # within one attribute: 0 key, 1 '="', 2 value, 3 '"'
    expected_close = ""

    for seg in segments:
        text = seg.text

        if seg.allow_special:
            if text not in _CONTROL_TOKENS:
                # A media prompt standing in for a customer image, or an
                # unknown control token: attribution is not guaranteed.
                raise ProvenanceUnavailable(f"unknown control segment {text!r}")
            out.append(_TEMPLATE)
            if text == _OPEN:
                if phase != "body":
                    raise ProvenanceUnavailable("open tag inside a tag header")
                phase = "tag_name"
            elif text == _CLOSE:
                if phase != "body" or not stack:
                    raise ProvenanceUnavailable("unbalanced close tag")
                expected_close = stack.pop()[0]
                phase = "closing_name"
            elif text == _SEP:
                if phase == "attrs":
                    stack.append([pending_tag, pending_attrs, False])
                    phase = "body"
                elif phase != "body":
                    raise ProvenanceUnavailable("separator inside a tag header")
            # _END_OF_MSG only ever appears in body phase.
            continue

        if phase == "tag_name":
            if text not in _TAG_ATTR_KEYS:
                raise ProvenanceUnavailable(f"unknown tag {text!r}")
            pending_tag = text
            pending_attrs = {}
            phase = "attrs"
            attr_key = None
            attr_pos = 0
            out.append(_TEMPLATE)
        elif phase == "attrs":
            # The renderer emits one attribute as up to four text segments:
            # " key", '="', value, '"'; an empty value omits the value
            # segment. Values never equal '"' (quotes are escaped).
            if attr_pos == 0:
                if not text.startswith(_ATTR_KEY_PREFIX):
                    raise ProvenanceUnavailable(f"expected attribute key, {text!r}")
                attr_key = text[len(_ATTR_KEY_PREFIX) :]
                if attr_key not in _TAG_ATTR_KEYS[pending_tag]:
                    raise ProvenanceUnavailable(
                        f"unknown attribute {attr_key!r} on tag {pending_tag!r}"
                    )
                out.append(_TEMPLATE)
                attr_pos = 1
            elif attr_pos == 1:
                if text != _ATTR_OPEN:
                    raise ProvenanceUnavailable(f"expected attribute open, {text!r}")
                out.append(_TEMPLATE)
                attr_pos = 2
            elif attr_pos == 2:
                if text == _ATTR_CLOSE:
                    # Empty attribute value: this segment is the closer.
                    out.append(_TEMPLATE)
                    attr_pos = 0
                else:
                    if attr_key is None:  # pragma: no cover - guarded above
                        raise ProvenanceUnavailable("attribute without key")
                    pending_attrs[attr_key] = text
                    out.append(_attr_value_is_template(pending_tag, attr_key, text))
                    attr_pos = 3
            else:  # attr_pos == 3: the closing quote.
                if text != _ATTR_CLOSE:
                    raise ProvenanceUnavailable(f"expected attribute close, {text!r}")
                out.append(_TEMPLATE)
                attr_pos = 0
        elif phase == "closing_name":
            if text != expected_close:
                raise ProvenanceUnavailable(
                    f"mismatched closing tag {text!r} (expected {expected_close!r})"
                )
            out.append(_TEMPLATE)
            phase = "body"
        else:  # body phase
            if not stack:
                raise ProvenanceUnavailable("body text outside any tag")
            tag, attrs, internal_body_seen = stack[-1]
            if tag == "message":
                message_type = attrs.get("type")
                if message_type is None:
                    out.append(_CUSTOMER)  # customer message body
                else:
                    if internal_body_seen:
                        raise ProvenanceUnavailable(
                            "multiple body segments in an internal message"
                        )
                    out.append(_classify_internal_body(message_type, text))
                    stack[-1][2] = True
            elif tag in ("think", "response", "argument", "json"):
                out.append(_CUSTOMER)
            else:
                # tools/call never take direct text bodies.
                raise ProvenanceUnavailable(f"text body inside <{tag}>")

    return out


# --------------------------------------------------------------------------
# Byte-accurate range construction
# --------------------------------------------------------------------------


def _token_bytes_fn(tokenizer: Any):
    """Resolve ``decode_single_token_bytes`` on the tokenizer's backend."""
    backend = getattr(tokenizer, "model", None)
    raw = getattr(backend, "decode_single_token_bytes", None)
    if not callable(raw):
        raise ProvenanceUnavailable(
            "tokenizer backend has no decode_single_token_bytes"
        )

    def token_bytes(token_id: int) -> bytes:
        try:
            return raw(token_id)
        except Exception as exc:  # noqa: BLE001 - any backend drift
            raise ProvenanceUnavailable("token bytes unavailable") from exc

    return token_bytes


def _fused_template_spans(
    fused: _FusedTemplate,
    segment_text: str,
    piece: list[int],
    token_bytes,
) -> tuple[tuple[int, int], ...]:
    """Token-index spans (within one segment) that the template owns.

    Uses the *actual* tokenization of the segment: a token is template iff
    its bytes lie fully inside the exact server prefix or suffix bytes. A
    token spanning the server/customer byte boundary is customer-owned. The
    text is never split or re-tokenized; token bytes come from the backend.
    """
    piece_bytes = [token_bytes(t) for t in piece]
    text_bytes = b"".join(piece_bytes)
    if text_bytes != segment_text.encode("utf-8"):
        raise ProvenanceUnavailable("decoded token bytes do not reproduce the text")
    prefix, suffix = fused.prefix, fused.suffix
    if (
        not text_bytes.startswith(prefix)
        or not text_bytes.endswith(suffix)
        or len(text_bytes) <= len(prefix) + len(suffix)
    ):
        raise ProvenanceUnavailable("fused body does not match the checkpoint text")

    n = len(piece)
    k_prefix = 0
    consumed = 0
    for chunk in piece_bytes:
        if consumed + len(chunk) <= len(prefix):
            consumed += len(chunk)
            k_prefix += 1
        else:
            break
    k_suffix = 0
    consumed = 0
    for chunk in reversed(piece_bytes):
        if consumed + len(chunk) <= len(suffix):
            consumed += len(chunk)
            k_suffix += 1
        else:
            break
    if k_prefix + k_suffix > n:
        raise ProvenanceUnavailable("prefix and suffix template claims overlap")

    spans = []
    if k_prefix:
        spans.append((0, k_prefix))
    if k_suffix:
        spans.append((n - k_suffix, n))
    return tuple(spans)


def _template_token_ranges(
    tokenizer: Any,
    segments: list[Any],
    classifications: list[str | _FusedTemplate],
    pieces: list[list[int]],
) -> tuple[tuple[int, int], ...]:
    """Merge per-segment template token spans into half-open ID ranges."""
    token_bytes = None
    ranges: list[tuple[int, int]] = []
    pos = 0
    for seg, cls, piece in zip(segments, classifications, pieces):
        n = len(piece)
        if n == 0:
            continue
        if cls is _TEMPLATE or cls == _TEMPLATE:
            spans = ((0, n),)
        elif isinstance(cls, _FusedTemplate):
            if token_bytes is None:
                token_bytes = _token_bytes_fn(tokenizer)
            spans = _fused_template_spans(cls, seg.text, piece, token_bytes)
        else:  # customer
            spans = ()
        for a, b in spans:
            start, end = pos + a, pos + b
            if ranges and ranges[-1][1] == start:
                ranges[-1] = (ranges[-1][0], end)
            else:
                ranges.append((start, end))
        pos += n
    return tuple(ranges)


# --------------------------------------------------------------------------
# Rendering with provenance (single tokenization pass)
# --------------------------------------------------------------------------


class _Helpers:
    """Renderer entry points of the checkpoint, resolved once per class."""

    __slots__ = ("build_chat_segments", "is_batched_conversation")

    def __init__(self, build: Any, is_batched: Any) -> None:
        self.build_chat_segments = build
        self.is_batched_conversation = is_batched


_HELPER_CACHE: dict[type, _Helpers | None] = {}
_COMPAT_CACHE: dict[tuple[Any, type], bool] = {}


def _checkpoint_helpers(tokenizer: Any) -> _Helpers | None:
    """Resolve the checkpoint's segment renderer for this tokenizer.

    Returns ``None`` unless the tokenizer class's own module provides
    ``build_chat_segments`` / ``is_batched_conversation`` and the instance
    implements ``_encode_text_piece`` -- i.e. it is a Kimi-K3(.1)
    TikTokenTokenizer. Other tokenizers never take the provenance path.
    """
    key = type(tokenizer)
    if key in _HELPER_CACHE:
        return _HELPER_CACHE[key]
    module = inspect.getmodule(key)
    build = getattr(module, "build_chat_segments", None)
    is_batched = getattr(module, "is_batched_conversation", None)
    encode_piece = getattr(tokenizer, "_encode_text_piece", None)
    if (
        module is None
        or not callable(build)
        or not callable(is_batched)
        or not callable(encode_piece)
    ):
        _HELPER_CACHE[key] = None
        return None
    helpers = _Helpers(build, is_batched)
    _HELPER_CACHE[key] = helpers
    return helpers


def _marshal_template_kwargs(
    template_kwargs: dict[str, Any],
) -> tuple[Any, ...] | None:
    """Replicate ``apply_chat_template``'s argument marshalling.

    Returns ``(tools, add_generation_prompt, thinking, image_prompts,
    remaining_kwargs)`` or ``None`` when the call uses a shape this module
    does not mirror (non-tokenizing call, or the batch/padding/truncation
    options Dingo never passes -- those stay on the stock path).
    """
    kwargs = dict(template_kwargs)
    if not kwargs.pop("tokenize", False):
        return None
    if _BAIL_KWARGS & kwargs.keys():
        return None
    tools = kwargs.pop("tools", None)
    add_generation_prompt = kwargs.pop("add_generation_prompt", True)
    thinking = kwargs.pop("thinking", True)
    image_prompts = kwargs.pop("image_prompts", None)
    # The checkpoint defaults the effort to "max" inside apply_chat_template.
    kwargs.setdefault("thinking_effort", "max")
    return tools, add_generation_prompt, thinking, image_prompts, kwargs


def _render_ids(
    tokenizer: Any,
    helpers: _Helpers,
    messages: list[Any],
    marshalled: tuple[Any, ...],
) -> tuple[list[Any], list[list[int]], list[int]]:
    """Tokenize via the checkpoint's own functions, keeping per-segment pieces.

    This is ``_encode_chat_segments`` with the loop unrolled once so each
    segment's tokens are observed in the same, single pass that produces
    the ids -- no second tokenization.
    """
    tools, add_generation_prompt, thinking, image_prompts, kwargs = marshalled
    segments = helpers.build_chat_segments(
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        thinking=thinking,
        image_prompts=image_prompts,
        **kwargs,
    )
    encode_piece = tokenizer._encode_text_piece
    pieces: list[list[int]] = []
    ids: list[int] = []
    for seg in segments:
        piece = encode_piece(seg.text, allow_special_tokens=seg.allow_special)
        pieces.append(piece)
        ids.extend(piece)
    return segments, pieces, ids


def _as_id_list(value: Any) -> list[int]:
    """Flatten ``apply_chat_template`` output the way prepost does."""
    ids = getattr(value, "input_ids", value)
    if isinstance(ids, dict):
        ids = ids.get("input_ids", value)
    return list(ids)


# Representative probe conversations covering the marshalling surface:
# default thinking, explicit thinking=False, tool declaration, image prompts.
# The image case keeps ids-only: provenance for images is refused later.
_PROBE_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Echo the input.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]
_PROBE_CASES: tuple[tuple[list[dict[str, Any]], dict[str, Any]], ...] = (
    ([{"role": "user", "content": "Hi"}], {}),
    ([{"role": "user", "content": "Hi"}], {"thinking": False}),
    (
        [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Hi"},
        ],
        {"tools": _PROBE_TOOL},
    ),
    (
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hi"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,x"},
                    },
                ],
            }
        ],
        {"image_prompts": ["<|media_pad|>"]},
    ),
)


def _compat_verified(tokenizer: Any, helpers: _Helpers) -> bool:
    """One-time probe: our render must equal ``apply_chat_template``.

    Any mismatch disables provenance for this tokenizer class (ranges
    omitted downstream, which consumers must treat as "missing", never
    zero) and the caller falls back to the stock rendering path, so token
    IDs can never drift.
    """
    key = (helpers.build_chat_segments, type(tokenizer))
    cached = _COMPAT_CACHE.get(key)
    if cached is not None:
        return cached
    verified = True
    for messages, extra in _PROBE_CASES:
        template_kwargs = {"add_generation_prompt": True, "tokenize": True}
        template_kwargs.update(extra)
        try:
            expected = _as_id_list(
                tokenizer.apply_chat_template(list(messages), **template_kwargs)
            )
            marshalled = _marshal_template_kwargs(template_kwargs)
            assert marshalled is not None
            _, _, actual = _render_ids(tokenizer, helpers, list(messages), marshalled)
        except Exception:  # noqa: BLE001 - any drift disables provenance
            verified = False
            break
        if expected != actual:
            verified = False
            break
    if not verified:
        logger.warning(
            "Kimi-K3 template provenance disabled: rendered token ids differ "
            "from tokenizer.apply_chat_template for %s",
            type(tokenizer).__name__,
        )
    _COMPAT_CACHE[key] = verified
    return verified


def render_kimi_k3_prompt_with_provenance(
    tokenizer: Any,
    messages: list[Any],
    template_kwargs: dict[str, Any],
) -> tuple[list[int], tuple[tuple[int, int], ...]] | None:
    """Render + tokenize a K3(.1) chat prompt, returning ids and template ranges.

    Returns ``(prompt_token_ids, template_token_ranges)`` where the ranges
    are half-open ``[start, end)`` coordinates into ``prompt_token_ids``
    covering exactly the template-owned tokens, or ``None`` when this
    tokenizer is not a segment-rendering Kimi K3 checkpoint, the call uses
    an unsupported shape, attribution cannot be guaranteed (multimodal
    input, unknown grammar, byte mismatch, drifted checkpoint strings), or
    the compatibility probe fails. On ``None`` the caller must fall back
    to ``tokenizer.apply_chat_template``; on success the token IDs are
    identical to it.
    """
    helpers = _checkpoint_helpers(tokenizer)
    if helpers is None:
        return None
    if helpers.is_batched_conversation(messages):
        return None
    marshalled = _marshal_template_kwargs(template_kwargs)
    if marshalled is None:
        return None
    tools, add_generation_prompt, thinking, image_prompts, kwargs = marshalled
    if image_prompts:
        # Image tokens expand inside the engine; physical coordinates
        # computed here would not describe the served sequence.
        return None
    if not _compat_verified(tokenizer, helpers):
        return None
    segments, pieces, ids = _render_ids(
        tokenizer,
        helpers,
        messages,
        (tools, add_generation_prompt, thinking, image_prompts, kwargs),
    )
    try:
        classifications = classify_kimi_k3_segments(segments)
        ranges = _template_token_ranges(tokenizer, segments, classifications, pieces)
    except ProvenanceUnavailable as exc:
        logger.debug("Kimi-K3 template provenance unavailable: %s", exc)
        return None
    return ids, ranges
