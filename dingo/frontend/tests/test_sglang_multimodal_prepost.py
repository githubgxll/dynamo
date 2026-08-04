# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multimodal content normalization on the sglang chat-processor path.

Regression coverage for the bug where ``apply_chat_template`` ran on raw
OpenAI messages: VLM templates branch on ``type == 'image'`` while OpenAI
sends ``image_url``, so placeholders never rendered.
"""

from typing import Any

import pytest

from dingo.frontend.sglang_prepost import (
    _normalize_messages_for_template,
    preprocess_chat_request,
)

# Needs sglang packages (gpu_1 container) but allocates no GPU VRAM.
pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
    pytest.mark.profiled_vram_gib(0),
]

# A list-iterating template; this is what makes sglang's detector report the
# "openai" content format that drives normalization.
_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message.content is iterable and message.content is not string %}"
    "{% for chunk in message.content %}"
    "{% if chunk.type == 'image' %}<IMG>"
    "{% elif chunk.type == 'video' %}<VID>"
    "{% elif chunk.type == 'audio' %}<AUD>"
    "{% elif chunk.type == 'text' %}{{ chunk.text }}"
    "{% endif %}{% endfor %}{% endif %}{% endfor %}"
)


def _url_chunk(kind: str) -> dict:
    return {"type": f"{kind}_url", f"{kind}_url": {"url": "x://m"}}


@pytest.fixture
def tokenizer():
    class Tokenizer:
        chat_template = _TEMPLATE

    return Tokenizer()


@pytest.mark.parametrize("kind", ["image", "video", "audio"])
def test_url_chunk_normalized_to_bare_type(tokenizer, kind):
    messages = [{"role": "user", "content": [_url_chunk(kind)]}]
    out = _normalize_messages_for_template(messages, tokenizer)
    assert [c["type"] for c in out[0]["content"]] == [kind]


def test_text_only_passes_through(tokenizer):
    messages = [{"role": "user", "content": "hello"}]
    assert _normalize_messages_for_template(messages, tokenizer) == messages


def test_preprocess_feeds_normalized_chunks_to_template():
    class Tokenizer:
        chat_template = _TEMPLATE

        def apply_chat_template(self, messages, **_):
            self.seen = messages[0]["content"]
            return [42 if c["type"] == "image" else 1 for c in self.seen]

        def encode(self, prompt):
            raise AssertionError("text path must not be taken")

    tok = Tokenizer()
    request = {
        "model": "fake-vlm",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    _url_chunk("image"),
                ],
            }
        ],
    }

    result = preprocess_chat_request(
        request,
        tokenizer=tok,
        tool_call_parser_name=None,
        reasoning_parser_name=None,
    )

    assert 42 in result.prompt_token_ids
    assert [c["type"] for c in tok.seen] == ["text", "image"]


def test_kimi_k3_preserves_image_url_and_injects_media_pad_prompt():
    class KimiK3Tokenizer:
        chat_template = None
        name_or_path = "/models/Kimi-K3"

        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: Any
        ) -> list[int]:
            self.seen = messages[0]["content"]
            self.kwargs = kwargs
            image_prompts = kwargs.get("image_prompts") or []
            image_parts = [
                c
                for c in self.seen
                if isinstance(c, dict) and c.get("type") == "image_url"
            ]
            if len(image_prompts) != len(image_parts):
                raise ValueError(
                    f"image prompt count {len(image_prompts)} != "
                    f"consumed placeholder count {len(image_parts)}"
                )
            return [1, *([163605] * len(image_prompts)), 2]

    tok = KimiK3Tokenizer()
    request = {
        "model": "kimi-k3",
        "chat_template_kwargs": {"image_prompts": ["bad", "extra"]},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    _url_chunk("image"),
                ],
            }
        ],
    }

    result = preprocess_chat_request(
        request,
        tokenizer=tok,
        tool_call_parser_name=None,
        reasoning_parser_name=None,
    )

    assert result.prompt_token_ids.count(163605) == 1
    assert tok.kwargs["image_prompts"] == ["<|media_pad|>"]
    assert [c["type"] for c in tok.seen] == ["text", "image_url"]


def test_kimi_k3_rejects_literal_image_placeholder_in_user_text():
    class KimiK3Tokenizer:
        chat_template = None
        name_or_path = "/models/Kimi-K3"

        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: Any
        ) -> list[int]:
            raise AssertionError("reserved placeholder should fail before rendering")

    request = {
        "model": "kimi-k3",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "literal <|kimi_image_placeholder|> should fail",
                    }
                ],
            }
        ],
    }

    with pytest.raises(ValueError, match="reserved for Kimi-K3 image input"):
        preprocess_chat_request(
            request,
            tokenizer=KimiK3Tokenizer(),
            tool_call_parser_name=None,
            reasoning_parser_name=None,
        )
