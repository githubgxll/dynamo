# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Separate GLM reasoning using control token IDs, not literal tag text."""

from collections.abc import Callable
from typing import Any


class GlmTokenReasoningSplitter:
    """Keep ordinary-token spellings of think tags in their original field.

    Once a real closing token or tool opener ends reasoning, tool arguments
    pass through unchanged, including any special tokens inside those values.
    """

    def __init__(self, markers: dict[int, str], *, force_reasoning: bool):
        self.markers = markers
        self.in_reasoning = force_reasoning
        self.ended = False

    @classmethod
    def from_tokenizer(cls, tokenizer: Any, *, force_reasoning: bool):
        convert = getattr(tokenizer, "convert_tokens_to_ids", None)
        if not callable(convert):
            return None
        markers = {}
        for marker in ("<think>", "</think>", "<tool_call>"):
            token_id = convert(marker)
            if (
                type(token_id) is not int
                or token_id == getattr(tokenizer, "unk_token_id", None)
                or tokenizer.decode([token_id], skip_special_tokens=False) != marker
            ):
                return None
            markers[token_id] = marker
        if len(markers) != 3:
            return None
        return cls(markers, force_reasoning=force_reasoning)

    def split(
        self,
        token_ids: list[int],
        decode: Callable[..., str],
        *,
        finished: bool = False,
    ) -> tuple[str | None, str, int]:
        reasoning, content = [], []
        reasoning_tokens = 0
        start = 0

        def emit(ids: list[int], flush: bool):
            nonlocal reasoning_tokens
            text = decode(ids, flush=flush)
            (reasoning if self.in_reasoning else content).append(text)
            if self.in_reasoning:
                reasoning_tokens += len(ids)

        for index, token_id in enumerate(token_ids):
            marker = self.markers.get(token_id)
            if marker is None or self.ended:
                continue
            emit(token_ids[start:index], True)
            marker_text = decode([token_id], flush=True)
            if marker == "<think>":
                self.in_reasoning = True
            elif marker == "</think>":
                self.in_reasoning = False
                self.ended = True
            else:
                self.in_reasoning = False
                self.ended = True
                content.append(marker_text)
            start = index + 1
        emit(token_ids[start:], finished)
        return "".join(reasoning) or None, "".join(content), reasoning_tokens
