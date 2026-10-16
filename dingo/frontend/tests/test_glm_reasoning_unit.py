# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU regressions for GLM control-token versus literal-text boundaries."""

import pytest

from dingo.frontend.glm_reasoning import GlmTokenReasoningSplitter


class Tokenizer:
    vocab = {
        1: "<think>",
        2: "</think>",
        3: "<tool_call>",
        4: "Use `",
        5: "</",
        6: "think",
        7: ">",
        8: "` literally.",
        9: "answer",
        10: 'submit<arg_key>code</arg_key><arg_value>"',
        11: '"</arg_value></tool_call>',
    }
    unk_token_id = -1

    def convert_tokens_to_ids(self, token):
        return next((k for k, v in self.vocab.items() if v == token), -1)

    def decode(self, ids, **kwargs):
        return "".join(self.vocab[i] for i in ids)


@pytest.mark.parametrize("batch_size", [1, 2, 4, 100])
def test_literal_close_does_not_end_reasoning(batch_size):
    tokenizer = Tokenizer()
    splitter = GlmTokenReasoningSplitter.from_tokenizer(tokenizer, force_reasoning=True)
    ids = [4, 5, 6, 7, 8, 2, 9]
    chunks = [
        splitter.split(
            ids[i : i + batch_size],
            tokenizer.decode,
            finished=i + batch_size >= len(ids),
        )
        for i in range(0, len(ids), batch_size)
    ]
    assert "".join(r or "" for r, _, _ in chunks) == "Use `</think>` literally."
    assert "".join(c for _, c, _ in chunks) == "answer"
    assert sum(n for _, _, n in chunks) == 5


@pytest.mark.parametrize("batch_size", [1, 3, 100])
def test_tool_opener_ends_reasoning_and_preserves_argument_markers(batch_size):
    tokenizer = Tokenizer()
    splitter = GlmTokenReasoningSplitter.from_tokenizer(tokenizer, force_reasoning=True)
    ids = [4, 8, 3, 10, 1, 2, 11]
    chunks = [
        splitter.split(
            ids[i : i + batch_size],
            tokenizer.decode,
            finished=i + batch_size >= len(ids),
        )
        for i in range(0, len(ids), batch_size)
    ]
    assert "".join(r or "" for r, _, _ in chunks) == "Use `` literally."
    assert "".join(c for _, c, _ in chunks) == tokenizer.decode(ids[2:])


def test_no_reasoning_and_unrecognized_tokenizer():
    tokenizer = Tokenizer()
    splitter = GlmTokenReasoningSplitter.from_tokenizer(
        tokenizer, force_reasoning=False
    )
    assert splitter.split([9], tokenizer.decode, finished=True) == (None, "answer", 0)
    assert (
        GlmTokenReasoningSplitter.from_tokenizer(object(), force_reasoning=True) is None
    )
