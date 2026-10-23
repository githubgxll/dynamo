# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Abort generation on exhaustion, rather than vLLM's forced reasoning close."""

import asyncio
import logging
import os
from functools import lru_cache

from dynamo.llm.exceptions import HttpError

from .generation_errors import generate_with_validation

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def default_thinking_budget():
    value = os.environ.get("DYN_VLLM_DEFAULT_MAX_THINKING_TOKENS")
    if value is None:
        return None
    if not value.isascii() or not value.isdecimal():
        raise ValueError("DYN_VLLM_DEFAULT_MAX_THINKING_TOKENS must be nonnegative")
    return int(value)


def _make_parser(engine, parser_kwargs):
    from vllm.reasoning import ReasoningParserManager
    from vllm.tokenizers import cached_tokenizer_from_config

    config = engine.vllm_config.reasoning_config
    tokenizer = cached_tokenizer_from_config(
        model_config=engine.vllm_config.model_config
    )
    return ReasoningParserManager.get_reasoning_parser(config.reasoning_parser)(
        tokenizer, **(parser_kwargs or {})
    )


class ThinkingBudget:
    def __init__(self, engine, budget, parser_kwargs=None):
        if type(budget) is not int or budget < 0:
            raise HttpError(400, "thinking_token_budget must be a nonnegative integer")
        config = getattr(engine.vllm_config, "reasoning_config", None)
        start = getattr(config, "reasoning_start_token_ids", None) or []
        end = getattr(config, "natural_reasoning_end_token_ids", None) or []
        if (
            not getattr(config, "enabled", False)
            or not getattr(config, "reasoning_parser", None)
            or len(start) != 1
            or len(end) != 1
        ):
            raise HttpError(
                400,
                "thinking_budget_unavailable: abort-on-budget requires a native "
                "reasoning parser with initialized single-token reasoning markers",
            )
        self.budget = budget
        self.start = start[0]
        self.parser = _make_parser(engine, parser_kwargs)
        self.ends = set(getattr(self.parser, "reasoning_end_token_ids", ()))
        self.ends.add(end[0])
        self.states = {}

    def check(self, result, prompt, delta):
        prompt_ids = getattr(result, "prompt_token_ids", None)
        if prompt_ids is None and isinstance(prompt, dict):
            prompt_ids = prompt.get("prompt_token_ids")
        for output in result.outputs:
            index = output.index
            if index not in self.states:
                if prompt_ids is None:
                    raise HttpError(
                        400,
                        "thinking_budget_unavailable: prompt token IDs are required "
                        "to identify the current reasoning state",
                    )
                # Use the model parser, not a scan of all prior conversation turns.
                active = not self.parser.is_reasoning_end(prompt_ids)
                self.states[index] = [active, 0, 0]
            active, count, offset = self.states[index]
            if active and count >= self.budget:
                self.exceeded(index)
            tokens = output.token_ids or []
            if not delta and len(tokens) < offset:
                raise HttpError(
                    400, "thinking_budget_unavailable: invalid token stream"
                )
            new_tokens = tokens if delta else tokens[offset:]
            for token in new_tokens:
                if token == self.start:
                    active = True
                elif token in self.ends:
                    active = False
                elif active:
                    count += 1
                if active and count >= self.budget:
                    self.exceeded(index)
            self.states[index] = [active, count, offset + len(new_tokens)]

    def exceeded(self, index):
        raise HttpError(
            400,
            "thinking_token_budget_exceeded: "
            f"choice {index} reached its thinking token budget ({self.budget}); "
            "generation was aborted. Try lowering reasoning effort.",
        )


async def generate_with_thinking_budget(
    engine,
    prompt,
    sampling_params,
    request_id,
    *,
    abort_guard=None,
    budget_parser_kwargs=None,
    **kwargs,
):
    budget = getattr(sampling_params, "thinking_token_budget", None)
    if budget is None:
        budget = default_thinking_budget()
    guard = (
        None if budget is None else ThinkingBudget(engine, budget, budget_parser_kwargs)
    )
    if guard is not None:
        # A native forced </think> must not turn budget exhaustion into success.
        sampling_params = sampling_params.clone()
        sampling_params.thinking_token_budget = None
        output_kind = getattr(sampling_params.output_kind, "name", "")
        if output_kind not in ("DELTA", "CUMULATIVE"):
            raise HttpError(
                400, "thinking_token_budget requires incremental engine output"
            )
        delta = output_kind == "DELTA"
    stream = generate_with_validation(
        engine.generate, prompt, sampling_params, request_id, **kwargs
    )
    try:
        async for result in stream:
            if guard is not None and any(output.token_ids for output in result.outputs):
                # Even a first-chunk budget error is now past the NIXL transfer.
                if abort_guard is not None:
                    abort_guard.signal_first_token()
                try:
                    guard.check(result, prompt, delta)
                except HttpError:
                    try:
                        if abort_guard is not None:
                            await abort_guard.abort()
                        else:
                            await asyncio.shield(engine.abort(request_id))
                    except Exception:
                        logger.exception("Failed to abort thinking-budget request")
                        raise
                    raise
            yield result
    finally:
        await stream.aclose()
