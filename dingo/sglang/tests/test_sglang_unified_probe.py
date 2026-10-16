# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin the SGLang PREFILL canary probe contract: drain stream, then one
terminal chunk — no bootstrap chunk."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from typing import Any, cast

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_1,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
    pytest.mark.skipif(
        importlib.util.find_spec("sglang") is None,
        reason="sglang not installed in this container",
    ),
]


class _FakeContext:
    @property
    def trace_id(self) -> str:
        return "probe-1"

    def is_stopped(self) -> bool:
        return False


def _build_prefill_engine(stream_factory):
    from dingo.common.constants import DisaggregationMode
    from dingo.sglang.llm_engine import SglangLLMEngine

    server_args = SimpleNamespace(skip_tokenizer_init=True)
    dynamo_args = SimpleNamespace(use_sglang_tokenizer=False)
    engine = SglangLLMEngine(server_args, dynamo_args, DisaggregationMode.PREFILL)
    engine._input_param_manager = SimpleNamespace(
        get_input_param=lambda req, use_tokenizer: req.get("token_ids", [])
    )
    engine._bootstrap_host = "127.0.0.1"
    engine._bootstrap_port = 18000

    async def _async_generate(**_kwargs):
        return stream_factory()

    engine.engine = SimpleNamespace(
        async_generate=_async_generate,
        tokenizer_manager=SimpleNamespace(abort_request=lambda **_: None),
    )
    return engine


async def _drain(gen):
    return [c async for c in gen]


def _probe_request() -> dict[str, Any]:
    return {
        "token_ids": [1],
        "_HEALTH_CHECK": True,
        "stop_conditions": {"max_tokens": 1},
        "sampling_options": {"temperature": 0.0},
    }


def test_unified_sampling_params_preserve_stop_controls_without_tokenizer():
    from dingo.sglang.llm_engine import SglangLLMEngine

    engine = SglangLLMEngine.__new__(SglangLLMEngine)
    engine._use_sglang_tokenizer = False
    engine._skip_tokenizer_init = True

    params = engine._build_sampling_params(
        {
            "sampling_options": {"include_stop_str_in_output": True},
            "stop_conditions": {
                "stop": ["END"],
                "stop_token_ids": [32],
                "stop_token_ids_hidden": [128001],
                "min_tokens": 4,
                "ignore_eos": True,
            },
        }
    )

    assert "stop" not in params
    assert "min_new_tokens" not in params
    assert set(params["stop_token_ids"]) == {32, 128001}
    assert params["ignore_eos"] is True
    assert params["no_stop_trim"] is True


def test_unified_sampling_params_preserve_text_stop_controls():
    from dingo.sglang.llm_engine import SglangLLMEngine

    engine = SglangLLMEngine.__new__(SglangLLMEngine)
    engine._use_sglang_tokenizer = True
    engine._skip_tokenizer_init = False

    params = engine._build_sampling_params(
        {
            "stop": ["END"],
            "min_tokens": 2,
            "ignore_eos": False,
            "include_stop_str_in_output": True,
        }
    )

    assert params["stop"] == ["END"]
    assert params["min_new_tokens"] == 2
    assert params["ignore_eos"] is False
    assert params["no_stop_trim"] is True


async def test_prefill_probe_drains_stream_then_yields_single_terminal():
    consumed: list[dict] = []

    async def stream():
        for item in (
            {"meta_info": {"finish_reason": None}, "output_ids": []},
            {"meta_info": {"finish_reason": {"type": "stop"}}, "output_ids": []},
        ):
            consumed.append(item)
            yield item

    engine = _build_prefill_engine(stream)
    chunks = await _drain(engine.generate(_probe_request(), cast(Any, _FakeContext())))

    assert len(chunks) == 1, f"expected single terminal, got {chunks}"
    assert chunks[0]["finish_reason"] == "stop"
    assert "disaggregated_params" not in chunks[0]
    assert len(consumed) == 2


async def test_prefill_probe_yields_error_terminal_when_stream_raises():
    async def stream():
        yield {"meta_info": {"finish_reason": None}, "output_ids": []}
        raise RuntimeError("nixl transport down")

    engine = _build_prefill_engine(stream)
    chunks = await _drain(engine.generate(_probe_request(), cast(Any, _FakeContext())))

    assert len(chunks) == 1
    assert chunks[0]["finish_reason"].startswith("error:")
    assert "nixl transport down" in chunks[0]["finish_reason"]
