# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Kimi as_sent history-arguments policy plumbing.

Covers the two frontend halves of the KVV build-6 fix set:

1. ``--kimi-history-arguments-models`` CLI/env routing into the entrypoint
   kwargs (the native ``HistoryArgumentsPolicy::AsSent`` gate), including the
   backend gate and the default-strict behavior.
2. ``SglangProcessor`` usage accounting: the model's ``uncounted_prompt_tokens``
   are deducted from the *reported* usage only (engine dict copied, physical
   prompt and cached-token statistics untouched), identically for the
   engine-reported usage and the local stop fallback, stream and non-stream.
"""

import argparse

import pytest

from dingo.frontend.frontend_args import FrontendArgGroup, FrontendConfig
from dingo.frontend.sglang_processor import (
    SglangPreprocessWorkerResult,
    _deduct_uncounted_prompt_tokens,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def config_for(args):
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    return FrontendConfig.from_cli_args(parser.parse_args(args))


class TestKimiHistoryArgumentsModelsConfig:
    def test_default_is_empty(self, monkeypatch):
        monkeypatch.delenv("DYN_KIMI_HISTORY_ARGUMENTS_MODELS", raising=False)
        config = config_for(["--dyn-chat-processor", "sglang"])
        config.validate()
        assert config.kimi_history_arguments_models == []

    def test_cli_parses_comma_separated(self):
        config = config_for(
            [
                "--dyn-chat-processor",
                "sglang",
                "--kimi-history-arguments-models",
                "kimi-k3.1, kimi-k3",
            ]
        )
        config.validate()
        assert config.kimi_history_arguments_models == ["kimi-k3.1", "kimi-k3"]

    def test_env_overridden_by_cli(self, monkeypatch):
        monkeypatch.setenv("DYN_KIMI_HISTORY_ARGUMENTS_MODELS", "env-kimi")
        assert config_for(
            ["--dyn-chat-processor", "sglang"]
        ).kimi_history_arguments_models == ["env-kimi"]
        config = config_for(
            ["--dyn-chat-processor", "sglang", "--kimi-history-arguments-models", "kimi-k3.1"]
        )
        config.validate()
        assert config.kimi_history_arguments_models == ["kimi-k3.1"]

    def test_empty_list_is_accepted(self):
        config = config_for(
            ["--dyn-chat-processor", "sglang", "--kimi-history-arguments-models", ""]
        )
        config.validate()
        assert config.kimi_history_arguments_models == []

    def test_requires_sglang_processor(self):
        config = config_for(
            ["--dyn-chat-processor", "dynamo", "--kimi-history-arguments-models", "kimi-k3.1"]
        )
        with pytest.raises(ValueError, match="requires --dyn-chat-processor sglang"):
            config.validate()

    def test_glm_gate_still_independent(self):
        config = config_for(
            ["--dyn-chat-processor", "dynamo", "--glm-history-arguments-models", "glm-5.3"]
        )
        with pytest.raises(ValueError, match="glm-history-arguments-models"):
            config.validate()


class TestUncountedPromptTokensDeduction:
    def test_zero_is_identity(self):
        usage = {
            "prompt_tokens": 42,
            "completion_tokens": 8,
            "total_tokens": 50,
        }
        assert _deduct_uncounted_prompt_tokens(usage, 0) is usage

    def test_deducts_from_prompt_and_total_only(self):
        usage = {
            "prompt_tokens": 42,
            "completion_tokens": 8,
            "total_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 10},
        }
        adjusted = _deduct_uncounted_prompt_tokens(usage, 3)
        assert adjusted["prompt_tokens"] == 39
        assert adjusted["total_tokens"] == 47
        assert adjusted["completion_tokens"] == 8
        # Cached statistics are not clamped or altered.
        assert adjusted["prompt_tokens_details"] == {"cached_tokens": 10}
        # The engine dict itself is never mutated.
        assert usage["prompt_tokens"] == 42
        assert usage["total_tokens"] == 50

    def test_clamps_at_zero(self):
        usage = {"prompt_tokens": 2, "completion_tokens": 5, "total_tokens": 7}
        adjusted = _deduct_uncounted_prompt_tokens(usage, 3)
        assert adjusted["prompt_tokens"] == 0
        assert adjusted["total_tokens"] == 5

    def test_non_integer_prompt_is_untouched(self):
        usage = {"prompt_tokens": None, "completion_tokens": 5}
        assert _deduct_uncounted_prompt_tokens(usage, 3) is usage
        usage = {"prompt_tokens": True, "completion_tokens": 5}
        assert _deduct_uncounted_prompt_tokens(usage, 3) is usage

    def test_missing_total_is_not_synthesized(self):
        usage = {"prompt_tokens": 42, "completion_tokens": 8}
        adjusted = _deduct_uncounted_prompt_tokens(usage, 3)
        assert adjusted["prompt_tokens"] == 39
        assert "total_tokens" not in adjusted


class TestWorkerResultCarriesPolicyMetadata:
    def test_defaults_are_neutral(self):
        result = SglangPreprocessWorkerResult(
            prompt_token_ids=[1],
            dynamo_preproc={"model": "m"},
            request={"model": "m"},
        )
        # None means "caller falls back to convert_tools(request.tools)" and
        # 0 means "no deduction"; both preserve legacy behavior.
        assert result.effective_tools is None
        assert result.uncounted_prompt_tokens == 0
