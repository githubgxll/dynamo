# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the configuration passed from Dynamo to vLLM-Omni."""

import dataclasses
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

try:
    from vllm_omni.config import DeployConfig, VllmOmniConfig
    from vllm_omni.config.config_factory import StageConfigFactory
    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.model_executor.models.qwen3_tts.pipeline import QWEN3_TTS_PIPELINE

    from dingo.vllm.omni.args import OmniDiffusionKwargs, OmniParallelKwargs
    from dingo.vllm.omni.base_handler import BaseOmniHandler
except ImportError:
    pytest.skip("vLLM omni dependencies not available", allow_module_level=True)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.multimodal,
    pytest.mark.gpu_0,
    pytest.mark.xpu_1,
    pytest.mark.pre_merge,
]

_SKIP_FIELDS = {
    "sequence_parallel_size",
    "enable_expert_parallel",
    "ulysses_mode",
    "mask_sp_padding",
    # 0.29 sequence-parallel transport tuning, independent of request slots.
    # Dingo does not currently expose these controls; retain upstream defaults
    # instead of changing the existing TP/SP topology during slot migration.
    "ulysses_a2a_permute",
    "allgather_degree",
}

_ENGINE_ARG_PARALLEL_FIELDS = {
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "data_parallel_size",
}

try:
    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.engine.arg_utils import OmniEngineArgs

    from dingo.vllm.omni.args import OmniDiffusionKwargs, OmniParallelKwargs
    from dingo.vllm.omni.base_handler import BaseOmniHandler
except ImportError:
    pytest.skip("vLLM omni dependencies not available", allow_module_level=True)


def _diffusion_parallel_fields() -> set:
    return {f.name for f in dataclasses.fields(DiffusionParallelConfig)}


def _make_config(**parallel_overrides):
    cfg = MagicMock()
    cfg.model = "test-model"
    cfg.stage_configs_path = None
    cfg.output_modalities = None
    cfg.engine_args.trust_remote_code = False
    cfg.engine_args.enable_lora = False
    cfg.engine_args.max_cpu_loras = None
    cfg.engine_args.max_loras = None
    cfg.engine_args.tensor_parallel_size = 1
    cfg.engine_args.pipeline_parallel_size = 1
    cfg.engine_args.data_parallel_size = 1
    cfg.diffusion = OmniDiffusionKwargs()
    cfg.parallel = dataclasses.replace(OmniParallelKwargs(), **parallel_overrides)
    return cfg


def _build_kwargs(config, stage_type="diffusion"):
    handler = BaseOmniHandler.__new__(BaseOmniHandler)
    stages = [SimpleNamespace(stage_type=stage_type)] if stage_type else []
    with patch(
        "dingo.vllm.omni.base_handler.resolve_stage_configs",
        return_value=(None, stages),
    ):
        return handler._build_omni_kwargs(config)


def _parallel_config_without(*excluded_fields):
    """Model an older upstream config that lacks newer Dynamo options."""
    fields = [
        (field.name, field.type, dataclasses.field(default=None))
        for field in dataclasses.fields(DiffusionParallelConfig)
        if field.name not in excluded_fields
    ]
    return dataclasses.make_dataclass("LegacyDiffusionParallelConfig", fields)


def test_init_initializes_pause_state():
    config = _make_config()
    with (
        patch.object(BaseOmniHandler, "_build_omni_kwargs", return_value={}),
        patch("dingo.vllm.omni.base_handler.AsyncOmni", return_value=MagicMock()),
    ):
        handler = BaseOmniHandler(None, config, {})

    assert handler._paused is False
    assert not handler._pause_lock.locked()


class TestDiffusionParallelConfigCoverage:
    def test_all_diffusion_parallel_config_fields_covered(self):
        """Every DiffusionParallelConfig field must be in OmniParallelKwargs, engine_args, or _SKIP_FIELDS.

        When vllm-omni adds a new parallelism field to DiffusionParallelConfig, this test fails.
        Fix by adding it to OmniParallelKwargs and OmniArgGroup, or to _SKIP_FIELDS
        """
        parallel_kwarg_fields = {f.name for f in dataclasses.fields(OmniParallelKwargs)}
        stale_skips = _SKIP_FIELDS & parallel_kwarg_fields
        if stale_skips:
            pytest.fail(
                f"Exposed parallel fields still marked as skipped: {sorted(stale_skips)}"
            )
        uncovered = [
            f
            for f in _diffusion_parallel_fields()
            if f not in _SKIP_FIELDS
            and f not in parallel_kwarg_fields
            and f not in _ENGINE_ARG_PARALLEL_FIELDS
        ]
        assert not uncovered, (
            f"DiffusionParallelConfig fields not covered: {uncovered}. "
            f"Add to OmniParallelKwargs and OmniArgGroup, or add to _SKIP_FIELDS with a reason."
        )

    def test_parallel_fields_forwarded_from_separate_configs(self):
        """Construct the real vLLM-Omni config from both argument groups."""
        parallel_overrides = {"text_encoder_tp_size": 2}
        supports_ulysses_a2a_permute = (
            "ulysses_a2a_permute" in _diffusion_parallel_fields()
        )
        if supports_ulysses_a2a_permute:
            parallel_overrides["ulysses_a2a_permute"] = True
        config = _make_config(**parallel_overrides)
        config.engine_args.tensor_parallel_size = 4
        config.engine_args.pipeline_parallel_size = 3
        config.engine_args.data_parallel_size = 5

        parallel_config = _build_kwargs(config)["parallel_config"]

        assert parallel_config.tensor_parallel_size == 4
        assert parallel_config.pipeline_parallel_size == 3
        assert parallel_config.data_parallel_size == 5
        assert parallel_config.text_encoder_tp_size == 2
        if supports_ulysses_a2a_permute:
            assert parallel_config.ulysses_a2a_permute is True

    def test_unsupported_default_parallel_field_is_omitted(self):
        """Older Omni releases accept configs when new options keep their defaults."""
        legacy_config = _parallel_config_without("ulysses_a2a_permute")

        with patch(
            "dingo.vllm.omni.base_handler.DiffusionParallelConfig", legacy_config
        ):
            parallel_config = _build_kwargs(_make_config())["parallel_config"]

        assert parallel_config.text_encoder_tp_size == 1
        assert not hasattr(parallel_config, "ulysses_a2a_permute")

    def test_unsupported_non_default_parallel_field_is_rejected(self):
        """Do not silently ignore options unavailable in the installed Omni."""
        legacy_config = _parallel_config_without("ulysses_a2a_permute")
        config = _make_config(ulysses_a2a_permute=True)

        with patch(
            "dingo.vllm.omni.base_handler.DiffusionParallelConfig", legacy_config
        ):
            with pytest.raises(
                ValueError,
                match=(
                    "Installed vLLM-Omni does not support non-default parallel "
                    "option.*ulysses_a2a_permute"
                ),
            ):
                _build_kwargs(config)

    def test_output_modalities_forwarded_to_async_omni(self):
        config = _make_config()
        config.output_modalities = ["image"]

        kwargs = _build_kwargs(config)

        assert kwargs["output_modalities"] == ["image"]

    @pytest.mark.parametrize("output_modality", ["image", "video"])
    @pytest.mark.parametrize("layerwise_offload", [None, False, True])
    def test_diffusion_kwargs_accepted_by_upstream(
        self, output_modality, layerwise_offload
    ):
        """Exercise Omni's real validator without loading weights or using a GPU."""
        config = _make_config()
        config.output_modalities = [output_modality]
        config.diffusion.enable_layerwise_offload = layerwise_offload

        stages = StageConfigFactory.create_default_diffusion(_build_kwargs(config))

        assert len(stages) == 1
        assert stages[0]["stage_type"] == "diffusion"
        if layerwise_offload is not None:
            assert (
                stages[0]["engine_args"]["enable_layerwise_offload"]
                is layerwise_offload
            )

    def test_tts_kwargs_accepted_by_upstream_pipeline(self):
        config = _make_config()
        config.diffusion.enforce_eager = True

        kwargs = _build_kwargs(config, stage_type="llm")

        VllmOmniConfig.from_pipeline_config(
            QWEN3_TTS_PIPELINE,
            user_deploy_config=DeployConfig(),
            cli_overrides=kwargs,
        )
        assert kwargs["enforce_eager"] is True

    def test_diffusion_kwargs_preserved_when_stage_detection_is_deferred(self):
        config = _make_config()
        config.diffusion.enable_cpu_offload = True
        config.diffusion.vae_use_tiling = True

        kwargs = _build_kwargs(config, stage_type=None)

        assert kwargs["enable_cpu_offload"] is True
        assert kwargs["vae_use_tiling"] is True

    def test_model_defined_diffusion_fields_forwarded_to_async_omni(self):
        config = _make_config()
        config.diffusion = dataclasses.replace(
            OmniDiffusionKwargs(),
            task_type="fl2va",
            lora_path=["/models/fasth3/adapter_model.safetensors"],
            diffusion_attention_backend="FASTVIDEO_VSA",
            fastvideo_vsa_topk=64,
        )

        kwargs = _build_kwargs(config)

        assert kwargs["task_type"] == "fl2va"
        assert kwargs["lora_path"] == ["/models/fasth3/adapter_model.safetensors"]
        assert kwargs["diffusion_attention_backend"] == "FASTVIDEO_VSA"
        assert kwargs["fastvideo_vsa_topk"] == 64

    def test_diffusion_only_defaults_not_forwarded_to_async_omni(self):
        kwargs = _build_kwargs(_make_config())

        for field in (
            "enable_layerwise_offload",
            "layerwise_num_gpu_layers",
            "vae_use_slicing",
            "vae_use_tiling",
            "boundary_ratio",
            "enable_cache_dit_summary",
            "enable_cpu_offload",
        ):
            assert field not in kwargs

    def test_explicit_false_diffusion_option_forwarded_to_async_omni(self):
        config = _make_config()
        config.diffusion = dataclasses.replace(
            OmniDiffusionKwargs(), vae_use_tiling=False
        )

        kwargs = _build_kwargs(config)

        assert kwargs["vae_use_tiling"] is False

    def test_lora_disabled_resolves_no_capacity(self):
        config = _make_config()
        handler = BaseOmniHandler.__new__(BaseOmniHandler)

        assert handler._resolve_lora_capacity(config) is None

    def test_lora_enabled_with_unset_max_loras_resolves_no_capacity_limit(self):
        config = _make_config()
        config.engine_args.enable_lora = True
        handler = BaseOmniHandler.__new__(BaseOmniHandler)

        assert handler._resolve_lora_capacity(config) is None

    def test_lora_enabled_uses_configured_max_cpu_loras(self):
        config = _make_config()
        config.engine_args.enable_lora = True
        config.engine_args.max_cpu_loras = 3
        handler = BaseOmniHandler.__new__(BaseOmniHandler)

        assert handler._resolve_lora_capacity(config) == 3

    def test_lora_enabled_falls_back_to_max_loras_when_max_cpu_loras_unset(self):
        config = _make_config()
        config.engine_args.enable_lora = True
        config.engine_args.max_cpu_loras = None
        config.engine_args.max_loras = 2
        handler = BaseOmniHandler.__new__(BaseOmniHandler)

        assert handler._resolve_lora_capacity(config) == 2

    def test_advertised_gpu_capacity_uses_max_loras_even_when_max_cpu_loras_set(self):
        config = _make_config()
        config.engine_args.enable_lora = True
        config.engine_args.max_cpu_loras = 8
        config.engine_args.max_loras = 2
        handler = BaseOmniHandler.__new__(BaseOmniHandler)

        assert handler._resolve_advertised_gpu_lora_capacity(config) == 2

    def test_advertised_gpu_capacity_none_when_lora_disabled(self):
        config = _make_config()
        config.engine_args.enable_lora = False
        handler = BaseOmniHandler.__new__(BaseOmniHandler)

        assert handler._resolve_advertised_gpu_lora_capacity(config) is None

    def test_tensor_parallel_size_read_from_engine_args(self):
        """tensor_parallel_size must come from engine_args (vLLM's --tensor-parallel-size),
        not from OmniParallelKwargs, so it applies to both LLM encoder and diffusion transformer.
        """
        config = _make_config()
        config.engine_args.tensor_parallel_size = 4
        with patch("dingo.vllm.omni.base_handler.DiffusionParallelConfig") as MockCfg:
            MockCfg.return_value = SimpleNamespace()
            _build_kwargs(config)
            _, kwargs = MockCfg.call_args
            assert kwargs.get("tensor_parallel_size") == 4


def _engine_args_fields() -> set:
    fields: set = set()
    for cls in OmniEngineArgs.__mro__:
        fields |= set(getattr(cls, "__annotations__", {}).keys())
    return fields


def test_step_execution_and_capacity_forwarded():
    config = _make_config()
    config.diffusion = dataclasses.replace(
        config.diffusion, step_execution=True, max_num_seqs=2
    )
    kwargs = _build_kwargs(config)
    assert kwargs["step_execution"] is True
    assert kwargs["max_num_seqs"] == 2


def test_unset_capacity_preserves_upstream_default():
    kwargs = _build_kwargs(_make_config())
    assert "max_num_seqs" not in kwargs


class TestErrorChunk:
    """B3: _error_chunk returns the correct failure schema per request type."""

    def _make_handler(self):
        handler = BaseOmniHandler.__new__(BaseOmniHandler)
        handler.config = MagicMock()
        handler.config.served_model_name = None
        handler.config.model = "test-model"
        return handler

    def test_video_error_returns_nv_videos_response_failed(self):
        from dingo.common.utils.output_modalities import RequestType

        handler = self._make_handler()
        chunk = handler._error_chunk(
            "req-1", "boom", request_type=RequestType.VIDEO_GENERATION
        )
        assert chunk["object"] == "video"
        assert chunk["status"] == "failed"
        assert chunk["model"] == "test-model"
        assert chunk["error"] == "boom"
        assert chunk["data"] == []

    def test_audio_error_returns_nv_audio_speech_response_failed(self):
        from dingo.common.utils.output_modalities import RequestType

        handler = self._make_handler()
        chunk = handler._error_chunk(
            "req-1", "boom", request_type=RequestType.AUDIO_GENERATION
        )
        assert chunk["status"] == "failed"
        assert chunk["error"] == "boom"

    def test_chat_error_returns_chat_completion_chunk(self):
        from dingo.common.utils.output_modalities import RequestType

        handler = self._make_handler()
        chunk = handler._error_chunk(
            "req-1", "boom", request_type=RequestType.CHAT_COMPLETION
        )
        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["choices"][0]["finish_reason"] == "error"
        assert "boom" in chunk["choices"][0]["delta"]["content"]

    def test_unknown_request_type_returns_chat_completion_chunk(self):
        handler = self._make_handler()
        chunk = handler._error_chunk("req-1", "boom", request_type=None)
        assert chunk["object"] == "chat.completion.chunk"
