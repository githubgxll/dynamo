"""Backport vLLM-Omni PR #5979 for MiniMax-H3 video-VAE compilation.

The pinned image already regionally compiles the H3 DiT through the generic
model-runner path.  PR #5979 gives H3 a model-specific ``setup_compile`` that
keeps that behavior and additionally compiles the repeated
``TransformerBlock`` modules in the video-VAE decoder.
"""

from __future__ import annotations

import os

_regionally_compile = None


def install() -> None:
    """Install the H3-specific compile setup before the pipeline is created."""
    global _regionally_compile

    from vllm_omni.diffusion.compile import regionally_compile
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3

    _regionally_compile = regionally_compile

    pipeline_class = pipeline_minimax_h3.MiniMaxH3Pipeline
    if "setup_compile" in pipeline_class.__dict__:
        raise RuntimeError(
            "MiniMaxH3Pipeline already defines setup_compile; "
            "the PR #5979 backport no longer matches the installed runtime"
        )

    def setup_compile(self) -> None:
        dynamic = self.od_config.diffusion_compile_dynamic
        granularity = self.od_config.diffusion_compile_granularity
        for attr_name in self._dit_modules:
            model = getattr(self, attr_name, None)
            if model is None:
                continue
            if granularity == "full":
                model.compile(dynamic=dynamic)
            else:
                _regionally_compile(model, dynamic=dynamic)

        decoder = self.video_vae.model.decoder
        decoder._repeated_blocks = ["TransformerBlock"]
        _regionally_compile(decoder, dynamic=dynamic)
        pipeline_minimax_h3.logger.info(
            "MiniMax H3 regional torch.compile enabled for video VAE decoder "
            "TransformerBlock modules (dynamic=%s)",
            dynamic,
        )

    setup_compile.__name__ = "setup_compile"
    setup_compile.__qualname__ = "MiniMaxH3Pipeline.setup_compile"
    setup_compile.__module__ = pipeline_minimax_h3.__name__
    pipeline_class.setup_compile = setup_compile
    print(
        "[compat-child] enabled MiniMax-H3 video-VAE regional compile "
        f"for pid={os.getpid()}",
        flush=True,
    )
