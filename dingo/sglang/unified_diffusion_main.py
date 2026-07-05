# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified entry point for the SGLang diffusion backend.

Usage:
    python -m dingo.sglang.unified_diffusion_main --image-diffusion-worker <args>
    python -m dingo.sglang.unified_diffusion_main --video-generation-worker <args>

The token-pipeline / diffusion-LLM entry point is dingo.sglang.unified_main.
"""

from dynamo.common.backend.run import run
from dingo.sglang.unified_diffusion import SglangDiffusionEngine


def main():
    run(SglangDiffusionEngine)


if __name__ == "__main__":
    main()
