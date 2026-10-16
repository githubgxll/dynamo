# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry point for the sample diffusion backend (CPU-only).

Usage:
    python -m dingo.common.backend.sample_diffusion_main --model-name sample-diffusion-model
"""

from dingo.common.backend.run import run
from dingo.common.backend.sample_diffusion_engine import SampleDiffusionEngine


def main():
    run(SampleDiffusionEngine)


if __name__ == "__main__":
    main()
