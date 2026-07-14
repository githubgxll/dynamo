# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry point for the sample backend.

Usage:
    python -m dingo.common.backend.sample_main --model-name test-model
"""

from dingo.common.backend.run import run
from dingo.common.backend.sample_engine import SampleLLMEngine


def main():
    run(SampleLLMEngine)


if __name__ == "__main__":
    main()
