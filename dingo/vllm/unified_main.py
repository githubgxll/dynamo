# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified entry point for the vLLM backend.

Usage:
    python -m dingo.vllm.unified_main <vllm args>

See dynamo/common/backend/README.md for architecture, response contract,
and feature gap details.
"""

from dingo.common.backend.run import run
from dingo.vllm.llm_engine import VllmLLMEngine


def main():
    run(VllmLLMEngine)


if __name__ == "__main__":
    main()
