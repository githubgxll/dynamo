# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry point for the aggregated vLLM-Omni backend on the unified Worker.

Usage:
    python -m dingo.vllm.unified_omni_main --model <omni-model> \
        --output-modalities {image,video,audio} [vLLM-Omni args...]
"""

from dingo.common.backend.run import run
from dingo.vllm.unified_omni import VllmOmniEngine


def main():
    run(VllmOmniEngine)


if __name__ == "__main__":
    main()
