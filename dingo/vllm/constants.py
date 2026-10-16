# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constants for vLLM backend.

DisaggregationMode is defined in dingo.common.constants and re-exported here
so that existing imports from dingo.vllm.constants continue to work.
"""

from dingo.common.constants import DisaggregationMode, EmbeddingTransferMode

__all__ = ["DisaggregationMode", "EmbeddingTransferMode"]
