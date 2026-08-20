# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in request adapters for model-specific vLLM-Omni contracts."""

from typing import Any, Optional


def create_request_adapter(
    name: Optional[str],
    workflow: Optional[str],
    media_dir: str,
    media_max_bytes: int,
) -> Any | None:
    """Create an adapter without importing model-specific code by default."""
    if name is None:
        return None
    if name == "minimax_h3":
        from dingo.vllm.omni.request_adapters.minimax_h3 import (
            MiniMaxH3RequestAdapter,
        )

        return MiniMaxH3RequestAdapter(
            workflow=workflow,
            media_dir=media_dir,
            media_max_bytes=media_max_bytes,
        )
    raise ValueError(f"Unknown Omni request adapter: {name}")


__all__ = ["create_request_adapter"]
