# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit video backend adapter registry."""

from __future__ import annotations

from dingo.video_gateway.adapters.base import VideoBackendAdapter
from dingo.video_gateway.config import PoolConfig


def create_adapter(pool: PoolConfig) -> VideoBackendAdapter:
    """Create only explicitly supported adapters; never scan/import plugins."""

    if pool.adapter.name == "minimax_h3":
        from dingo.video_gateway.adapters.minimax_h3 import MiniMaxH3VideoAdapter

        return MiniMaxH3VideoAdapter(pool)
    raise ValueError(
        f"unknown video adapter {pool.adapter.name!r} for pool {pool.pool_id!r}"
    )


__all__ = ["VideoBackendAdapter", "create_adapter"]
