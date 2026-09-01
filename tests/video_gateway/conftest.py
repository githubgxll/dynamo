# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from dingo.video_gateway.config import GatewayConfig, parse_config


@pytest.fixture(autouse=True)
def inline_to_thread(monkeypatch):
    """Keep unit tests deterministic; production still offloads blocking media I/O."""

    async def _inline(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _inline)


def gateway_config(
    root: Path,
    *,
    pools: list[dict[str, Any]] | None = None,
    accept_without_workers: bool = False,
    http: dict[str, Any] | None = None,
    media: dict[str, Any] | None = None,
) -> GatewayConfig:
    if pools is None:
        pools = [
            {
                "pool_id": "fl-pool",
                "served_models": ["public-fl"],
                "backend_model": "worker-fl",
                "backend_target": "dyn://scope-a.backend.generate",
                "adapter": {
                    "name": "minimax_h3",
                    "workflow": "fl2va",
                    "flow_shift": 12.0,
                    "audio_flow_shift": 3.0,
                    "validate_media": False,
                },
                "scheduling": {
                    "worker_capacity": 1,
                    "queue_limit": 32,
                    "accept_without_workers": accept_without_workers,
                    "execution_timeout_s": 5,
                    "abort_grace_s": 0.1,
                    "discovery_interval_s": 0.01,
                    "dispatch_interval_s": 0.01,
                },
            }
        ]
    return parse_config(
        {
            "schema_version": 1,
            "deployment_id": "deployment-under-test",
            "runtime": {
                "discovery_backend": "etcd",
                "request_plane": "tcp",
                "event_plane": "zmq",
            },
            "http": http
            or {"host": "127.0.0.1", "port": 18000, "sync_timeout_s": 2},
            "media": media or {},
            "task_store": {"kind": "memory"},
            "artifact_store": {"kind": "filesystem", "root": str(root)},
            "pools": pools,
        }
    )


@pytest.fixture
def make_gateway_config(tmp_path: Path):
    def _make(**kwargs: Any) -> GatewayConfig:
        return gateway_config(tmp_path / "artifacts", **kwargs)

    return _make
