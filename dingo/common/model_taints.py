# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Startup configuration and worker-local updates for model routing taints."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dynamo.llm import update_model_taints
from dynamo.runtime import DistributedRuntime, Endpoint

MODEL_TAINT_ROUTE = "update/model_taints"
TOPOLOGY_TAINT_PREFIX = "dynamo.topology/"


def _validate_taints(taints: Any) -> list[str]:
    if not isinstance(taints, list) or not all(
        isinstance(taint, str) for taint in taints
    ):
        raise ValueError("'taints' must be a JSON array of strings")
    if reserved := next(
        (taint for taint in taints if taint.startswith(TOPOLOGY_TAINT_PREFIX)),
        None,
    ):
        raise ValueError(
            f"taint '{reserved}' uses reserved prefix '{TOPOLOGY_TAINT_PREFIX}'"
        )
    return sorted(set(taints))


def resolve_startup_model_taints(value: str | None) -> list[str] | None:
    """Resolve a JSON taint list or exact POD_NAME-to-list map, optionally @file.

    Read once during argument validation, before engine creation. A configured
    map must contain this Pod; never silently register an unassigned worker.
    Runtime updates do not modify this external source of restart defaults.
    """
    if value is None:
        return None
    try:
        raw = (
            Path(value[1:]).read_text(encoding="utf-8")
            if value.startswith("@")
            else value
        )
        config = json.loads(raw)
        if isinstance(config, dict):
            for pod_name, taints in config.items():
                if not pod_name.strip():
                    raise ValueError("Pod names must be non-empty")
                _validate_taints(taints)
            pod_name = os.environ.get("POD_NAME")
            if not pod_name:
                raise ValueError("POD_NAME is required for a per-Pod taint map")
            if pod_name not in config:
                raise ValueError(f"no taint mapping for POD_NAME={pod_name!r}")
            config = config[pod_name]
        return _validate_taints(config)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid --model-taints / DYN_MODEL_TAINTS: {exc}") from exc


def register_model_taint_route(runtime: DistributedRuntime, endpoint: Endpoint) -> None:
    """Register POST /engine/update/model_taints on the system status server."""

    async def _update_model_taints(body: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise ValueError("request body must be a JSON object")

        unique_taints = set(_validate_taints(body.get("taints")))
        await update_model_taints(endpoint, unique_taints)
        return {
            "status": "ok",
            "taints": sorted(unique_taints),
        }

    runtime.register_engine_route(MODEL_TAINT_ROUTE, _update_model_taints)
