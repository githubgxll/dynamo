# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Builtin Dynamo Planner factory for the Arena's common engine seam."""

from __future__ import annotations

from dingo.planner.config.planner_config import PlannerConfig
from dingo.planner.core.engine_protocol import EngineProtocol
from dingo.planner.core.types import WorkerCapabilities
from dingo.planner.plugins.clock import VirtualClock
from dingo.planner.plugins.orchestrator.engine_adapter import OrchestratorEngineAdapter


def planner_engine_factory(
    config: PlannerConfig, capabilities: WorkerCapabilities
) -> EngineProtocol:
    """Build the builtin Planner engine on the replay's virtual clock."""

    return OrchestratorEngineAdapter(
        config,
        capabilities,
        clock=VirtualClock(),
    )


__all__ = ["planner_engine_factory"]
