# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from vllm.v1.engine.exceptions import EngineDeadError

from dingo.common.engine_monitor import EngineHealthMonitorConfig
from dingo.vllm.omni.engine_monitor import OmniEngineMonitor

pytestmark = [pytest.mark.unit, pytest.mark.vllm, pytest.mark.pre_merge]


def _monitor(engine):
    monitor = object.__new__(OmniEngineMonitor)
    monitor.runtime = MagicMock()
    monitor.engine_client = engine
    monitor.shutdown_event = None
    monitor.health_config = EngineHealthMonitorConfig(
        interval=0.01,
        check_timeout=0.01,
        shutdown_timeout=0.01,
    )
    monitor._monitor_task = asyncio.get_event_loop().create_future()
    return monitor


@pytest.mark.asyncio
async def test_health_check_timeout_is_fatal():
    async def blocked_health_check():
        await asyncio.sleep(1)

    engine = AsyncMock()
    engine.check_health = blocked_health_check
    monitor = _monitor(engine)

    with pytest.raises(asyncio.TimeoutError):
        await monitor._run_health_check()


@pytest.mark.asyncio
async def test_engine_death_shuts_down_runtime_and_worker():
    engine = AsyncMock()
    engine.check_health.side_effect = EngineDeadError("engine is dead")
    monitor = _monitor(engine)

    with (
        patch.object(monitor, "_shutdown_engine") as shutdown_engine,
        patch(
            "dingo.vllm.omni.engine_monitor.os._exit",
            side_effect=SystemExit(1),
        ) as exit_process,
        pytest.raises(SystemExit, match="1"),
    ):
        await monitor._check_engine_health()

    shutdown_engine.assert_called_once_with()
    monitor.runtime.shutdown.assert_called_once_with()
    exit_process.assert_called_once_with(1)
