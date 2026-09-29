# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Health monitor isolated to the vLLM-Omni worker process."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import signal
import traceback

from vllm.v1.engine.exceptions import EngineDeadError
from vllm_omni.entrypoints.async_omni import AsyncOmni

from dingo.common.engine_monitor import EngineHealthMonitorConfig
from dynamo.runtime import DistributedRuntime

logger = logging.getLogger(__name__)


class OmniEngineMonitor:
    """Exit an Omni Worker when AsyncOmni reports a dead stage or rank."""

    def __init__(
        self,
        runtime: DistributedRuntime,
        engine_client: AsyncOmni,
        shutdown_event: asyncio.Event | None = None,
    ) -> None:
        if not isinstance(runtime, DistributedRuntime):
            raise ValueError(
                f"{self.__class__.__name__} requires a DistributedRuntime"
            )
        if not isinstance(engine_client, AsyncOmni):
            raise ValueError(f"{self.__class__.__name__} requires an AsyncOmni")
        self.runtime = runtime
        self.engine_client = engine_client
        self.shutdown_event = shutdown_event
        self.health_config = EngineHealthMonitorConfig.from_env()
        self._monitor_task = asyncio.create_task(self._check_engine_health())
        logger.info("OmniEngineMonitor initialized and health check task started")

    def __del__(self) -> None:
        monitor_task = getattr(self, "_monitor_task", None)
        if monitor_task is not None:
            monitor_task.cancel()

    def _shutdown_engine(self) -> None:
        def timeout_handler(signum, frame):
            del signum, frame
            raise TimeoutError("Omni engine shutdown timed out")

        previous_handler = None
        if self.health_config.shutdown_timeout > 0:
            previous_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(math.ceil(self.health_config.shutdown_timeout))
        try:
            self.engine_client.shutdown()
        except Exception as exc:
            logger.warning("vLLM-Omni engine shutdown failed: %s", exc)
        finally:
            if self.health_config.shutdown_timeout > 0:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous_handler)

    async def _check_engine_health(self) -> None:
        while True:
            try:
                if self.shutdown_event and self.shutdown_event.is_set():
                    logger.info(
                        "OmniEngineMonitor: shutdown event detected; stopping monitor"
                    )
                    return
                await self._run_health_check()
                if self.shutdown_event:
                    try:
                        await asyncio.wait_for(
                            self.shutdown_event.wait(),
                            timeout=self.health_config.interval,
                        )
                        return
                    except asyncio.TimeoutError:
                        pass
                else:
                    await asyncio.sleep(self.health_config.interval)
            except (EngineDeadError, asyncio.TimeoutError) as exc:
                logger.error("Traceback: %s", traceback.format_exc())
                logger.error("vLLM-Omni health check failed: %s", exc)
                logger.warning("Initiating Dynamo Runtime shutdown")
                self._shutdown_engine()
                self.runtime.shutdown()
                os._exit(1)
            except asyncio.CancelledError:
                logger.debug("OmniEngineMonitor health check task cancelled")
                return

    async def _run_health_check(self) -> None:
        health_check = self.engine_client.check_health()
        if self.health_config.check_timeout > 0:
            await asyncio.wait_for(
                health_check,
                timeout=self.health_config.check_timeout,
            )
            return
        await health_check
