# SPDX-License-Identifier: Apache-2.0
"""Opt-in Dingo endpoint backed by already-loaded native H3/SR services."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib.util
import logging
import os
from pathlib import Path
import sys
import time

from dingo.common.video_result_file import BINARY_RESULT_WRITER

logger = logging.getLogger(__name__)
HEALTH_MARKER = "_h3sr_native_health_v1"


def load_bridge_module():
    root = Path("/opt/h3-sr")
    # The source root contains only our bridge/control modules, never a second
    # vllm_omni package or a sitecustomize hook.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    name = "_dingo_h3_sr_production_gateway"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, root / "gateway.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("H3/SR production bridge was not installed")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


class _NoOutputFormatter:
    def close(self):
        pass


class ProductionOmniHandler:
    """Preserves Dingo Context/detached/artifact contracts without AsyncOmni."""
    def __init__(self, runtime, config, default_sampling_params=None,
                 shutdown_event=None, media_output_fs=None, media_output_http_url=None,
                 *, bridge=None):
        self.runtime, self.config, self.shutdown_event = runtime, config, shutdown_event
        self.bridge = bridge or load_bridge_module().GatewayBridge.from_profile(config.h3_sr_profile)
        if config.request_adapter_workflow != self.bridge.workflow:
            raise ValueError("Gateway workflow differs from supervisor-owned profile")
        self.media_fs, self.media_http_url = media_output_fs, media_output_http_url
        self.output_formatter = _NoOutputFormatter()
        self._monitor = None

    async def check_ready(self):
        await self.bridge.check_ready()

    async def publish_registration(self):
        """Prove the supervisor-owned Dingo PID registered its real endpoint."""
        from service.state import atomic_json, process_birth, read_json
        profile_hash = hashlib.sha256(Path(self.config.h3_sr_profile).read_bytes()).hexdigest()
        executor = self.bridge.executor
        state = read_json(executor.root / "state.json")
        if state.get("service_epoch") != executor.epoch or state.get("profile_sha256") != profile_hash:
            raise RuntimeError("Gateway registration profile/epoch differs from supervisor")
        receipt = {"schema": "h3sr-gateway-registration/1", "status": "REGISTERED",
            "service_epoch": executor.epoch, "profile_sha256": profile_hash,
            "pid": os.getpid(), "birth": process_birth()}
        path = executor.root / "gateway-ready.json"
        if path.exists():
            raise RuntimeError("Existing Gateway registration receipt must not be replaced")
        await asyncio.to_thread(atomic_json, path, receipt)
        return receipt

    def start_monitor(self, runtime, shutdown_event):
        async def monitor():
            while not shutdown_event.is_set():
                try:
                    await self.check_ready()
                    await asyncio.wait_for(shutdown_event.wait(), timeout=5)
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    return
                except Exception:
                    logger.exception("Supervisor-owned H3/SR readiness lost; stopping dispatch")
                    shutdown_event.set()
                    runtime.shutdown()
                    return
        self._monitor = asyncio.create_task(monitor())
        return self._monitor

    def health_check_payload(self):
        return {HEALTH_MARKER: True}

    async def generate(self, request, context):
        if request.get(HEALTH_MARKER) is True:
            await self.check_ready()
            yield {"status": "ready", "native_backends_ready": True}
            return
        request_id = context.id()
        if request_id is None:
            raise ValueError("Dingo request identity is required")
        if context.is_stopped():
            raise asyncio.CancelledError
        started = time.time()

        async def write_output(path, durations, timing=None):
            # Runs inside GatewayExecutor's permit, including durable artifact
            # completion. Cancellation therefore trips the same blocked latch.
            if not 0 < path.stat().st_size <= 128 * 1024 * 1024:
                raise ValueError("Completed video exceeds configured result limit")
            data = await asyncio.to_thread(path.read_bytes)
            writer = BINARY_RESULT_WRITER.get()
            if writer is not None:
                entry = {"output_format": "mp4", "artifact": await writer.write(data)}
            elif request.get("response_format") == "b64_json":
                entry = {"output_format": "mp4", "b64_json": base64.b64encode(data).decode("ascii")}
            else:
                from dingo.common.storage import upload_to_fs
                if self.media_fs is None:
                    raise ValueError("URL video responses require configured media storage")
                url = await upload_to_fs(self.media_fs, f"videos/{request_id}.mp4", data, self.media_http_url)
                entry = {"output_format": "mp4", "url": url}
            response = {"id": request_id, "object": "video",
                "model": self.config.served_model_name or self.config.model,
                "status": "completed", "progress": 100, "created": int(started),
                "data": [entry], "inference_time_s": time.time() - started}
            if self.bridge.stages:
                response["stage_durations"] = durations
                if timing is not None:
                    response["h3_sr_timing"] = timing
            return response

        work = asyncio.create_task(self.bridge.execute(request, request_id, write_output))
        stopped = asyncio.ensure_future(context.async_killed_or_stopped())
        try:
            done, _ = await asyncio.wait((work, stopped), return_when=asyncio.FIRST_COMPLETED)
            if stopped in done:
                work.cancel()
                await asyncio.gather(work, return_exceptions=True)
                raise asyncio.CancelledError
            yield await work
        except BaseException:
            # A disconnected caller must await bridge cancellation/latch
            # persistence; it never implies the native job stopped generating.
            if not work.done():
                work.cancel()
            await asyncio.gather(work, return_exceptions=True)
            raise
        finally:
            stopped.cancel()
            await asyncio.gather(stopped, return_exceptions=True)

    async def aclose(self):
        if self._monitor is not None:
            self._monitor.cancel()
            await asyncio.gather(self._monitor, return_exceptions=True)
        await self.bridge.close()

    def cleanup(self):
        # Model processes belong to service.launcher, not this endpoint.
        pass
