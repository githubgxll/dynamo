# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, non-retrying client for an owned loopback SeedVR2 process.

DELETE acknowledges intent, not GPU completion. Callers must retain their GPU
permit until a settled status is observed, or quarantine/terminate the process.
An interrupted POST is ambiguous: never retry it automatically.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import aiohttp

ROUTE = "/v1/seedvr2/restore-long"
JOB_ID = re.compile(r"[0-9a-f]{32}\Z")
SETTLED = frozenset({"completed", "failed", "cancelled"})


class SubmissionUnknown(RuntimeError):
    """The server may have started GPU work without returning its identity."""


class SeedVR2Client:
    def __init__(self, port: int = 8124, *, timeout_s: float = 30):
        if type(port) is not int or not 1024 <= port <= 65535:
            raise ValueError("invalid loopback SR port")
        if timeout_s <= 0:
            raise ValueError("SR HTTP timeout must be positive")
        self.base_url = f"http://127.0.0.1:{port}"
        self.timeout = aiohttp.ClientTimeout(total=timeout_s)

    @staticmethod
    def job_route(job_id: str) -> str:
        if not isinstance(job_id, str) or JOB_ID.fullmatch(job_id) is None:
            raise ValueError("invalid native SR job id")
        return f"{ROUTE}/{job_id}"

    async def _json(self, method: str, route: str, *, data=None, status=200, max_bytes=64 * 1024):
        # Do not inherit cluster/user proxies or follow a redirect off loopback.
        async with aiohttp.ClientSession(
            trust_env=False, timeout=self.timeout
        ) as session:
            async with session.request(
                method, self.base_url + route, data=data, allow_redirects=False
            ) as response:
                if response.status != status:
                    raise RuntimeError(f"SR {method} returned HTTP {response.status}")
                body = bytearray()
                async for chunk in response.content.iter_chunked(8192):
                    body.extend(chunk)
                    if len(body) > max_bytes:
                        raise RuntimeError("SR status exceeds metadata limit")
                value = json.loads(body)
                if not isinstance(value, dict):
                    raise RuntimeError("SR response must be an object")
                return value

    async def manifest(self, job_id):
        return await self._json("GET", self.job_route(job_id) + "/manifest", max_bytes=512 * 1024)

    async def metrics(self, job_id):
        # Native publication and rank telemetry finalization are separate.
        deadline = asyncio.get_running_loop().time() + 30
        while True:
            try:
                value = await self._json("GET", self.job_route(job_id) + "/metrics", max_bytes=4 * 1024 * 1024)
            except RuntimeError as exc:
                if "returned HTTP 404" not in str(exc):
                    raise
                value = None
            if value is not None and value.get("metrics_status") == "PASS_COMPLETE":
                return value
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError("SR metrics did not reach PASS_COMPLETE")
            await asyncio.sleep(.5)

    async def submit(
        self,
        source: Path,
        *,
        width: int,
        height: int,
        frames: int,
        seed: int | None = None,
        options: dict | None = None,
    ) -> str:
        if any(type(v) is not int or v <= 0 for v in (width, height, frames)):
            raise ValueError("SR dimensions and frames must be positive integers")
        if width % 16 or height % 16:
            raise ValueError("SR internal dimensions must be 16-aligned")
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
            raise ValueError("SR seed must be an unsigned 32-bit integer")
        if (
            source.is_symlink()
            or not source.is_file()
            or source.stat().st_size > 128 * 1024**2
        ):
            raise ValueError("invalid or oversized Base artifact")
        fields = {
            "prompt": " ",
            "size": f"{width}x{height}",
            "num_frames": str(frames),
            "loop_input": "false",
            "extra_params": '{"color_correction_method":"lab"}',
        }
        if options is not None:
            if set(options) != {"window_frames", "delivery_size", "encoding_mode", "measurement_mode", "measurement_context"}:
                raise ValueError("invalid SR request options")
            if options["window_frames"] not in (33, 65) or options["encoding_mode"] != "single_encode" or options["measurement_mode"] != "basic":
                raise ValueError("invalid SR profile options")
            fields.update({k: json.dumps(v) if k == "measurement_context" else str(v) for k, v in options.items()})
        if seed is not None:
            fields["seed"] = str(seed)
        with source.open("rb") as stream:
            form = aiohttp.FormData()
            for key, value in fields.items():
                form.add_field(key, value)
            form.add_field(
                "input_references",
                stream,
                filename="base.mp4",
                content_type="video/mp4",
            )
            try:
                value = await self._json("POST", ROUTE, data=form, status=202)
                job_id = value.get("id")
                self.job_route(job_id)
                if value.get("status") != "queued":
                    raise RuntimeError("invalid SR submission acknowledgement")
                return job_id
            except asyncio.CancelledError:
                # Preserve cancellation; the caller's persisted submit intent
                # and quarantine path account for the unknown job identity.
                raise
            except Exception as exc:
                raise SubmissionUnknown(
                    "SR submission outcome unknown; do not repeat POST"
                ) from exc

    async def status(self, job_id: str) -> dict[str, Any]:
        value = await self._json("GET", self.job_route(job_id))
        if (
            value.get("status") not in SETTLED | {"queued", "running"}
            or type(value.get("frames")) is not int
            or value["frames"] < 0
        ):
            raise RuntimeError("invalid native SR status")
        return value

    async def cancel_and_wait(
        self, job_id: str, *, settle_timeout_s: float, poll_s: float = 1
    ):
        if settle_timeout_s <= 0 or poll_s <= 0:
            raise ValueError("invalid SR cancellation timeout")
        async with asyncio.timeout(settle_timeout_s):
            state = await self.status(job_id)
            if state["status"] in SETTLED:
                return state
            try:
                await self._json("DELETE", self.job_route(job_id), status=202)
            except Exception:
                # Completion may race DELETE. Only a subsequent settled GET
                # proves cancellation/drain, never the DELETE status itself.
                state = await self.status(job_id)
                if state["status"] in SETTLED:
                    return state
                raise
            while True:
                state = await self.status(job_id)
                if state["status"] in SETTLED:
                    return state
                await asyncio.sleep(poll_s)

    async def download(
        self, job_id: str, destination: Path, *, max_bytes: int = 128 * 1024**2
    ):
        route = self.job_route(job_id) + "/content"
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("invalid SR download limit")
        # Create-only, caller-owned attempt path. Failed downloads remain .part
        # evidence and must never be mistaken for a published final artifact.
        async with aiohttp.ClientSession(
            trust_env=False, timeout=self.timeout
        ) as session:
            async with session.get(
                self.base_url + route, allow_redirects=False
            ) as response:
                if response.status != 200 or response.content_type != "video/mp4":
                    raise RuntimeError("invalid SR content response")
                written = 0
                with destination.open("xb") as stream:
                    async for chunk in response.content.iter_chunked(1024 * 1024):
                        written += len(chunk)
                        if written > max_bytes:
                            raise RuntimeError("SR artifact exceeds result size limit")
                        writing = asyncio.create_task(
                            asyncio.to_thread(stream.write, chunk)
                        )
                        try:
                            await asyncio.shield(writing)
                        except asyncio.CancelledError:
                            # Do not close a stream while its filesystem write
                            # is still running in a thread.
                            await writing
                            raise
                if not written:
                    raise RuntimeError("empty SR content")
        return written
