# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable SR submission state machine, independent of CUDA and H3 imports.

The caller owns a physical-worker execution permit across this entire operation.
Its quarantine callback must disable ALL generation, not only SR admission.
This module never infers GPU termination from an HTTP timeout or DELETE ack.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dingo.vllm.omni.seedvr2_client import SETTLED, SeedVR2Client


class UnsafeRecovery(RuntimeError):
    """Existing submission lacks sufficient evidence to safely repeat work."""


def write_journal(path: Path, value: dict[str, Any]) -> None:
    """Publish metadata only after file AND containing-directory persistence."""
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("SR journal path cannot be a symlink")
    temporary = path.with_name(path.name + ".part-" + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def read_journal(path: Path) -> dict[str, Any] | None:
    if path.is_symlink():
        raise ValueError("SR journal cannot be a symlink")
    try:
        with path.open("rb") as stream:
            payload = stream.read(64 * 1024 + 1)
    except FileNotFoundError:
        return None
    if len(payload) > 64 * 1024:
        raise ValueError("SR journal exceeds metadata limit")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("invalid SR journal")
    return value


class SeedVR2Execution:
    def __init__(
        self,
        client: SeedVR2Client,
        *,
        owner: str,
        quarantine: Callable[[str], None],
        timeout_s: float = 10800,
        cancel_timeout_s: float = 30,
        poll_s: float = 1,
        pause: Callable[[], None] = lambda: None,
        resume: Callable[[], None] = lambda: None,
    ):
        if not owner or min(timeout_s, cancel_timeout_s, poll_s) <= 0:
            raise ValueError("invalid SR execution settings")
        self.client = client
        # Unique per supervised SR process incarnation, never just a Pod name.
        self.owner = owner
        self.quarantine = quarantine
        self.timeout_s = timeout_s
        self.cancel_timeout_s = cancel_timeout_s
        self.poll_s = poll_s
        self.pause, self.resume = pause, resume

    async def _cancel_and_settle(self, job_id):
        """Keep a bounded native drain alive across repeated caller cancellation."""
        cleanup = asyncio.create_task(self.client.cancel_and_wait(
            job_id, settle_timeout_s=self.cancel_timeout_s, poll_s=self.poll_s
        ))
        deadline = asyncio.get_running_loop().time() + self.cancel_timeout_s
        try:
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError("SR cancellation settlement timed out")
                try:
                    return await asyncio.wait_for(asyncio.shield(cleanup), remaining)
                except asyncio.CancelledError:
                    # The detached task's abort enforcer can cancel us again
                    # while native GPU cancellation is still settling. Keep
                    # admission paused and retain the original fixed deadline.
                    if cleanup.done():
                        return cleanup.result()
        finally:
            if not cleanup.done():
                cleanup.cancel()
                await asyncio.gather(cleanup, return_exceptions=True)
            elif not cleanup.cancelled():
                cleanup.exception()  # Retrieve a timeout/result race exception.

    async def run(
        self,
        source: Path,
        journal: Path,
        *,
        fingerprint: str,
        width: int,
        height: int,
        frames: int,
        stopped: Callable[[], bool],
        seed: int | None = None,
        options: dict | None = None,
    ) -> str:
        """Return a completed native job id; never turn Base into an SR result.

        A same-incarnation known job can be reattached without POST. Unknown
        identity or a new process incarnation fails closed. Cross-attempt Base
        recovery is a separate layer and must not erase these journals.
        """
        contract = {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "width": width,
            "height": height,
            "frames": frames,
            "seed": seed,
        }
        if options is not None:
            contract["options"] = options
        existing = read_journal(journal)
        if existing is not None and any(
            existing.get(k) != v for k, v in contract.items()
        ):
            raise UnsafeRecovery("SR checkpoint belongs to a different request")
        if existing is not None:
            if existing.get("owner") != self.owner or not existing.get("job_id"):
                self.quarantine(
                    "unresolved SR submission or changed process incarnation"
                )
                raise UnsafeRecovery("SR submission cannot be safely repeated")
            self.client.job_route(existing["job_id"])
            if existing.get("phase") in {"failed", "cancelled"}:
                raise UnsafeRecovery(
                    "settled unsuccessful SR attempt requires a new attempt"
                )
        record = existing or {**contract, "owner": self.owner, "phase": "submit_intent"}
        job_id = record.get("job_id")
        may_be_running = bool(existing and record.get("phase") != "completed")
        settled = False
        try:
            if stopped():
                raise asyncio.CancelledError
            async with asyncio.timeout(self.timeout_s):
                if job_id is None:
                    # Synchronous bounded metadata write is deliberate: there
                    # must not be an orphan thread publishing after cancellation.
                    write_journal(journal, record)
                    may_be_running = True
                    job_id = await self.client.submit(
                        source,
                        **({"options": options} if options is not None else {}), width=width, height=height, frames=frames, seed=seed
                    )
                    record.update(job_id=job_id, phase="submitted")
                    write_journal(journal, record)
                while True:
                    if stopped():
                        raise asyncio.CancelledError
                    state = await self.client.status(job_id)
                    if state["status"] in SETTLED:
                        settled = True
                        record.update(phase=state["status"], settled_at=time.time())
                        write_journal(journal, record)
                        if state["status"] != "completed" or state["frames"] != frames:
                            raise RuntimeError("SR did not complete every Base frame")
                        return job_id
                    if state["frames"] > frames:
                        raise RuntimeError("native SR processed more frames than Base")
                    await asyncio.sleep(self.poll_s)
        except BaseException:
            if may_be_running and not settled:
                # Pause admission while determining settlement. A confirmed
                # terminal cancellation/failure is request-scoped, not fatal.
                self.pause()
                if job_id is not None:
                    try:
                        state = await self._cancel_and_settle(job_id)
                        if state["status"] not in SETTLED:
                            raise RuntimeError("SR cancellation did not settle")
                        settled = True
                        record.update(phase=state["status"], settled_at=time.time())
                        write_journal(journal, record)
                    except BaseException:
                        # Unknown GPU state remains fenced until process exit.
                        logging.getLogger(__name__).warning(
                            "Native SR cancellation did not confirm settlement", exc_info=True
                        )
                    finally:
                        if settled:
                            self.resume()
                        else:
                            self.quarantine("SR interrupted before confirmed settlement")
                else:
                    self.quarantine("SR submission identity is unknown")
            raise
