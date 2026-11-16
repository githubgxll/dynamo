# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discovery/cache lifecycle shared by task types; wire policy is injected."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from dingo.common.task_protocol import WorkerCapabilities


async def refresh_instances(
    *,
    list_instances,
    cache,
    probe,
    publish,
    needs_probe=True,
    refresh_ahead_s=3.0,
    probe_parallelism=8,
    on_error=None,
):
    """Publish discovery before probing; remove stale instance evidence first.

    The callback owns type-specific caches (video SR, shared inputs, etc.). Probe
    failure policy belongs to the caller: v1 video may fall back to one slot,
    whereas v2 compilation must never infer capacity from a failed handshake.
    """
    try:
        instances = sorted(set(list_instances()))
        healthy = True
    except Exception:  # noqa: BLE001 - discovery failures must clear all eligibility
        instances, healthy = [], False
        if on_error is not None:
            on_error()
    live = set(instances)
    for instance in tuple(cache):
        if instance not in live:
            del cache[instance]
    publish(instances, healthy)
    if not needs_probe:
        return
    semaphore = asyncio.Semaphore(probe_parallelism)

    async def refresh(instance):
        if cache.get(instance, (0, 0))[1] > time.monotonic() + refresh_ahead_s:
            return
        async with semaphore:
            cache[instance] = await probe(instance)

    await asyncio.gather(*(refresh(instance) for instance in instances))


@dataclass(frozen=True)
class RegisteredWorker:
    capabilities: WorkerCapabilities
    expires_at: float


class CapabilityRegistry:
    """Strict v2 observations, separately scoped to a deployment/endpoint.

    A changed process generation replaces all previous capabilities. Missing,
    expired or malformed observations are unusable, never legacy single-slot.
    """

    def __init__(self, ttl_s=5.0):
        if ttl_s <= 0:
            raise ValueError("capability TTL must be positive")
        self.ttl_s = ttl_s
        self.records = {}

    def observe(self, instance_id, response, *, now=None):
        self.records.pop(instance_id, None)
        caps = WorkerCapabilities.parse(response)
        if caps.instance_id != instance_id or 2 not in caps.protocol_versions:
            raise ValueError("capability response identity/version mismatch")
        now = time.monotonic() if now is None else now
        self.records[instance_id] = RegisteredWorker(caps, now + self.ttl_s)
        return caps

    def reconcile(self, live_ids):
        live = set(live_ids)
        self.records = {k: v for k, v in self.records.items() if k in live}

    def eligible(
        self,
        *,
        task_kind,
        storage_id,
        model_adapter,
        protocol_adapter,
        queue=False,
        now=None,
    ):
        now = time.monotonic() if now is None else now
        result = []
        for record in self.records.values():
            c = record.capabilities
            if (
                record.expires_at > now
                and task_kind in c.task_kinds
                and c.storage_id == storage_id
                and not c.draining
                and (c.queue_allowed if queue else c.accepting)
                and model_adapter in c.model_adapters
                and protocol_adapter in c.protocol_adapters
            ):
                result.append(c)
        return result
