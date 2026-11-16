# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local candidate selection only: a candidate is not an acquired etcd slot."""

from __future__ import annotations

from fractions import Fraction

from dingo.common.task_protocol import integer
from dingo.video_gateway.task_runtime.repository import RuntimeKeys


def expand_slots(instances, capacity_for):
    """Expand already-eligible registrations; type-specific rules stay outside."""
    return [
        (instance, slot)
        for instance in instances
        for slot in range(capacity_for(instance))
    ]


def available_candidates(
    workers,
    occupied_keys,
    *,
    deployment_id,
    configured_capacity=16,
    blocked_domains=(),
    cursor=0,
):
    """Prefer free fraction, rotate ties, and deduplicate logical pool aliases.

    Conflicting registrations and occupied slots from an earlier registration
    block that physical backend domain until reconciliation releases them.
    Actual acquisition still compares snapshots in one transaction.
    """
    integer(configured_capacity, "configured_capacity")
    keys = RuntimeKeys(deployment_id)
    occupied, blocked = set(occupied_keys), set(blocked_domains)
    registrations = {}
    prefix = keys.root + "/slots/"
    for key in occupied:
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix) :].split("/")
        if len(parts) != 4:
            raise ValueError("invalid runtime slot key")
        domain, instance, generation, _ = parts
        registrations.setdefault(domain, set()).add((instance, generation))
    grouped = {}
    for worker in workers:
        ref = worker.ref
        physical = ref.capacity_domain
        if registrations.get(physical, set()) - {(ref.instance_id, ref.generation)}:
            blocked.add(physical)
        if ref.capacity_domain in blocked:
            continue
        prior = grouped.get(physical)
        if prior is None:
            grouped[physical] = (worker, min(configured_capacity, worker.capacity))
        elif (
            prior[0].ref.generation != ref.generation
            or prior[0].ref.instance_id != ref.instance_id
        ):
            blocked.add(ref.capacity_domain)
        else:
            grouped[physical] = (prior[0], min(prior[1], worker.capacity))
    choices = []
    for worker, capacity in grouped.values():
        if worker.ref.capacity_domain in blocked:
            continue
        free = [s for s in range(capacity) if keys.slot(worker.ref, s) not in occupied]
        if free:
            choices.append((Fraction(len(free), capacity), worker, free))
    # Stable registration order, then rotation, then free-fraction preference.
    if choices:
        offset = cursor % len(choices)
        choices = choices[offset:] + choices[:offset]
    choices.sort(key=lambda entry: entry[0], reverse=True)
    return [(worker, slot) for _, worker, free in choices for slot in free]
