# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video-specific capability policy retained while using shared slot expansion."""
import time
from collections.abc import Mapping
from typing import Any

from .capacity import expand_slots


def video_worker_slots(
    pool, normalized: Mapping[str, Any] | None = None
) -> list[tuple[int, int]]:
    sr = (normalized or {}).get("super_resolution")
    needs_sr = normalized is not None and "super_resolution" in normalized
    required_shared = (normalized or {}).get("shared_inputs")
    adapter_options = getattr(getattr(pool.config, "adapter", None), "options", {})
    if (
        normalized is None
        and adapter_options.get("reference_transport") == "shared_files_v1"
    ):
        required_shared = {"storage_id": adapter_options.get("shared_storage_id")}
    ceiling = pool.config.scheduling.worker_capacity + getattr(
        pool.config.scheduling, "worker_prefetch_capacity", 0
    )
    eligible = [
        instance
        for instance in pool.instance_ids
        if not required_shared
        or (
            getattr(pool, "shared_inputs", {}).get(instance, {}).get("storage_id")
            == required_shared.get("storage_id")
            and getattr(pool, "shared_inputs", {}).get(instance, {}).get("workflow")
            == pool.config.adapter.workflow
            and pool.capacity_cache.get(instance, (0, 0))[1] > time.monotonic()
        )
        if pool.capacity_cache.get(instance, (1, 0))[0] > 0
        if not getattr(pool.config, "sr_profile", None)
        or getattr(pool.config, "unified_pipeline", False)
        or pool.capacity_cache.get(instance, (0, 0))[1] > time.monotonic()
        if not needs_sr
        or (
            isinstance(sr, Mapping)
            and sr.get("profile") == getattr(pool.config, "sr_profile", None)
            and sr.get("profile") in getattr(pool, "sr_profiles", {}).get(instance, ())
            and pool.capacity_cache.get(instance, (0, 0))[0] > 0
            and pool.capacity_cache.get(instance, (0, 0))[1] > time.monotonic()
        )
    ]
    return expand_slots(
        eligible,
        lambda instance: 1
        if ceiling == 1
        else min(ceiling, pool.capacity_cache.get(instance, (0, 0))[0]),
    )
