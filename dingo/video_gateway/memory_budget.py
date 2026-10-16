# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fair process-local weighted budget for legacy Base64 media transport."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MemoryBudgetSnapshot:
    capacity_bytes: int
    used_bytes: int
    peak_bytes: int
    waiting_tasks: int
    active_tasks: int


class WeightedMemoryBudget:
    def __init__(self, capacity_bytes: int) -> None:
        if capacity_bytes <= 0:
            raise ValueError("memory budget capacity must be positive")
        self.capacity_bytes = capacity_bytes
        self._lock = asyncio.Lock()
        self._used_bytes = 0
        self._peak_bytes = 0
        self._allocations: dict[str, int] = {}
        self._waiters: OrderedDict[str, int] = OrderedDict()

    async def try_acquire(self, task_id: str, weight_bytes: int) -> bool:
        if weight_bytes <= 0:
            raise ValueError("memory budget weight must be positive")
        if weight_bytes > self.capacity_bytes:
            raise ValueError("task weight exceeds the process memory budget")
        async with self._lock:
            if task_id in self._allocations:
                return True
            self._waiters.setdefault(task_id, weight_bytes)
            if self._waiters[task_id] != weight_bytes:
                raise ValueError("task memory budget weight changed while queued")
            first_task_id = next(iter(self._waiters))
            if first_task_id != task_id:
                return False
            if self._used_bytes + weight_bytes > self.capacity_bytes:
                return False
            self._waiters.pop(task_id)
            self._allocations[task_id] = weight_bytes
            self._used_bytes += weight_bytes
            self._peak_bytes = max(self._peak_bytes, self._used_bytes)
            return True

    async def cancel_waiter(self, task_id: str) -> None:
        async with self._lock:
            self._waiters.pop(task_id, None)

    async def shrink(self, task_id: str, weight_bytes: int) -> int:
        """Reduce an active allocation and return the released byte count."""
        if weight_bytes <= 0:
            raise ValueError("memory budget weight must be positive")
        async with self._lock:
            current = self._allocations.get(task_id)
            if current is None or current == weight_bytes:
                return 0
            if weight_bytes > current:
                raise ValueError("memory budget allocation cannot grow while active")
            released = current - weight_bytes
            self._allocations[task_id] = weight_bytes
            self._used_bytes -= released
            if self._used_bytes < 0:  # pragma: no cover - invariant guard
                raise RuntimeError("memory budget usage became negative")
            return released

    async def release(self, task_id: str) -> bool:
        async with self._lock:
            weight = self._allocations.pop(task_id, None)
            if weight is None:
                self._waiters.pop(task_id, None)
                return False
            self._used_bytes -= weight
            if self._used_bytes < 0:  # pragma: no cover - invariant guard
                raise RuntimeError("memory budget usage became negative")
            return True

    async def snapshot(self) -> MemoryBudgetSnapshot:
        async with self._lock:
            return MemoryBudgetSnapshot(
                capacity_bytes=self.capacity_bytes,
                used_bytes=self._used_bytes,
                peak_bytes=self._peak_bytes,
                waiting_tasks=len(self._waiters),
                active_tasks=len(self._allocations),
            )
