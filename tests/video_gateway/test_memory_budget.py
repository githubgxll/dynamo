# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dingo.video_gateway.memory_budget import WeightedMemoryBudget


async def test_weighted_budget_applies_fifo_backpressure_and_tracks_peak():
    budget = WeightedMemoryBudget(100)

    assert await budget.try_acquire("large-active", 70) is True
    assert await budget.try_acquire("first-waiter", 50) is False
    assert await budget.try_acquire("small-later", 20) is False
    waiting = await budget.snapshot()
    assert waiting.used_bytes == 70
    assert waiting.waiting_tasks == 2

    assert await budget.release("large-active") is True
    assert await budget.try_acquire("small-later", 20) is False
    assert await budget.try_acquire("first-waiter", 50) is True
    assert await budget.release("first-waiter") is True
    assert await budget.try_acquire("small-later", 20) is True

    snapshot = await budget.snapshot()
    assert snapshot.used_bytes == 20
    assert snapshot.peak_bytes == 70
    assert snapshot.waiting_tasks == 0


async def test_weighted_budget_cancelled_waiter_does_not_block_followers():
    budget = WeightedMemoryBudget(10)

    assert await budget.try_acquire("active", 10) is True
    assert await budget.try_acquire("cancelled", 5) is False
    assert await budget.try_acquire("next", 5) is False
    await budget.cancel_waiter("cancelled")
    await budget.release("active")

    assert await budget.try_acquire("next", 5) is True
