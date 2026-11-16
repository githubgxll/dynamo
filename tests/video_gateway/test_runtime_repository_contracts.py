# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError, replace

import pytest

from dingo.video_gateway.task_runtime.repository import (
    AttemptFence,
    CasResult,
    CasStatus,
    ClaimResult,
    ClaimStatus,
    KeySnapshot,
    OwnerSnapshot,
    RuntimeKeys,
    TaskRef,
    ValidatedOutcome,
    WorkerRef,
)


def test_physical_slot_keys_do_not_multiply_capacity_for_pool_aliases():
    keys = RuntimeKeys("deployment")
    worker = WorkerRef("h3-fl", "instance", "boot-1", "qwen")
    assert keys.slot(worker, 0) == keys.slot(replace(worker, pool_id="h3-ref"), 0)
    assert keys.slot(worker, 0) != keys.slot(replace(worker, instance_id="other"), 0)
    assert keys.slot(worker, 0) != keys.slot(replace(worker, generation="boot-2"), 0)
    with pytest.raises((ValueError, TypeError)):
        keys.slot(worker, -1)
    with pytest.raises((ValueError, TypeError)):
        keys.active(TaskRef("other", "context_ir", "task"))


def test_fence_and_snapshots_are_validated_and_immutable():
    task = TaskRef("deployment", "context_ir", "task")
    worker = WorkerRef("pool", "w", "g", "q")
    fence = AttemptFence(task, worker, 1, "a" * 32)
    assert fence.identity().task_id == "task"
    assert RuntimeKeys("deployment").attempt(fence).endswith("/1-" + "a" * 32)
    with pytest.raises(FrozenInstanceError):
        fence.attempt = 2
    for create in [
        lambda: KeySnapshot("/task", 0, b"{}"),
        lambda: KeySnapshot("/task", 1, {}),
        lambda: AttemptFence(task, worker, True, "a" * 32),
        lambda: OwnerSnapshot("new", KeySnapshot("/owner", 1, b"old")),
    ]:
        with pytest.raises((ValueError, TypeError)):
            create()


def test_unknown_is_not_a_conflict_or_a_confirmed_claim():
    assert CasResult(CasStatus.UNKNOWN).revision is None
    assert ClaimResult(ClaimStatus.UNKNOWN).attempt is None
    for create in [
        lambda: CasResult(CasStatus.UNKNOWN, 123),
        lambda: CasResult(CasStatus.APPLIED),
        lambda: ClaimResult(ClaimStatus.CLAIMED),
        lambda: ValidatedOutcome("running", None, b"{}"),
    ]:
        with pytest.raises((ValueError, TypeError)):
            create()
