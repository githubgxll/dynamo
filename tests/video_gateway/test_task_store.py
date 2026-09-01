# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import pytest

from dingo.video_gateway.errors import GatewayError, StoreConflict
from dingo.video_gateway.models import TaskStatus, VideoTask, WorkerLease, now_ms
from dingo.video_gateway.task_store import MemoryTaskStore, worker_key


def _task(task_id: str, *, pool_id="pool-a", digest="sha256:one") -> VideoTask:
    timestamp = now_ms()
    return VideoTask(
        schema_version=1,
        id=task_id,
        deployment_id="deployment",
        pool_id=pool_id,
        model=f"model-{pool_id}",
        backend_model=f"backend-{pool_id}",
        backend_target=f"dyn://scope.{pool_id}.generate",
        configuration_revision="sha256:config",
        delivery_mode="async",
        status=TaskStatus.QUEUED,
        request_digest=digest,
        request_path=f"/artifacts/{task_id}/request.json",
        input_manifest_path=f"/artifacts/{task_id}/manifest.json",
        created_at_ms=timestamp,
        queued_at_ms=timestamp,
        expires_at_ms=timestamp + 100_000,
    )


def _lease(task: VideoTask, instance_id=7) -> WorkerLease:
    key = worker_key(task.backend_target, instance_id)
    return WorkerLease(
        pool_id=task.pool_id,
        worker_key=key,
        worker_instance_id=instance_id,
        backend_target=task.backend_target,
        task_id=task.id,
        owner_generation="generation",
        state="reserved",
        heartbeat_at_ms=now_ms(),
        owner_expires_at_ms=now_ms() + 15_000,
    )


def test_public_task_reports_final_seed_and_validated_media_metadata():
    task = _task("video-metadata")
    task.status = TaskStatus.COMPLETED
    task.completed_at_ms = task.created_at_ms + 1_000
    task.result_bytes = 1234
    task.result_sha256 = "a" * 64
    task.normalized_request = {
        "width": 1344,
        "height": 768,
        "fps": 24,
        "seconds": 5.0,
        "num_frames": 120,
        "seed": 1101,
        "seed_generated": True,
    }
    task.normalized_request["_result_media"] = {
        "container": "mp4",
        "fps": 24.0,
        "frames": 124,
        "duration_s": 5.175,
        "video_duration_s": 124 / 24,
        "audio_duration_s": 5.175,
    }

    public = task.public_dict()

    assert public["seed"] == 1101
    assert public["seed_generated"] is True
    assert public["requested_num_frames"] == 120
    assert public["num_frames"] == 124
    assert public["requested_seconds"] == 5.0
    assert public["seconds"] == 5.175
    assert public["duration_s"] == 5.175
    assert public["video_duration_s"] == 124 / 24
    assert public["audio_duration_s"] == 5.175


async def test_idempotency_returns_original_task_and_detects_conflict():
    store = MemoryTaskStore()
    first, created = await store.create_task(
        _task("video-01"),
        principal_hash="principal",
        idempotency_hash="key",
        queue_limit=2,
    )
    duplicate, duplicate_created = await store.create_task(
        _task("video-02"),
        principal_hash="principal",
        idempotency_hash="key",
        queue_limit=2,
    )

    assert created is True
    assert duplicate_created is False
    assert duplicate.task.id == first.task.id
    assert await store.queue_depth("pool-a") == 1

    with pytest.raises(GatewayError) as error:
        await store.create_task(
            _task("video-03", digest="sha256:different"),
            principal_hash="principal",
            idempotency_hash="key",
            queue_limit=2,
        )
    assert error.value.code == "idempotency_conflict"


async def test_queue_limit_is_atomic_for_concurrent_creates():
    store = MemoryTaskStore()

    results = await asyncio.gather(
        *(
            store.create_task(
                _task(f"video-{index:02d}"),
                principal_hash="principal",
                idempotency_hash=None,
                queue_limit=2,
            )
            for index in range(3)
        ),
        return_exceptions=True,
    )

    assert sum(not isinstance(result, Exception) for result in results) == 2
    errors = [result for result in results if isinstance(result, GatewayError)]
    assert [error.code for error in errors] == ["queue_full"]
    assert await store.queue_depth("pool-a") == 2


async def test_one_queued_task_can_be_reserved_only_once():
    store = MemoryTaskStore()
    stored, _ = await store.create_task(
        _task("video-reserve"),
        principal_hash="principal",
        idempotency_hash=None,
        queue_limit=2,
    )
    lease = _lease(stored.task)

    left, right = await asyncio.gather(
        store.reserve(stored, lease, deadline_at_ms=now_ms() + 1000),
        store.reserve(stored, lease, deadline_at_ms=now_ms() + 1000),
    )

    assert sum(value is not None for value in (left, right)) == 1
    assert await store.queue_depth("pool-a") == 0
    assert len(await store.list_leases("pool-a")) == 1


async def test_same_numeric_instance_in_different_targets_has_independent_leases():
    store = MemoryTaskStore()
    task_a = _task("video-a", pool_id="pool-a")
    task_b = _task("video-b", pool_id="pool-b")
    stored_a, _ = await store.create_task(
        task_a, principal_hash="p", idempotency_hash=None, queue_limit=2
    )
    stored_b, _ = await store.create_task(
        task_b, principal_hash="p", idempotency_hash=None, queue_limit=2
    )

    reserved_a, reserved_b = await asyncio.gather(
        store.reserve(stored_a, _lease(task_a, 7), deadline_at_ms=now_ms() + 1000),
        store.reserve(stored_b, _lease(task_b, 7), deadline_at_ms=now_ms() + 1000),
    )

    assert reserved_a is not None and reserved_b is not None
    assert reserved_a.task.worker_key != reserved_b.task.worker_key
    assert len(await store.list_leases("pool-a")) == 1
    assert len(await store.list_leases("pool-b")) == 1


async def test_late_task_transition_never_releases_new_owner_lease():
    store = MemoryTaskStore()
    old_task = _task("video-old")
    old, _ = await store.create_task(
        old_task, principal_hash="p", idempotency_hash=None, queue_limit=2
    )
    reserved = await store.reserve(
        old, _lease(old_task), deadline_at_ms=now_ms() + 10_000
    )
    assert reserved is not None and reserved.task.worker_key is not None

    new_lease = _lease(_task("video-new"))
    new_lease.worker_key = reserved.task.worker_key
    store._leases[(old_task.pool_id, reserved.task.worker_key)] = new_lease

    await store.transition(
        old_task.id,
        expected={TaskStatus.DISPATCHING},
        expected_revision=reserved.revision,
        patch={"status": TaskStatus.FAILED},
        release_lease=True,
        quarantine_until_ms=now_ms() + 10_000,
    )

    leases = await store.list_leases(old_task.pool_id)
    assert len(leases) == 1
    assert leases[0].task_id == "video-new"
    assert leases[0].state == "reserved"


async def test_missing_execution_lease_is_replaced_by_quarantine_with_cas_semantics():
    store = MemoryTaskStore()
    task = _task("video-lost-lease")
    stored, _ = await store.create_task(
        task, principal_hash="p", idempotency_hash=None, queue_limit=1
    )
    reserved = await store.reserve(
        stored, _lease(task), deadline_at_ms=now_ms() + 10_000
    )
    assert reserved is not None and reserved.task.worker_key is not None
    await store.release_lease(task.pool_id, reserved.task.worker_key)

    failed = await store.transition(
        task.id,
        expected={TaskStatus.DISPATCHING},
        expected_revision=reserved.revision,
        patch={"status": TaskStatus.FAILED},
        release_lease=True,
        quarantine_until_ms=now_ms() + 10_000,
    )

    leases = await store.list_leases(task.pool_id)
    assert failed.task.status == TaskStatus.FAILED
    assert len(leases) == 1
    assert leases[0].task_id == task.id
    assert leases[0].state == "quarantined"


async def test_illegal_state_transition_is_rejected():
    store = MemoryTaskStore()
    stored, _ = await store.create_task(
        _task("video-state"),
        principal_hash="p",
        idempotency_hash=None,
        queue_limit=1,
    )

    with pytest.raises(StoreConflict, match="illegal task transition"):
        await store.transition(
            stored.task.id,
            expected={TaskStatus.QUEUED},
            expected_revision=stored.revision,
            patch={"status": TaskStatus.COMPLETED},
        )


async def test_running_cancel_request_is_idempotent():
    store = MemoryTaskStore()
    task = _task("video-cancel-idempotent")
    stored, _ = await store.create_task(
        task,
        principal_hash="p",
        idempotency_hash=None,
        queue_limit=1,
    )
    reserved = await store.reserve(
        stored, _lease(task), deadline_at_ms=now_ms() + 10_000
    )
    assert reserved is not None

    first = await store.request_cancel(task.id)
    second = await store.request_cancel(task.id)

    assert first.task.cancel_requested_at_ms is not None
    assert second.task.cancel_requested_at_ms == first.task.cancel_requested_at_ms
    assert second.revision == first.revision


async def test_expired_task_and_idempotency_index_are_deleted_together():
    store = MemoryTaskStore()
    stored, _ = await store.create_task(
        _task("video-expire"),
        principal_hash="principal",
        idempotency_hash="key",
        queue_limit=1,
    )
    cancelled = await store.request_cancel(stored.task.id)
    expired = await store.transition(
        cancelled.task.id,
        expected={TaskStatus.CANCELLED},
        expected_revision=cancelled.revision,
        patch={"status": TaskStatus.EXPIRED},
    )

    assert await store.delete_expired(expired) is True
    assert await store.get_task(stored.task.id) is None

    replacement, created = await store.create_task(
        _task("video-replacement"),
        principal_hash="principal",
        idempotency_hash="key",
        queue_limit=1,
    )
    assert created is True
    assert replacement.task.id == "video-replacement"
