# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from dingo.video_gateway.adapters import create_adapter
from dingo.video_gateway.artifact_store import FileArtifactStore
from dingo.video_gateway.dispatcher import VideoDispatcher
from dingo.video_gateway.etcd_http import EtcdHttpClient
from dingo.video_gateway.models import TaskStatus
from dingo.video_gateway.service import VideoGatewayService
from dingo.video_gateway.task_store import EtcdTaskStore
from dingo.vllm.omni.detached_tasks import DetachedOmniTaskManager
from tests.video_gateway.test_dispatcher import (
    FakeClient,
    FakeContext,
    _DetachedClient,
    _DetachedHandler,
    _pool,
    _submit,
)
from tests.video_gateway.test_task_store import _lease, _task

_ETCD_URL = os.environ.get("DINGO_VIDEO_TEST_ETCD_URL")


@pytest.mark.skipif(not _ETCD_URL, reason="requires a real etcd v3 endpoint")
async def test_result_handoff_and_owner_takeover_never_touch_reused_slot():
    from dingo.video_gateway.models import now_ms
    from tests.video_gateway.test_result_handoff import patch_for

    client = EtcdHttpClient(str(_ETCD_URL), timeout_s=5.0)
    prefix = f"/dingo/continuous-execution-contract/{uuid.uuid4().hex}"
    store = EtcdTaskStore(client, prefix=prefix, deployment_id="contract")
    owners = []
    try:
        for generation in ["generation", "gateway-b", "gateway-c"]:
            owners.append(await store.register_gateway(generation, ttl_s=30))
        first = _task("first")
        stored, _ = await store.create_task(
            first, principal_hash="p", idempotency_hash=None, queue_limit=8
        )
        lease = _lease(first)
        lease.execution_token = "c" * 32
        stored = await store.reserve(
            stored, lease, deadline_at_ms=now_ms() + 60000, reserve_retry=True
        )
        stored = await store.transition(
            first.id,
            expected={TaskStatus.DISPATCHING},
            patch={"status": TaskStatus.IN_PROGRESS},
        )
        ready = await store.transition(
            first.id,
            expected={TaskStatus.IN_PROGRESS},
            expected_revision=stored.revision,
            patch=patch_for(stored),
            release_lease=True,
            release_execution=True,
        )
        assert not await store.list_leases(first.pool_id)
        assert await client.get(store._retry_credit_key(ready.task)) is None
        assert (
            int(
                (
                    await client.get(store._retry_counter_key(first.pool_id, "credits"))
                ).value
            )
            == 0
        )
        second = _task("second")
        queued, _ = await store.create_task(
            second, principal_hash="p", idempotency_hash=None, queue_limit=8
        )
        next_lease = _lease(second)
        next_lease.owner_generation = "gateway-b"
        next_lease.execution_token = "d" * 32
        reserved = await store.reserve(
            queued, next_lease, deadline_at_ms=now_ms() + 60000, reserve_retry=True
        )
        assert reserved is not None
        key = store._lease_key(second.pool_id, reserved.task.worker_key)
        occupant = await client.get(key)
        assert (
            await store.claim_finalizing(ready, new_owner_generation="gateway-b")
            is None
        )
        await store.unregister_gateway(owners.pop(0))
        claims = await asyncio.gather(
            *[
                store.claim_orphaned_active(ready, new_owner_generation=owner)
                for owner in ["gateway-b", "gateway-c"]
            ]
        )
        winners = [claim for claim in claims if claim is not None]
        assert len(winners) == 1
        claimed = winners[0]
        assert claimed.task.worker_lease_id is None
        assert await client.get(key) == occupant
        await store.transition(
            first.id,
            expected={TaskStatus.FINALIZING},
            expected_revision=claimed.revision,
            patch={"status": TaskStatus.COMPLETED},
            release_lease=False,
        )
        assert await client.get(key) == occupant
        assert (
            int(
                (
                    await client.get(store._retry_counter_key(first.pool_id, "credits"))
                ).value
            )
            == 1
        )
    finally:
        for lease_id in owners:
            await store.unregister_gateway(lease_id)
        await _delete_prefix(client, prefix)
        await client.close()


async def _delete_prefix(client: EtcdHttpClient, prefix: str) -> None:
    values, _revision = await client.range_all(prefix, prefix=True)
    for offset in range(0, len(values), 100):
        await client.txn(
            [],
            [client.delete(value.key) for value in values[offset : offset + 100]],
        )


@pytest.mark.skipif(
    not _ETCD_URL,
    reason="set DINGO_VIDEO_TEST_ETCD_URL to run the real etcd v3 contract",
)
async def test_real_etcd_create_idempotency_queue_and_cleanup_contract():
    client = EtcdHttpClient(str(_ETCD_URL), timeout_s=5.0)
    prefix = f"/dingo/video-gateway-contract-tests/{uuid.uuid4().hex}"
    store = EtcdTaskStore(client, prefix=prefix, deployment_id="contract")
    try:
        await store.health()
        original, created = await store.create_task(
            _task("video-etcd-contract"),
            principal_hash="principal",
            idempotency_hash="key",
            queue_limit=1,
        )
        duplicate, duplicate_created = await store.create_task(
            _task("video-etcd-duplicate"),
            principal_hash="principal",
            idempotency_hash="key",
            queue_limit=1,
        )
        assert created is True
        assert original.task.created_seq > 0
        assert duplicate_created is False
        assert duplicate.task.id == original.task.id
        assert await store.queue_depth(original.task.pool_id) == 1
        counts = await store.task_counts(original.task.pool_id)
        assert counts[TaskStatus.QUEUED] == 1
        listed = await store.list_tasks(
            pool_id=original.task.pool_id,
            status=TaskStatus.QUEUED,
            limit=10,
        )
        assert [stored.task.id for stored in listed] == [original.task.id]

        cancelled = await store.request_cancel(original.task.id)
        assert cancelled.task.status == TaskStatus.CANCELLED
        expired = await store.transition(
            original.task.id,
            expected={TaskStatus.CANCELLED},
            expected_revision=cancelled.revision,
            patch={"status": TaskStatus.EXPIRED},
        )
        assert await store.delete_expired(expired) is True
        assert await store.get_task(original.task.id) is None
    finally:
        await _delete_prefix(client, prefix)
        await client.close()


@pytest.mark.skipif(
    not _ETCD_URL,
    reason="set DINGO_VIDEO_TEST_ETCD_URL to run the real etcd v3 contract",
)
async def test_real_etcd_range_all_reads_every_page_from_one_snapshot():
    client = EtcdHttpClient(str(_ETCD_URL), timeout_s=5.0)
    prefix = f"/dingo/video-gateway-pagination-tests/{uuid.uuid4().hex}/"
    expected = {
        f"{prefix}video-{index:05d}": f"value-{index}".encode()
        for index in range(1_025)
    }
    try:
        items = sorted(expected.items())
        for offset in range(0, len(items), 100):
            succeeded, _revision = await client.txn(
                [],
                [client.put(key, value) for key, value in items[offset : offset + 100]],
            )
            assert succeeded is True

        first_page = await client.range_page(prefix, prefix=True, limit=128)
        assert first_page.more is True
        assert len(first_page.values) == 128

        values, snapshot_revision = await client.range_all(
            prefix,
            prefix=True,
            page_size=128,
        )

        assert snapshot_revision > 0
        assert {value.key: value.value for value in values} == expected
    finally:
        await _delete_prefix(client, prefix)
        await client.close()


@pytest.mark.skipif(
    not _ETCD_URL,
    reason="set DINGO_VIDEO_TEST_ETCD_URL to run the real etcd v3 contract",
)
async def test_real_etcd_lease_and_watch_contract():
    client = EtcdHttpClient(str(_ETCD_URL), timeout_s=5.0)
    prefix = f"/dingo/video-gateway-watch-tests/{uuid.uuid4().hex}/"
    owner_key = prefix + "owner"
    watch = client.watch_prefix(prefix, previous=True, progress_notify=False)
    try:
        created = await asyncio.wait_for(anext(watch), timeout=5.0)
        assert created.created is True

        lease = await client.lease_grant(10)
        succeeded, _revision = await client.txn(
            [],
            [client.put(owner_key, "owner-a", lease_id=lease.lease_id)],
        )
        assert succeeded is True

        put_event = await asyncio.wait_for(anext(watch), timeout=5.0)
        assert put_event.events[0].event_type == "PUT"
        assert put_event.events[0].value.key == owner_key
        assert put_event.events[0].value.lease == lease.lease_id

        remaining = await client.lease_time_to_live(lease.lease_id, keys=True)
        assert remaining.ttl > 0
        assert owner_key in remaining.keys
        alive = await client.lease_keepalive(lease.lease_id)
        assert alive.ttl > 0

        await client.lease_revoke(lease.lease_id)
        delete_event = await asyncio.wait_for(anext(watch), timeout=5.0)
        assert delete_event.events[0].event_type == "DELETE"
        assert delete_event.events[0].value.key == owner_key
        assert await client.get(owner_key) is None
    finally:
        await watch.aclose()
        await _delete_prefix(client, prefix)
        await client.close()


@pytest.mark.skipif(
    not _ETCD_URL,
    reason="set DINGO_VIDEO_TEST_ETCD_URL to run the real etcd v3 contract",
)
async def test_real_etcd_count_descending_and_batch_get_contract():
    client = EtcdHttpClient(str(_ETCD_URL), timeout_s=5.0)
    prefix = f"/dingo/video-gateway-batch-tests/{uuid.uuid4().hex}/"
    keys = [f"{prefix}{index:03d}" for index in range(5)]
    try:
        succeeded, revision = await client.txn(
            [], [client.put(key, key) for key in keys]
        )
        assert succeeded is True
        assert await client.count_prefix(prefix) == len(keys)
        descending = await client.range(prefix, prefix=True, limit=2, descending=True)
        assert [value.key for value in descending] == list(reversed(keys[-2:]))
        values, snapshot_revision = await client.get_many(
            [keys[1], prefix + "missing", keys[3]], revision=revision
        )
        assert [value.key if value else None for value in values] == [
            keys[1],
            None,
            keys[3],
        ]
        assert snapshot_revision >= revision
    finally:
        await _delete_prefix(client, prefix)
        await client.close()


@pytest.mark.skipif(
    not _ETCD_URL,
    reason="set DINGO_VIDEO_TEST_ETCD_URL to run the real etcd v3 contract",
)
async def test_real_etcd_worker_lease_snapshot_watch_and_keepalive_contract():
    client = EtcdHttpClient(str(_ETCD_URL), timeout_s=5.0)
    prefix = f"/dingo/video-gateway-worker-lease-tests/{uuid.uuid4().hex}"
    store = EtcdTaskStore(client, prefix=prefix, deployment_id="contract")
    try:
        task = _task("video-etcd-worker-lease")
        stored, _created = await store.create_task(
            task,
            principal_hash="principal",
            idempotency_hash=None,
            queue_limit=2,
        )
        worker_lease = _lease(stored.task)
        gateway_lease_id = await store.register_gateway(
            worker_lease.owner_generation, ttl_s=15
        )
        reserved = await store.reserve(
            stored,
            worker_lease,
            deadline_at_ms=stored.task.created_at_ms + 60_000,
        )
        assert reserved is not None
        assert reserved.task.worker_lease_id is not None

        leases, revision = await store.lease_snapshot(task.pool_id)
        assert revision > 0
        assert reserved.task.worker_key in leases
        await store.heartbeat_lease(
            task.pool_id,
            str(reserved.task.worker_key),
            task.id,
            reserved.task.worker_lease_id,
        )

        watch = store.watch_leases(task.pool_id, start_revision=revision + 1)
        created = await asyncio.wait_for(anext(watch), timeout=5.0)
        assert created.created is True
        await store.release_lease(task.pool_id, str(reserved.task.worker_key))
        deleted = await asyncio.wait_for(anext(watch), timeout=5.0)
        assert deleted.worker_key == reserved.task.worker_key
        assert deleted.lease is None
        await watch.aclose()

        assert [item async for item in store.iter_orphaned_active_tasks()] == []
        await store.unregister_gateway(gateway_lease_id)
        orphaned = [item async for item in store.iter_orphaned_active_tasks()]
        assert [item.task.id for item in orphaned] == [stored.task.id]
    finally:
        await _delete_prefix(client, prefix)
        await client.close()


@pytest.mark.skipif(
    not _ETCD_URL,
    reason="set DINGO_VIDEO_TEST_ETCD_URL to run the real etcd v3 contract",
)
async def test_real_etcd_task_watch_contract():
    client = EtcdHttpClient(str(_ETCD_URL), timeout_s=5.0)
    prefix = f"/dingo/video-gateway-task-watch-tests/{uuid.uuid4().hex}"
    store = EtcdTaskStore(client, prefix=prefix, deployment_id="contract")
    watch = None
    try:
        revision = await store.task_watch_revision()
        watch = store.watch_tasks(start_revision=revision + 1)
        opened = await asyncio.wait_for(anext(watch), timeout=5.0)
        assert opened.created is True

        stored, created = await store.create_task(
            _task("video-etcd-task-watch"),
            principal_hash="principal",
            idempotency_hash=None,
            queue_limit=2,
        )
        assert created is True
        task_created = await asyncio.wait_for(anext(watch), timeout=5.0)
        assert task_created.task_id == stored.task.id
        assert task_created.deleted is False

        cancelled = await store.request_cancel(stored.task.id)
        task_updated = await asyncio.wait_for(anext(watch), timeout=5.0)
        assert task_updated.task_id == cancelled.task.id
        assert task_updated.revision >= task_created.revision
    finally:
        if watch is not None:
            await watch.aclose()
        await _delete_prefix(client, prefix)
        await client.close()


@pytest.mark.skipif(
    not _ETCD_URL,
    reason="set DINGO_VIDEO_TEST_ETCD_URL to run the real etcd v3 contract",
)
async def test_two_gateways_preserve_task_owned_by_healthy_peer(
    make_gateway_config,
):
    prefix = f"/dingo/video-gateway-ha-tests/{uuid.uuid4().hex}"
    config = make_gateway_config()
    cleanup_client = EtcdHttpClient(str(_ETCD_URL), timeout_s=5.0)
    store_a = EtcdTaskStore(
        EtcdHttpClient(str(_ETCD_URL), timeout_s=5.0),
        prefix=prefix,
        deployment_id=config.deployment_id,
    )
    store_b = EtcdTaskStore(
        EtcdHttpClient(str(_ETCD_URL), timeout_s=5.0),
        prefix=prefix,
        deployment_id=config.deployment_id,
    )
    artifacts = FileArtifactStore(config.artifact_store.root)
    adapters = {pool.pool_id: create_adapter(pool) for pool in config.pools}
    client_a = FakeClient(block=True)
    client_b = FakeClient(block=True)
    dispatcher_a = VideoDispatcher(
        config,
        store_a,
        artifacts,
        {"fl-pool": client_a},
        adapters,
        context_factory=FakeContext,
        generation="gateway-a",
    )
    dispatcher_b = VideoDispatcher(
        config,
        store_b,
        artifacts,
        {"fl-pool": client_b},
        adapters,
        context_factory=FakeContext,
        generation="gateway-b",
    )
    service_a = VideoGatewayService(config, store_a, artifacts, dispatcher_a, adapters)
    started_a = False
    started_b = False
    try:
        await dispatcher_a.start()
        started_a = True
        submission = await _submit(service_a, "public-fl")
        for _ in range(100):
            if client_a.calls:
                break
            await asyncio.sleep(0.01)
        assert len(client_a.calls) == 1

        await dispatcher_b.start()
        started_b = True
        active = await store_b.get_task(submission.stored.task.id)
        assert active is not None
        assert active.task.status == TaskStatus.IN_PROGRESS
        assert active.task.owner_generation == "gateway-a"
        assert client_b.calls == []

        client_a.release.set()
        terminal = await dispatcher_b.wait_terminal(active.task.id, 2)
        assert terminal.task.status == TaskStatus.COMPLETED
        assert len(client_a.calls) + len(client_b.calls) == 1
    finally:
        client_a.release.set()
        client_b.release.set()
        if started_b:
            await dispatcher_b.stop()
        if started_a:
            await dispatcher_a.stop()
        await _delete_prefix(cleanup_client, prefix)
        await cleanup_client.close()


@pytest.mark.skipif(
    not _ETCD_URL,
    reason="set DINGO_VIDEO_TEST_ETCD_URL to run the real etcd v3 contract",
)
async def test_detached_task_survives_owner_gateway_shutdown_and_is_claimed(
    make_gateway_config,
):
    prefix = f"/dingo/video-gateway-detached-ha-tests/{uuid.uuid4().hex}"
    pool_raw = _pool("fl-pool", "public-fl", "dyn://scope-a.backend.generate")
    pool_raw["execution_mode"] = "detached"
    config = make_gateway_config(pools=[pool_raw])
    cleanup_client = EtcdHttpClient(str(_ETCD_URL), timeout_s=5.0)
    store_a = EtcdTaskStore(
        EtcdHttpClient(str(_ETCD_URL), timeout_s=5.0),
        prefix=prefix,
        deployment_id=config.deployment_id,
        execution_lease_ttl_s=5,
    )
    store_b = EtcdTaskStore(
        EtcdHttpClient(str(_ETCD_URL), timeout_s=5.0),
        prefix=prefix,
        deployment_id=config.deployment_id,
        execution_lease_ttl_s=5,
    )
    artifacts = FileArtifactStore(config.artifact_store.root)
    adapters = {pool.pool_id: create_adapter(pool) for pool in config.pools}
    handler = _DetachedHandler(block=True)
    manager = DetachedOmniTaskManager(
        handler, config.artifact_store.root, drain_timeout_s=15
    )
    dispatcher_a = VideoDispatcher(
        config,
        store_a,
        artifacts,
        {"fl-pool": _DetachedClient(manager)},
        adapters,
        context_factory=FakeContext,
        generation="detached-gateway-a",
    )
    dispatcher_b = VideoDispatcher(
        config,
        store_b,
        artifacts,
        {"fl-pool": _DetachedClient(manager)},
        adapters,
        context_factory=FakeContext,
        generation="detached-gateway-b",
    )
    service_a = VideoGatewayService(config, store_a, artifacts, dispatcher_a, adapters)
    started_a = False
    started_b = False
    try:
        await dispatcher_a.start()
        started_a = True
        # Submitting through A does not force A to win a shared queue. First
        # establish the intended owner, then start its standby before failure.
        submission = await _submit(service_a, "public-fl")
        await asyncio.wait_for(handler.started.wait(), timeout=2)
        active = await store_b.get_task(submission.stored.task.id)
        assert active is not None
        assert active.task.status == TaskStatus.IN_PROGRESS
        assert active.task.owner_generation == "detached-gateway-a"

        await dispatcher_b.start()
        started_b = True
        await dispatcher_a.stop()
        started_a = False
        handler.release.set()
        terminal = await dispatcher_b.wait_terminal(active.task.id, 12)
        assert terminal.task.status == TaskStatus.COMPLETED
        assert terminal.task.owner_generation == "detached-gateway-b"
        assert handler.calls == 1
    finally:
        handler.release.set()
        if started_b:
            await dispatcher_b.stop()
        if started_a:
            await dispatcher_a.stop()
        await manager.shutdown()
        await _delete_prefix(cleanup_client, prefix)
        await cleanup_client.close()
