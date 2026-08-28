# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from dingo.video_gateway.etcd_http import EtcdHttpClient
from dingo.video_gateway.models import TaskStatus
from dingo.video_gateway.task_store import EtcdTaskStore
from tests.video_gateway.test_task_store import _lease, _task

_ETCD_URL = os.environ.get("DINGO_VIDEO_TEST_ETCD_URL")


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
                [
                    client.put(key, value)
                    for key, value in items[offset : offset + 100]
                ],
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
        descending = await client.range(
            prefix, prefix=True, limit=2, descending=True
        )
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
        reserved = await store.reserve(
            stored,
            _lease(stored.task),
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
