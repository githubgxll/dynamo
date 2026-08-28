# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64

from dingo.video_gateway.etcd_http import EtcdHttpClient, EtcdLease, EtcdValue
from dingo.video_gateway.models import TaskStatus, now_ms
from dingo.video_gateway.task_store import EtcdTaskStore
from tests.video_gateway.test_task_store import _task


def _decode(value: str) -> bytes:
    return base64.b64decode(value)


class FakeEtcd:
    """Execute the exact compare/request dictionaries emitted by EtcdTaskStore."""

    def __init__(self) -> None:
        self.values: dict[str, EtcdValue] = {}
        self.revision = 0
        self.range_all_calls: list[tuple[str, int, int]] = []
        self.next_lease_id = 100
        self.native_leases: set[int] = set()
        self.keepalive_calls: list[int] = []

    compare_version = staticmethod(EtcdHttpClient.compare_version)
    compare_mod = staticmethod(EtcdHttpClient.compare_mod)
    compare_value = staticmethod(EtcdHttpClient.compare_value)
    put = staticmethod(EtcdHttpClient.put)
    delete = staticmethod(EtcdHttpClient.delete)
    prefix_end = staticmethod(EtcdHttpClient.prefix_end)

    async def close(self):
        return None

    async def lease_grant(self, ttl, *, lease_id=0):
        assigned = lease_id or self.next_lease_id
        self.next_lease_id = max(self.next_lease_id + 1, assigned + 1)
        self.native_leases.add(assigned)
        return EtcdLease(assigned, ttl, self.revision, granted_ttl=ttl)

    async def lease_keepalive(self, lease_id):
        if lease_id not in self.native_leases:
            raise RuntimeError("unknown lease")
        self.keepalive_calls.append(lease_id)
        return EtcdLease(lease_id, 15, self.revision)

    async def lease_revoke(self, lease_id):
        self.native_leases.discard(lease_id)
        for key, value in list(self.values.items()):
            if value.lease == lease_id:
                del self.values[key]
        self.revision += 1
        return self.revision

    async def get(self, key):
        return self.values.get(key)

    async def range(
        self,
        key,
        *,
        prefix=False,
        limit=0,
        keys_only=False,
        descending=False,
    ):
        matches = [
            value
            for stored_key, value in sorted(
                self.values.items(), reverse=descending
            )
            if stored_key == key or (prefix and stored_key.startswith(key))
        ]
        if limit:
            matches = matches[:limit]
        if keys_only:
            return [
                EtcdValue(
                    item.key, b"", item.create_revision, item.mod_revision, item.version
                )
                for item in matches
            ]
        return matches

    async def range_page(
        self,
        key,
        *,
        prefix=False,
        limit=0,
        keys_only=False,
        count_only=False,
        descending=False,
        revision=0,
        range_end=None,
    ):
        from dingo.video_gateway.etcd_http import EtcdRangePage

        key_bytes = key.encode() if isinstance(key, str) else key
        if prefix:
            range_end = self.prefix_end(key_bytes)
        end_bytes = range_end or key_bytes + b"\0"
        matches = [
            value
            for stored_key, value in sorted(
                self.values.items(), reverse=descending
            )
            if key_bytes <= stored_key.encode() < end_bytes
        ]
        selected = matches[:limit] if limit else matches
        if count_only:
            selected = []
        elif keys_only:
            selected = [
                EtcdValue(
                    item.key,
                    b"",
                    item.create_revision,
                    item.mod_revision,
                    item.version,
                    item.lease,
                )
                for item in selected
            ]
        return EtcdRangePage(
            tuple(selected),
            bool(limit and len(matches) > limit),
            revision or self.revision,
            len(matches),
        )

    async def count_prefix(self, prefix, *, revision=0):
        page = await self.range_page(
            prefix, prefix=True, count_only=True, revision=revision
        )
        return page.count

    async def get_many(self, keys, *, revision=0):
        return [self.values.get(key) for key in keys], revision or self.revision

    async def range_all(
        self,
        key,
        *,
        prefix=False,
        page_size=512,
        keys_only=False,
        revision=0,
    ):
        self.range_all_calls.append((key, page_size, revision))
        matches = await self.range(key, prefix=prefix, keys_only=keys_only)
        return matches, revision or self.revision

    def _compare(self, item):
        key = _decode(item["key"]).decode()
        current = self.values.get(key)
        target = item["target"]
        if target == "VERSION":
            actual = current.version if current else 0
            expected = int(item["version"])
        elif target == "MOD":
            actual = current.mod_revision if current else 0
            expected = int(item["mod_revision"])
        elif target == "VALUE":
            actual = current.value if current else b""
            expected = _decode(item["value"])
        else:  # pragma: no cover - contract guard
            raise AssertionError(target)
        if item["result"] == "EQUAL":
            return actual == expected
        if item["result"] == "GREATER":
            return actual > expected
        raise AssertionError(item["result"])

    async def txn(self, compare, success, failure=()):
        succeeded = all(self._compare(item) for item in compare)
        operations = success if succeeded else failure
        if operations:
            self.revision += 1
        for operation in operations:
            if "request_put" in operation:
                request = operation["request_put"]
                key = _decode(request["key"]).decode()
                value = _decode(request["value"])
                old = self.values.get(key)
                self.values[key] = EtcdValue(
                    key=key,
                    value=value,
                    create_revision=old.create_revision if old else self.revision,
                    mod_revision=self.revision,
                    version=(old.version + 1) if old else 1,
                    lease=int(request.get("lease", 0)),
                )
            elif "request_delete_range" in operation:
                key = _decode(operation["request_delete_range"]["key"]).decode()
                self.values.pop(key, None)
            else:  # pragma: no cover - contract guard
                raise AssertionError(operation)
        return succeeded, self.revision


async def test_etcd_store_create_reserve_cancel_and_reconcile_are_transactional():
    client = FakeEtcd()
    store = EtcdTaskStore(client, prefix="/isolated/video", deployment_id="arbitrary")
    task = _task("video-etcd")
    stored, created = await store.create_task(
        task,
        principal_hash="principal",
        idempotency_hash="key",
        queue_limit=1,
    )

    assert created is True
    assert await store.queue_depth(task.pool_id) == 1
    queue_key = store._queue_key(task.pool_id, task.id)
    client.values.pop(queue_key)
    await store.reconcile_pool(task.pool_id)
    assert queue_key in client.values
    assert await store.queue_depth(task.pool_id) == 1

    from tests.video_gateway.test_task_store import _lease

    reserved = await store.reserve(
        stored, _lease(task), deadline_at_ms=now_ms() + 10_000
    )
    assert reserved is not None
    assert reserved.task.status == TaskStatus.DISPATCHING
    assert reserved.task.worker_lease_id is not None
    lease_record = await client.get(
        store._lease_key(task.pool_id, reserved.task.worker_key)
    )
    assert lease_record is not None
    assert lease_record.lease == reserved.task.worker_lease_id
    await store.heartbeat_lease(
        task.pool_id,
        reserved.task.worker_key,
        task.id,
        reserved.task.worker_lease_id,
    )
    assert client.keepalive_calls == [reserved.task.worker_lease_id]
    assert await store.queue_depth(task.pool_id) == 0

    cancelled_request = await store.request_cancel(task.id)
    assert cancelled_request.task.cancel_requested_at_ms is not None
    cancelled = await store.transition(
        task.id,
        expected={TaskStatus.DISPATCHING},
        expected_revision=cancelled_request.revision,
        patch={"status": TaskStatus.CANCELLED},
        release_lease=True,
    )
    expired = await store.transition(
        task.id,
        expected={TaskStatus.CANCELLED},
        expected_revision=cancelled.revision,
        patch={"status": TaskStatus.EXPIRED},
    )
    assert await store.delete_expired(expired) is True
    assert await store.get_task(task.id) is None
    assert await client.get(store._idempotency_key("principal", "key")) is None


async def test_reconcile_pool_does_not_truncate_after_ten_thousand_tasks():
    client = FakeEtcd()
    store = EtcdTaskStore(client, prefix="/isolated/video", deployment_id="arbitrary")
    pool_id = "pool-a"
    queued_ids: list[str] = []

    for index in range(10_050):
        task = _task(f"video-{index:05d}", pool_id=pool_id)
        if index < 10_000:
            task.status = TaskStatus.FAILED
        else:
            queued_ids.append(task.id)
        client.revision += 1
        task_key = store._task_key(task.id)
        client.values[task_key] = EtcdValue(
            task_key,
            store._encode(task.to_dict()).encode(),
            client.revision,
            client.revision,
            1,
        )
        if task.status == TaskStatus.QUEUED:
            queue_key = store._queue_key(pool_id, task.id)
            client.values[queue_key] = EtcdValue(
                queue_key,
                str(task.queued_at_ms).encode(),
                client.revision,
                client.revision,
                1,
            )

    counter_key = store._counter_key(pool_id)
    client.revision += 1
    client.values[counter_key] = EtcdValue(
        counter_key,
        str(len(queued_ids)).encode(),
        client.revision,
        client.revision,
        1,
    )

    await store.reconcile_pool(pool_id)

    assert await store.queue_depth(pool_id) == len(queued_ids)
    for task_id in queued_ids:
        assert await client.get(store._queue_key(pool_id, task_id)) is not None
    assert any(
        page_size == 512
        for _key, page_size, _revision in client.range_all_calls
    )


async def test_indexed_list_uses_created_sequence_status_and_bounded_pages():
    client = FakeEtcd()
    store = EtcdTaskStore(client, prefix="/isolated/video", deployment_id="indexed")
    created = []
    for task_id in ["video-z", "video-a", "video-m"]:
        stored, was_created = await store.create_task(
            _task(task_id),
            principal_hash="principal",
            idempotency_hash=None,
            queue_limit=10,
        )
        assert was_created is True
        created.append(stored)

    assert [stored.task.created_seq for stored in created] == [1, 2, 3]
    ascending = await store.list_tasks(limit=10)
    descending = await store.list_tasks(limit=10, descending=True)
    after = await store.list_tasks(after="video-z", limit=10)

    assert [stored.task.id for stored in ascending] == [
        "video-z",
        "video-a",
        "video-m",
    ]
    assert [stored.task.id for stored in descending] == [
        "video-m",
        "video-a",
        "video-z",
    ]
    assert [stored.task.id for stored in after] == ["video-a", "video-m"]

    failed = await store.transition(
        created[1].task.id,
        expected={TaskStatus.QUEUED},
        expected_revision=created[1].revision,
        patch={"status": TaskStatus.FAILED},
    )
    assert failed.task.status == TaskStatus.FAILED
    assert [
        stored.task.id
        for stored in await store.list_tasks(status=TaskStatus.QUEUED, limit=10)
    ] == ["video-z", "video-m"]
    assert [
        stored.task.id
        for stored in await store.list_tasks(status=TaskStatus.FAILED, limit=10)
    ] == ["video-a"]
    counts = await store.task_counts("pool-a")
    assert counts[TaskStatus.QUEUED] == 2
    assert counts[TaskStatus.FAILED] == 1


async def test_prepare_backfills_legacy_task_indexes_and_sequence():
    client = FakeEtcd()
    store = EtcdTaskStore(client, prefix="/isolated/video", deployment_id="legacy")
    legacy = _task("video-legacy")
    legacy.created_seq = 0
    client.revision += 1
    task_key = store._task_key(legacy.id)
    client.values[task_key] = EtcdValue(
        task_key,
        store._encode(legacy.to_dict()).encode(),
        client.revision,
        client.revision,
        1,
    )
    queue_key = store._queue_key(legacy.pool_id, legacy.id)
    client.values[queue_key] = EtcdValue(
        queue_key,
        str(legacy.queued_at_ms).encode(),
        client.revision,
        client.revision,
        1,
    )
    counter_key = store._counter_key(legacy.pool_id)
    client.values[counter_key] = EtcdValue(
        counter_key,
        b"1",
        client.revision,
        client.revision,
        1,
    )

    await store.prepare()

    migrated = await store.get_task(legacy.id)
    assert migrated is not None and migrated.task.created_seq == 1
    assert await client.get(store._index_schema_key()) is not None
    assert await client.get(store._ordered_queue_key(migrated.task)) is not None
    assert [
        stored.task.id for stored in await store.list_queued(legacy.pool_id)
    ] == [legacy.id]


async def test_reconcile_repairs_missing_queued_status_index():
    client = FakeEtcd()
    store = EtcdTaskStore(client, prefix="/isolated/video", deployment_id="repair")
    stored, created = await store.create_task(
        _task("video-missing-status-index"),
        principal_hash="principal",
        idempotency_hash=None,
        queue_limit=8,
    )
    assert created is True
    await store.prepare()

    status_key = store._task_index_key(
        stored.task,
        pool_id=stored.task.pool_id,
        status=TaskStatus.QUEUED,
    )
    client.values.pop(status_key)

    await store.reconcile_pool(stored.task.pool_id)

    assert await client.get(status_key) is not None
    assert [
        item.task.id for item in await store.list_queued(stored.task.pool_id)
    ] == [stored.task.id]
    assert await store.queue_depth(stored.task.pool_id) == 1


async def test_due_task_query_uses_ordered_expiry_index_and_limit():
    client = FakeEtcd()
    store = EtcdTaskStore(client, prefix="/isolated/video", deployment_id="expiry")
    current = now_ms()
    tasks = []
    for index, expires_at_ms in enumerate(
        (current - 1_000, current - 2_000, current + 1_000)
    ):
        stored, created = await store.create_task(
            _task(f"video-expiry-{index}"),
            principal_hash="principal",
            idempotency_hash=None,
            queue_limit=8,
        )
        assert created is True
        tasks.append(
            await store.transition(
                stored.task.id,
                expected={TaskStatus.QUEUED},
                expected_revision=stored.revision,
                patch={"expires_at_ms": expires_at_ms},
            )
        )

    first = await store.list_due_tasks(current, limit=1)
    all_due = await store.list_due_tasks(current, limit=8)

    assert [stored.task.id for stored in first] == [tasks[1].task.id]
    assert [stored.task.id for stored in all_due] == [
        tasks[1].task.id,
        tasks[0].task.id,
    ]
