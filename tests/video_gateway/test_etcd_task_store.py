# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64

from dingo.video_gateway.etcd_http import EtcdHttpClient, EtcdValue
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

    compare_version = staticmethod(EtcdHttpClient.compare_version)
    compare_mod = staticmethod(EtcdHttpClient.compare_mod)
    compare_value = staticmethod(EtcdHttpClient.compare_value)
    put = staticmethod(EtcdHttpClient.put)
    delete = staticmethod(EtcdHttpClient.delete)

    async def close(self):
        return None

    async def get(self, key):
        return self.values.get(key)

    async def range(self, key, *, prefix=False, limit=0, keys_only=False):
        matches = [
            value
            for stored_key, value in sorted(self.values.items())
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
