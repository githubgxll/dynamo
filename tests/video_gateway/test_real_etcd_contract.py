# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import uuid

import pytest

from dingo.video_gateway.etcd_http import EtcdHttpClient
from dingo.video_gateway.models import TaskStatus
from dingo.video_gateway.task_store import EtcdTaskStore
from tests.video_gateway.test_task_store import _task

_ETCD_URL = os.environ.get("DINGO_VIDEO_TEST_ETCD_URL")


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
        assert duplicate_created is False
        assert duplicate.task.id == original.task.id
        assert await store.queue_depth(original.task.pool_id) == 1

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
        values = await client.range(prefix, prefix=True)
        if values:
            await client.txn([], [client.delete(value.key) for value in values])
        await client.close()
