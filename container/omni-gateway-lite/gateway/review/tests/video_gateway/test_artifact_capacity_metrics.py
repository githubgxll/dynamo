# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Storage metrics and admission use the same shared PVC capacity observation."""

import errno
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dingo.video_gateway.adapters import create_adapter
from dingo.video_gateway.api import _SERVICE_KEY, metrics
from dingo.video_gateway.artifact_store import FileArtifactStore
from dingo.video_gateway.dispatcher import VideoDispatcher
from dingo.video_gateway.errors import GatewayError
from dingo.video_gateway.service import VideoGatewayService
from dingo.video_gateway.task_store import MemoryTaskStore


@pytest.fixture
def storage_service(make_gateway_config, monkeypatch):
    config = make_gateway_config()
    artifacts = FileArtifactStore(config.artifact_store.root)
    store = MemoryTaskStore()
    adapters = {pool.pool_id: create_adapter(pool) for pool in config.pools}
    dispatcher = VideoDispatcher(
        config,
        store,
        artifacts,
        {pool.pool_id: object() for pool in config.pools},
        adapters,
    )
    service = VideoGatewayService(config, store, artifacts, dispatcher, adapters)
    quota_root = artifacts.root.parent
    usage = {"bytes": 250_000}

    def getxattr(path, name):
        if Path(path) == quota_root:
            if name == "ceph.quota.max_bytes":
                return b"1000000"
            if name == "ceph.dir.rbytes":
                return str(usage["bytes"]).encode()
        raise OSError(errno.ENODATA, "no attribute")

    monkeypatch.setattr(os, "getxattr", getxattr)
    monkeypatch.setattr(Path, "is_mount", lambda path: path == quota_root)
    return service, usage


async def test_metrics_refresh_shared_quota_when_usage_grows_and_shrinks(
    storage_service,
):
    service, usage = storage_service
    request = SimpleNamespace(app={_SERVICE_KEY: service})
    for used in (250_000, 300_000, 100_000):
        usage["bytes"] = used
        response = await metrics(request)
        assert response.status == 200
        assert "dingo_video_artifact_total_bytes 1000000\n" in response.text
        assert f"dingo_video_artifact_free_bytes {1_000_000 - used}\n" in response.text


async def test_admission_accounts_for_other_directories_under_shared_quota(
    storage_service,
):
    service, usage = storage_service
    usage["bytes"] = 900_000
    service.dispatcher.sweep_now = AsyncMock()

    await service.ensure_submission_capacity(50_000)
    with pytest.raises(GatewayError) as raised:
        await service.ensure_submission_capacity(200_000)

    assert raised.value.status == 507
    assert raised.value.code == "insufficient_artifact_storage"
    service.dispatcher.sweep_now.assert_awaited_once()
