# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer

from dingo.video_gateway.app import create_app
from dingo.video_gateway.config import PerformanceCacheConfig, parse_config
from dingo.video_gateway.models import TaskStatus
from dingo.video_gateway.performance_cache import PerformanceCache
from dingo.video_gateway.service import VideoGatewayService
from dingo.video_gateway.telemetry import GatewayTelemetry
from tests.video_gateway.test_config import _raw
from tests.video_gateway.test_task_store import _task


def test_snapshot_is_immutable_and_pagination_has_finite_upper_bound():
    cache = PerformanceCache(PerformanceCacheConfig())
    event = {"value": 1}
    cache.append(event)
    event["value"] = 999
    cache.append({"value": 2})
    first = cache.page(limit=1)
    assert first["records"][0]["event"]["value"] == 1
    assert first["has_more"]
    cache.append({"value": 3})
    second = cache.page(first["next_cursor"], limit=1)
    assert not second["has_more"]
    assert second["records"][0]["event"]["value"] == 2
    third = cache.page(second["next_cursor"])
    assert third["records"][0]["event"]["value"] == 3
    assert cache.page(third["next_cursor"])["records"] == []
    assert len(cache.page()["records"]) == 3  # Reads do not consume data.


def test_expiry_capacity_and_restart_are_explicit():
    clock = [0.0]
    cache = PerformanceCache(
        PerformanceCacheConfig(retention_s=60, max_records=2), clock=lambda: clock[0]
    )
    initial = cache.page()
    for i in range(3):
        cache.append({"i": i})
    page = cache.page(initial["next_cursor"])
    assert page["gap_detected"] and len(page["records"]) == 2
    assert cache.stats()["capacity_dropped_total"] == 1
    clock[0] = 60
    assert cache.stats()["records"] == 0
    assert cache.stats()["expired_total"] == 2
    assert cache.page(initial["next_cursor"])["gap_detected"]
    fresh = PerformanceCache(PerformanceCacheConfig())
    assert fresh.page(page["next_cursor"])["instance_changed"]


def test_default_retention_expires_at_five_minutes():
    clock = [0.0]
    cache = PerformanceCache(PerformanceCacheConfig(), clock=lambda: clock[0])
    cursor = cache.page()["next_cursor"]
    cache.append({"value": 1})
    for timestamp in (60, 299.999):
        clock[0] = timestamp
        page = cache.page(cursor)
        assert page["retention_s"] == 300
        assert len(page["records"]) == 1
        assert not page["gap_detected"]
    clock[0] = 300
    page = cache.page(cursor)
    assert page["records"] == []
    assert page["gap_detected"]
    assert cache.stats()["expired_total"] == 1


def test_byte_capacity_evicts_oldest_before_expiry():
    event = {"value": "example"}
    size = len(json.dumps(event, separators=(",", ":")).encode())
    cache = PerformanceCache(
        PerformanceCacheConfig(max_bytes=2 * size), clock=lambda: 0.0
    )
    for _ in range(3):
        assert cache.append(event)
    page = cache.page()
    assert [item["seq"] for item in page["records"]] == [2, 3]
    assert page["gap_detected"]
    assert cache.stats() == {
        "records": 2,
        "bytes": 2 * size,
        "expired_total": 0,
        "capacity_dropped_total": 1,
    }


def test_byte_limit_oversize_and_bad_cursors():
    cache = PerformanceCache(PerformanceCacheConfig(max_bytes=20))
    assert not cache.append({"value": "x" * 30})
    assert cache.stats()["bytes"] == 0
    assert cache.page()["gap_detected"]
    for cursor in ("bad cursor", "W10=", "bnVsbA==", "" * 1):
        with pytest.raises(ValueError):
            cache.page(cursor)
    for limit in (0, 501, True):
        with pytest.raises(ValueError):
            cache.page(limit=limit)


def test_rate_limit_and_cache_metrics():
    clock = [0.0]
    cache = PerformanceCache(PerformanceCacheConfig(), clock=lambda: clock[0])
    assert all(cache.admit_read() for _ in range(20))
    assert not cache.admit_read()
    clock[0] = 1
    assert cache.admit_read()
    telemetry = GatewayTelemetry()
    telemetry.performance_cache = cache
    assert "dingo_video_performance_cache_records 0" in telemetry.render_prometheus()


def test_cache_is_independent_from_log_flag_and_excludes_private_text(monkeypatch):
    monkeypatch.delenv("DINGO_VIDEO_PERFORMANCE_RECORDS", raising=False)
    telemetry = GatewayTelemetry()
    telemetry.performance_cache = PerformanceCache(PerformanceCacheConfig())
    task = _task("video-cache")
    task.status = TaskStatus.COMPLETED
    task.completed_at_ms = 123456
    task.normalized_request = {"prompt": "sensitive prompt", "seconds": 5}
    telemetry.audit_task(
        "completed", task, gateway_generation="test", reason="sensitive reason"
    )
    result = telemetry.performance_cache.page()
    assert len(result["records"]) == 1
    event = result["records"][0]["event"]
    assert event["timestamp_ms"] == 123456
    assert "performance" in event and "sensitive" not in json.dumps(result)
    telemetry.audit_task("expired", task, gateway_generation="test")
    assert len(telemetry.performance_cache.page()["records"]) == 1


@pytest.mark.parametrize(
    "value",
    [
        {"retention_s": 0},
        {"retention_s": float("nan")},
        {"retention_s": True},
        {"max_bytes": True},
        {"max_records": 0},
        {"token_env": "invalid-name"},
        {"unknown": 1},
        [],
    ],
)
def test_config_is_strict(tmp_path, value):
    raw = _raw(tmp_path)
    raw["performance_cache"] = value
    with pytest.raises((ValueError, TypeError)):
        parse_config(raw)


@pytest.mark.parametrize("section", [None, {}, {"enabled": True}])
def test_cache_config_defaults_match_dataclass(tmp_path, section):
    raw = _raw(tmp_path)
    if section is not None:
        raw["performance_cache"] = section
    config = parse_config(raw).performance_cache
    assert config == PerformanceCacheConfig()
    assert config.retention_s == 300
    assert config.max_records == 50000
    assert config.max_bytes == 64 * 1024 * 1024


def test_cache_explicit_legacy_limits_are_preserved(tmp_path):
    raw = _raw(tmp_path)
    raw["performance_cache"] = {
        "retention_s": 60,
        "max_records": 10000,
        "max_bytes": 16 * 1024 * 1024,
    }
    assert parse_config(raw).performance_cache == PerformanceCacheConfig(
        retention_s=60, max_records=10000, max_bytes=16 * 1024 * 1024
    )


async def test_endpoint_is_authenticated_and_enabled_by_default(
    make_gateway_config, monkeypatch
):
    async def noop():
        pass

    def stack(config):
        dispatcher = SimpleNamespace(
            telemetry=GatewayTelemetry(), start=noop, stop=noop
        )
        return VideoGatewayService(config, None, None, dispatcher, {})

    config = make_gateway_config()
    assert config.performance_cache.enabled
    assert config.performance_cache.retention_s == 300
    assert PerformanceCacheConfig().enabled
    service = stack(
        replace(config, performance_cache=PerformanceCacheConfig(enabled=False))
    )
    async with TestClient(TestServer(create_app(service))) as client:
        assert (await client.get("/internal/v1/performance/records")).status == 404
    config = replace(config, performance_cache=PerformanceCacheConfig(enabled=True))
    service = stack(config)
    monkeypatch.delenv("DINGO_VIDEO_PERFORMANCE_TOKEN", raising=False)
    async with TestClient(TestServer(create_app(service))) as client:
        assert (await client.get("/internal/v1/performance/records")).status == 401
        assert service.performance_cache is not None
    service = stack(config)
    monkeypatch.setenv("DINGO_VIDEO_PERFORMANCE_TOKEN", "short-token")
    with pytest.raises(ValueError, match="token"):
        create_app(service)
    monkeypatch.setenv("DINGO_VIDEO_PERFORMANCE_TOKEN", "test-token-long-enough")
    service.performance_cache.append({"hello": "world"})
    async with TestClient(TestServer(create_app(service))) as client:
        assert (await client.get("/internal/v1/performance/records")).status == 401
        headers = {"Authorization": "Bearer test-token-long-enough"}
        response = await client.get("/internal/v1/performance/records", headers=headers)
        assert (
            response.status == 200 and response.headers["Cache-Control"] == "no-store"
        )
        page = await response.json()
        assert page["records"][0]["event"] == {"hello": "world"}
        assert page["source"]["deployment_id"] == config.deployment_id
        assert (
            await client.post("/internal/v1/performance/records", headers=headers)
        ).status == 405
        for query in ("?limit=0", "?cursor=invalid", "?limit=1&limit=2", "?secret=bad"):
            assert (
                await client.get(
                    "/internal/v1/performance/records" + query, headers=headers
                )
            ).status == 400


async def test_real_gateway_response_is_readable_by_http_puller(
    make_gateway_config, monkeypatch
):
    monkeypatch.syspath_prepend(
        str(
            Path(__file__).resolve().parents[2]
            / "deploy/observability/video-performance"
        )
    )
    pull = importlib.import_module("http_pull")
    monkeypatch.setenv("DINGO_VIDEO_PERFORMANCE_TOKEN", "test-token-long-enough")
    monkeypatch.delenv("DINGO_VIDEO_PERFORMANCE_RECORDS", raising=False)

    async def noop():
        pass

    config = replace(
        make_gateway_config(), performance_cache=PerformanceCacheConfig(enabled=True)
    )
    dispatcher = SimpleNamespace(telemetry=GatewayTelemetry(), start=noop, stop=noop)
    service = VideoGatewayService(config, None, None, dispatcher, {})
    task = _task("video-http-e2e")
    task.deployment_id = config.deployment_id
    task.status = TaskStatus.COMPLETED
    task.completed_at_ms = 1791977739000
    service.telemetry.audit_task("completed", task, gateway_generation="test")
    async with TestClient(TestServer(create_app(service))) as client:
        target = pull.configuration(
            {
                "targets": [
                    {
                        "name": "gateway-test",
                        "cluster_id": "example-region",
                        "namespace": "example-video",
                        "deployment_id": config.deployment_id,
                        "url": str(client.make_url("/")).rstrip("/"),
                        "token_env": "DINGO_VIDEO_PERFORMANCE_TOKEN",
                        "allow_insecure_http": True,
                    }
                ]
            }
        )["targets"][0]
        loop = asyncio.get_running_loop()
        page = await loop.run_in_executor(None, pull.fetch, target, None)
        source = pull.validate_page(target, page)
        row = pull.collector.event_row(page["records"][0]["event"], source)
        assert row["task_id"] == "video-http-e2e"
        assert row["namespace"] == "example-video"
        next_page = await loop.run_in_executor(
            None, pull.fetch, target, page["next_cursor"]
        )
        assert next_page["records"] == []
