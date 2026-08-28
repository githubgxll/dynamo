# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import aiohttp
from aiohttp.test_utils import TestClient, TestServer

from dingo.video_gateway.app import create_app
from tests.video_gateway.test_dispatcher import FakeClient, _stack


def _form(*, fps="24", model="public-fl"):
    form = aiohttp.MultipartWriter("form-data")
    for name, value in (
        ("model", model),
        ("prompt", "an API integration test"),
        ("seconds", "5"),
        ("size", "1344x768"),
        ("fps", fps),
        ("seed", "55"),
    ):
        if value is None:
            continue
        part = form.append(value, {"Content-Type": "text/plain; charset=utf-8"})
        part.set_content_disposition("form-data", name=name)
    return form


async def _client(make_gateway_config, endpoint_client=None, *, media=None):
    config = make_gateway_config(media=media)
    endpoint_client = endpoint_client or FakeClient()
    _store, _artifacts, _dispatcher, service = _stack(
        config, {"fl-pool": endpoint_client}
    )
    client = TestClient(TestServer(create_app(service)))
    await client.start_server()
    return client


async def _compat_client(make_gateway_config, endpoint_client=None):
    config = make_gateway_config(
        http={
            "host": "127.0.0.1",
            "port": 18000,
            "sync_timeout_s": 2,
            "default_model": "public-fl",
            "async_submit_status_code": 200,
        }
    )
    endpoint_client = endpoint_client or FakeClient()
    _store, _artifacts, _dispatcher, service = _stack(
        config, {"fl-pool": endpoint_client}
    )
    client = TestClient(TestServer(create_app(service)))
    await client.start_server()
    return client


async def _wait_completed(client: TestClient, task_id: str):
    for _ in range(200):
        response = await client.get(f"/v1/videos/{task_id}")
        payload = await response.json()
        if payload["status"] in {"completed", "failed", "cancelled"}:
            return response, payload
        await asyncio.sleep(0.01)
    raise AssertionError(f"task {task_id} did not reach a terminal state")


async def test_async_submit_poll_head_range_and_delete(make_gateway_config):
    client = await _client(make_gateway_config)
    try:
        response = await client.post(
            "/v1/videos", data=_form(), headers={"Idempotency-Key": "api-1"}
        )
        submitted = await response.json()

        assert response.status == 202
        assert submitted["status"] == "queued"
        assert response.headers["Location"] == f"/v1/videos/{submitted['id']}"

        _status_response, completed = await _wait_completed(client, submitted["id"])
        assert completed["status"] == "completed"
        assert completed["media_type"] == "video/mp4"
        assert completed["size"] == "1344x768"
        assert completed["metrics"]["queue_wait_s"] is not None

        head = await client.head(f"/v1/videos/{submitted['id']}/content")
        ranged = await client.get(
            f"/v1/videos/{submitted['id']}/content",
            headers={"Range": "bytes=0-3"},
        )
        assert head.status == 200
        assert head.headers["Content-Type"] == "video/mp4"
        assert head.headers["Accept-Ranges"] == "bytes"
        assert head.headers["ETag"] == f'"sha256-{completed["sha256"]}"'
        assert await head.read() == b""
        assert ranged.status == 206
        assert ranged.headers["Content-Range"].startswith("bytes 0-3/")
        assert await ranged.read() == b"\x00\x00\x00\x18"

        metrics = await client.get("/metrics")
        metrics_text = await metrics.text()
        assert "dingo_video_media_memory_used_bytes 0" in metrics_text
        assert "dingo_video_media_payload_build_total 1" in metrics_text
        assert "dingo_video_media_finalize_total 1" in metrics_text
        assert "dingo_video_process_rss_bytes " in metrics_text

        not_modified = await client.get(
            f"/v1/videos/{submitted['id']}/content",
            headers={"If-None-Match": head.headers["ETag"]},
        )
        invalid_range = await client.get(
            f"/v1/videos/{submitted['id']}/content",
            headers={"Range": "bytes=0-1,4-5"},
        )
        assert not_modified.status == 304
        assert await not_modified.read() == b""
        assert invalid_range.status == 416
        assert invalid_range.headers["Content-Range"].startswith("bytes */")
        assert (await invalid_range.json())["error"]["code"] == (
            "range_not_satisfiable"
        )

        deleted, duplicate_delete = await asyncio.gather(
            client.delete(f"/v1/videos/{submitted['id']}"),
            client.delete(f"/v1/videos/{submitted['id']}"),
        )
        deleted_payload = await deleted.json()
        duplicate_payload = await duplicate_delete.json()
        expired = await client.get(f"/v1/videos/{submitted['id']}")
        expired_payload = await expired.json()
        assert deleted_payload == {
            "id": submitted["id"],
            "object": "video.deleted",
            "deleted": True,
        }
        assert duplicate_delete.status == 200
        assert duplicate_payload == deleted_payload
        assert expired_payload["status"] == "expired"
        gone = await client.get(f"/v1/videos/{submitted['id']}/content")
        assert gone.status == 410
    finally:
        await client.close()


async def test_vllm_omni_health_default_model_and_submit_status(make_gateway_config):
    client = await _compat_client(make_gateway_config)
    try:
        health = await client.get("/health")
        form = _form(model=None)
        # Native vLLM-Omni deployments represent one checkpoint and their
        # clients therefore omit model. The configured default supplies it.
        response = await client.post("/v1/videos", data=form)
        submitted = await response.json()

        assert health.status == 200
        assert (await health.json())["status"] == "ready"
        assert response.status == 200
        assert submitted["model"] == "public-fl"

        _status_response, completed = await _wait_completed(client, submitted["id"])
        assert completed["inference_time_s"] >= 0
        assert completed["stage_durations"]["queue_wait"] >= 0
        assert completed["stage_durations"]["finalize"] >= 0
    finally:
        await client.close()


async def test_sync_uses_same_task_pipeline_and_returns_mp4(make_gateway_config):
    client = await _client(make_gateway_config)
    try:
        response = await client.post("/v1/videos/sync", data=_form())
        payload = await response.read()

        assert response.status == 200
        assert response.headers["Content-Type"] == "video/mp4"
        assert response.headers["X-Video-Id"].startswith("video-")
        assert payload.startswith(b"\x00\x00\x00\x18ftyp")
    finally:
        await client.close()


async def test_invalid_fps_and_unimplemented_stream_return_stable_errors(
    make_gateway_config,
):
    client = await _client(make_gateway_config)
    try:
        invalid = await client.post("/v1/videos", data=_form(fps="30"))
        invalid_payload = await invalid.json()
        stream = await client.post("/v1/videos/stream", data=_form())
        stream_payload = await stream.json()

        assert invalid.status == 400
        assert invalid_payload["error"]["code"] == "invalid_fps"
        assert stream.status == 404
        assert stream_payload["error"]["code"] == "unsupported_endpoint"
    finally:
        await client.close()


async def test_multipart_parser_uses_effective_media_config(make_gateway_config):
    text_limited = await _client(
        make_gateway_config,
        media={"max_text_field_bytes": 8},
    )
    try:
        response = await text_limited.post("/v1/videos", data=_form())
        payload = await response.json()
        assert response.status == 413
        assert payload["error"]["code"] == "field_too_large"
    finally:
        await text_limited.close()

    part_limited = await _client(make_gateway_config, media={"max_parts": 5})
    try:
        response = await part_limited.post("/v1/videos", data=_form())
        payload = await response.json()
        assert response.status == 413
        assert payload["error"]["code"] == "too_many_parts"
    finally:
        await part_limited.close()


async def test_idempotency_replay_does_not_create_second_task(make_gateway_config):
    endpoint_client = FakeClient()
    client = await _client(make_gateway_config, endpoint_client)
    try:
        first = await client.post(
            "/v1/videos", data=_form(), headers={"Idempotency-Key": "same"}
        )
        endpoint_client.available = False
        for _ in range(100):
            model_response = await client.get("/v1/models")
            if not (await model_response.json())["data"][0]["available"]:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("fake Worker did not disappear from discovery")
        second = await client.post(
            "/v1/videos", data=_form(), headers={"Idempotency-Key": "same"}
        )
        first_payload = await first.json()
        second_payload = await second.json()

        assert first.status == 202
        assert second.status == 200
        assert second_payload["id"] == first_payload["id"]
        listing = await client.get("/v1/videos")
        listing_payload = await listing.json()
        assert [item["id"] for item in listing_payload["data"]] == [first_payload["id"]]
    finally:
        await client.close()


async def test_sync_and_async_requests_share_one_worker_lease(make_gateway_config):
    endpoint_client = FakeClient(block=True)
    client = await _client(make_gateway_config, endpoint_client)
    try:
        asynchronous = await client.post("/v1/videos", data=_form())
        async_payload = await asynchronous.json()
        while not endpoint_client.calls:
            await asyncio.sleep(0.01)

        sync_request = asyncio.create_task(client.post("/v1/videos/sync", data=_form()))
        await asyncio.sleep(0.05)

        assert endpoint_client.active == 1
        assert endpoint_client.max_active == 1
        assert len(endpoint_client.calls) == 1

        endpoint_client.release.set()
        sync_response = await asyncio.wait_for(sync_request, timeout=2)
        _, async_completed = await _wait_completed(client, async_payload["id"])

        assert sync_response.status == 200
        assert (await sync_response.read()).startswith(b"\x00\x00\x00\x18ftyp")
        assert async_completed["status"] == "completed"
        assert endpoint_client.max_active == 1
        assert len(endpoint_client.calls) == 2
    finally:
        await client.close()


async def test_ready_without_workers_but_submission_fails_fast(make_gateway_config):
    endpoint_client = FakeClient(available=False)
    client = await _client(make_gateway_config, endpoint_client)
    try:
        ready_response = await client.get("/ready")
        submission = await client.post("/v1/videos", data=_form())

        assert ready_response.status == 200
        assert submission.status == 503
        assert (await submission.json())["error"]["code"] == "no_worker_available"
        assert endpoint_client.calls == []
    finally:
        await client.close()
