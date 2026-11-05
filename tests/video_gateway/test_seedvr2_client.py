# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import asynccontextmanager

import pytest
from aiohttp import web

from dingo.vllm.omni.seedvr2_client import ROUTE, SeedVR2Client, SubmissionUnknown

JOB = "a" * 32


@asynccontextmanager
async def server(handler):
    app = web.Application()
    app.router.add_route("*", "/{path:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield SeedVR2Client(port)
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", [None, 0, 2**32-1])
async def test_native_multipart_preserves_seed_omission(tmp_path, seed):
    requests = []

    async def handler(request):
        assert request.path == ROUTE and request.method == "POST"
        form = await request.post()
        requests.append(dict(form))
        assert form["input_references"].file.read() == b"base-video"
        return web.json_response({"id": JOB, "status": "queued"}, status=202)

    source = tmp_path / "base.mp4"
    source.write_bytes(b"base-video")
    async with server(handler) as client:
        assert await client.submit(source, width=2048, height=1152, frames=122, seed=seed) == JOB
    assert len(requests) == 1
    assert requests[0]["loop_input"] == "false"
    assert ("seed" in requests[0]) == (seed is not None)
    if seed is not None:
        assert requests[0]["seed"] == str(seed)


@pytest.mark.asyncio
async def test_delete_ack_is_not_completion():
    calls = []
    states = iter(["running", "running", "cancelled"])

    async def handler(request):
        calls.append(request.method)
        if request.method == "DELETE":
            return web.json_response({"id": JOB, "status": "cancelling"}, status=202)
        return web.json_response({"status": next(states), "frames": 2})

    async with server(handler) as client:
        state = await client.cancel_and_wait(JOB, settle_timeout_s=1, poll_s=0.001)
    assert state["status"] == "cancelled"
    assert calls == ["GET", "DELETE", "GET", "GET"]


@pytest.mark.asyncio
async def test_delete_completion_race_is_rechecked():
    calls = []

    async def handler(request):
        calls.append(request.method)
        if request.method == "DELETE":
            return web.Response(status=409)
        return web.json_response({"status": "running" if len(calls) == 1 else "completed", "frames": 122})

    async with server(handler) as client:
        state = await client.cancel_and_wait(JOB, settle_timeout_s=1, poll_s=0.001)
    assert state["status"] == "completed"
    assert calls == ["GET", "DELETE", "GET"]


@pytest.mark.asyncio
async def test_post_redirect_is_not_followed_or_retried(tmp_path):
    calls = []

    async def handler(request):
        calls.append(request.path)
        return web.Response(status=307, headers={"Location": "/elsewhere"})

    source = tmp_path / "base.mp4"
    source.write_bytes(b"base")
    async with server(handler) as client:
        with pytest.raises(SubmissionUnknown):
            await client.submit(source, width=2048, height=1152, frames=122)
    assert calls == [ROUTE]


@pytest.mark.asyncio
async def test_chunked_status_response_is_complete():
    async def handler(request):
        response = web.StreamResponse(headers={"Content-Type": "application/json"})
        await response.prepare(request)
        await response.write(b'{"status":')
        await response.write(b'"completed","frames":122}')
        await response.write_eof()
        return response

    async with server(handler) as client:
        assert (await client.status(JOB))["frames"] == 122


@pytest.mark.asyncio
async def test_download_limit_and_create_only(tmp_path):
    async def handler(request):
        return web.Response(body=b"12345678", content_type="video/mp4")

    output = tmp_path / "raw.mp4"
    async with server(handler) as client:
        with pytest.raises(RuntimeError, match="size limit"):
            await client.download(JOB, output, max_bytes=4)
        with pytest.raises(FileExistsError):
            await client.download(JOB, output)


@pytest.mark.asyncio
async def test_cancel_timeout_does_not_claim_success():
    async def handler(request):
        if request.method == "DELETE":
            return web.json_response({"status": "cancelling"}, status=202)
        return web.json_response({"status": "running", "frames": 2})

    async with server(handler) as client:
        with pytest.raises(TimeoutError):
            await client.cancel_and_wait(JOB, settle_timeout_s=0.02, poll_s=0.001)
