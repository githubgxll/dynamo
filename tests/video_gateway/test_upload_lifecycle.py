import asyncio
import dataclasses
import os
import threading
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer

from dingo.video_gateway import api
from dingo.video_gateway.app import create_app
from dingo.video_gateway.artifact_store import UPLOAD_HEARTBEAT_NAME, FileArtifactStore
from dingo.video_gateway.errors import GatewayError
from dingo.video_gateway.file_io import run_file_io
from tests.video_gateway.test_dispatcher import FakeClient, _stack

_REAL_TO_THREAD = asyncio.to_thread


def stack(make_gateway_config):
    config = make_gateway_config()
    config = dataclasses.replace(
        config, lifecycle=dataclasses.replace(config.lifecycle, upload_grace_s=0.3)
    )
    return _stack(config, {"fl-pool": FakeClient()})


def pending_heartbeats():
    return [t for t in asyncio.all_tasks() if t.get_name() == "video-upload-heartbeat"]


async def test_real_slow_multipart_survives_other_gateway_cleanup(make_gateway_config):
    _, artifacts, _, service = stack(make_gateway_config)
    cleaner = FileArtifactStore(artifacts.root)
    client = TestClient(TestServer(create_app(service)))
    await client.start_server()

    async def body():
        yield b'--slow\r\nContent-Disposition: form-data; name="prompt"\r\n\r\n'
        for _ in range(12):
            await asyncio.sleep(0.05)
            for upload in artifacts.upload_root.iterdir():
                if upload.is_dir():
                    os.utime(upload, (0, 0))
            assert await cleaner.cleanup_orphan_uploads(minimum_age_s=0.3) == 0
            yield b"a slow request "
        yield b"\r\n"
        for key, value in [
            ("model", "public-fl"),
            ("seconds", "5"),
            ("size", "1344x768"),
        ]:
            yield (
                f'--slow\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n'
                f"{value}\r\n"
            ).encode()
        yield b"--slow--\r\n"

    try:
        response = await client.post(
            "/v1/videos",
            data=body(),
            headers={"Content-Type": "multipart/form-data; boundary=slow"},
        )
        assert response.status == 202, await response.text()
        assert not list(artifacts.upload_root.iterdir())
        assert not pending_heartbeats()
    finally:
        await client.close()


@pytest.mark.parametrize("phase", ["parsing", "validation"])
async def test_cancel_submission_stops_heartbeat_and_discards_staging(
    make_gateway_config, monkeypatch, phase
):
    _, artifacts, _, service = stack(make_gateway_config)
    entered = asyncio.Event()

    async def parse(*args, upload_root, **kwargs):
        if phase == "parsing":
            entered.set()
            await asyncio.Event().wait()
        return SimpleNamespace(fields={}, uploads=[], upload_root=upload_root)

    async def submit(**kwargs):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(api, "parse_multipart", parse)
    monkeypatch.setattr(service, "submit", submit)
    request = SimpleNamespace(
        app={api._SERVICE_KEY: service}, content_length=10, headers={}
    )
    operation = asyncio.create_task(api._submit(request, delivery_mode="async"))
    await asyncio.wait_for(entered.wait(), 1)
    operation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await operation
    assert not pending_heartbeats()
    assert not list(artifacts.upload_root.iterdir())


async def test_heartbeat_error_aborts_upload_instead_of_leaving_unprotected_request(
    make_gateway_config, monkeypatch
):
    _, artifacts, _, service = stack(make_gateway_config)
    unwound = asyncio.Event()

    async def parse(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            unwound.set()

    async def failed_heartbeat(*args):
        await asyncio.sleep(0)
        raise OSError("injected storage failure")

    monkeypatch.setattr(api, "parse_multipart", parse)
    monkeypatch.setattr(api, "_keep_upload_active", failed_heartbeat)
    request = SimpleNamespace(
        app={api._SERVICE_KEY: service}, content_length=10, headers={}
    )
    with pytest.raises(GatewayError) as raised:
        await api._submit(request, delivery_mode="async")
    assert raised.value.status == 503 and raised.value.code == "upload_heartbeat_failed"
    assert unwound.is_set() and not pending_heartbeats()
    assert not list(artifacts.upload_root.iterdir())


async def test_cleanup_failure_does_not_keep_request_heartbeat_alive(
    make_gateway_config, monkeypatch
):
    _, artifacts, _, service = stack(make_gateway_config)
    real_discard = artifacts.discard

    async def parse(*args, **kwargs):
        raise ValueError("bad multipart")

    async def failed_discard(*args, **kwargs):
        raise OSError("unlink temporarily unavailable")

    monkeypatch.setattr(api, "parse_multipart", parse)
    monkeypatch.setattr(artifacts, "discard", failed_discard)
    request = SimpleNamespace(
        app={api._SERVICE_KEY: service}, content_length=10, headers={}
    )
    with pytest.raises(OSError):
        await api._submit(request, delivery_mode="async")
    assert not pending_heartbeats()
    [residual] = list(artifacts.upload_root.iterdir())
    os.utime(residual, (0, 0))
    os.utime(residual / UPLOAD_HEARTBEAT_NAME, (0, 0))
    monkeypatch.setattr(artifacts, "discard", real_discard)
    assert (
        await FileArtifactStore(artifacts.root).cleanup_orphan_uploads(minimum_age_s=60)
        == 1
    )


async def test_repeated_cancel_drains_file_work_before_unlink(
    make_gateway_config, monkeypatch
):
    monkeypatch.setattr(asyncio, "to_thread", _REAL_TO_THREAD)
    _, artifacts, _, service = stack(make_gateway_config)
    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()

    def file_work():
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(3)

    async def parse(*args, **kwargs):
        await run_file_io(file_work)

    monkeypatch.setattr(api, "parse_multipart", parse)
    request = SimpleNamespace(
        app={api._SERVICE_KEY: service}, content_length=10, headers={}
    )
    operation = asyncio.create_task(api._submit(request, delivery_mode="async"))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        operation.cancel()
        await asyncio.sleep(0.02)
        operation.cancel()
        await asyncio.sleep(0.02)
        assert not operation.done()
        assert len(list(artifacts.upload_root.iterdir())) == 1
    finally:
        release.set()
        await asyncio.gather(operation, return_exceptions=True)
    assert not pending_heartbeats()
    assert not list(artifacts.upload_root.iterdir())
