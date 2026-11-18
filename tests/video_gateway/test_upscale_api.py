# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public-model independent-upscale routing and strict HTTP input contract."""

import aiohttp
import pytest
from aiohttp.test_utils import TestClient, TestServer

from dingo.video_gateway.app import create_app
from dingo.video_gateway.config import parse_config
from tests.video_gateway.test_config import _raw
from tests.video_gateway.test_dispatcher import FakeClient, _stack
from tests.video_gateway.test_seedvr2_adapter import video


def raw_config(tmp_path):
    raw = _raw(tmp_path / "artifacts")
    raw["pools"][0]["served_models"] = ["minimax-h3"]
    raw["pools"].append({
        "pool_id": "sr-pool", "served_models": ["internal-sr"],
        "backend_model": "seedvr2-backend", "backend_target": "dyn://sr.backend.generate",
        "execution_mode": "detached", "adapter": {
            "name": "seedvr2", "workflow": "sr_only", "storage_id": "test-store"}})
    raw["upscale_routes"] = {"minimax-h3": "sr-pool"}
    return raw


def upscale_source(tmp_path, *, size=(1344, 768), frames=107, audio=True):
    from fractions import Fraction
    from types import SimpleNamespace

    import av
    import numpy as np

    source = tmp_path / "upscale-source.mp4"
    with av.open(str(source), "w", format="mp4") as media:
        video = media.add_stream("libx264", rate=24)
        video.width, video.height, video.pix_fmt = *size, "yuv420p"
        video.options = {"preset": "ultrafast"}
        video.codec_context.thread_count = 2
        sound = media.add_stream("aac", rate=48000) if audio else None
        if sound:
            sound.layout = "mono"
        image = np.zeros((size[1] * 3 // 2, size[0]), dtype=np.uint8)
        image[size[1]:] = 128
        for index in range(frames):
            frame = av.VideoFrame.from_ndarray(image, format="yuv420p")
            frame.pts, frame.time_base = index, Fraction(1, 24)
            media.mux(video.encode(frame))
        media.mux(video.encode())
        if sound:
            samples = av.AudioFrame.from_ndarray(
                np.zeros((1, frames * 2000), dtype=np.float32), format="fltp", layout="mono"
            )
            samples.pts, samples.time_base, samples.sample_rate = 0, Fraction(1, 48000), 48000
            media.mux(sound.encode(samples))
            media.mux(sound.encode())
    return SimpleNamespace(path=source)


def form(source, *, model="minimax-h3", resolution="1080P", file_field="base_video", extra=None, files=1):
    data = aiohttp.FormData(default_to_multipart=True)
    if model is not None:
        data.add_field("model", model)
    if resolution is not None:
        data.add_field("resolution", resolution)
    for key, value in extra or []:
        data.add_field(key, value)
    for _ in range(files):
        data.add_field(file_field, source.path.read_bytes(), filename="source.mp4", content_type="video/mp4")
    return data


def stack(tmp_path, monkeypatch, *, sr_available=True):
    config = parse_config(raw_config(tmp_path))
    clients = {p.pool_id: FakeClient(available=False) for p in config.pools}
    store, artifacts, dispatcher, service = _stack(config, clients)
    # Keep real task admission/persistence, while modelling a ready SR Worker and no base Worker.
    monkeypatch.setattr(dispatcher, "has_sr_admission_workers", lambda pid, request: pid == "sr-pool" and sr_available)
    monkeypatch.setattr(dispatcher, "has_admission_workers", lambda pid: pid == "sr-pool" and sr_available)
    return store, service, clients


def test_upscale_route_parsing_is_optional_and_does_not_change_generation(tmp_path):
    raw = raw_config(tmp_path)
    config = parse_config(raw)
    assert config.upscale_routes == {"minimax-h3": "sr-pool"}
    assert config.pool_for_model("minimax-h3").pool_id == "first-pool"
    assert parse_config({k: v for k, v in raw.items() if k != "upscale_routes"}).upscale_routes == {}


@pytest.mark.parametrize("routes", [None, [], {"missing-model": "sr-pool"},
    {"internal-sr": "sr-pool"}, {"minimax-h3": "missing-pool"},
    {"minimax-h3": "first-pool"}, {"minimax-h3": []}])
def test_invalid_upscale_routes_rejected(tmp_path, routes):
    raw = raw_config(tmp_path)
    raw["upscale_routes"] = routes
    with pytest.raises((TypeError, ValueError)):
        parse_config(raw)


async def test_public_model_routes_to_sr_without_base_worker_and_is_idempotent(tmp_path, monkeypatch):
    store, service, clients = stack(tmp_path, monkeypatch)
    source = upscale_source(tmp_path)
    async with TestClient(TestServer(create_app(service))) as client:
        response = await client.post("/v1/videos/regenerate", data=form(source), headers={"Idempotency-Key": "same-source"})
        body = await response.json()
        assert response.status == 202, body
        assert body["model"] == "minimax-h3" and body["operation"] == "sr_only"
        assert body["task"] is None and "seed" not in body
        assert body["id"].startswith("video-")
        assert response.headers["Location"] == "/v1/videos/" + body["id"]
        stored = await store.get_task(body["id"])
        assert stored.task.pool_id == "sr-pool"
        assert "_pipeline" not in stored.task.normalized_request
        replay = await client.post("/v1/videos/regenerate", data=form(source), headers={"Idempotency-Key": "same-source"})
        assert replay.status == 200 and (await replay.json())["id"] == body["id"]
        changed = await client.post("/v1/videos/regenerate", data=form(source, resolution="2K"), headers={"Idempotency-Key": "same-source"})
        assert changed.status == 409
        queried = await (await client.get("/v1/videos/" + body["id"])).json()
        assert queried["model"] == "minimax-h3" and queried["operation"] == "sr_only"
        assert all(not clients[p].calls for p in ["first-pool", "second-pool"])


@pytest.mark.parametrize("options", [
    {"model": None}, {"model": ""}, {"model": "internal-sr"}, {"model": "alias-b"},
    {"resolution": None}, {"resolution": "768P"},
    {"file_field": "input_reference"}, {"file_field": "input_references"},
    {"files": 2}, {"files": 0},
    {"extra": [("operation", "sr_only")]}, {"extra": [("prompt", "hello")]},
    {"extra": [("seed", "1")]}, {"extra": [("model", "minimax-h3")]},
    {"extra": [("resolution", "2K")]}, {"extra": [("base_video", "http://example.invalid/source.mp4")]},
])
async def test_upscale_input_contract(tmp_path, monkeypatch, options):
    store, service, _ = stack(tmp_path, monkeypatch)
    source = video(tmp_path)
    async with TestClient(TestServer(create_app(service))) as client:
        response = await client.post("/v1/videos/regenerate", data=form(source, **options))
        assert response.status == 400, await response.text()
        assert (await (await client.get("/v1/videos")).json())["data"] == []
        assert not list(service.artifacts.root.glob("_uploads/*/inputs/*.bin"))


async def test_old_creation_routes_reject_independent_sr(tmp_path, monkeypatch):
    _, service, _ = stack(tmp_path, monkeypatch)
    source = video(tmp_path)
    async with TestClient(TestServer(create_app(service))) as client:
        for route in ["/v1/videos", "/v1/videos/sync"]:
            response = await client.post(route, data=form(source, model="internal-sr", file_field="input_references", extra=[("operation", "sr_only")]))
            assert response.status == 400
            assert (await response.json())["error"]["code"] == "unsupported_operation"


async def test_no_sr_worker_does_not_create_task(tmp_path, monkeypatch):
    _, service, _ = stack(tmp_path, monkeypatch, sr_available=False)
    async with TestClient(TestServer(create_app(service))) as client:
        response = await client.post("/v1/videos/regenerate", data=form(upscale_source(tmp_path)))
        assert response.status == 503
        assert (await (await client.get("/v1/videos")).json())["data"] == []


async def test_models_advertises_sr_availability_independently(tmp_path, monkeypatch):
    _, service, _ = stack(tmp_path, monkeypatch)
    async with TestClient(TestServer(create_app(service))) as client:
        data = (await (await client.get("/v1/models")).json())["data"]
        assert "internal-sr" not in [x["id"] for x in data]
        model = next(x for x in data if x["id"] == "minimax-h3")
        assert model["available"] is False  # No FL2VA/Ref2VA registration required for SR.
        assert model["video_capabilities"]["regeneration"]["available"] is True
        upscale = model["video_capabilities"]["regeneration"]
        assert upscale["endpoint"] == "/v1/videos/regenerate"
        assert upscale["max_video_bytes"] == service.config.media.max_single_file_bytes == 50 * 1024**2
        assert (upscale["min_source_frames"], upscale["max_source_frames"], upscale["source_frame_step"]) == (107, 362, 17)
        assert upscale["audio_required"] is True


@pytest.mark.parametrize("resolution,delivery,internal", [
    ("1080P", (2208, 936), (2256, 960)),
    ("2K", (2944, 1248), (3008, 1280)),
])
async def test_upscale_prepares_source_for_existing_worker(tmp_path, monkeypatch, resolution, delivery, internal):
    import av
    import hashlib
    import importlib.util
    import json
    from pathlib import Path
    from dingo.common.video_sr_only import SOURCE_KEY
    from dingo.common.video_super_resolution import SR_PAYLOAD_KEY

    def audio_packets(path):
        with av.open(str(path)) as media:
            return [(p.pts * p.time_base, p.duration * p.time_base, bytes(p))
                    for p in media.demux(audio=0) if p.pts is not None]

    source = upscale_source(tmp_path, size=(1536, 672))
    source_path = source.path
    store, service, _ = stack(tmp_path, monkeypatch)
    async with TestClient(TestServer(create_app(service))) as client:
        response = await client.post("/v1/videos/regenerate", data=form(source, resolution=resolution),
                                     headers={"Idempotency-Key": "wide-source"})
        body = await response.json()
        assert response.status == 202, body
        assert body["size"] == f"{delivery[0]}x{delivery[1]}"
        stored = await store.get_task(body["id"])
        task = stored.task
        manifest_path = Path(task.input_manifest_path)
        manifest = json.loads(manifest_path.read_text())
        adapter = service.adapters["sr-pool"]
        payload = adapter.build_worker_payload(task.normalized_request, manifest, manifest_path.parent)
        assert payload["size"] == "1504x640"
        assert payload[SOURCE_KEY]["media"] == {"width": 1504, "height": 640, "frames": 107, "fps": 24, "audio": True}
        assert (payload[SR_PAYLOAD_KEY]["sr_width"], payload[SR_PAYLOAD_KEY]["sr_height"]) == internal
        prepared_path = manifest_path.parent / manifest[0]["path"]
        assert manifest[0]["sha256"] == hashlib.sha256(prepared_path.read_bytes()).hexdigest()
        assert audio_packets(prepared_path) == audio_packets(source_path)
        overlay = Path(__file__).resolve().parents[2] / (
            "container/deps/vllm/patches/h3_seedvr2_v3/sources/"
            "vllm_omni/diffusion/models/seedvr2/long_video_support.py"
        )
        spec = importlib.util.spec_from_file_location("native_sr_validation", overlay)
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        options = native.options("single_encode", body["size"])
        with pytest.raises(ValueError):
            native.validate_source(options, 1536, 672, *internal)
        assert native.validate_source(options, 1504, 640, *internal) is None
        def unexpected_encode(*args, **kwargs):
            pytest.fail("idempotent replay must not re-encode its input")
        monkeypatch.setattr(adapter, "prepare_upscale_source", unexpected_encode)
        response = await client.post("/v1/videos/regenerate", data=form(source, resolution=resolution),
                                     headers={"Idempotency-Key": "wide-source"})
        assert response.status == 200 and (await response.json())["id"] == body["id"]


@pytest.mark.parametrize("content_type,body", [
    ("multipart/form-data", b"not a multipart upload"),
    ("multipart/form-data; boundary=test", b"--another-boundary\r\n"),
    ("multipart/form-data; boundary=test", b"--test\r\nInvalid header\r\n\r\n--test--\r\n"),
    ("multipart/form-data; boundary=test",
     b'--test\r\nContent-Disposition: form-data; name="base_video"; filename="partial.mp4"\r\n'
     b'Content-Type: video/mp4\r\n\r\ntruncated file body'),
])
async def test_malformed_upload_is_400_and_cleans_staging(tmp_path, monkeypatch, content_type, body):
    store, service, _ = stack(tmp_path, monkeypatch)
    async with TestClient(TestServer(create_app(service))) as client:
        response = await client.post("/v1/videos/regenerate", data=body, headers={"Content-Type": content_type})
        error = await response.json()
        assert response.status == 400, error
        assert error["error"]["code"] == "invalid_multipart"
        assert not await store.list_all_tasks()
        assert not list((service.artifacts.root / "_uploads").glob("*"))

@pytest.mark.parametrize("frames,expected", [(106, 400), (108, 400), (362, 202), (363, 400)])
async def test_upscale_h3_frame_range(tmp_path, monkeypatch, frames, expected):
    store, service, _ = stack(tmp_path, monkeypatch)
    source = upscale_source(tmp_path, size=(768, 768), frames=frames)
    async with TestClient(TestServer(create_app(service))) as client:
        response = await client.post("/v1/videos/regenerate", data=form(source))
        body = await response.json()
        assert response.status == expected, body
        if expected == 400:
            assert body["error"]["code"] == "invalid_source_video"
            assert "frames" in body["error"]["message"]
            assert not await store.list_all_tasks()


@pytest.mark.parametrize("source_options,message", [
    ({"audio": False}, "AAC audio"),
    ({"size": (1330, 768)}, "multiples of 32"),
    ({"size": (704, 704)}, "pixels per frame"),
    ({"size": (1024, 1024)}, "pixels per frame"),
])
async def test_upscale_h3_source_constraints(tmp_path, monkeypatch, source_options, message):
    store, service, _ = stack(tmp_path, monkeypatch)
    source = upscale_source(tmp_path, **source_options)
    async with TestClient(TestServer(create_app(service))) as client:
        response = await client.post("/v1/videos/regenerate", data=form(source))
        body = await response.json()
        assert response.status == 400, body
        assert body["error"]["code"] == "invalid_source_video"
        assert message in body["error"]["message"]
        assert not await store.list_all_tasks()

async def test_retired_upscale_path_cannot_create_regeneration_task(tmp_path, monkeypatch):
    store, service, _ = stack(tmp_path, monkeypatch)
    source = upscale_source(tmp_path)
    async with TestClient(TestServer(create_app(service))) as client:
        response = await client.post("/v1/videos/upscale", data=form(source))
        assert response.status in (404, 405)
        assert not await store.list_all_tasks()
