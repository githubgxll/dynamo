# SPDX-License-Identifier: Apache-2.0
"""H3 sizing, actual CPU subprocesses, model-input wiring and cancellation."""
import asyncio
import base64
import hashlib
import io
import json
import os
from types import SimpleNamespace

import pytest
from PIL import Image

from dingo.vllm.omni import h3_image_preprocess as sizing
from dingo.vllm.omni.detached_tasks import DetachedOmniTaskManager
from dingo.vllm.omni.h3_shared_inputs import CURRENT, SharedInputSession
from dingo.vllm.omni.request_adapters.minimax_h3 import MiniMaxH3RequestAdapter
from tests.video_gateway.test_h3_shared_inputs import fixture, wait_until


def png(size=(3840, 2160)):
    out = io.BytesIO()
    with Image.new("RGB", size, "red") as im:
        im.save(out, format="PNG")
    return out.getvalue()


def adapter(tmp_path, monkeypatch, workflow="fl2va"):
    monkeypatch.setenv("DINGO_H3_IMAGE_PREPROCESS_PROFILE", sizing.PROFILE)
    monkeypatch.setenv("DINGO_H3_SHARED_INPUTS_ENABLED", "1")
    monkeypatch.setenv("DINGO_H3_SHARED_INPUTS_STORAGE_ID", "storage")
    return MiniMaxH3RequestAdapter(workflow, str(tmp_path / "media"), 256 * 1024**2)


class Loader:
    @staticmethod
    def _open_image_sync(stream):
        with Image.open(stream) as image:
            return image.convert("RGB")

    async def load_image(self, url):
        return await asyncio.to_thread(
            self._open_image_sync, io.BytesIO(base64.b64decode(url.split(",", 1)[1]))
        )


@pytest.mark.parametrize(
    "source,target",
    [
        ((3840, 2160), (1920, 1080)),
        ((2160, 3840), (1080, 1920)),
        ((5760, 5760), (1080, 1080)),
        ((4000, 3000), (1440, 1080)),
        ((5040, 2160), (1920, 823)),
        ((1920, 1080), (1920, 1080)),
        ((1280, 720), (1280, 720)),
        ((1921, 1080), (1920, 1079)),
    ],
)
def test_exact_fit_rules(source, target):
    assert sizing.fit_size(*source) == target
    scale = min(1, 1920 / max(source), 1080 / min(source))
    assert all(abs(b - a * scale) <= 0.5 for a, b in zip(source, target))


def test_no_crop_no_independent_stretch_and_small_identity():
    with Image.new("RGB", (4000, 3000)) as image:
        # A quadrant marker must survive at every corner, with no crop or pad.
        colors = ["red", "green", "blue", "white"]
        for color, box in zip(
            colors,
            [
                (0, 0, 2000, 1500),
                (2000, 0, 4000, 1500),
                (0, 1500, 2000, 3000),
                (2000, 1500, 4000, 3000),
            ],
        ):
            image.paste(color, box)
        fitted = sizing.fit_image(image)
        try:
            assert fitted.size == (1440, 1080)
            assert [
                fitted.getpixel(p) for p in [(0, 0), (1439, 0), (0, 1079), (1439, 1079)]
            ] == [(255, 0, 0), (0, 128, 0), (0, 0, 255), (255, 255, 255)]
        finally:
            fitted.close()
    with Image.new("RGB", (720, 720)) as small:
        assert sizing.fit_image(small) is small


@pytest.mark.parametrize("size", [(200, 200), (6000, 4000), (1000, 4000)])
def test_invalid_original_not_repaired_by_sizing(size):
    with Image.new("RGB", size) as image, pytest.raises(ValueError, match="dimensions"):
        sizing.fit_image(image)


@pytest.mark.parametrize(
    "workflow,kind",
    [
        ("fl2va", "first"),
        ("fl2va", "last"),
        ("fl2va", "pair"),
        ("fl2va", "reverse"),
        ("ref2va", "mixed"),
    ],
)
async def test_legacy_model_input_images_are_sized_and_roles_preserved(
    tmp_path, monkeypatch, workflow, kind
):
    a = adapter(tmp_path, monkeypatch, workflow)
    uri = "data:image/png;base64," + base64.b64encode(png()).decode()
    indices = None
    if kind == "first":
        payload = uri
    elif kind == "last":
        indices = [-1]
        payload = json.dumps(
            {"type": "fl2va_keyframes_v1", "images": [uri], "frame_indices": indices}
        )
    elif kind == "pair":
        payload = None  # The existing two-frame contract is input_references.
    elif kind == "reverse":
        payload = json.dumps([uri, uri])
    else:
        payload = json.dumps({"type": "ref2va_mixed_v1", "images": [uri]})
    req = SimpleNamespace(
        input_reference=payload, input_references=[uri, uri] if kind == "pair" else None
    )
    async with a.request_scope("inline") as scope:
        result = await a.load_reference(req, Loader(), scope)
        images = (
            result.images
            if hasattr(result, "images")
            else result
            if isinstance(result, list)
            else [result]
        )
        assert all(image.size == (1920, 1080) for image in images)
        if indices is not None:
            assert tuple(result.frame_indices) == tuple(indices)
        for image in images:
            image.close()


async def test_shared_prefetch_receipt_capacity_and_original_unchanged(
    tmp_path, monkeypatch
):
    a = adapter(tmp_path, monkeypatch)
    original = png()
    identity, refs, root = fixture(tmp_path / "pvc", contents=[original])
    session = SharedInputSession(refs, a, tmp_path / "pvc", identity)
    try:
        await session.prepare()
        row = session.image_receipt["images"][0]
        assert row["source_size"] == [3840, 2160] and row["target_size"] == [1920, 1080]
        assert session.path(0).suffix == ".png"
        assert (root / "inputs/0.jpg").read_bytes() == original
        assert a._media_capacity._used == len(original) + row["bytes"]
        assert hashlib.sha256(session.path(0).read_bytes()).hexdigest() == row["sha256"]
        token = CURRENT.set(session)
        try:
            reference = await a.load_reference(
                SimpleNamespace(
                    input_reference=json.dumps(refs), input_references=None
                ),
                Loader(),
            )
            assert reference.images[0].size == (1920, 1080)
            reference.images[0].close()
        finally:
            CURRENT.reset(token)
    finally:
        session.cleanup()
    assert a._media_capacity._used == 0 and a.image_prepare_slots._value == 1


async def test_shared_video_audio_bytes_unchanged(tmp_path, monkeypatch):
    a = adapter(tmp_path, monkeypatch, "ref2va")
    identity, refs, root = fixture(
        tmp_path / "pvc", contents=[png(), b"video unchanged", b"audio unchanged"]
    )
    refs.update(workflow="ref2va", frame_indices=None)
    refs["entries"][1]["content_type"] = "video/mp4"
    refs["entries"][2]["content_type"] = "audio/wav"
    raw = json.dumps(refs["entries"]).encode()
    (root / "input-manifest.json").write_bytes(raw)
    refs["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    session = SharedInputSession(refs, a, tmp_path / "pvc", identity)
    try:
        await session.prepare()
        assert session.path(1).read_bytes() == b"video unchanged"
        assert session.path(2).read_bytes() == b"audio unchanged"
        assert len(session.image_receipt["images"]) == 1
    finally:
        session.cleanup()


async def test_corrupt_input_never_calls_model_or_leaks_capacity(tmp_path, monkeypatch):
    a = adapter(tmp_path, monkeypatch)

    class Handler:
        sr_ready = True
        request_adapter = a
        calls = 0

        async def generate(self, payload, context):
            self.calls += 1
            yield {"status": "completed"}

    h = Handler()
    root = tmp_path / "pvc"
    identity, refs, _ = fixture(root, contents=[b"broken jpeg"])
    manager = DetachedOmniTaskManager(
        h, root, prefetch_capacity=1, binary_results=False, inline_results=False
    )
    try:
        await manager._submit(identity, {"input_reference": json.dumps(refs)})
        await wait_until(lambda: not manager._running)
        status = await manager._status(identity)
        assert status["error"]["code"] == "input_processing_failed" and h.calls == 0
        assert h.sr_ready and a._media_capacity._used == 0
    finally:
        await manager.shutdown()


async def test_cancel_joins_real_cpu_child_before_cleanup(tmp_path, monkeypatch):
    a = adapter(tmp_path, monkeypatch)
    identity, refs, _ = fixture(tmp_path / "pvc", contents=[png()])
    child = tmp_path / "slow_child.py"
    marker = tmp_path / "pid"
    child.write_text(
        "import os,time\nfrom pathlib import Path\nPath("
        + repr(str(marker))
        + ").write_text(str(os.getpid()))\ntime.sleep(60)\n"
    )
    monkeypatch.setattr(sizing, "__file__", str(child))
    session = SharedInputSession(refs, a, tmp_path / "pvc", identity)
    task = asyncio.create_task(session.prepare())
    try:
        await wait_until(marker.exists)
        pid = int(marker.read_text())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        session.cleanup()
    assert a._media_capacity._used == 0 and a.image_prepare_slots._value == 1


async def test_small_shared_image_not_reencoded(tmp_path, monkeypatch):
    a = adapter(tmp_path, monkeypatch)
    raw = png((1280, 720))
    identity, refs, _ = fixture(tmp_path / "pvc", contents=[raw])
    session = SharedInputSession(refs, a, tmp_path / "pvc", identity)
    try:
        await session.prepare()
        assert not session.image_receipt["images"][0]["resized"]
        assert session.path(0).name == "0" and session.path(0).read_bytes() == raw
        assert session.image_receipt["derived_bytes"] == 0
    finally:
        session.cleanup()


async def test_nine_large_images_prepared_while_previous_task_runs(
    tmp_path, monkeypatch
):
    a = adapter(tmp_path, monkeypatch, "ref2va")
    root = tmp_path / "pvc"
    data = png((2736, 2736))
    identity, refs, task_root = fixture(root, contents=[data] * 9)
    refs.update(workflow="ref2va", frame_indices=None)

    class Handler:
        sr_ready = True
        request_adapter = a

        def __init__(self):
            self.release = asyncio.Event()
            self.calls = []

        async def generate(self, payload, context):
            session = CURRENT.get()
            self.calls.append(session.value["task_id"])
            assert len(session.image_receipt["images"]) == 9
            assert all(
                r["target_size"] == [1080, 1080]
                for r in session.image_receipt["images"]
            )
            if len(self.calls) == 1:
                await self.release.wait()
            yield {"status": "completed"}

    h = Handler()
    manager = DetachedOmniTaskManager(
        h, root, prefetch_capacity=1, binary_results=False, inline_results=False
    )
    try:
        await manager._submit(identity, {"input_reference": json.dumps(refs)})
        await wait_until(lambda: len(h.calls) == 1)
        other, other_refs, _ = fixture(root, task="video-B", contents=[data] * 9)
        other_refs.update(workflow="ref2va", frame_indices=None)
        await manager._submit(other, {"input_reference": json.dumps(other_refs)})
        await wait_until(
            lambda: len(
                list(
                    (a.media_root / "h3-shared-inputs").glob(
                        "request-*/image-preprocess.json"
                    )
                )
            )
            == 2
        )
        assert len(h.calls) == 1
        h.release.set()
        await wait_until(lambda: not manager._running)
        assert len(h.calls) == 2 and a._media_capacity._used == 0
        assert (task_root / "inputs/0.jpg").read_bytes() == data
    finally:
        h.release.set()
        await manager.shutdown()


def test_exif_orientation_follows_existing_loader_pixels(tmp_path):
    stream = io.BytesIO()
    with Image.new("RGB", (4000, 3000), "red") as image:
        exif = image.getexif()
        exif[274] = 6
        image.save(stream, format="JPEG", exif=exif)
    raw = stream.getvalue()
    (tmp_path / "0").write_bytes(raw)
    result = sizing.prepare_images(
        tmp_path,
        [
            {
                "ordinal": 0,
                "content_type": "image/jpeg",
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        ],
    )
    assert result["images"][0]["target_size"] == [1440, 1080]
    with Image.open(tmp_path / "image-0.png") as out:
        assert (
            out.size == (1440, 1080) and out.mode == "RGB" and 274 not in out.getexif()
        )


async def test_legacy_url_cache_is_not_closed_or_mutated():
    with Image.new("RGB", (3840, 2160), "red") as cached:

        class CachedLoader:
            async def load_image(self, url):
                return cached

        loader = sizing.SizingLoader(CachedLoader(), asyncio.Semaphore(1))
        for _ in range(2):
            result = await loader.load_image("https://example.invalid/image.jpg")
            assert result.size == (1920, 1080)
            result.close()
            assert cached.size == (3840, 2160) and cached.getpixel((0, 0)) == (
                255,
                0,
                0,
            )
