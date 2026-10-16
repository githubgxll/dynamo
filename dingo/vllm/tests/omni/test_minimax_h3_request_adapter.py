# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import json
import os
import stat
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from dingo.vllm.omni.request_adapters import create_request_adapter
from dingo.vllm.omni.request_adapters.minimax_h3 import MiniMaxH3RequestAdapter

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class _Loader:
    def __init__(self):
        self.references = []

    async def load_image(self, reference):
        self.references.append(reference)
        return Image.new("RGB", (64, 64), color="red")


def _request(*, input_reference=None, input_references=None):
    return SimpleNamespace(
        input_reference=input_reference, input_references=input_references
    )


def _engine_inputs():
    sampling = SimpleNamespace(width=1344, height=768, extra_args={})
    return SimpleNamespace(prompt={"prompt": "test"}, sampling_params_list=[sampling])


def _builder(req, image=None):
    inputs = _engine_inputs()
    if image is not None:
        inputs.prompt["multi_modal_data"] = {"image": image}
    return inputs


def _data_url(prefix, payload):
    return prefix + base64.b64encode(payload).decode()


def test_disabled_factory_does_not_import_minimax_adapter():
    module = "dingo.vllm.omni.request_adapters.minimax_h3"
    previous = sys.modules.pop(module, None)
    try:
        assert create_request_adapter(None, None, "/tmp/unused", 1024) is None
        assert module not in sys.modules
    finally:
        if previous is not None:
            sys.modules[module] = previous


@pytest.mark.asyncio
async def test_fl2va_two_keyframes_preserve_order(tmp_path):
    loader = _Loader()
    adapter = MiniMaxH3RequestAdapter("fl2va", str(tmp_path), 1024 * 1024)
    reference = json.dumps(["data:image/png;base64,one", "data:image/png;base64,two"])
    loaded = await adapter.load_reference(_request(input_reference=reference), loader)
    inputs = adapter.build_engine_inputs(_request(), loaded, _builder)

    assert loader.references == [
        "data:image/png;base64,one",
        "data:image/png;base64,two",
    ]
    assert len(inputs.prompt["multi_modal_data"]["image"]) == 2


@pytest.mark.asyncio
async def test_fl2va_tail_keyframe_sets_semantic_index(tmp_path):
    loader = _Loader()
    adapter = MiniMaxH3RequestAdapter("fl2va", str(tmp_path), 1024 * 1024)
    envelope = json.dumps(
        {
            "type": "fl2va_keyframes_v1",
            "images": ["data:image/png;base64,last"],
            "frame_indices": [-1],
        }
    )
    loaded = await adapter.load_reference(_request(input_reference=envelope), loader)
    inputs = adapter.build_engine_inputs(_request(), loaded, _builder)

    assert inputs.sampling_params_list[0].extra_args["frame_indices"] == [-1]
    assert len(inputs.prompt["multi_modal_data"]["image"]) == 1


def test_fl2va_t2va_derives_aspect_ratio(tmp_path):
    adapter = MiniMaxH3RequestAdapter("fl2va", str(tmp_path), 1024 * 1024)
    inputs = adapter.build_engine_inputs(_request(), None, _builder)
    assert inputs.sampling_params_list[0].extra_args["aspect_ratio"] == "16:9"


@pytest.mark.asyncio
async def test_ref2va_mp4_is_validated_and_persisted_private(tmp_path):
    adapter = MiniMaxH3RequestAdapter("ref2va", str(tmp_path), 1024 * 1024)
    mp4 = b"\x00\x00\x00\x18ftypisom" + b"0" * 32
    async with adapter.request_scope("request-private") as scope:
        loaded = await adapter.load_reference(
            _request(input_reference=_data_url("data:video/mp4;base64,", mp4)),
            _Loader(),
            scope,
        )
        inputs = adapter.build_engine_inputs(_request(), loaded, _builder)
        path = inputs.prompt["multi_modal_data"]["video"]

        assert path.endswith(".mp4")
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert not os.path.exists(path)


@pytest.mark.asyncio
async def test_ref2va_duplicate_media_is_isolated_between_request_scopes(tmp_path):
    adapter = MiniMaxH3RequestAdapter("ref2va", str(tmp_path), 1024 * 1024)
    mp4 = b"\x00\x00\x00\x18ftypisom" + b"0" * 32
    request = _request(input_reference=_data_url("data:video/mp4;base64,", mp4))

    first_context = adapter.request_scope("request-first")
    second_context = adapter.request_scope("request-second")
    first_scope = await first_context.__aenter__()
    second_scope = await second_context.__aenter__()
    try:
        first, second = await asyncio.gather(
            adapter.load_reference(request, _Loader(), first_scope),
            adapter.load_reference(request, _Loader(), second_scope),
        )

        assert first.path != second.path
        assert (
            os.path.getsize(first.path)
            == os.path.getsize(second.path)
            == len(mp4)
        )
        await first_context.__aexit__(None, None, None)
        assert not os.path.exists(first.path)
        assert os.path.exists(second.path)
    finally:
        if first_scope.root is not None:
            await first_context.__aexit__(None, None, None)
        await second_context.__aexit__(None, None, None)

    assert not os.path.exists(second.path)
    assert not list((tmp_path / "ref2va-requests").glob("**/*.tmp"))


@pytest.mark.asyncio
async def test_ref2va_media_cache_rejects_new_file_over_limit(tmp_path):
    mp4 = b"\x00\x00\x00\x18ftypisom" + b"0" * 32
    adapter = MiniMaxH3RequestAdapter("ref2va", str(tmp_path), len(mp4) - 1)

    async with adapter.request_scope("request-over-limit") as scope:
        with pytest.raises(ValueError, match="media cache capacity exceeded"):
            await adapter.load_reference(
                _request(input_reference=_data_url("data:video/mp4;base64,", mp4)),
                _Loader(),
                scope,
            )


@pytest.mark.asyncio
async def test_ref2va_mixed_reference_attaches_all_modalities(tmp_path):
    adapter = MiniMaxH3RequestAdapter("ref2va", str(tmp_path), 1024 * 1024)
    mp4 = _data_url("data:video/mp4;base64,", b"\x00\x00\x00\x18ftypisom")
    wav = _data_url("data:audio/wav;base64,", b"RIFF\x04\x00\x00\x00WAVE")
    envelope = json.dumps(
        {
            "type": "ref2va_mixed_v1",
            "images": ["data:image/png;base64,image"],
            "videos": [mp4],
            "audios": [wav],
        }
    )
    async with adapter.request_scope("request-mixed") as scope:
        loaded = await adapter.load_reference(
            _request(input_reference=envelope), _Loader(), scope
        )
        inputs = adapter.build_engine_inputs(_request(), loaded, _builder)
        media = inputs.prompt["multi_modal_data"]

        assert len(media["image"]) == len(media["video"]) == len(media["audio"]) == 1
        assert media["video"][0].endswith(".mp4")
        assert media["audio"][0].endswith(".wav")
        paths = [*media["video"], *media["audio"]]
    assert all(not os.path.exists(path) for path in paths)


@pytest.mark.asyncio
async def test_ref2va_scope_cleans_media_when_request_fails(tmp_path):
    adapter = MiniMaxH3RequestAdapter("ref2va", str(tmp_path), 1024 * 1024)
    mp4 = b"\x00\x00\x00\x18ftypisom" + b"0" * 32
    path = None

    with pytest.raises(RuntimeError, match="engine failed"):
        async with adapter.request_scope("request-failure") as scope:
            loaded = await adapter.load_reference(
                _request(input_reference=_data_url("data:video/mp4;base64,", mp4)),
                _Loader(),
                scope,
            )
            path = loaded.path
            raise RuntimeError("engine failed")

    assert path is not None
    assert not os.path.exists(path)


@pytest.mark.asyncio
async def test_ref2va_cancelled_scope_defers_cleanup_gracefully(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "dingo.vllm.omni.request_adapters.minimax_h3._CANCELLED_CLEANUP_GRACE_S",
        0.05,
    )
    adapter = MiniMaxH3RequestAdapter("ref2va", str(tmp_path), 1024 * 1024)
    context = SimpleNamespace(is_stopped=lambda: True)
    mp4 = b"\x00\x00\x00\x18ftypisom" + b"0" * 32

    async with adapter.request_scope("request-cancelled", context=context) as scope:
        loaded = await adapter.load_reference(
            _request(input_reference=_data_url("data:video/mp4;base64,", mp4)),
            _Loader(),
            scope,
        )
        path = loaded.path

    assert os.path.exists(path)
    await asyncio.sleep(0.1)
    assert not os.path.exists(path)
    assert not adapter._deferred_cleanups


@pytest.mark.asyncio
async def test_fl2va_scope_does_not_create_media_directory(tmp_path):
    adapter = MiniMaxH3RequestAdapter("fl2va", str(tmp_path), 1024 * 1024)
    async with adapter.request_scope("fl2va-request") as scope:
        assert scope is None
        loaded = await adapter.load_reference(
            _request(
                input_references=[
                    "data:image/png;base64,first",
                    "data:image/png;base64,last",
                ]
            ),
            _Loader(),
            scope,
        )
        assert len(loaded) == 2
    assert not (tmp_path / "ref2va-requests").exists()


@pytest.mark.asyncio
async def test_workflow_specific_adapter_rejects_other_envelope(tmp_path):
    adapter = MiniMaxH3RequestAdapter("fl2va", str(tmp_path), 1024 * 1024)
    envelope = json.dumps(
        {"type": "ref2va_mixed_v1", "images": [], "videos": [], "audios": []}
    )
    with pytest.raises(ValueError, match="FL2VA keyframe envelope type"):
        await adapter.load_reference(_request(input_reference=envelope), _Loader())


@pytest.mark.asyncio
async def test_normal_image_reference_uses_generic_loader(tmp_path):
    loader = _Loader()
    adapter = MiniMaxH3RequestAdapter("ref2va", str(tmp_path), 1024 * 1024)
    loaded = await adapter.load_reference(
        _request(input_reference="https://example.test/reference.png"), loader
    )
    assert isinstance(loaded, Image.Image)
    assert loader.references == ["https://example.test/reference.png"]
