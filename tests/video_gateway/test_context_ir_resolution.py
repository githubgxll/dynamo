# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile geometry must use the video policy and remain reusable for generation."""

import hashlib
from dataclasses import replace
from types import SimpleNamespace

import pytest
from PIL import Image

from dingo.video_gateway.adapters.base import UploadedArtifact
from dingo.video_gateway.adapters.h3_resolution import BASE_CANVASES
from dingo.video_gateway.adapters.minimax_h3 import MiniMaxH3VideoAdapter
from dingo.video_gateway.context_ir.config import BackendConfig, ContextIRConfig
from dingo.video_gateway.context_ir.protocol import prepare
from dingo.video_gateway.context_ir.validation import normalize, request_digest
from dingo.video_gateway.errors import GatewayError


def compiler_service(make_gateway_config, *, workflow="fl2va", sr=False):
    cfg = make_gateway_config()
    pool = cfg.pools[0]
    options = dict(pool.adapter.options)
    if sr:
        options["super_resolution"] = {"enabled": True, "profile": "minimax_h3_seedvr2.v3"}
    pool = replace(pool, adapter=replace(pool.adapter, workflow=workflow, options=options))
    cfg = replace(cfg, context_ir=ContextIRConfig(
        enabled=True, backends={"h3": BackendConfig(url="http://compiler")},
        model_bindings={"public-fl": "h3"},
    ))
    adapter = MiniMaxH3VideoAdapter(pool)
    return SimpleNamespace(config=cfg, resolve_pool=lambda model: pool, adapters={pool.pool_id: adapter}), adapter


def fields(**extra):
    return {k: [v] for k, v in {"model": "public-fl", "prompt": "Animate the provided scene.", "seconds": "5", **extra}.items()}


def image(root, size, ordinal=0):
    path = root / "inputs" / f"image-{ordinal}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "red").save(path)
    payload = path.read_bytes()
    return UploadedArtifact(field_name="input_references", ordinal=ordinal,
        filename=path.name, content_type="image/png", path=path,
        size=len(payload), sha256=hashlib.sha256(payload).hexdigest())


def replay_video(adapter, normalized, uploads):
    values = {"model": "public-fl", "prompt": "Compiled output.", **normalized["generation_parameters"]}
    values = {k: [str(v)] for k, v in values.items()}
    # frame_indices uses JSON, not Python's string form in general.
    result = adapter.normalize_request(values, uploads, "public-fl")
    assert [result["width"], result["height"]] == normalized["canvas"]
    return result


@pytest.mark.parametrize("ratio,size", list(BASE_CANVASES.items()))
@pytest.mark.parametrize("ratio_key", ["ratio", "aspect_ratio"])
def test_context_ir_resolution_presets_canonicalize_like_explicit_size(make_gateway_config, ratio, size, ratio_key):
    service, adapter = compiler_service(make_gateway_config)
    named = normalize(service, fields(resolution="768P", **{ratio_key: ratio}), [])
    explicit = normalize(service, fields(size=f"{size[0]}x{size[1]}"), [])
    assert named == explicit
    assert request_digest(named) == request_digest(explicit)
    replay_video(adapter, named, [])


@pytest.mark.parametrize("choice", [{}, {"ratio": "adaptive"}, {"aspect_ratio": "adaptive"}, {"ratio": "adaptive", "aspect_ratio": "adaptive"}])
@pytest.mark.parametrize("source,expected", [((506, 960), [736, 1408]), ((1536, 1024), [1152, 768])])
def test_context_ir_adaptive_canvas_survives_compile_and_video_handoff(make_gateway_config, tmp_path, choice, source, expected):
    service, adapter = compiler_service(make_gateway_config)
    upload = image(tmp_path, source)
    normalized = normalize(service, fields(resolution="768P", **choice), [upload])
    assert normalized["canvas"] == expected
    assert normalized["generation_parameters"] == {"resolution": "768P", "ratio": "adaptive", "seconds": 5.0, "fps": 24, "frame_indices": [0]}
    payload, _, _ = prepare(normalized, tmp_path, tmp_path / "attempt", 64 * 1024**2)
    assert payload["aspect"] == f"{expected[0]}:{expected[1]}"
    assert payload["assets"][0]["sha256"] == upload.sha256
    assert replay_video(adapter, normalized, [upload])["frame_indices"] == [0]
    # A size-only replay is intentionally insufficient for these non-preset canvases.
    with pytest.raises(GatewayError):
        adapter.normalize_request(fields(size=f"{expected[0]}x{expected[1]}"), [upload], "public-fl")
    implicit = normalize(service, fields(resolution="768P"), [upload])
    assert request_digest(normalized) == request_digest(implicit)


def test_context_ir_adaptive_unequal_first_last_and_sole_last(make_gateway_config, tmp_path):
    service, adapter = compiler_service(make_gateway_config)
    first, last = image(tmp_path, (1536, 1024)), image(tmp_path, (506, 960), 1)
    normalized = normalize(service, fields(resolution="768P"), [first, last])
    assert normalized["canvas"] == [1152, 768]
    assert replay_video(adapter, normalized, [first, last])["frame_indices"] == [0, -1]
    tail = image(tmp_path / "tail", (506, 960))
    normalized = normalize(service, fields(resolution="768P", ratio="adaptive", reference_options='[{"index":0,"role":"frame_anchor_last"}]'), [tail])
    assert normalized["canvas"] == [736, 1408]
    assert normalized["mode"] == "l2va"
    assert replay_video(adapter, normalized, [tail])["frame_indices"] == [-1]


def test_context_ir_ref2va_concrete_ratio_preserves_reference_role(make_gateway_config, tmp_path):
    service, adapter = compiler_service(make_gateway_config, workflow="ref2va")
    upload = image(tmp_path, (506, 960))
    normalized = normalize(service, fields(resolution="768P", ratio="4:3"), [upload])
    assert normalized["mode"] == "ref2va" and normalized["canvas"] == [1024, 768]
    assert normalized["references"][0]["role"] == "subject"
    assert normalized["references"][0]["role_stated"] is False
    replay_video(adapter, normalized, [upload])
    adaptive = normalize(service, fields(ratio="adaptive"), [upload])
    assert adaptive["canvas"] == [768, 1344]
    assert adaptive == normalize(service, fields(), [upload])


@pytest.mark.parametrize("extra,code", [
    ({"resolution": "4K", "ratio": "16:9"}, "invalid_resolution"),
    ({"resolution": "768P"}, "invalid_aspect_ratio"),
    ({"resolution": "768P", "ratio": "adaptive"}, "invalid_aspect_ratio"),
    ({"resolution": "768P", "ratio": "5:3"}, "invalid_aspect_ratio"),
    ({"resolution": "768P", "ratio": "16:9", "aspect_ratio": "9:16"}, "conflicting_parameter"),
    ({"size": "1344x768", "ratio": "9:16"}, "invalid_aspect_ratio"),
])
def test_context_ir_reuses_video_shape_conflict_checks(make_gateway_config, extra, code):
    service, _ = compiler_service(make_gateway_config)
    with pytest.raises(GatewayError) as error:
        normalize(service, fields(**extra), [])
    assert error.value.code == code


@pytest.mark.parametrize("tier", [None, "768P", "1080P", "2K"])
@pytest.mark.parametrize("ratio,canvas", list(BASE_CANVASES.items()))
@pytest.mark.parametrize("workflow", ["fl2va", "ref2va"])
@pytest.mark.parametrize("sr", [False, True])
def test_compilation_is_independent_of_resolution_and_sr(make_gateway_config, tmp_path, tier, ratio, canvas, workflow, sr):
    service, _ = compiler_service(make_gateway_config, workflow=workflow, sr=sr)
    refs = [image(tmp_path, (506, 960))] if workflow == "ref2va" else []
    params = {"ratio": ratio}
    if tier is not None:
        params["resolution"] = tier
    normalized = normalize(service, fields(**params), refs)
    baseline = normalize(service, fields(ratio=ratio), refs)
    assert normalized == baseline
    assert normalized["canvas"] == list(canvas)
    assert request_digest(normalized) == request_digest(baseline)
    assert "super_resolution" not in normalized


@pytest.mark.parametrize("tier", [None, "768P", "1080P", "2K"])
def test_compiler_adaptive_never_uses_generation_high_resolution_whitelist(make_gateway_config, tmp_path, tier):
    service, _ = compiler_service(make_gateway_config, sr=False)
    upload = image(tmp_path, (506, 960))
    params = {"ratio": "adaptive"}
    if tier is not None:
        params["resolution"] = tier
    n = normalize(service, fields(**params), [upload])
    assert n["canvas"] == [736, 1408]
    assert "super_resolution" not in n
    assert n == normalize(service, fields(ratio="adaptive"), [upload])


def test_compiler_never_reads_sr_profile(make_gateway_config, tmp_path, monkeypatch):
    service, adapter = compiler_service(make_gateway_config)
    def forbidden(_):
        raise AssertionError("compiler must not inspect SR configuration")
    monkeypatch.setattr(type(adapter.pool), "sr_profile", property(forbidden))
    assert normalize(service, fields(resolution="2K", ratio="16:9"), [])["canvas"] == [1344, 768]
    upload = image(tmp_path, (1536, 1024))
    assert normalize(service, fields(resolution="1080P"), [upload])["canvas"] == [1152, 768]


def test_compiler_keyframes_ignore_concrete_ratio_like_hosted_context_ir(make_gateway_config, tmp_path):
    service, _ = compiler_service(make_gateway_config)
    first, tail = image(tmp_path, (506, 960)), image(tmp_path, (1000, 1000), 1)
    n = normalize(service, fields(ratio="16:9"), [first, tail])
    assert n["canvas"] == [736, 1408]
    assert n["references"][0]["role"] == "frame_anchor_first"
    assert n["references"][1]["role"] == "frame_anchor_last"


@pytest.mark.parametrize("size,ratio,canvas", [("1920x1080","16:9",[1344,768]),("2560x1440","16:9",[1344,768]),("1080x1920","9:16",[768,1344])])
def test_legacy_high_resolution_sizes_only_select_compilation_aspect(make_gateway_config, size, ratio, canvas):
    service, _ = compiler_service(make_gateway_config, sr=False)
    n = normalize(service, fields(size=size), [])
    assert n["canvas"] == canvas
    assert n == normalize(service, fields(ratio=ratio), [])
    assert normalize(service, fields(size=size, resolution="768P"), []) == n


def test_compiler_ref_adaptive_prefers_video_over_image(make_gateway_config, tmp_path, monkeypatch):
    from dingo.video_gateway.adapters import minimax_h3
    service, _ = compiler_service(make_gateway_config, workflow="ref2va")
    first = image(tmp_path, (1600, 900))
    second = image(tmp_path, (506, 960), 1)
    real = minimax_h3._probe_upload
    def probe(upload, limits):
        value = real(upload, limits)
        return replace(value, kind="video", fps=24, duration_s=5) if upload.ordinal == 1 else value
    monkeypatch.setattr(minimax_h3, "_probe_upload", probe)
    from dingo.video_gateway.context_ir import validation
    monkeypatch.setattr(validation,"audio_present",lambda path:False)
    n = normalize(service, fields(reference_options='[{"index":1,"soundtrack_policy":"ignore"}]'), [first, second])
    assert n["canvas"] == [768, 1344]


def test_public_result_hides_all_pixel_fields_without_mutating_evidence():
    from copy import deepcopy
    from dingo.video_gateway.context_ir.protocol import public_result
    stored = {"content":{"prompt":"compiled"},"generation_parameters":{"size":"1344x768","resolution":"768P","width":1344,"height":768,"canvas":[1344,768],"seconds":5,"fps":24},"target":{"frames":124,"fps":24,"canvas":[1344,768]},"warnings":[]}
    original = deepcopy(stored)
    exposed = public_result(stored)
    assert exposed["generation_parameters"] == {"ratio":"16:9","seconds":5,"fps":24}
    assert exposed["target"] == {"frames":124,"fps":24}
    assert stored == original
    stored["references"]=[{"role":"frame_anchor_first"}]
    assert public_result(stored)["generation_parameters"]["ratio"] == "adaptive"
    assert public_result({"content":{"prompt":"compiled"}}) == {"content":{"prompt":"compiled"}}
