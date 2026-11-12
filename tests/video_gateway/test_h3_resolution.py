# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import pytest
from PIL import Image

from dingo.video_gateway.adapters.h3_resolution import RESOLUTION_CANVASES
from dingo.video_gateway.adapters.minimax_h3 import MiniMaxH3VideoAdapter
from dingo.video_gateway.errors import GatewayError
from dingo.video_gateway.models import VideoTask
from tests.video_gateway.test_minimax_h3_adapter import _base_fields, _png_upload
from tests.video_gateway.test_task_store import _task


def adapter(make_gateway_config, workflow="fl2va", sr=True):
    pool = make_gateway_config().pools[0]
    options = dict(pool.adapter.options)
    if sr:
        options["super_resolution"] = {
            "enabled": True,
            "profile": "minimax_h3_seedvr2.v3",
        }
    return MiniMaxH3VideoAdapter(
        dataclasses.replace(
            pool,
            adapter=dataclasses.replace(
                pool.adapter, options=options, workflow=workflow
            ),
        )
    )


def fields(**kw):
    value = _base_fields()
    value.pop("size")
    value.update({k: [v] for k, v in kw.items()})
    return value


@pytest.mark.parametrize(
    "tier,ratio,size",
    [(t, r, s) for t, ratios in RESOLUTION_CANVASES.items() for r, s in ratios.items()],
)
def test_all_presets_match_explicit_request_and_worker_payload(
    make_gateway_config, tmp_path, tier, ratio, size
):
    a = adapter(make_gateway_config)
    named = a.normalize_request(fields(resolution=tier, ratio=ratio), [], "public-fl")
    explicit = a.normalize_request(fields(size=f"{size[0]}x{size[1]}"), [], "public-fl")
    assert named == explicit  # Identical canonical request and idempotency digest.
    payload = a.build_worker_payload(named, [], tmp_path)
    assert payload == a.build_worker_payload(explicit, [], tmp_path)
    task = _task("tier-test")
    task.normalized_request = named
    restored = VideoTask.from_dict(task.to_dict())
    assert restored.public_dict()["resolution"] == tier
    assert restored.public_dict()["ratio"] == ratio
    assert restored.public_dict()["size"] == f"{size[0]}x{size[1]}"


@pytest.mark.parametrize(
    "kw,code",
    [
        ({"resolution": "4K", "ratio": "16:9"}, "invalid_resolution"),
        ({"resolution": "2K", "ratio": "5:3"}, "invalid_aspect_ratio"),
        ({"resolution": "2K"}, "invalid_aspect_ratio"),
        ({"ratio": "16:9"}, "missing_size"),
        ({"resolution": "2K", "ratio": "adaptive"}, "unsupported_adaptive_ratio"),
        ({"size": "1920x1080", "resolution": "2K"}, "conflicting_resolution"),
        ({"size": "864x480", "resolution": "768P"}, "conflicting_resolution"),
        ({"size": "2560x1440", "ratio": "9:16"}, "invalid_aspect_ratio"),
        (
            {"size": "2560x1440", "ratio": "16:9", "aspect_ratio": "9:16"},
            "conflicting_parameter",
        ),
        ({"size": "2560x1440", "width": "2560", "height": "1440"}, "conflicting_size"),
        ({"resolution": "2K", "ratio": "16:9", "width": "2560"}, "incomplete_size"),
    ],
)
def test_conflicts(make_gateway_config, kw, code):
    with pytest.raises(GatewayError) as error:
        adapter(make_gateway_config).normalize_request(fields(**kw), [], "public-fl")
    assert error.value.code == code


def test_legacy_custom_size_and_compatibility_canvas(make_gateway_config):
    a = adapter(make_gateway_config)
    value = a.normalize_request(fields(size="864x480"), [], "public-fl")
    task = _task("custom")
    task.normalized_request = value
    assert "resolution" not in task.public_dict()
    assert (
        a.normalize_request(fields(size="2400x1440", resolution="2K"), [], "public-fl")[
            "width"
        ]
        == 2400
    )
    assert (
        a.normalize_request(
            fields(width="2560", height="1440", ratio="16:9", aspect_ratio="16:9"),
            [],
            "public-fl",
        )["height"]
        == 1440
    )


@pytest.mark.parametrize("ratio", [None, "adaptive", "16:9"])
@pytest.mark.parametrize("tier", ["768P", "1080P", "2K"])
def test_fl2va_keyframe_adaptive(make_gateway_config, tmp_path, ratio, tier):
    upload = _png_upload(tmp_path)
    # A real keyframe near the nominal 16:9 delivery ratio.
    request = fields(resolution=tier)
    if ratio is not None:
        request["ratio"] = [ratio]
    value = adapter(make_gateway_config).normalize_request(
        request, [upload], "public-fl"
    )
    assert (value["width"], value["height"]) == RESOLUTION_CANVASES[tier]["16:9"]


@pytest.mark.parametrize("tier", ["768P", "1080P", "2K"])
def test_adaptive_accepts_different_keyframe_dimensions(make_gateway_config, tmp_path, tier):
    first = _png_upload(tmp_path)
    second = _png_upload(tmp_path, ordinal=1)
    Image.new("RGB", (768, 768)).save(second.path)
    a = adapter(make_gateway_config)
    value = a.normalize_request(
        fields(resolution=tier, ratio="adaptive"), [first, second], "public-fl"
    )
    assert (value["width"], value["height"]) == RESOLUTION_CANVASES[tier]["16:9"]
    # The explicit-size path still requires identical keyframe dimensions.
    with pytest.raises(GatewayError, match="identical dimensions"):
        a.normalize_request(fields(size="1344x768"), [first, second], "public-fl")


def test_adaptive_preserves_high_resolution_whitelist(make_gateway_config, tmp_path):
    first = _png_upload(tmp_path)
    a = adapter(make_gateway_config)
    Image.new("RGB", (1200, 800)).save(first.path)
    with pytest.raises(GatewayError, match="no preset matches"):
        a.normalize_request(
            fields(resolution="2K", ratio="adaptive"), [first], "public-fl"
        )
    with pytest.raises(GatewayError, match="output size must match"):
        a.normalize_request(fields(size="1440x2560"), [first], "public-fl")


@pytest.mark.parametrize("tier", ["768P", "1080P", "2K"])
def test_ref2va_requires_concrete_ratio(make_gateway_config, tmp_path, tier):
    a = adapter(make_gateway_config, "ref2va")
    upload = _png_upload(tmp_path)
    for kw in ({"resolution": tier}, {"resolution": tier, "ratio": "adaptive"}):
        with pytest.raises(GatewayError):
            a.normalize_request(fields(**kw), [upload], "public-fl")
    assert (
        a.normalize_request(
            fields(resolution=tier, ratio="9:16"), [upload], "public-fl"
        )["aspect_ratio"]
        == "9:16"
    )


def test_existing_policy_remains_authoritative(make_gateway_config):
    a = adapter(make_gateway_config, sr=False)
    assert set(a.capabilities(max_result_bytes=100)["resolution_presets"]) == {"768P"}
    with pytest.raises(GatewayError):
        a.normalize_request(fields(resolution="2K", ratio="16:9"), [], "public-fl")


async def test_equivalent_syntax_replays_same_task_after_gateway_recreation(
    make_gateway_config, monkeypatch
):
    from dingo.video_gateway.service import VideoGatewayService
    from tests.video_gateway.test_dispatcher import FakeClient, _stack

    config = make_gateway_config(accept_without_workers=True)
    store, artifacts, dispatcher, service = _stack(config, {"fl-pool": FakeClient()})

    async def submit(target, request):
        root = await artifacts.create_upload()
        return await target.submit(
            fields=request,
            uploads=[],
            upload_root=root,
            delivery_mode="async",
            idempotency_key="same-resolution",
        )

    first = await submit(service, fields(resolution="768P", ratio="16:9"))
    replacement = VideoGatewayService(
        config, store, artifacts, dispatcher, service.adapters
    )
    replay = await submit(replacement, fields(size="1344x768"))
    assert first.created and not replay.created
    assert first.stored.task.id == replay.stored.task.id
    assert first.stored.task.request_digest == replay.stored.task.request_digest
    assert first.stored.task.normalized_request == replay.stored.task.normalized_request
    with pytest.raises(GatewayError) as error:
        await submit(replacement, fields(resolution="768P", ratio="1:1"))
    assert error.value.code == "idempotency_conflict"


@pytest.mark.parametrize("field,value", [("ratio", "16:9"), ("aspect_ratio", "16:9")])
def test_alias_and_explicit_dimensions_are_canonical(make_gateway_config, field, value):
    a = adapter(make_gateway_config)
    assert a.normalize_request(
        fields(resolution="2K", **{field: value}), [], "public-fl"
    ) == a.normalize_request(
        fields(size="2560x1440", **{field: value}), [], "public-fl"
    )


def test_tail_frame_and_explicit_adaptive(make_gateway_config, tmp_path):
    a = adapter(make_gateway_config)
    upload = _png_upload(tmp_path)
    f = fields(resolution="2K", ratio="adaptive", frame_indices="[-1]")
    normalized = a.normalize_request(f, [upload], "public-fl")
    assert normalized["frame_indices"] == [-1]
    assert normalized == a.normalize_request(
        fields(size="2560x1440", ratio="adaptive", frame_indices="[-1]"),
        [upload],
        "public-fl",
    )


def test_resolution_map_matches_published_contract():
    ratios = ("16:9", "21:9", "4:3", "1:1", "3:4", "9:16")
    expected = {
        "768P": (
            (1344, 768),
            (1536, 672),
            (1024, 768),
            (768, 768),
            (768, 1024),
            (768, 1344),
        ),
        "1080P": (
            (1920, 1080),
            (2208, 936),
            (1440, 1080),
            (1080, 1080),
            (1080, 1440),
            (1080, 1920),
        ),
        "2K": (
            (2560, 1440),
            (2944, 1248),
            (1920, 1440),
            (1440, 1440),
            (1440, 1920),
            (1440, 2560),
        ),
    }
    assert {tier: dict(values) for tier, values in RESOLUTION_CANVASES.items()} == {
        tier: dict(zip(ratios, sizes)) for tier, sizes in expected.items()
    }


@pytest.mark.parametrize(
    "size,ratio",
    [
        ((1504, 640), "21:9"),
        ((1152, 864), "4:3"),
        ((992, 992), "1:1"),
        ((864, 1152), "3:4"),
    ],
)
def test_previous_explicit_base_sizes_stay_valid(make_gateway_config, size, ratio):
    a = adapter(make_gateway_config)
    normalized = a.normalize_request(
        fields(size=f"{size[0]}x{size[1]}"), [], "public-fl"
    )
    by_axes = a.normalize_request(
        fields(width=str(size[0]), height=str(size[1])), [], "public-fl"
    )
    assert normalized == by_axes
    assert (normalized["width"], normalized["height"]) == size
    assert normalized["aspect_ratio"] == ratio
    assert "super_resolution" not in normalized


@pytest.mark.parametrize(
    "size,base",
    [
        ((2208, 936), (1504, 640)),
        ((2944, 1248), (1504, 640)),
        ((1440, 1080), (1152, 864)),
        ((1920, 1440), (1152, 864)),
        ((1080, 1080), (992, 992)),
        ((1440, 1440), (992, 992)),
        ((1080, 1440), (864, 1152)),
        ((1440, 1920), (864, 1152)),
    ],
)
def test_official_768p_alignment_does_not_change_sr_base_canvas(
    make_gateway_config, size, base
):
    a = adapter(make_gateway_config)
    normalized = a.normalize_request(
        fields(size=f"{size[0]}x{size[1]}"), [], "public-fl"
    )
    sr = normalized["super_resolution"]
    assert (sr["base_width"], sr["base_height"]) == base


def test_adaptive_allows_strict_21_9_with_three_percent_tolerance(
    make_gateway_config, tmp_path
):
    a = adapter(make_gateway_config)
    upload = _png_upload(tmp_path)
    Image.new("RGB", (1536, 672)).save(upload.path)
    normalized = a.normalize_request(
        fields(resolution="768P", ratio="adaptive"), [upload], "public-fl"
    )
    assert (normalized["width"], normalized["height"]) == (1536, 672)
    Image.new("RGB", (1792, 768)).save(upload.path)
    # Strict 21:9 is ~2.04% away from the rounded official output: now admitted.
    normalized = a.normalize_request(
        fields(resolution="768P", ratio="adaptive"), [upload], "public-fl"
    )
    assert (normalized["width"], normalized["height"]) == (1536, 672)


@pytest.mark.parametrize(
    "size_fields",
    [
        {"resolution": "768P", "ratio": "21:9"},
        {"size": "1536x672"},
        {"width": "1536", "height": "672"},
    ],
)
@pytest.mark.parametrize("ratio_error,accepted", [(0.029, True), (0.031, False)])
def test_fl2va_ratio_tolerance_still_applies_to_explicit_choices(
    make_gateway_config, tmp_path, size_fields, ratio_error, accepted
):
    a = adapter(make_gateway_config)
    upload = _png_upload(tmp_path)
    # Make the image slightly wider than the output, with a controlled error.
    source_ratio = (1536 / 672) / (1 - ratio_error)
    Image.new("RGB", (round(source_ratio * 1000), 1000)).save(upload.path)
    if accepted:
        value = a.normalize_request(fields(**size_fields), [upload], "public-fl")
        assert (value["width"], value["height"]) == (1536, 672)
    else:
        with pytest.raises(GatewayError) as error:
            a.normalize_request(fields(**size_fields), [upload], "public-fl")
        assert error.value.code == "reference_aspect_ratio_mismatch"


@pytest.mark.parametrize("ratio_field", [{}, {"ratio": "adaptive"}, {"aspect_ratio": "adaptive"}])
@pytest.mark.parametrize(
    "source,expected",
    [
        ((1792, 768), (1536, 672)),
        ((1536, 1024), (1152, 768)),
        ((768, 1792), (672, 1536)),
        ((1024, 1536), (768, 1152)),
        ((1001, 1001), (768, 768)),
        ((5760, 2304), (1600, 640)),
    ],
)
def test_native_adaptive_uses_image_ratio_not_six_presets(
    make_gateway_config, tmp_path, source, expected, ratio_field
):
    upload = _png_upload(tmp_path)
    Image.new("RGB", source).save(upload.path)
    a = adapter(make_gateway_config, sr=False)
    normalized = a.normalize_request(
        fields(resolution="768P", **ratio_field), [upload], "public-fl"
    )
    assert (normalized["width"], normalized["height"]) == expected
    assert "super_resolution" not in normalized
    payload = a.build_worker_payload(normalized, [], tmp_path)
    assert payload["size"] == f"{expected[0]}x{expected[1]}"
    # An adaptive non-preset canvas survives persistence without lying about
    # its ratio or broadening classification of legacy explicit sizes.
    task = _task("adaptive-shape")
    task.normalized_request = normalized
    public = VideoTask.from_dict(task.to_dict()).public_dict()
    assert public["resolution"] == "768P"
    assert public["size"] == payload["size"]
    if source == (1536, 1024):
        assert public["ratio"] == "3:2"
        with pytest.raises(GatewayError):
            a.normalize_request(fields(size="1152x768"), [], "public-fl")


def test_adaptive_canvas_tracks_first_keyframe_or_sole_tail(make_gateway_config, tmp_path):
    a = adapter(make_gateway_config)
    first, tail = _png_upload(tmp_path), _png_upload(tmp_path, ordinal=1)
    Image.new("RGB", (1792, 768)).save(first.path)
    Image.new("RGB", (768, 1792)).save(tail.path)
    request = fields(resolution="768P", ratio="adaptive")
    for refs, expected in [([first, tail], (1536, 672)), ([tail, first], (672, 1536))]:
        normalized = a.normalize_request(request, refs, "public-fl")
        assert (normalized["width"], normalized["height"]) == expected
        assert normalized["frame_indices"] == [0, -1]
    normalized = a.normalize_request(
        fields(resolution="768P", ratio="adaptive", frame_indices="[-1]"),
        [tail], "public-fl",
    )
    assert (normalized["width"], normalized["height"]) == (672, 1536)
    assert normalized["frame_indices"] == [-1]
    with pytest.raises(GatewayError) as error:
        a.normalize_request(
            fields(resolution="768P", ratio="adaptive", frame_indices="[-1,0]"),
            [first, tail], "public-fl",
        )
    assert error.value.code == "invalid_frame_indices"


@pytest.mark.parametrize("tier", ["1080P", "2K"])
def test_high_resolution_adaptive_rejects_unlisted_ratio(make_gateway_config, tmp_path, tier):
    upload = _png_upload(tmp_path)
    Image.new("RGB", (1536, 1024)).save(upload.path)
    with pytest.raises(GatewayError, match="no preset matches"):
        adapter(make_gateway_config).normalize_request(
            fields(resolution=tier, ratio="adaptive"), [upload], "public-fl"
        )


def test_native_adaptive_does_not_apply_preset_three_percent_limit(make_gateway_config, tmp_path):
    upload = _png_upload(tmp_path)
    Image.new("RGB", (2360, 1000)).save(upload.path)
    value = adapter(make_gateway_config).normalize_request(
        fields(resolution="768P", ratio="adaptive"), [upload], "public-fl"
    )
    assert (value["width"], value["height"]) == (1568, 672)
