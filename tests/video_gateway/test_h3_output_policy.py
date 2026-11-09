# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError

import pytest

from dingo.video_gateway.adapters.h3_output_policy import (
    H3_SEEDVR2_PROFILE,
    H3_SR_BASE_SIZES,
    plan_h3_output,
)
from dingo.video_gateway.adapters.h3_shape import resolve_output_shape

CASES = [
    ((1890, 1080), (1344, 768), (1904, 1088), "16:9"),
    ((2520, 1440), (1344, 768), (2576, 1472), "16:9"),
    ((1080, 1890), (768, 1344), (1088, 1904), "9:16"),
    ((1440, 2520), (768, 1344), (1472, 2576), "9:16"),
]


@pytest.mark.parametrize("output,base,internal,ratio", CASES)
def test_exact_mapping_and_alignment(output, base, internal, ratio):
    plan = plan_h3_output(*output, sr_profile=H3_SEEDVR2_PROFILE)
    assert (plan.base_width, plan.base_height) == base
    assert (plan.output_width, plan.output_height) == output
    assert (plan.sr_width, plan.sr_height) == internal
    assert plan.aspect_ratio == ratio
    assert plan.sr_profile == H3_SEEDVR2_PROFILE
    assert all(v % 32 == 0 for v in base)
    assert all(v % 16 == 0 for v in internal)
    assert base[0] * output[1] == output[0] * base[1]
    assert internal[0] * output[1] == output[0] * internal[1]
    assert internal[0] >= output[0] and internal[1] >= output[1]


@pytest.mark.parametrize("output,base,internal,ratio", CASES)
def test_sr_disabled_by_default_and_ratio_conflicts(output, base, internal, ratio):
    with pytest.raises(ValueError):
        plan_h3_output(*output)
    with pytest.raises(ValueError, match="conflicts"):
        plan_h3_output(
            *output, sr_profile=H3_SEEDVR2_PROFILE, requested_aspect_ratio="1:1"
        )


@pytest.mark.parametrize(
    "size",
    [(1344, 768), (768, 1344), (1024, 576), (576, 1024), (512, 512), (1024, 768)],
)
def test_base_plan_unchanged_even_with_sr_profile(size):
    old = resolve_output_shape(*size)
    for profile in (None, H3_SEEDVR2_PROFILE):
        plan = plan_h3_output(*size, sr_profile=profile)
        assert (plan.base_width, plan.base_height) == size
        assert (plan.output_width, plan.output_height) == size
        assert plan.aspect_ratio == old.aspect_ratio
        assert plan.sr_profile is plan.sr_width is plan.sr_height is None


@pytest.mark.parametrize(
    "size",
    [
        (1920, 1080),
        (1920, 1088),
        (2560, 1440),
        (1080, 1920),
        (1440, 2560),
        (2433, 100),
        (199, 361),
        (1024, 1024),
        (1920, 1104),
        (3840, 2160),
        (576, 1344),
        (0, 768),
        (-32, 768),
    ],
)
def test_no_arbitrary_high_resolution_or_new_aspect_ratios(size):
    with pytest.raises(ValueError):
        plan_h3_output(*size, sr_profile=H3_SEEDVR2_PROFILE)


@pytest.mark.parametrize("width", [True, 1024.0, "1024", None])
def test_non_integer_dimensions_rejected(width):
    with pytest.raises(ValueError, match="integers"):
        plan_h3_output(width, 576)


def test_unknown_model_profile_fails_closed():
    with pytest.raises(ValueError, match="profile"):
        plan_h3_output(512, 512, sr_profile="other_model.v1")


def test_policy_and_plan_are_immutable():
    assert set(H3_SR_BASE_SIZES) == {c[0] for c in CASES}
    with pytest.raises(TypeError):
        H3_SR_BASE_SIZES[(2433, 100)] = (512, 512)
    plan = plan_h3_output(512, 512)
    with pytest.raises(FrozenInstanceError):
        plan.output_width = 1024


@pytest.mark.parametrize("output,base,internal,ratio", CASES)
def test_public_parser_does_not_enable_sr_before_worker_support(
    output, base, internal, ratio
):
    from dingo.video_gateway.adapters.minimax_h3 import _parse_size
    from dingo.video_gateway.errors import GatewayError

    for fields in (
        {"size": [f"{output[0]}x{output[1]}"]},
        {"width": [str(output[0])], "height": [str(output[1])]},
    ):
        with pytest.raises(GatewayError):
            _parse_size(fields)


def test_public_parser_base_aliases_and_conflict_unchanged():
    from dingo.video_gateway.adapters.minimax_h3 import _parse_size
    from dingo.video_gateway.errors import GatewayError

    assert _parse_size({"size": ["1344x768"]}) == (1344, 768)
    assert _parse_size({"width": ["1344"], "height": ["768"]}) == (1344, 768)
    for fields in (
        {"size": ["1344x768"], "width": ["1344"], "height": ["768"]},
        {"width": ["1344"]},
        {"short_edge": ["768"], "aspect_ratio": ["16:9"]},
    ):
        with pytest.raises(GatewayError):
            _parse_size(fields)
