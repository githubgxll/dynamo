# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Common admission and exact-aspect pixel cropping for uploaded SR sources."""
import importlib.util
from fractions import Fraction
from pathlib import Path

import pytest

from dingo.common.video_sr_only import source_crop_box
from dingo.video_gateway.adapters.h3_resolution import BASE_CANVASES, RESOLUTION_CANVASES
from dingo.video_gateway.adapters.h3_output_policy import plan_h3_output
from dingo.common.video_super_resolution import H3_SR_PROFILE_V3

OVERLAY = Path(__file__).resolve().parents[2] / 'container/deps/vllm/patches/h3_seedvr2_v3/sources/vllm_omni/diffusion/models/seedvr2'
spec = importlib.util.spec_from_file_location('uploaded_sr_support_test', OVERLAY / 'long_video_support.py')
support = importlib.util.module_from_spec(spec); spec.loader.exec_module(support)


@pytest.mark.parametrize('ratio', list(BASE_CANVASES))
@pytest.mark.parametrize('tier', ['1080P', '2K'])
def test_all_normal_768p_outputs_are_admitted_with_exact_model_aspect(ratio, tier):
    sw, sh = BASE_CANVASES[ratio]; dw, dh = RESOLUTION_CANVASES[tier][ratio]
    plan = plan_h3_output(dw, dh, sr_profile=H3_SR_PROFILE_V3)
    config = support.options('single_encode', f'{dw}x{dh}', plan.window_frames, source_mode='uploaded_video')
    crop = support.validate_source(config, sw, sh, plan.sr_width, plan.sr_height)
    assert crop == source_crop_box(sw, sh, dw, dh, plan.sr_width, plan.sr_height)
    x, y, w, h = crop
    assert all(v % 2 == 0 for v in crop)
    assert x + w <= sw and y + h <= sh
    assert w * plan.sr_height == h * plan.sr_width


def test_failed_21_9_case_crops_without_resizing_source_pixels():
    assert source_crop_box(1536, 672, 2208, 936, 2256, 960) == (16, 16, 1504, 640)


@pytest.mark.parametrize('size', [(256, 256), (1048, 1048), (1000, 1000)])
def test_bounded_noncanonical_square_sources_are_supported(size):
    crop = source_crop_box(*size, 1080, 1080, 1088, 1088)
    assert crop == (0, 0, *size)


@pytest.mark.parametrize('source, output, internal', [
    ((1536,672), (1920,1080), (2016,1152)),
    ((254,256), (1080,1080), (1088,1088)),
    ((1080,1080), (1080,1080), (1088,1088)),
    ((768,768), (1080,1080), (1920,1080)),
])
def test_uploaded_mode_does_not_widen_source_bounds_or_internal_canvases(source, output, internal):
    with pytest.raises(ValueError):source_crop_box(*source, *output, *internal)


def test_default_generation_mode_still_requires_its_fixed_base():
    config = support.options('single_encode', '2208x936', 33)
    with pytest.raises(ValueError):support.validate_source(config,1536,672,2256,960)
    assert support.validate_source(config,1504,640,2256,960) is None


def test_uploaded_pixel_crop_preserves_visible_yuv_planes_and_timestamps():
    import av
    import numpy as np
    raw = np.arange(672*3//2*1536, dtype=np.uint8).reshape(672*3//2,1536)
    frame = av.VideoFrame.from_ndarray(raw, format='yuv420p')
    frame.pts = 5; frame.time_base = Fraction(1,24)
    crop = support.crop_uploaded_frame(frame,(16,16,1504,640))
    assert (crop.width,crop.height,crop.pts,crop.time_base)==(1504,640,5,Fraction(1,24))
    original = raw.reshape(-1);actual=crop.to_ndarray().reshape(-1)
    y = original[:672*1536].reshape(672,1536)[16:656,16:1520]
    u = original[672*1536:672*1536*5//4].reshape(336,768)[8:328,8:760]
    v = original[672*1536*5//4:].reshape(336,768)[8:328,8:760]
    assert np.array_equal(actual,np.concatenate([y.reshape(-1),u.reshape(-1),v.reshape(-1)]))


@pytest.mark.parametrize('output', [(1800,1080),(2400,1440)])
def test_internal_legacy_pipeline_canvases_remain_supported(output):
    plan = plan_h3_output(*output, sr_profile=H3_SR_PROFILE_V3)
    assert source_crop_box(1280,768,*output,plan.sr_width,plan.sr_height)==(0,0,1280,768)


def test_dynamic_sr_preserves_edges_when_internal_grid_exceeds_delivery_size():
    import numpy as np

    config = support.options("single_encode", "1800x1080", 33, geometry_mode="full_frame_resize")
    assert support.validate_source(config, 1280, 768, 1808, 1088) is None
    writer = support.writer_spec(config, 1808, 1088)
    assert writer["crop_box"] == [0, 0, 1808, 1088]
    assert writer["delivery_transform"] == "lanczos"
    image = np.zeros((1088, 1808, 3), dtype=np.uint8)
    image[:, :3] = (255, 0, 0)
    image[-3:, :] = (0, 255, 0)
    frame = support.delivery_array(image, 1800, 1080, support.NullMetrics(), crop_box=writer["crop_box"])
    out = frame.to_ndarray(format="rgb24")
    assert out.shape == (1080, 1800, 3)
    assert out[100, 0, 0] > 200 and out[-1, 100, 1] > 200
