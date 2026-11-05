# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""H3-only output geometry contract; SR execution is not enabled by this module.

The adapter selects this policy, never a substring of a public model name.
The adapter selects a configured profile for validation. Submission and dispatch
separately require a compatible, ready Worker; geometry alone is not execution.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import gcd, lcm
from types import MappingProxyType

from dingo.common.video_super_resolution import H3_SR_PROFILE, H3_SR_PROFILE_V2, H3_SR_PROFILE_V3
from dingo.video_gateway.adapters.h3_shape import resolve_output_shape

H3_SEEDVR2_PROFILE = H3_SR_PROFILE
H3_SR_BASE_SIZES = MappingProxyType(
    {
        (1890, 1080): (1344, 768),
        (2520, 1440): (1344, 768),
        (1080, 1890): (768, 1344),
        (1440, 2520): (768, 1344),
    }
)


H3_SR_V3_PRESETS = MappingProxyType({(1920, 1080): (1344, 768, 2016, 1152, 33, '16:9'), (2560, 1440): (1344, 768, 2576, 1472, 65, '16:9'), (2208, 936): (1504, 640, 2256, 960, 33, '21:9'), (2944, 1248): (1504, 640, 3008, 1280, 65, '21:9'), (1440, 1080): (1152, 864, 1472, 1104, 33, '4:3'), (1920, 1440): (1152, 864, 1920, 1440, 65, '4:3'), (1080, 1080): (992, 992, 1088, 1088, 33, '1:1'), (1440, 1440): (992, 992, 1440, 1440, 65, '1:1'), (1080, 1440): (864, 1152, 1104, 1472, 33, '3:4'), (1440, 1920): (864, 1152, 1440, 1920, 65, '3:4'), (1080, 1920): (768, 1344, 1152, 2016, 33, '9:16'), (1440, 2560): (768, 1344, 1472, 2576, 65, '9:16'), (1800, 1080): (1280, 768, 1840, 1104, 33, '16:9'), (2400, 1440): (1280, 768, 2400, 1440, 65, '16:9')})

# Product tiers are exact canvas groups, independent of orientation, short edge
# (the 21:9 presets use 936/1248), or a tunable SR window size.
_SR_1080P_CANVASES = frozenset({
    (1920, 1080), (2208, 936), (1440, 1080), (1080, 1080),
    (1080, 1440), (1080, 1920), (1800, 1080),
    (1890, 1080), (1080, 1890),
})
_SR_2K_CANVASES = frozenset({
    (2560, 1440), (2944, 1248), (1920, 1440), (1440, 1440),
    (1440, 1920), (1440, 2560), (2400, 1440),
    (2520, 1440), (1440, 2520),
})


def h3_sr_tier(normalized: Mapping) -> str | None:
    """Classify only SR-enabled requests admitted by the selected H3 policy."""
    sr = normalized.get("super_resolution")
    if not isinstance(sr, Mapping) or not isinstance(sr.get("profile"), str):
        return None
    size = (normalized.get("width"), normalized.get("height"))
    try:
        plan = plan_h3_output(*size, sr_profile=sr["profile"])
    except (ValueError, TypeError):
        return None
    if plan.sr_profile is None:
        return None
    if size in _SR_1080P_CANVASES:
        return "1080p"
    if size in _SR_2K_CANVASES:
        return "2k"
    return None

@dataclass(frozen=True, slots=True)
class H3OutputPlan:
    output_width: int
    output_height: int
    base_width: int
    base_height: int
    aspect_ratio: str
    sr_profile: str | None = None
    sr_width: int | None = None
    sr_height: int | None = None
    window_frames: int | None = None


def plan_h3_output(
    width: int,
    height: int,
    *,
    requested_aspect_ratio: str | None = None,
    sr_profile: str | None = None,
) -> H3OutputPlan:
    """Resolve an exact delivery size without rounding/correcting user input.

    No profile means precisely the existing base geometry policy. Unknown profiles
    fail closed, including for otherwise valid base sizes. Internal SR alignment is
    calculated on the reduced ratio, not by independently rounding width/height."""
    if type(width) is not int or type(height) is not int:
        raise ValueError("width and height must be integers")
    if sr_profile == H3_SR_PROFILE_V3 and (width, height) in H3_SR_V3_PRESETS:
        bw, bh, sw, sh, window, ratio = H3_SR_V3_PRESETS[(width, height)]
        if requested_aspect_ratio not in (None, ratio):
            raise ValueError("aspect_ratio conflicts with the requested output size")
        return H3OutputPlan(width, height, bw, bh, ratio, sr_profile, sw, sh, window)
    if sr_profile == H3_SR_PROFILE_V3 and (width, height) == (1280, 768):
        if requested_aspect_ratio not in (None, "16:9"):
            raise ValueError("aspect_ratio conflicts with the verified base canvas")
        return H3OutputPlan(width,height,width,height,"16:9")
    if sr_profile == H3_SR_PROFILE_V2:
        # The 0.32 handoff explicitly validated this 5:3 base canvas. Its
        # native aspect_ratio compatibility label is 16:9; dimensions win.
        if (width, height) in {(1800, 1080), (2400, 1440)}:
            if requested_aspect_ratio not in (None, "16:9"):
                raise ValueError("aspect_ratio conflicts with the verified SR profile")
            internal = (1840, 1104) if height == 1080 else (2400, 1440)
            return H3OutputPlan(width, height, 1280, 768, "16:9", sr_profile, *internal)
        if (width, height) == (1280, 768):
            if requested_aspect_ratio not in (None, "16:9"):
                raise ValueError("aspect_ratio conflicts with the verified base canvas")
            return H3OutputPlan(width, height, width, height, "16:9")
    if sr_profile not in (None, H3_SEEDVR2_PROFILE, H3_SR_PROFILE_V2, H3_SR_PROFILE_V3):
        raise ValueError("unsupported MiniMax-H3 output profile")
    base = H3_SR_BASE_SIZES.get((width, height)) if sr_profile == H3_SEEDVR2_PROFILE else None
    if base is None:
        shape = resolve_output_shape(
            width, height, requested_aspect_ratio=requested_aspect_ratio
        )
        return H3OutputPlan(width, height, width, height, shape.aspect_ratio)

    base_width, base_height = base
    shape = resolve_output_shape(
        base_width, base_height, requested_aspect_ratio=requested_aspect_ratio
    )
    if width * base_height != height * base_width:
        raise ValueError("H3 SR policy must preserve the exact base aspect ratio")
    divisor = gcd(width, height)
    ratio_width, ratio_height = width // divisor, height // divisor
    alignment = lcm(16 // gcd(ratio_width, 16), 16 // gcd(ratio_height, 16))
    multiplier = ((divisor + alignment - 1) // alignment) * alignment
    sr_width, sr_height = ratio_width * multiplier, ratio_height * multiplier
    # The pinned SeedVR2 resident profile uses a 17-frame window.
    if sr_width * sr_height > 3791872 or sr_width * sr_height * 17 > 64461824:
        raise ValueError("H3 SR internal canvas exceeds the verified profile budget")
    return H3OutputPlan(
        width,
        height,
        base_width,
        base_height,
        shape.aspect_ratio,
        H3_SEEDVR2_PROFILE,
        sr_width,
        sr_height,
    )
