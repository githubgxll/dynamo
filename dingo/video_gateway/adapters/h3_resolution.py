# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named delivery tiers; all admission still belongs to h3_output_policy."""

from math import gcd, sqrt
from types import MappingProxyType

from dingo.common.video_super_resolution import H3_SR_PROFILE_V3
from dingo.video_gateway.adapters.h3_output_policy import (
    H3OutputPlan, H3_SR_V3_PRESETS, h3_sr_tier, plan_h3_output,
)
from dingo.video_gateway.adapters.h3_shape import relative_ratio_error

# Official API output measurements (4s text-to-video, 2026-11-10).
# Delivery presets are independent of the SR profile's internal base canvases.
BASE_CANVASES = MappingProxyType(
    {
        "16:9": (1344, 768),
        "21:9": (1536, 672),
        "4:3": (1024, 768),
        "1:1": (768, 768),
        "3:4": (768, 1024),
        "9:16": (768, 1344),
    }
)
# Read high-resolution geometry from the existing whitelist, not a second table.
RESOLUTION_CANVASES = MappingProxyType(
    {
        "768P": BASE_CANVASES,
        **{
            tier: MappingProxyType(
                {
                    preset[-1]: size
                    for size, preset in H3_SR_V3_PRESETS.items()
                    if size not in {(1800, 1080), (2400, 1440)}
                    and h3_sr_tier(
                        {
                            "width": size[0],
                            "height": size[1],
                            "super_resolution": {"profile": H3_SR_PROFILE_V3},
                        }
                    )
                    == wire_tier
                }
            )
            for tier, wire_tier in (("1080P", "1080p"), ("2K", "2k"))
        },
    }
)


def plan_fl2va_adaptive(
    source_width: int,
    source_height: int,
    resolution: str,
    *,
    sr_profile: str | None,
) -> H3OutputPlan:
    """Resolve image-derived geometry without widening explicit-size admission.

    Native 768P follows the measured H3 short-edge/pixel-budget/32-alignment
    rule. Higher tiers retain the verified SR whitelist and 3% match limit.
    The caller supplies the first keyframe (or the sole last keyframe).
    """
    ratio = source_width / source_height
    if resolution != "768P":
        canvases = RESOLUTION_CANVASES[resolution]
        name = min(
            canvases,
            key=lambda key: relative_ratio_error(*canvases[key], ratio),
        )
        width, height = canvases[name]
        if relative_ratio_error(width, height, ratio) > 0.03:
            raise ValueError(
                "no preset matches the keyframe aspect ratio within 3%; "
                "provide a compatible image or explicit size"
            )
        return plan_h3_output(
            width, height, requested_aspect_ratio=name, sr_profile=sr_profile
        )

    # Preserve profile validation while keeping adaptive geometry local to H3.
    plan_h3_output(768, 768, sr_profile=sr_profile)
    width, height = (768 * ratio, 768.0) if ratio >= 1 else (768.0, 768 / ratio)
    if width * height > 768 * 1344:
        scale = sqrt((768 * 1344) / (width * height))
        width, height = width * scale, height * scale
    width, height = round(width / 32) * 32, round(height / 32) * 32
    # Reuse names only for exact preset canvases. Other ratios must never be
    # mislabeled as the nearest named ratio (e.g. 1152x768 is 3:2).
    divisor = gcd(width, height)
    name = next(
        (key for key, size in BASE_CANVASES.items() if size == (width, height)),
        f"{width // divisor}:{height // divisor}",
    )
    return H3OutputPlan(width, height, width, height, name)


def classify_resolution(
    width: int, height: int, *, super_resolution: bool
) -> str | None:
    size = (width, height)
    if super_resolution:
        for tier in ("1080P", "2K"):
            if size in RESOLUTION_CANVASES[tier].values():
                return tier
        # Exact compatibility sizes remain valid explicit-size requests.
        return {
            (1800, 1080): "1080P",
            (2400, 1440): "2K",
            (1890, 1080): "1080P",
            (1080, 1890): "1080P",
            (2520, 1440): "2K",
            (1440, 2520): "2K",
        }.get(size)
    return "768P" if size in BASE_CANVASES.values() else None
