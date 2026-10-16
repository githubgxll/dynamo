# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure MiniMax-H3 output-shape validation shared by adapter tests."""

from __future__ import annotations

from dataclasses import dataclass

NAMED_ASPECT_RATIOS: dict[str, float] = {
    "21:9": 21 / 9,
    "16:9": 16 / 9,
    "4:3": 4 / 3,
    "1:1": 1.0,
    "3:4": 3 / 4,
    "9:16": 9 / 16,
}


def align_frame_count(frame_count: int) -> int:
    """Snap upward to the pinned MiniMax-H3 ``17n+5`` frame boundary."""

    if frame_count <= 0:
        return 1
    current = int(frame_count)
    remainder = current % 17
    return current + ((5 - remainder) % 17)


@dataclass(frozen=True, slots=True)
class H3Shape:
    width: int
    height: int
    aspect_ratio: str


def relative_ratio_error(width: int, height: int, ratio: float) -> float:
    return abs((width / height) - ratio) / ratio


def resolve_output_shape(
    width: int,
    height: int,
    *,
    requested_aspect_ratio: str | None = None,
    max_ratio_error: float = 0.05,
) -> H3Shape:
    """Validate the bounded H3 verification shape and resolve its named ratio."""

    if not (256 <= width <= 2048 and 256 <= height <= 2048):
        raise ValueError("width and height must each be between 256 and 2048")
    if width % 32 or height % 32:
        raise ValueError("width and height must be multiples of 32")
    if width * height > 768 * 1344:
        raise ValueError("output pixels must not exceed 768x1344")

    nearest_name, nearest_ratio = min(
        NAMED_ASPECT_RATIOS.items(),
        key=lambda item: relative_ratio_error(width, height, item[1]),
    )
    if relative_ratio_error(width, height, nearest_ratio) > max_ratio_error:
        raise ValueError("size does not match a supported MiniMax-H3 aspect ratio")
    if requested_aspect_ratio is not None:
        if requested_aspect_ratio not in NAMED_ASPECT_RATIOS:
            raise ValueError("aspect_ratio is not supported by MiniMax-H3")
        if requested_aspect_ratio != nearest_name:
            raise ValueError("aspect_ratio conflicts with the requested output size")
    return H3Shape(width=width, height=height, aspect_ratio=nearest_name)
