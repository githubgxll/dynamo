# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dingo.video_gateway.ids import MonotonicULIDGenerator


def test_video_ids_are_unique_and_lexically_monotonic_within_a_millisecond():
    generator = MonotonicULIDGenerator()

    values = [generator.new(now_ms=1_787_000_000_000) for _ in range(100)]

    assert values == sorted(values)
    assert len(set(values)) == 100
    assert all(value.startswith("video-") and len(value) == 32 for value in values)
