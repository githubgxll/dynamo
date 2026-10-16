# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt native H3 profiler names to the versioned public timing contract."""

from collections.abc import Mapping
from typing import Any

from dingo.common.video_timing_schemas import (
    MINIMAX_H3_SCHEMA,
    normalize_model_execution,
)

NATIVE_STAGE_NAMES = {
    "encode_prompt": "encode_prompt",
    "diffuse": "diffuse",
    "decode": "decode",
    "_prepare_reference_videos": "reference_video_prepare",
    "_encode_visual_conditions": "reference_visual_encode",
    "_encode_reference_audio_conditions": "reference_audio_encode",
    "video_vae.decode_latent": "video_decode",
    "audio_vae.decode_latent": "audio_decode",
}


def extract_model_execution(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    # Do not mix orchestration *_ms fields or other models into this schema.
    return normalize_model_execution({
        "schema": MINIMAX_H3_SCHEMA,
        "unit": "seconds",
        "stages": {
            public: raw[f"MiniMaxH3Pipeline.{native}"]
            for native, public in NATIVE_STAGE_NAMES.items()
            if f"MiniMaxH3Pipeline.{native}" in raw
        },
    })
