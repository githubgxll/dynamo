# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned public contracts for metrics.model_execution.

Stage names and boundaries are immutable within a schema version. Add a new
schema for breaking changes; clients may ignore unfamiliar optional stages.
Engine-specific field extraction belongs in the backend adapter, not here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

MODEL_EXECUTION_KEY = "_model_execution_v1"
MINIMAX_H3_SCHEMA = "minimax_h3.v1"


@dataclass(frozen=True, slots=True)
class StageDefinition:
    description: str
    parent: str | None = None


@dataclass(frozen=True, slots=True)
class TimingSchema:
    id: str
    description: str
    stages: Mapping[str, StageDefinition]
    unit: str = "seconds"

    def json_schema(self) -> dict[str, Any]:
        """Export a strict producer schema; all individual stages are optional."""
        properties = {}
        for name, stage in self.stages.items():
            definition = {
                "type": "number", "minimum": 0,
                "description": stage.description,
            }
            if stage.parent is not None:
                definition["x-parent-stage"] = stage.parent
            properties[name] = definition
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": self.id,
            "description": self.description,
            "type": "object",
            "required": ["schema", "unit", "stages"],
            "additionalProperties": False,
            "properties": {
                "schema": {"const": self.id},
                "unit": {"const": self.unit},
                "stages": {
                    "type": "object", "minProperties": 1,
                    "additionalProperties": False, "properties": properties,
                },
            },
        }


SCHEMAS: Mapping[str, TimingSchema] = MappingProxyType({
    MINIMAX_H3_SCHEMA: TimingSchema(
        id=MINIMAX_H3_SCHEMA,
        description=(
            "MiniMax-H3 request-mode pipeline timings for the current attempt. "
            "Stages may overlap or nest; values are not a partition of total time. "
            "Only the engine-returned profiler sample is represented, not a max "
            "or sum across GPU ranks. Missing stages mean unavailable."
        ),
        stages=MappingProxyType({
            "encode_prompt": StageDefinition(
                "Prompt/condition encoder, including visual inputs when used."
            ),
            "diffuse": StageDefinition("Diffusion denoising stage."),
            "decode": StageDefinition("Combined output video/audio decoding."),
            "video_decode": StageDefinition("Video VAE latent decoding.", "decode"),
            "audio_decode": StageDefinition("Audio VAE latent decoding.", "decode"),
            "reference_video_prepare": StageDefinition(
                "Reference-video preparation method: probe, validation and "
                "transcoding when performed here. Rank 0 does this work; "
                "other ranks may report an empty invocation."
            ),
            "reference_visual_encode": StageDefinition(
                "Combined reference-image and reference-video VAE condition encoding."
            ),
            "reference_audio_encode": StageDefinition(
                "Combined embedded-video and standalone-audio condition encoding."
            ),
        }),
    ),
})


def normalize_model_execution(value: Any) -> dict[str, Any] | None:
    """Bound optional telemetry without making missing/new metrics fail a task.

    Old Gateways omit unknown schemas. Invalid/unknown stages are omitted;
    never coerce strings, booleans, NaN or infinity into valid measurements.
    """
    if not isinstance(value, Mapping):
        return None
    schema_id = value.get("schema")
    if not isinstance(schema_id, str):
        return None
    schema = SCHEMAS.get(schema_id)
    if schema is None or value.get("unit") != schema.unit:
        return None
    raw = value.get("stages")
    if not isinstance(raw, Mapping) or len(raw) > 32:
        return None
    stages = {}
    for name in schema.stages:
        duration = raw.get(name)
        if type(duration) not in (int, float):
            continue
        try:
            seconds = float(duration)
        except OverflowError:
            continue
        if math.isfinite(seconds) and seconds >= 0:
            stages[name] = seconds
    if not stages:
        return None
    return {"schema": schema.id, "unit": schema.unit, "stages": stages}


def with_model_execution(
    request: Mapping[str, Any], value: Any
) -> dict[str, Any]:
    """Store optional telemetry in an existing extensible map for rolling upgrades."""
    result = dict(request)
    result.pop(MODEL_EXECUTION_KEY, None)
    normalized = normalize_model_execution(value)
    if normalized is not None:
        result[MODEL_EXECUTION_KEY] = normalized
    return result


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Export a model timing JSON Schema")
    parser.add_argument("schema", choices=sorted(SCHEMAS))
    args = parser.parse_args()
    print(json.dumps(SCHEMAS[args.schema].json_schema(), ensure_ascii=False, indent=2))
