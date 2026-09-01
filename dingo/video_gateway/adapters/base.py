# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-specific adapter contracts kept outside the gateway core."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class UploadedArtifact:
    field_name: str
    ordinal: int
    filename: str
    content_type: str
    path: Path
    size: int
    sha256: str

    def manifest_entry(self, *, relative_to: Path) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "ordinal": self.ordinal,
            "filename": self.filename,
            "content_type": self.content_type,
            "path": str(self.path.relative_to(relative_to)),
            "bytes": self.size,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class WorkerVideoResult:
    b64_json: str
    output_format: str
    inference_time_s: float | None = None
    stage_durations: Mapping[str, float] | None = None


class WorkerStreamConsumer(Protocol):
    """Incrementally consume one Worker response without retaining progress."""

    def consume(self, chunk: Any) -> None: ...

    def finish(self) -> WorkerVideoResult: ...


class VideoBackendAdapter(Protocol):
    """The only model-specific surface consumed by gateway core."""

    def normalize_request(
        self,
        fields: Mapping[str, list[str]],
        uploads: Sequence[UploadedArtifact],
        public_model: str,
    ) -> dict[str, Any]: ...

    def build_worker_payload(
        self,
        normalized: Mapping[str, Any],
        manifest: Sequence[Mapping[str, Any]],
        task_root: Path,
    ) -> dict[str, Any]: ...

    def estimate_encoded_reference_bytes(
        self, manifest: Sequence[Mapping[str, Any]]
    ) -> int: ...

    @property
    def max_encoded_reference_bytes(self) -> int: ...

    def capabilities(self, *, max_result_bytes: int) -> Mapping[str, Any]: ...

    def create_worker_stream_consumer(self) -> WorkerStreamConsumer: ...

    def consume_worker_stream(self, chunks: Sequence[Any]) -> WorkerVideoResult: ...

    def validate_artifact(
        self, path: Path, normalized: Mapping[str, Any]
    ) -> dict[str, Any]: ...

    def prepare_artifact(
        self, path: Path, normalized: Mapping[str, Any]
    ) -> None: ...
