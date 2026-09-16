"""Private, task-scoped binary result handoff; no public request path accepts it."""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import math
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

BINARY_RESULT_WRITER: contextvars.ContextVar[BinaryResultWriter | None] = (
    contextvars.ContextVar("dingo_binary_result_writer", default=None)
)
INLINE_RESULT_FORMAT = "binary_mp4_inline_v1"


def normalize_inline_result(value: Any) -> dict[str, Any]:
    """Bound the small private status payload; never admit Base64 or paths."""
    if not isinstance(value, dict) or value.get("status") not in {
        "completed",
        "failed",
        "cancelled",
    }:
        raise ValueError("inline Worker response is not terminal")
    result = {"status": value["status"]}
    if value["status"] == "completed":
        data = value.get("data")
        if (
            not isinstance(data, list)
            or len(data) != 1
            or not isinstance(data[0], dict)
            or set(data[0]) != {"output_format", "artifact"}
            or data[0]["output_format"] != "mp4"
        ):
            raise ValueError(
                "inline Worker response must contain one binary MP4 descriptor"
            )
        result["data"] = [
            {
                "output_format": "mp4",
                "artifact": validate_descriptor(data[0]["artifact"]),
            }
        ]
    elif "error" in value:
        error = value["error"]
        if isinstance(error, str) and len(error) <= 1024:
            result["error"] = error
        elif isinstance(error, dict) and all(
            isinstance(error.get(k), str) and len(error[k]) <= limit
            for k, limit in [("code", 128), ("message", 1024)]
        ):
            result["error"] = {k: error[k] for k in ["code", "message"]}
        else:
            raise ValueError("invalid inline Worker error")
    if value.get("inference_time_s") is not None:
        duration = value["inference_time_s"]
        if (
            type(duration) not in {int, float}
            or not math.isfinite(duration)
            or duration < 0
        ):
            raise ValueError("invalid inline inference duration")
        result["inference_time_s"] = duration
    if value.get("stage_durations") is not None:
        stages = value["stage_durations"]
        if (
            not isinstance(stages, dict)
            or len(stages) > 32
            or any(
                not isinstance(k, str)
                or len(k) > 128
                or type(v) not in {int, float}
                or not math.isfinite(v)
                or v < 0
                for k, v in stages.items()
            )
        ):
            raise ValueError("invalid inline stage durations")
        result["stage_durations"] = dict(stages)
    if len(json.dumps(result, ensure_ascii=False).encode()) > 8192:
        raise ValueError("inline Worker result exceeds metadata limit")
    return result


def validate_descriptor(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "filename",
        "bytes",
        "sha256",
    }:
        raise ValueError("invalid binary result descriptor")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or not isinstance(value["filename"], str)
        or re.fullmatch(r"worker-video-[0-9a-f]{32}\.mp4", value["filename"]) is None
        or type(value["bytes"]) is not int
        or value["bytes"] <= 0
        or not isinstance(value["sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None
    ):
        raise ValueError("invalid binary result descriptor")
    return dict(value)


class BinaryResultWriter:
    def __init__(self, root: Path, max_bytes: int = 128 * 1024 * 1024):
        self.root = root
        self.max_bytes = max_bytes
        self.used = False
        self.stage_durations: dict[str, float] = {}
        self._submitted_at: float | None = None

    def _write(self, data: bytes) -> dict[str, Any]:
        started = time.perf_counter()
        self.stage_durations["artifact_queue_s"] = max(
            0.0, started - (self._submitted_at or started)
        )
        name = f"worker-video-{uuid.uuid4().hex}.mp4"
        temporary = self.root / (name + ".part")
        final = self.root / name
        renamed = False
        try:
            mark = time.perf_counter()
            digest = hashlib.sha256(data).hexdigest()
            self.stage_durations["artifact_hash_s"] = time.perf_counter() - mark
            mark = time.perf_counter()
            with temporary.open("xb") as stream:
                self.stage_durations["artifact_open_s"] = time.perf_counter() - mark
                mark = time.perf_counter()
                stream.write(data)
                stream.flush()
                self.stage_durations["artifact_write_s"] = time.perf_counter() - mark
                mark = time.perf_counter()
                os.fsync(stream.fileno())
                self.stage_durations["artifact_file_fsync_s"] = (
                    time.perf_counter() - mark
                )
                mark = time.perf_counter()
            self.stage_durations["artifact_close_s"] = time.perf_counter() - mark
            mark = time.perf_counter()
            os.replace(temporary, final)
            self.stage_durations["artifact_rename_s"] = time.perf_counter() - mark
            renamed = True
            mark = time.perf_counter()
            descriptor = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self.stage_durations["artifact_dir_fsync_s"] = time.perf_counter() - mark
            return dict(schema_version=1, filename=name, bytes=len(data), sha256=digest)
        finally:
            if not renamed:
                temporary.unlink(missing_ok=True)
            self._finished_at = time.perf_counter()
            self.stage_durations["artifact_work_s"] = self._finished_at - started

    async def write(self, data: bytes) -> dict[str, Any]:
        if self.used:
            raise RuntimeError("multiple binary video results are not supported")
        if not isinstance(data, bytes) or not 0 < len(data) <= self.max_bytes:
            raise ValueError("binary video result exceeds configured size or is empty")
        self.used = True
        self._submitted_at = time.perf_counter()
        work = asyncio.create_task(asyncio.to_thread(self._write, data))
        cancelled = False
        while True:
            try:
                result = await asyncio.shield(work)
                break
            except asyncio.CancelledError:
                cancelled = True
                if work.done():
                    work.result()
                    raise
        if cancelled:
            raise asyncio.CancelledError
        self.stage_durations["artifact_resume_s"] = max(
            0.0, time.perf_counter() - self._finished_at
        )
        return result
