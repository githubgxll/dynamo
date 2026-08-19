# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atomic filesystem storage for video task inputs and results."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import os
import shutil
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


class FileArtifactStore:
    """Store artifacts below one configured root with exact-path deletion."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.upload_root = self.root / "_uploads"
        self.upload_root.mkdir(mode=0o700, parents=True, exist_ok=True)

    def _contained(self, path: Path) -> Path:
        resolved = path.resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise RuntimeError(f"artifact path escaped configured root: {resolved}")
        return resolved

    def _lexically_contained(self, path: Path) -> Path:
        absolute = path.absolute()
        if absolute != self.root and self.root not in absolute.parents:
            raise RuntimeError(f"artifact path escaped configured root: {absolute}")
        return absolute

    async def health(self) -> None:
        """Verify that the configured root remains writable without broad cleanup."""

        probe = self.upload_root / f".health-{uuid.uuid4().hex}"

        def _probe() -> None:
            with probe.open("xb") as stream:
                stream.write(b"ok")
                stream.flush()
                os.fsync(stream.fileno())
            probe.unlink()

        await asyncio.to_thread(_probe)

    async def create_upload(self) -> Path:
        path = self.upload_root / uuid.uuid4().hex
        await asyncio.to_thread(path.mkdir, 0o700, True, False)
        inputs = path / "inputs"
        await asyncio.to_thread(inputs.mkdir, 0o700, False, False)
        return self._contained(path)

    async def discard(self, path: Path) -> None:
        target = self._lexically_contained(path)
        if target.is_symlink():
            await asyncio.to_thread(target.unlink)
        elif target.exists():
            resolved = self._contained(target)
            await asyncio.to_thread(shutil.rmtree, resolved)

    async def cleanup_orphan_uploads(self, *, minimum_age_s: float = 3600.0) -> int:
        """Remove only stale staging directories that were never task-owned."""

        cutoff = time.time() - minimum_age_s

        def _cleanup() -> int:
            removed = 0
            for candidate in self.upload_root.iterdir():
                try:
                    metadata = candidate.lstat()
                except FileNotFoundError:
                    continue
                if metadata.st_mtime > cutoff:
                    continue
                if candidate.is_symlink():
                    candidate.unlink(missing_ok=True)
                elif candidate.is_dir():
                    shutil.rmtree(self._contained(candidate))
                else:
                    candidate.unlink(missing_ok=True)
                removed += 1
            return removed

        return await asyncio.to_thread(_cleanup)

    def task_root(self, deployment_id: str, pool_id: str, task_id: str) -> Path:
        return self._contained(
            self.root / deployment_id / "v1" / "pools" / pool_id / "tasks" / task_id
        )

    async def commit_upload(
        self, upload_root: Path, deployment_id: str, pool_id: str, task_id: str
    ) -> Path:
        source = self._contained(upload_root)
        target = self.task_root(deployment_id, pool_id, task_id)
        await asyncio.to_thread(target.parent.mkdir, 0o750, True, True)
        if target.exists():
            raise FileExistsError(f"task artifact directory already exists: {task_id}")
        await asyncio.to_thread(os.replace, source, target)
        return target

    async def write_json(self, path: Path, value: Any) -> None:
        target = self._contained(path)
        await asyncio.to_thread(target.parent.mkdir, 0o750, True, True)
        payload = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        temporary = target.with_name(target.name + ".part-" + uuid.uuid4().hex)

        def _write() -> None:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)

        await asyncio.to_thread(_write)

    async def read_json(self, path: str | Path) -> Any:
        target = self._contained(Path(path))

        def _read() -> Any:
            with target.open("r", encoding="utf-8") as stream:
                return json.load(stream)

        return await asyncio.to_thread(_read)

    async def finalize_b64_mp4(
        self,
        task_root: Path,
        b64_json: str,
        normalized: Mapping[str, Any],
        validator: Callable[[Path, Mapping[str, Any]], dict[str, Any]],
        processor: Callable[[Path, Mapping[str, Any]], None] | None = None,
        *,
        max_result_bytes: int = 128 * 1024 * 1024,
    ) -> tuple[Path, int, str, dict[str, Any]]:
        root = self._contained(task_root)
        result_dir = root / "result"
        temporary_dir = root / "tmp"
        await asyncio.to_thread(result_dir.mkdir, 0o750, True, True)
        await asyncio.to_thread(temporary_dir.mkdir, 0o750, True, True)
        estimated = (len(b64_json) // 4) * 3
        if estimated > max_result_bytes:
            raise RuntimeError("Worker base64 result exceeds configured maximum")
        if len(b64_json) % 4:
            raise RuntimeError("Worker base64 result has invalid padding length")
        temporary = temporary_dir / f"result-{uuid.uuid4().hex}.part"
        final = result_dir / "video.mp4"

        def _decode() -> tuple[int, str]:
            digest = hashlib.sha256()
            written = 0
            chunk_chars = 4 * 1024 * 1024
            with temporary.open("xb") as output:
                for offset in range(0, len(b64_json), chunk_chars):
                    encoded = b64_json[offset : offset + chunk_chars]
                    try:
                        decoded = base64.b64decode(encoded, validate=True)
                    except (binascii.Error, ValueError) as exc:
                        raise RuntimeError(
                            "Worker returned invalid base64 video data"
                        ) from exc
                    written += len(decoded)
                    if written > max_result_bytes:
                        raise RuntimeError("Worker result exceeds configured maximum")
                    digest.update(decoded)
                    output.write(decoded)
                output.flush()
                os.fsync(output.fileno())
            return written, digest.hexdigest()

        def _digest() -> tuple[int, str]:
            digest = hashlib.sha256()
            size = 0
            with temporary.open("rb") as stream:
                for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    size += len(chunk)
                    digest.update(chunk)
            return size, digest.hexdigest()

        try:
            await asyncio.to_thread(_decode)
            if processor is not None:
                await asyncio.to_thread(processor, temporary, normalized)
            media = await asyncio.to_thread(validator, temporary, normalized)
            size, sha256 = await asyncio.to_thread(_digest)
            await asyncio.to_thread(os.replace, temporary, final)
            return final, size, sha256, media
        finally:
            if temporary.exists():
                await asyncio.to_thread(temporary.unlink)

    def result_path(self, path: str | Path) -> Path:
        original = self._lexically_contained(Path(path))
        relative = original.relative_to(self.root)
        cursor = self.root
        for component in relative.parts:
            cursor /= component
            if cursor.is_symlink():
                raise FileNotFoundError(original)
        result = self._contained(original)
        if not result.is_file():
            raise FileNotFoundError(result)
        return result
