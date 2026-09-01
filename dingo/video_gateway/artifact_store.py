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
import re
import shutil
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dingo.common.video_task_protocol import detached_attempt_root
from dingo.video_gateway.errors import ResultTooLarge


@dataclass(frozen=True, slots=True)
class ArtifactCapacity:
    total_bytes: int
    used_bytes: int
    free_bytes: int


@dataclass(frozen=True, slots=True)
class TaskArtifactCandidate:
    task_id: str
    path: Path
    age_s: float
    manifest_valid: bool


class FileArtifactStore:
    """Store artifacts below one configured root with exact-path deletion."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.upload_root = self.root / "_uploads"
        self.upload_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.trash_root = self.root / "_trash"
        self.trash_root.mkdir(mode=0o700, parents=True, exist_ok=True)

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

    def _symlink_free(self, path: Path) -> Path:
        absolute = self._lexically_contained(path)
        relative = absolute.relative_to(self.root)
        cursor = self.root
        for component in relative.parts:
            cursor /= component
            if cursor.is_symlink():
                raise RuntimeError(f"artifact path contains a symlink: {cursor}")
        return self._contained(absolute)

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

    @staticmethod
    def _tree_size(path: Path) -> int:
        if path.is_symlink():
            return 0
        if path.is_file():
            return path.stat().st_size
        total = 0
        for root, directories, files in os.walk(path, followlinks=False):
            directories[:] = [
                name for name in directories if not (Path(root) / name).is_symlink()
            ]
            for name in files:
                candidate = Path(root) / name
                try:
                    if not candidate.is_symlink():
                        total += candidate.stat().st_size
                except FileNotFoundError:
                    continue
        return total

    async def discard(self, path: Path) -> int:
        target = self._lexically_contained(path)
        if target.is_symlink():
            try:
                await asyncio.to_thread(target.unlink)
            except FileNotFoundError:
                # Concurrent idempotent DELETE/sweeper cleanup won the race.
                pass
            return 0
        elif target.exists():
            resolved = self._contained(target)
            size = await asyncio.to_thread(self._tree_size, resolved)

            def _remove_tree() -> bool:
                try:
                    shutil.rmtree(resolved)
                except FileNotFoundError:
                    # rmtree may observe a concurrently removed child even
                    # when the root existed at the containment check above.
                    return False
                return True

            removed = await asyncio.to_thread(_remove_tree)
            return size if removed else 0
        return 0

    async def capacity(self) -> ArtifactCapacity:
        usage = await asyncio.to_thread(shutil.disk_usage, self.root)
        return ArtifactCapacity(usage.total, usage.used, usage.free)

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
        self,
        upload_root: Path,
        deployment_id: str,
        pool_id: str,
        task_id: str,
        *,
        artifact_manifest: Mapping[str, Any] | None = None,
    ) -> Path:
        source = self._contained(upload_root)
        target = self.task_root(deployment_id, pool_id, task_id)
        await asyncio.to_thread(target.parent.mkdir, 0o750, True, True)
        if target.exists():
            raise FileExistsError(f"task artifact directory already exists: {task_id}")
        await asyncio.to_thread(os.replace, source, target)
        if artifact_manifest is not None:
            try:
                await self.write_json(target / "_artifact.json", artifact_manifest)
            except Exception:
                await self.discard(target)
                raise
        return target

    async def orphan_task_candidates(
        self,
        deployment_id: str,
        pool_ids: tuple[str, ...],
        *,
        minimum_age_s: float,
    ) -> list[TaskArtifactCandidate]:
        cutoff = time.time() - minimum_age_s
        root_device = (await asyncio.to_thread(self.root.stat)).st_dev

        def _scan() -> list[TaskArtifactCandidate]:
            candidates: list[TaskArtifactCandidate] = []
            for pool_id in pool_ids:
                try:
                    tasks_root = self._symlink_free(
                        self.root
                        / deployment_id
                        / "v1"
                        / "pools"
                        / pool_id
                        / "tasks"
                    )
                except (FileNotFoundError, RuntimeError):
                    continue
                if not tasks_root.is_dir():
                    continue
                with os.scandir(tasks_root) as entries:
                    for entry in entries:
                        try:
                            metadata = entry.stat(follow_symlinks=False)
                        except FileNotFoundError:
                            continue
                        if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                            continue
                        if metadata.st_dev != root_device or metadata.st_mtime > cutoff:
                            continue
                        path = self._lexically_contained(Path(entry.path))
                        manifest_valid = False
                        manifest_path = path / "_artifact.json"
                        try:
                            if manifest_path.is_symlink():
                                raise ValueError("artifact manifest is a symlink")
                            manifest = json.loads(manifest_path.read_text("utf-8"))
                            manifest_valid = (
                                manifest.get("schema_version") == 1
                                and manifest.get("task_id") == entry.name
                                and manifest.get("deployment_id") == deployment_id
                                and manifest.get("pool_id") == pool_id
                            )
                        except (
                            AttributeError,
                            OSError,
                            ValueError,
                            TypeError,
                            json.JSONDecodeError,
                        ):
                            manifest_valid = False
                        candidates.append(
                            TaskArtifactCandidate(
                                task_id=entry.name,
                                path=path,
                                age_s=max(0.0, time.time() - metadata.st_mtime),
                                manifest_valid=manifest_valid,
                            )
                        )
            return candidates

        return await asyncio.to_thread(_scan)

    async def trash_orphan(
        self, candidate: TaskArtifactCandidate, *, dry_run: bool = False
    ) -> Path | None:
        source = self._lexically_contained(candidate.path)
        if source.name != candidate.task_id or source.parent.name != "tasks":
            raise RuntimeError("orphan candidate is not a task directory")
        if dry_run:
            return source
        if not source.exists() and not source.is_symlink():
            return None
        source_parent = self._symlink_free(source.parent)
        target = self.trash_root / f"{uuid.uuid4().hex}-{candidate.task_id}"

        def _move() -> None:
            directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            nofollow = getattr(os, "O_NOFOLLOW", 0)
            source_fd = os.open(source_parent, directory_flags | nofollow)
            trash_fd = os.open(self.trash_root, directory_flags | nofollow)
            try:
                os.rename(
                    source.name,
                    target.name,
                    src_dir_fd=source_fd,
                    dst_dir_fd=trash_fd,
                )
            finally:
                os.close(source_fd)
                os.close(trash_fd)
            os.utime(target, None, follow_symlinks=False)

        await asyncio.to_thread(_move)
        return self._contained(target)

    async def cleanup_trash(self, *, minimum_age_s: float) -> tuple[int, int]:
        cutoff = time.time() - minimum_age_s

        def _cleanup() -> tuple[int, int]:
            removed = 0
            released = 0
            with os.scandir(self.trash_root) as entries:
                for entry in entries:
                    try:
                        metadata = entry.stat(follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    if metadata.st_mtime > cutoff:
                        continue
                    candidate = self._lexically_contained(Path(entry.path))
                    if entry.is_symlink():
                        candidate.unlink(missing_ok=True)
                    elif entry.is_dir(follow_symlinks=False):
                        released += self._tree_size(candidate)
                        shutil.rmtree(self._contained(candidate))
                    else:
                        released += metadata.st_size
                        candidate.unlink(missing_ok=True)
                    removed += 1
            return removed, released

        return await asyncio.to_thread(_cleanup)

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

    def detached_attempt_root(
        self,
        deployment_id: str,
        pool_id: str,
        task_id: str,
        attempt: int,
        execution_token: str,
    ) -> Path:
        path = detached_attempt_root(
            self.root,
            deployment_id,
            pool_id,
            task_id,
            attempt,
            execution_token,
        )
        return self._symlink_free(path)

    async def read_detached_status(
        self,
        deployment_id: str,
        pool_id: str,
        task_id: str,
        attempt: int,
        execution_token: str,
    ) -> dict[str, Any] | None:
        attempt_root = self.detached_attempt_root(
            deployment_id, pool_id, task_id, attempt, execution_token
        )
        path = attempt_root / "worker-status.json"

        def _read() -> dict[str, Any] | None:
            try:
                if path.is_symlink():
                    raise RuntimeError("detached Worker status is a symlink")
                if path.stat().st_size > 64 * 1024:
                    raise RuntimeError("detached Worker status exceeds 64 KiB")
                with path.open("r", encoding="utf-8") as stream:
                    value = json.load(stream)
            except FileNotFoundError:
                return None
            if not isinstance(value, dict):
                raise RuntimeError("detached Worker status is not an object")
            expected = {
                "deployment_id": deployment_id,
                "pool_id": pool_id,
                "task_id": task_id,
                "attempt": attempt,
                "execution_token": execution_token,
            }
            if any(
                value.get(key) != expected_value
                for key, expected_value in expected.items()
            ):
                raise RuntimeError("detached Worker status identity mismatch")
            if value.get("schema_version") != 1:
                raise RuntimeError("unsupported detached Worker status schema")
            return value

        return await asyncio.to_thread(_read)

    async def request_detached_cancel(
        self,
        deployment_id: str,
        pool_id: str,
        task_id: str,
        attempt: int,
        execution_token: str,
    ) -> None:
        attempt_root = self.detached_attempt_root(
            deployment_id, pool_id, task_id, attempt, execution_token
        )
        path = attempt_root / "cancel.requested"

        def _write() -> None:
            attempt_root.mkdir(mode=0o750, parents=True, exist_ok=True)
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
            os.close(descriptor)

        await asyncio.to_thread(_write)

    async def consume_detached_response(
        self,
        deployment_id: str,
        pool_id: str,
        task_id: str,
        attempt: int,
        execution_token: str,
        consumer: Any,
        *,
        expected_sha256: str,
        max_response_bytes: int,
    ) -> int:
        attempt_root = self.detached_attempt_root(
            deployment_id, pool_id, task_id, attempt, execution_token
        )
        path = attempt_root / "worker-response.jsonl"

        def _consume() -> int:
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("detached Worker response is not a regular file")
            digest = hashlib.sha256()
            consumed = 0
            with path.open("rb") as stream:
                while True:
                    remaining = max_response_bytes - consumed
                    if remaining < 0:
                        raise RuntimeError(
                            "detached Worker response exceeds configured maximum"
                        )
                    encoded = stream.readline(remaining + 1)
                    if not encoded:
                        break
                    consumed += len(encoded)
                    if consumed > max_response_bytes:
                        raise RuntimeError(
                            "detached Worker response exceeds configured maximum"
                        )
                    digest.update(encoded)
                    try:
                        item = json.loads(encoded)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(
                            "detached Worker response contains invalid JSON"
                        ) from exc
                    if not isinstance(item, dict):
                        raise RuntimeError(
                            "detached Worker response chunk is not an object"
                        )
                    consumer.consume(item)
            if digest.hexdigest() != expected_sha256:
                raise RuntimeError("detached Worker response checksum mismatch")
            return consumed

        return await asyncio.to_thread(_consume)

    async def finalize_b64_mp4(
        self,
        task_root: Path,
        b64_json: str,
        normalized: Mapping[str, Any],
        validator: Callable[[Path, Mapping[str, Any]], dict[str, Any]],
        processor: Callable[[Path, Mapping[str, Any]], None] | None = None,
        *,
        max_result_bytes: int = 128 * 1024 * 1024,
        publication_scope: str | None = None,
    ) -> tuple[Path, int, str, dict[str, Any]]:
        root = self._contained(task_root)
        result_dir = root / "result"
        temporary_dir = root / "tmp"
        await asyncio.to_thread(result_dir.mkdir, 0o750, True, True)
        await asyncio.to_thread(temporary_dir.mkdir, 0o750, True, True)
        estimated = (len(b64_json) // 4) * 3
        if estimated > max_result_bytes:
            raise ResultTooLarge(
                "Worker base64 result exceeds configured maximum"
            )
        if len(b64_json) % 4:
            raise RuntimeError("Worker base64 result has invalid padding length")
        temporary = temporary_dir / f"result-{uuid.uuid4().hex}.part"
        scope = publication_scope or "legacy"
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", scope) is None:
            raise ValueError("result publication scope is invalid")
        # The task CAS publishes one unique candidate. A stale Gateway can
        # safely unlink its own losing candidate without touching the winner.
        final = result_dir / f"video-{scope}-{uuid.uuid4().hex}.mp4"

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
                        raise ResultTooLarge(
                            "Worker result exceeds configured maximum"
                        )
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
