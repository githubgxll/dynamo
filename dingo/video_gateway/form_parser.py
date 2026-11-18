# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming multipart parser that persists uploads before model dispatch."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from aiohttp import BodyPartReader, web
from aiohttp.http_exceptions import BadHttpMessage

from dingo.video_gateway.adapters.base import UploadedArtifact
from dingo.video_gateway.artifact_store import FileArtifactStore
from dingo.video_gateway.config import MediaConfig
from dingo.video_gateway.errors import GatewayError
from dingo.video_gateway.file_io import opened_file, run_file_io


@dataclass(slots=True)
class ParsedMultipart:
    fields: dict[str, list[str]]
    uploads: list[UploadedArtifact]
    upload_root: Path
    total_bytes: int


async def _read_multipart(awaitable):
    # Scope these exceptions to aiohttp parsing. Filesystem failures still
    # propagate as server errors.
    try:
        return await awaitable
    except (ValueError, AssertionError, BadHttpMessage) as exc:
        raise GatewayError(400, "invalid_multipart", "malformed multipart/form-data body") from exc


async def _read_text(part, *, limit: int) -> str:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await _read_multipart(part.read_chunk(size=64 * 1024))
        if not chunk:
            break
        size += len(chunk)
        if size > limit:
            raise GatewayError(
                413,
                "field_too_large",
                f"multipart field {part.name!r} exceeds {limit} bytes",
                part.name,
            )
        chunks.append(chunk)
    try:
        return b"".join(chunks).decode(part.get_charset(default="utf-8"))
    except (UnicodeDecodeError, LookupError) as exc:
        raise GatewayError(
            400,
            "invalid_field_encoding",
            f"multipart field {part.name!r} is not valid text",
            part.name,
        ) from exc


async def parse_multipart(
    request: web.Request,
    artifacts: FileArtifactStore,
    limits: MediaConfig,
    *,
    upload_root: Path | None = None,
    allowed_file_fields: frozenset[str] | None = None,
    max_files: int | None = None,
) -> ParsedMultipart:
    if request.content_type != "multipart/form-data":
        raise GatewayError(
            415,
            "unsupported_media_type",
            "video requests require multipart/form-data",
        )
    if upload_root is None:
        upload_root = await artifacts.create_upload()
    fields: dict[str, list[str]] = {}
    uploads: list[UploadedArtifact] = []
    total_file_bytes = 0
    part_count = 0
    try:
        reader = await _read_multipart(request.multipart())
        while True:
            part = await _read_multipart(reader.next())
            if part is None:
                break
            if not isinstance(part, BodyPartReader):
                raise GatewayError(400, "invalid_multipart", "nested multipart parts are not supported")
            part_count += 1
            if part_count > limits.max_parts:
                raise GatewayError(413, "too_many_parts", "too many multipart parts")
            name = part.name
            if not name:
                raise GatewayError(
                    400, "invalid_multipart", "multipart part has no name"
                )
            if part.filename is None:
                fields.setdefault(name, []).append(
                    await _read_text(part, limit=limits.max_text_field_bytes)
                )
                continue

            if allowed_file_fields is not None and name not in allowed_file_fields:
                raise GatewayError(400, "unsupported_parameter",
                                   f"unsupported file field {name!r}; use base_video", name)
            if max_files is not None and len(uploads) >= max_files:
                raise GatewayError(400, "invalid_source_video", "upload exactly one base_video MP4", "base_video")

            ordinal = len(uploads)
            destination = upload_root / "inputs" / f"{ordinal:03d}.bin"
            temporary = destination.with_suffix(".part")
            digest = hashlib.sha256()
            size = 0
            async with opened_file(temporary.open, "xb") as output:
                while True:
                    chunk = await _read_multipart(part.read_chunk(size=1024 * 1024))
                    if not chunk:
                        break
                    size += len(chunk)
                    total_file_bytes += len(chunk)
                    if size > limits.max_single_file_bytes:
                        raise GatewayError(
                            413,
                            "file_too_large",
                            f"file field {name!r} exceeds "
                            f"{limits.max_single_file_bytes} bytes",
                            name,
                        )
                    if total_file_bytes > limits.max_total_file_bytes:
                        raise GatewayError(
                            413,
                            "payload_too_large",
                            "reference files exceed the aggregate upload limit",
                        )
                    digest.update(chunk)
                    await run_file_io(output.write, chunk)
                await run_file_io(output.flush)
                await run_file_io(os.fsync, output.fileno())
            await run_file_io(os.replace, temporary, destination)
            filename = Path(part.filename).name or f"upload-{ordinal}"
            uploads.append(
                UploadedArtifact(
                    field_name=name,
                    ordinal=ordinal,
                    filename=filename,
                    content_type=part.headers.get(
                        "Content-Type", "application/octet-stream"
                    ),
                    path=destination,
                    size=size,
                    sha256=digest.hexdigest(),
                )
            )
        return ParsedMultipart(
            fields=fields,
            uploads=uploads,
            upload_root=upload_root,
            total_bytes=total_file_bytes,
        )
    except BaseException:
        await artifacts.discard(upload_root)
        raise
