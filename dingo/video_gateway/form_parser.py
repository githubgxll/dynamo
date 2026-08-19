# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming multipart parser that persists uploads before model dispatch."""

from __future__ import annotations

import asyncio
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from aiohttp import web

from dingo.video_gateway.adapters.base import UploadedArtifact
from dingo.video_gateway.artifact_store import FileArtifactStore
from dingo.video_gateway.errors import GatewayError


@dataclass(slots=True)
class ParsedMultipart:
    fields: dict[str, list[str]]
    uploads: list[UploadedArtifact]
    upload_root: Path
    total_bytes: int


async def _read_text(part, *, limit: int = 64 * 1024) -> str:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await part.read_chunk(size=64 * 1024)
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
    except UnicodeDecodeError as exc:
        raise GatewayError(
            400,
            "invalid_field_encoding",
            f"multipart field {part.name!r} is not valid text",
            part.name,
        ) from exc


async def parse_multipart(
    request: web.Request,
    artifacts: FileArtifactStore,
    *,
    max_total_file_bytes: int = 256 * 1024 * 1024,
    max_single_file_bytes: int = 50 * 1024 * 1024,
    max_parts: int = 32,
) -> ParsedMultipart:
    if request.content_type != "multipart/form-data":
        raise GatewayError(
            415,
            "unsupported_media_type",
            "video requests require multipart/form-data",
        )
    upload_root = await artifacts.create_upload()
    fields: dict[str, list[str]] = {}
    uploads: list[UploadedArtifact] = []
    total_file_bytes = 0
    part_count = 0
    try:
        reader = await request.multipart()
        while True:
            part = await reader.next()
            if part is None:
                break
            part_count += 1
            if part_count > max_parts:
                raise GatewayError(413, "too_many_parts", "too many multipart parts")
            name = part.name
            if not name:
                raise GatewayError(
                    400, "invalid_multipart", "multipart part has no name"
                )
            if part.filename is None:
                fields.setdefault(name, []).append(await _read_text(part))
                continue

            ordinal = len(uploads)
            destination = upload_root / "inputs" / f"{ordinal:03d}.bin"
            temporary = destination.with_suffix(".part")
            digest = hashlib.sha256()
            size = 0
            output = temporary.open("xb")
            try:
                while True:
                    chunk = await part.read_chunk(size=1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    total_file_bytes += len(chunk)
                    if size > max_single_file_bytes:
                        raise GatewayError(
                            413,
                            "file_too_large",
                            f"file field {name!r} exceeds {max_single_file_bytes} bytes",
                            name,
                        )
                    if total_file_bytes > max_total_file_bytes:
                        raise GatewayError(
                            413,
                            "payload_too_large",
                            "reference files exceed the aggregate upload limit",
                        )
                    digest.update(chunk)
                    await asyncio.to_thread(output.write, chunk)
                await asyncio.to_thread(output.flush)
                await asyncio.to_thread(os.fsync, output.fileno())
            finally:
                output.close()
            await asyncio.to_thread(os.replace, temporary, destination)
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
    except Exception:
        await artifacts.discard(upload_root)
        raise
