"""H3-only shared input descriptors and bounded, cancellable raw-file staging.

The executable path uses only the standard library. Staging never imports a
model, decodes an image, or initializes CUDA. Original task files are read-only.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
import time

TYPE = "dingo_h3_shared_inputs_v1"
CAPABILITY = "shared_input_files.v1"
CURRENT = contextvars.ContextVar("h3_shared_input_session", default=None)
MAX_DESCRIPTOR = 256 * 1024
MAX_TOTAL = 256 * 1024 * 1024
_IDENT = re.compile(r"^[A-Za-z0-9_.-]{1,160}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")


def descriptor(payload):
    value = payload.get("input_reference")
    if not isinstance(value, str) or not value.lstrip().startswith("{"):
        return None
    # Other existing H3 JSON envelopes continue through their own adapters.
    if TYPE not in value:
        return None
    if len(value.encode()) > MAX_DESCRIPTOR:
        raise ValueError("shared input descriptor too large")
    result = json.loads(value)
    if result.get("type") != TYPE:
        return None
    if payload.get("input_references") is not None:
        raise ValueError("shared and inline inputs are mutually exclusive")
    return result


def validate(value, *, storage_id, workflow, identity):
    allowed = {
        "type",
        "schema_version",
        "storage_id",
        "workflow",
        "deployment_id",
        "pool_id",
        "task_id",
        "attempt",
        "execution_token",
        "manifest_sha256",
        "frame_indices",
        "entries",
    }
    if (
        set(value) != allowed
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
    ):
        raise ValueError("invalid shared input schema")
    if value["storage_id"] != storage_id or value["workflow"] != workflow:
        raise ValueError("shared input storage/workflow mismatch")
    for key in ("deployment_id", "pool_id", "task_id", "attempt", "execution_token"):
        if value[key] != getattr(identity, key):
            raise ValueError("shared input execution identity mismatch")
    for key in ("deployment_id", "pool_id", "task_id"):
        if (
            not isinstance(value[key], str)
            or not _IDENT.fullmatch(value[key])
            or value[key] in {".", ".."}
        ):
            raise ValueError("invalid shared input identity")
    if not isinstance(value["manifest_sha256"], str) or not _SHA.fullmatch(
        value["manifest_sha256"]
    ):
        raise ValueError("invalid shared input manifest hash")
    entries = value["entries"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= 12:
        raise ValueError("invalid shared input count")
    total = 0
    ordinals = set()
    paths = set()
    kinds = []
    limits = {"image": 30 * 1024**2, "video": 50 * 1024**2, "audio": 15 * 1024**2}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "ordinal",
            "path",
            "content_type",
            "bytes",
            "sha256",
        }:
            raise ValueError("invalid shared input entry")
        ordinal = entry["ordinal"]
        path = entry["path"]
        if type(ordinal) is not int or ordinal < 0 or ordinal in ordinals:
            raise ValueError("invalid shared input ordinal")
        if (
            not isinstance(path, str)
            or len(path) > 512
            or "\x00" in path
            or "\\" in path
        ):
            raise ValueError("invalid shared input path")
        parts = path.split("/")
        if (
            len(parts) != 2
            or parts[0] != "inputs"
            or parts[1] in {"", ".", ".."}
            or path in paths
        ):
            raise ValueError("shared input path escaped task inputs")
        if not isinstance(entry["content_type"], str):
            raise ValueError("invalid shared input content type")
        if entry["content_type"] not in {
            "image/jpeg",
            "image/png",
            "image/webp",
            "video/mp4",
            "video/quicktime",
            "audio/wav",
        }:
            raise ValueError("unsupported shared input media format")
        kind = entry["content_type"].split("/", 1)[0]
        if (
            kind not in limits
            or type(entry["bytes"]) is not int
            or not 0 < entry["bytes"] <= limits[kind]
        ):
            raise ValueError("invalid shared input byte limit")
        if not isinstance(entry["sha256"], str) or not _SHA.fullmatch(entry["sha256"]):
            raise ValueError("invalid shared input hash")
        ordinals.add(ordinal)
        paths.add(path)
        kinds.append(kind)
        total += entry["bytes"]
    if total > MAX_TOTAL or [e["ordinal"] for e in entries] != sorted(ordinals):
        raise ValueError("invalid shared input total/order")
    indices = value["frame_indices"]
    if workflow == "fl2va":
        if (
            not isinstance(indices, list)
            or any(type(i) is not int for i in indices)
            or kinds not in (["image"], ["image", "image"])
            or indices not in ([0], [-1], [0, -1])
            or len(indices) != len(entries)
        ):
            raise ValueError("invalid shared FL2VA references")
    elif workflow == "ref2va":
        if (
            indices not in (None, [])
            or kinds.count("image") > 9
            or kinds.count("video") > 3
            or kinds.count("audio") > 3
            or not any(k in {"image", "video"} for k in kinds)
        ):
            raise ValueError("invalid shared Ref2VA references")
    else:
        raise ValueError("invalid shared input workflow")
    return total


def _open_beneath(root_fd, parts):
    """Open each component relative to held FDs; never follow symlinks."""
    current = os.dup(root_fd)
    device = os.fstat(current).st_dev
    try:
        for index, part in enumerate(parts):
            if part in {"", ".", ".."} or "/" in part:
                raise ValueError("invalid shared path component")
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            if index != len(parts) - 1:
                flags |= os.O_DIRECTORY
            fd = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = fd
            if os.fstat(fd).st_dev != device:
                raise ValueError("shared input crosses a mount")
        if not stat.S_ISREG(os.fstat(current).st_mode):
            raise ValueError("shared input is not a regular file")
        return current
    except BaseException:
        os.close(current)
        raise


def stage_files(root, destination, value):
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    prefix = [
        value["deployment_id"],
        "v1",
        "pools",
        value["pool_id"],
        "tasks",
        value["task_id"],
    ]
    try:
        with os.fdopen(
            _open_beneath(root_fd, prefix + ["input-manifest.json"]), "rb"
        ) as stream:
            raw = stream.read(MAX_DESCRIPTOR + 1)
        if (
            len(raw) > MAX_DESCRIPTOR
            or hashlib.sha256(raw).hexdigest() != value["manifest_sha256"]
        ):
            raise ValueError("shared input manifest integrity mismatch")
        manifest = json.loads(raw)
        fields = ("ordinal", "path", "content_type", "bytes", "sha256")
        if not isinstance(manifest, list):
            raise ValueError("invalid persisted input manifest")
        expected = [
            {k: e[k] for k in fields}
            for e in sorted(manifest, key=lambda e: e["ordinal"])
        ]
        if expected != value["entries"]:
            raise ValueError("shared input entries mismatch")
        for entry in value["entries"]:
            path = Path(destination) / str(entry["ordinal"])
            with os.fdopen(
                _open_beneath(root_fd, prefix + entry["path"].split("/")), "rb"
            ) as source:
                if os.fstat(source.fileno()).st_size != entry["bytes"]:
                    raise ValueError("shared input size mismatch")
                digest = hashlib.sha256()
                length = 0
                with path.with_suffix(".part").open("xb") as target:
                    while chunk := source.read(1024 * 1024):
                        length += len(chunk)
                        if length > entry["bytes"]:
                            raise ValueError("shared input grew during read")
                        digest.update(chunk)
                        target.write(chunk)
                if length != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
                    raise ValueError("shared input integrity mismatch")
                os.replace(path.with_suffix(".part"), path)
    finally:
        os.close(root_fd)


class SharedInputSession:
    def __init__(self, value, adapter, root, identity):
        self.value = value
        self.adapter = adapter
        self.source_root = root
        self.total = validate(
            value,
            storage_id=adapter.shared_storage_id,
            workflow=adapter.workflow,
            identity=identity,
        )
        self.capacity = adapter._media_capacity
        self.root = None
        self.owned_bytes = 0
        self.prepared = False

    async def prepare(self, timeout=120):
        self.capacity.reserve(self.total)
        self.owned_bytes = self.total
        spool = self.adapter.media_root / "h3-shared-inputs"
        spool.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root = Path(tempfile.mkdtemp(prefix="request-", dir=spool))
        started = time.monotonic()
        spawn = asyncio.create_task(
            asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                str(Path(__file__).resolve()),
                "--stage",
                str(self.source_root),
                str(self.root),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
        )
        try:
            process = await asyncio.shield(spawn)
        except asyncio.CancelledError:
            process = await spawn
            if process.returncode is None:
                process.kill()
                await process.wait()
            raise
        try:
            _, stderr = await asyncio.wait_for(
                process.communicate(json.dumps(self.value).encode()),
                timeout=max(0.001, timeout),
            )
            if process.returncode:
                # Detail remains internal; no model/GPU work has started.
                logging.getLogger(__name__).warning(
                    "shared_input_prepare_failed detail=%s",
                    stderr.decode(errors="replace")[-2048:],
                )
                raise ValueError("shared input file preparation failed")
            self.prepared = True
            logging.getLogger(__name__).info(
                "shared_input_prepared task=%s attempt=%s bytes=%s elapsed_s=%.3f gpu_started=false",
                self.value["task_id"],
                self.value["attempt"],
                self.total,
                time.monotonic() - started,
            )
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

    def cleanup(self):
        if self.root is not None:
            shutil.rmtree(self.root)
            self.root = None
        if self.owned_bytes:
            self.capacity.release(self.owned_bytes)
            self.owned_bytes = 0

    def path(self, ordinal):
        if not self.prepared or self.root is None:
            raise ValueError("shared inputs not prepared")
        return self.root / str(ordinal)


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--stage":
        raise SystemExit(2)
    stage_files(
        sys.argv[2], sys.argv[3], json.loads(sys.stdin.buffer.read(MAX_DESCRIPTOR + 1))
    )
