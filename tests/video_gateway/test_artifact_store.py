# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import hashlib
import os

import pytest

from dingo.video_gateway.artifact_store import FileArtifactStore


async def test_finalize_decodes_validated_mp4_atomically(tmp_path):
    store = FileArtifactStore(tmp_path / "artifacts")
    upload = await store.create_upload()
    task_root = await store.commit_upload(upload, "deployment", "pool", "video-id")
    payload = b"\x00\x00\x00\x18ftypisomvalidated"

    final, size, sha256, media = await store.finalize_b64_mp4(
        task_root,
        base64.b64encode(payload).decode(),
        {"width": 1},
        lambda path, _normalized: {"magic": path.read_bytes()[4:8].decode("ascii")},
    )

    assert final.read_bytes() == payload
    assert size == len(payload)
    assert sha256 == hashlib.sha256(payload).hexdigest()
    assert media == {"magic": "ftyp"}
    assert list((task_root / "tmp").glob("*.part")) == []


async def test_invalid_base64_leaves_no_partial_result(tmp_path):
    store = FileArtifactStore(tmp_path / "artifacts")
    upload = await store.create_upload()
    task_root = await store.commit_upload(upload, "deployment", "pool", "video-id")

    with pytest.raises(RuntimeError, match="invalid base64"):
        await store.finalize_b64_mp4(
            task_root, "====", {}, lambda _path, _normalized: {}
        )

    assert list((task_root / "tmp").glob("*.part")) == []
    assert not (task_root / "result" / "video.mp4").exists()


async def test_result_path_rejects_symlinks_even_when_target_is_under_root(tmp_path):
    store = FileArtifactStore(tmp_path / "artifacts")
    real = store.root / "real.mp4"
    real.write_bytes(b"video")
    linked = store.root / "linked.mp4"
    linked.symlink_to(real)

    with pytest.raises(FileNotFoundError):
        store.result_path(linked)

    await store.discard(linked)
    assert real.read_bytes() == b"video"
    assert not linked.exists()


async def test_orphan_cleanup_only_removes_stale_staging_directories(tmp_path):
    store = FileArtifactStore(tmp_path / "artifacts")
    stale = await store.create_upload()
    current = await store.create_upload()
    os.utime(stale, (0, 0))

    removed = await store.cleanup_orphan_uploads(minimum_age_s=60)

    assert removed == 1
    assert not stale.exists()
    assert current.is_dir()
