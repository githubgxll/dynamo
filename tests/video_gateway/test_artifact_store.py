# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import hashlib
import os
import json

import pytest

from dingo.video_gateway.artifact_store import FileArtifactStore
from dingo.video_gateway.errors import ResultTooLarge


class _Consumer:
    def __init__(self):
        self.values = []

    def consume(self, value):
        self.values.append(value)


async def test_detached_status_and_response_are_identity_and_checksum_checked(tmp_path):
    store = FileArtifactStore(tmp_path / "artifacts")
    token = "a" * 32
    root = store.detached_attempt_root("deployment", "pool", "task", 1, token)
    root.mkdir(parents=True)
    response = root / "worker-response.jsonl"
    payload = (json.dumps({"status": "completed"}) + "\n").encode()
    response.write_bytes(payload)
    status = {
        "schema_version": 1,
        "deployment_id": "deployment",
        "pool_id": "pool",
        "task_id": "task",
        "attempt": 1,
        "execution_token": token,
        "state": "completed",
        "response_bytes": len(payload),
        "response_sha256": hashlib.sha256(payload).hexdigest(),
    }
    (root / "worker-status.json").write_text(json.dumps(status))

    loaded = await store.read_detached_status(
        "deployment", "pool", "task", 1, token
    )
    consumer = _Consumer()
    consumed = await store.consume_detached_response(
        "deployment",
        "pool",
        "task",
        1,
        token,
        consumer,
        expected_sha256=status["response_sha256"],
        max_response_bytes=1024,
    )
    assert loaded == status
    assert consumed == len(payload)
    assert consumer.values == [{"status": "completed"}]

    with pytest.raises(RuntimeError, match="checksum"):
        await store.consume_detached_response(
            "deployment",
            "pool",
            "task",
            1,
            token,
            _Consumer(),
            expected_sha256="0" * 64,
            max_response_bytes=1024,
        )


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


async def test_finalize_publishes_distinct_immutable_candidates(tmp_path):
    store = FileArtifactStore(tmp_path / "artifacts")
    upload = await store.create_upload()
    task_root = await store.commit_upload(upload, "deployment", "pool", "video-id")
    left_payload = b"\x00\x00\x00\x18ftypisomleft"
    right_payload = b"\x00\x00\x00\x18ftypisomright"

    left, *_ = await store.finalize_b64_mp4(
        task_root,
        base64.b64encode(left_payload).decode(),
        {},
        lambda _path, _normalized: {},
        publication_scope="a1-token",
    )
    right, *_ = await store.finalize_b64_mp4(
        task_root,
        base64.b64encode(right_payload).decode(),
        {},
        lambda _path, _normalized: {},
        publication_scope="a1-token",
    )

    assert left != right
    assert left.parent == right.parent == task_root / "result"
    assert left.name.startswith("video-a1-token-")
    assert right.name.startswith("video-a1-token-")
    assert left.read_bytes() == left_payload
    assert right.read_bytes() == right_payload

    # A stale owner discards only its own candidate; the CAS winner survives.
    left.unlink()
    assert right.read_bytes() == right_payload


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


async def test_oversize_result_has_stable_error_and_leaves_no_partial_file(tmp_path):
    store = FileArtifactStore(tmp_path / "artifacts")
    upload = await store.create_upload()
    task_root = await store.commit_upload(upload, "deployment", "pool", "video-id")
    payload = b"\x00\x00\x00\x18ftypisomtoo-large"

    with pytest.raises(ResultTooLarge, match="configured maximum"):
        await store.finalize_b64_mp4(
            task_root,
            base64.b64encode(payload).decode(),
            {},
            lambda _path, _normalized: {},
            max_result_bytes=len(payload) - 1,
        )

    assert list((task_root / "tmp").glob("*.part")) == []
    assert list((task_root / "result").glob("*.mp4")) == []


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


async def test_task_orphan_is_manifested_trashed_and_deleted_in_two_steps(tmp_path):
    store = FileArtifactStore(tmp_path / "artifacts")
    upload = await store.create_upload()
    task_root = await store.commit_upload(
        upload,
        "deployment",
        "pool",
        "video-orphan",
        artifact_manifest={
            "schema_version": 1,
            "task_id": "video-orphan",
            "deployment_id": "deployment",
            "pool_id": "pool",
            "created_at_ms": 1,
            "expires_at_ms": 2,
        },
    )
    (task_root / "payload.bin").write_bytes(b"orphan-payload")
    os.utime(task_root, (0, 0))

    candidates = await store.orphan_task_candidates(
        "deployment", ("pool",), minimum_age_s=60
    )

    assert len(candidates) == 1
    assert candidates[0].task_id == "video-orphan"
    assert candidates[0].manifest_valid is True
    assert await store.trash_orphan(candidates[0], dry_run=True) == task_root
    assert task_root.exists()

    trashed = await store.trash_orphan(candidates[0])
    assert trashed is not None and trashed.parent == store.trash_root
    assert not task_root.exists()
    assert trashed.exists()
    os.utime(trashed, (0, 0))
    removed, released = await store.cleanup_trash(minimum_age_s=60)
    assert removed == 1
    assert released >= len(b"orphan-payload")
    assert not trashed.exists()


async def test_orphan_scan_skips_symlinks_and_reports_missing_manifest(tmp_path):
    store = FileArtifactStore(tmp_path / "artifacts")
    tasks = store.root / "deployment" / "v1" / "pools" / "pool" / "tasks"
    tasks.mkdir(parents=True)
    unmanifested = tasks / "video-unmanifested"
    unmanifested.mkdir()
    os.utime(unmanifested, (0, 0))
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tasks / "video-linked"
    linked.symlink_to(outside, target_is_directory=True)

    candidates = await store.orphan_task_candidates(
        "deployment", ("pool",), minimum_age_s=60
    )

    assert [(item.task_id, item.manifest_valid) for item in candidates] == [
        ("video-unmanifested", False)
    ]
    assert outside.exists()
