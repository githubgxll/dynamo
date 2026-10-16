"""Read-only, offline byte identity gate for FastH3's exact FL2VA teacher.

This is not GPU, adapter/HSDP compatibility, or media-quality acceptance.
The lock contains upstream LFS payload SHA256 and ordinary Git blob SHA1.
Hugging Face snapshot symlinks are allowed: the target bytes, not path names,
are verified. No model file is downloaded, modified, or merged by this module.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys


LOCK_SHA256 = "c3267aade02feb54c4cb2391c4b67a12fa8ae42b1e128b8c5225a2291ad9c8db"
REPOSITORY = "MiniMaxAI/MiniMax-H3"
REVISION = "9bfb6693f2cf6de171db46d1aa586f67d773a1da"
CHUNK_BYTES = 8 * 1024 * 1024
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf"}
RUNTIME_SUFFIXES = WEIGHT_SUFFIXES | {".json", ".yaml", ".yml", ".py"}


class BaseVerificationError(RuntimeError):
    def __init__(self, message, evidence=None):
        super().__init__(message)
        self.evidence = evidence or {"status": "FAIL_EXACT_BASE_REVISION_BYTES", "error": message}


def _require(condition, message):
    if not condition:
        raise BaseVerificationError(message)


def _safe_path(value):
    _require(isinstance(value, str) and value and "\\" not in value and ":" not in value
             and "\x00" not in value, "Unsafe Base lock relative path")
    path = PurePosixPath(value)
    _require(not path.is_absolute() and all(part not in ("", ".", "..") for part in value.split("/"))
             and path.as_posix() == value, "Unsafe Base lock relative path: " + value)
    return path


def _validate_entries(entries):
    _require(isinstance(entries, list) and entries, "Base lock needs a nonempty file inventory")
    seen = set()
    for item in entries:
        _require(isinstance(item, dict) and set(item) in (
            {"path", "size_bytes", "git_blob_sha1"},
            {"path", "size_bytes", "git_blob_sha1", "sha256"}), "Unknown Base lock entry fields")
        path = _safe_path(item["path"]).as_posix()
        _require(path not in seen, "Duplicate Base lock path: " + path)
        seen.add(path)
        _require(type(item["size_bytes"]) is int and item["size_bytes"] >= 0,
                 "Invalid Base lock size: " + path)
        _require(isinstance(item["git_blob_sha1"], str) and re.fullmatch(r"[0-9a-f]{40}", item["git_blob_sha1"]),
                 "Invalid Git blob SHA1: " + path)
        if "sha256" in item:
            _require(isinstance(item["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]),
                     "Invalid full payload SHA256: " + path)
        _require(path.endswith(".safetensors") == ("sha256" in item),
                 "Unsupported or missing payload hash semantics: " + path)
    return entries


def load_lock(lock_path):
    raw = Path(lock_path).read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == LOCK_SHA256,
             "FastH3 Base lock changed; cannot verify against an untrusted inventory")
    data = json.loads(raw)
    _require(data.get("schema") == "h3-fasth3-exact-base-lock/1"
             and data.get("repository") == REPOSITORY and data.get("revision") == REVISION
             and data.get("partition") == "FL2VA", "Wrong FastH3 Base revision contract")
    entries = _validate_entries(data.get("files"))
    _require(len(entries) == data.get("file_count") == 81
             and sum("sha256" in row for row in entries) == data.get("weight_file_count") == 29
             and sum(row["size_bytes"] for row in entries) == data.get("total_bytes") == 144051182625,
             "Incomplete FastH3 Base inventory")
    return data


def _stat_identity(value):
    # Windows stat/fstat can expose different ctime semantics (birth/change
    # time). Size/mtime/inode plus the full content digest are portable here.
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _check_extra_weights(root, expected):
    """Reject alternate payloads/configuration; allow HF cache bookkeeping.

    Walking follows directory links (some installations link components), with
    explicit ancestor cycle detection. External blob links are allowed because
    official HF snapshots normally point outside their snapshot directory.
    """
    stack = [(root, frozenset())]
    while stack:
        folder, ancestors = stack.pop()
        resolved = folder.resolve(strict=True)
        _require(resolved not in ancestors, "Base directory symlink cycle: " + str(folder))
        ancestors = ancestors | {resolved}
        for path in folder.iterdir():
            rel = path.relative_to(root).as_posix()
            if path.name in (".cache", ".git") and path.is_dir():
                continue
            if path.is_dir():
                stack.append((path, ancestors))
            elif path.suffix.lower() in RUNTIME_SUFFIXES:
                _require(rel in expected, "Unexpected Base runtime/checkpoint file: " + rel)


def _verify_files(model_root, entries, progress=None):
    """Stream every supplied entry; production callers must use verify_model."""
    entries = _validate_entries(entries)
    root = Path(model_root)
    _require(root.is_dir(), "Mounted FL2VA Base directory is unavailable: " + str(root))
    expected = {row["path"] for row in entries}
    _check_extra_weights(root, expected)
    verified = []
    for item in entries:
        path = root.joinpath(*_safe_path(item["path"]).parts)
        _require(path.is_file(), "Missing/non-regular Base file: " + item["path"])
        resolved_before = path.resolve(strict=True)
        algorithm = "sha256" if "sha256" in item else "git_blob_sha1"
        digest = hashlib.sha256() if algorithm == "sha256" else hashlib.sha1()
        if algorithm == "git_blob_sha1":
            digest.update(("blob " + str(item["size_bytes"]) + "\0").encode("ascii"))
        total = 0
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            _require(stat.S_ISREG(before.st_mode), "Non-regular Base file: " + item["path"])
            _require(before.st_size == item["size_bytes"], "Base byte-size mismatch: " + item["path"])
            while True:
                chunk = stream.read(CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
                total += len(chunk)
            after = os.fstat(stream.fileno())
        _require(total == item["size_bytes"] and _stat_identity(before) == _stat_identity(after)
                 and _stat_identity(after) == _stat_identity(path.stat())
                 and resolved_before == path.resolve(strict=True),
                 "Base file changed during read-only verification: " + item["path"])
        observed = digest.hexdigest()
        _require(observed == item[algorithm], "Base " + algorithm + " mismatch: " + item["path"])
        verified.append({"path": item["path"], "size_bytes": total, algorithm: observed,
                         "resolved_path": str(resolved_before), "identity_verified": True})
        if progress is not None:
            progress(verified[-1], len(verified), len(entries))
    return verified


def verify_model(model_root, lock_path, progress=None):
    """Return exact-byte identity evidence, or raise before model loading.

    The fixed lock checksum prevents replacing expected hashes with hashes of
    unrelated local weights. File-name/revision-name matching is never enough.
    The mounted base must stay read-only and unchanged after this check.
    """
    started = datetime.now(timezone.utc).isoformat()
    evidence = {"schema": "h3-fasth3-base-check/1", "repository": REPOSITORY,
                "required_revision": REVISION, "partition": "FL2VA",
                "model_root": str(model_root), "started_at": started,
                "lock_sha256": LOCK_SHA256, "status": "VERIFYING_EXACT_BASE_REVISION_BYTES",
                "verification_scope": "Every upstream FL2VA file; full weight payloads, not sampled headers",
                "compatibility_gpu_media_validated": False,
                "model_directory_modified": False, "network_access": False,
                "symlink_policy": "Follow snapshot/component links and verify target bytes; record resolved paths"}
    verified = []
    def on_file(row, current, count):
        verified.append(row)
        if progress is not None:
            progress(row, current, count)
    try:
        lock = load_lock(lock_path)
        _verify_files(model_root, lock["files"], on_file)
        evidence.update(status="PASS_EXACT_BASE_REVISION_BYTES", files=verified,
                        file_count=len(verified), weight_file_count=lock["weight_file_count"],
                        verified_bytes=sum(row["size_bytes"] for row in verified),
                        finished_at=datetime.now(timezone.utc).isoformat())
        return evidence
    except (OSError, ValueError, BaseVerificationError) as exc:
        evidence.update(status="FAIL_EXACT_BASE_REVISION_BYTES", files=verified, error=str(exc),
                        finished_at=datetime.now(timezone.utc).isoformat())
        raise BaseVerificationError(str(exc), evidence) from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--lock", required=True)
    args = parser.parse_args()
    try:
        result = verify_model(args.model_root, args.lock)
    except BaseVerificationError as exc:
        print(json.dumps(exc.evidence, indent=2, ensure_ascii=False))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
