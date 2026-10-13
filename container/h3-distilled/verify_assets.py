#!/usr/bin/env python3
"""Offline, stdlib-only verification of the four pinned H3 adapters."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import struct
import sys
from urllib.parse import quote, urlsplit


class AssetError(RuntimeError):
    pass


def relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise AssetError("Invalid relative asset path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in {"", ".", ".."} for p in value.split("/")):
        raise AssetError("Unsafe relative asset path")
    return path


def safe_path(root: Path, value: str) -> Path:
    relative = relative_path(value)
    root = Path(root).absolute()
    # Reject aliases at every existing ancestor, including a caller-supplied root.
    for part in [root, *root.parents]:
        if part.is_symlink():
            raise AssetError("Symlink roots are not allowed")
    target = root.joinpath(*relative.parts)
    for part in [target, *target.parents]:
        if part == root:
            break
        if part.is_symlink():
            raise AssetError("Symlink asset paths are not allowed")
    if not target.resolve().is_relative_to(root.resolve()):
        raise AssetError("Asset path escapes root")
    return target


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def canonical_header_digest(header: dict) -> str:
    raw = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AssetError("Duplicate JSON key")
        result[key] = value
    return result


def load_lock(path: Path) -> dict:
    lock = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicate_keys)
    if lock.get("schema") != "h3-distilled-assets/1" or not lock.get("assets"):
        raise AssetError("Unsupported or empty asset lock")
    targets = set()
    identities = set()
    for asset in lock["assets"]:
        if asset["id"] in identities or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", asset["id"]):
            raise AssetError("Duplicate or invalid asset id")
        identities.add(asset["id"])
        if not re.fullmatch(r"[0-9a-f]{40}", asset["revision"]):
            raise AssetError("Asset revision is not a fixed commit")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", asset["repo"]):
            raise AssetError("Invalid official repository")
        relative_path(asset["filename"])
        expected = "https://huggingface.co/{}/resolve/{}/{}".format(
            asset["repo"], asset["revision"], quote(asset["filename"], safe="/"))
        if asset["url"] != expected:
            raise AssetError("Asset URL must be the exact official HTTPS commit URL")
        if type(asset["size_bytes"]) is not int or asset["size_bytes"] <= 8:
            raise AssetError("Invalid expected asset size")
        if not isinstance(asset.get("header_contract"), dict):
            raise AssetError("Missing safetensors header contract")
    for item in lock["assets"] + lock.get("documents", []):
        relative_path(item["relative_path"])
        # Casefold also rejects alias collisions on Windows preparation hosts.
        target = item["relative_path"].casefold()
        if target in targets:
            raise AssetError("Duplicate output path")
        targets.add(target)
        if not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise AssetError("Invalid SHA256")
    for document in lock.get("documents", []):
        relative_path(document["local_source"])
        uri = urlsplit(document["source_url"])
        if uri.scheme != "https" or uri.hostname not in {"huggingface.co", "raw.githubusercontent.com"} or uri.query or uri.fragment or uri.username or uri.password:
            raise AssetError("Unexpected document provenance URL")
        if not re.fullmatch(r"[0-9a-f]{40}", document["source_revision"]):
            raise AssetError("Document revision must be a fixed commit")
        pinned_path = (f'/resolve/{document["source_revision"]}/' if uri.hostname == "huggingface.co"
                       else f'/githubgxll/dynamo/{document["source_revision"]}/')
        if pinned_path not in uri.path:
            raise AssetError("Document provenance does not pin its revision")
    return lock


def check_file(path: Path, item: dict) -> None:
    if not path.is_file() or path.is_symlink():
        raise AssetError("Missing regular asset file: " + item.get("id", item["relative_path"]))
    if "size_bytes" in item and path.stat().st_size != item["size_bytes"]:
        raise AssetError("File size mismatch: " + item.get("id", item["relative_path"]))
    if sha256_file(path) != item["sha256"]:
        raise AssetError("SHA256 mismatch: " + item.get("id", item["relative_path"]))


def check_header(path: Path, asset: dict) -> dict:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise AssetError("Truncated safetensors header")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= 16 * 1024 * 1024 or 8 + length >= path.stat().st_size:
            raise AssetError("Invalid safetensors header length")
        raw = stream.read(length)
    header = json.loads(raw, object_pairs_hook=no_duplicate_keys)
    if not isinstance(header, dict):
        raise AssetError("Invalid safetensors header object")
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    contract = asset["header_contract"]
    if header.get("__metadata__") != contract["metadata"]:
        raise AssetError("Safetensors metadata mismatch: " + asset["id"])
    if len(tensors) != contract["tensor_count"] or canonical_header_digest(header) != contract["header_json_sha256"]:
        raise AssetError("Safetensors tensor contract mismatch: " + asset["id"])
    dtype_width = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
    counts = {}
    ranges = []
    for tensor in tensors.values():
        dtype, shape, offsets = tensor["dtype"], tensor["shape"], tensor["data_offsets"]
        if dtype not in dtype_width or not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
            raise AssetError("Invalid tensor dtype or shape")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(v) is not int for v in offsets):
            raise AssetError("Invalid tensor offsets")
        start, end = offsets
        if start < 0 or end - start != math.prod(shape) * dtype_width[dtype]:
            raise AssetError("Tensor byte length mismatch")
        counts[dtype] = counts.get(dtype, 0) + 1
        ranges.append((start, end))
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise AssetError("Overlapping or noncontiguous tensor data")
        cursor = end
    if cursor + 8 + length != path.stat().st_size or counts != contract["dtype_counts"]:
        raise AssetError("Safetensors payload layout mismatch")
    return {"metadata": header["__metadata__"], "tensor_count": len(tensors), "dtype_counts": counts}


def verify(root: Path, lock: dict) -> dict:
    report = {"status": "PASS_ASSETS_ONLY", "assets": [], "documents": [],
              "not_validated": ["mounted H3 Base identity", "GPU loading", "generated video/audio"]}
    for asset in lock["assets"]:
        path = safe_path(root, asset["relative_path"])
        check_file(path, asset)
        contract = check_header(path, asset)
        report["assets"].append({"id": asset["id"], "relative_path": asset["relative_path"],
                                 "sha256": asset["sha256"], "size_bytes": path.stat().st_size,
                                 "safetensors": contract})
    for document in lock.get("documents", []):
        check_file(safe_path(root, document["relative_path"]), document)
        report["documents"].append({"relative_path": document["relative_path"], "sha256": document["sha256"]})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = verify(args.root, load_lock(args.lock))
        report["lock_sha256"] = sha256_file(args.lock)
        text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        print(text, end="")
    except (AssetError, OSError, ValueError, KeyError, TypeError) as exc:
        # AssetError is controlled text; unexpected exceptions may contain URLs.
        print("Asset verification failed: " + (str(exc) if isinstance(exc, AssetError) else type(exc).__name__), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
