#!/usr/bin/env python3
"""Small, credential-free helpers for VBench's prepare/publish workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

BASE_TAG = "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04"
BASE_REPOSITORY = "nvidia/cuda"
PLATFORM = "linux/amd64"
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
SHA = re.compile(r"^[0-9a-f]{40}$")
IMAGE_TAG = re.compile(r"^[a-z0-9][a-z0-9.:-]*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+:[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")
REQUIRED_LOCKS = ("base.lock.json", "requirements.lock", "assets.lock.json", "input-fingerprint.json", "preparation-license-status.json")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def amd64_digest(manifest: dict) -> str:
    """Select the actual Linux/amd64 manifest, excluding attestations/other CPUs."""
    matching = [
        item for item in manifest.get("manifests", [])
        if item.get("platform", {}).get("os") == "linux"
        and item.get("platform", {}).get("architecture") == "amd64"
        and item.get("platform", {}).get("variant", "") in {"", "v1"}
    ]
    if len(matching) != 1 or not DIGEST.fullmatch(matching[0].get("digest", "")):
        raise ValueError("Expected exactly one Linux/amd64 image digest in the CUDA image index")
    return matching[0]["digest"]


def validate_base_lock(lock: dict) -> str:
    digest = lock.get("digest", "")
    if (lock.get("schema_version") != 1 or lock.get("source_tag") != BASE_TAG
            or lock.get("platform") != PLATFORM or not DIGEST.fullmatch(digest)
            or lock.get("image") != f"{BASE_REPOSITORY}@{digest}"):
        raise ValueError("base.lock.json must pin the expected CUDA 12.1 Linux/amd64 image by digest")
    return lock["image"]


def write_base_output(image: str, path: Path) -> None:
    # Only a validated, newline-free digest reference enters GITHUB_OUTPUT.
    with path.open("a", encoding="utf-8") as output:
        output.write(f"base_image={image}\n")


def resolve_base(lock_path: Path, output_path: Path) -> None:
    result = subprocess.run(
        ["docker", "buildx", "imagetools", "inspect", "--raw", BASE_TAG],
        check=True, capture_output=True, text=True,
    )
    digest = amd64_digest(json.loads(result.stdout))
    lock = {
        "schema_version": 1, "source_tag": BASE_TAG, "platform": PLATFORM,
        "digest": digest, "image": f"{BASE_REPOSITORY}@{digest}",
        "resolved_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(lock_path, lock)
    write_base_output(validate_base_lock(lock), output_path)
    print(f"Resolved {BASE_TAG} ({PLATFORM}) to {lock['image']}")


def verify_locks(directory: Path, output_path: Path) -> None:
    for name in REQUIRED_LOCKS:
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Publish requires a nonempty reviewed lock: {path}")
        if name.endswith(".json"):
            read_json(path)
    image = validate_base_lock(read_json(directory / "base.lock.json"))
    if read_json(directory / "preparation-license-status.json").get("policy_pass") is not True:
        raise ValueError("Preparation license review has not passed; publish is blocked")
    write_base_output(image, output_path)
    print("Lock files present; Docker build will enforce fingerprints, hashes and license decisions.")


def record_image(args: argparse.Namespace) -> None:
    if not IMAGE_TAG.fullmatch(args.image) or not DIGEST.fullmatch(args.digest):
        raise ValueError("Invalid published image tag or digest")
    if not SHA.fullmatch(args.source_commit):
        raise ValueError("Source commit must be a full lowercase Git SHA")
    if not args.run_id.isdecimal() or not args.run_attempt.isdecimal():
        raise ValueError("Run ID and attempt must be numeric")
    dockerfile = Path(args.dockerfile)
    manifest = {
        "source_commit": args.source_commit, "run_id": args.run_id,
        "run_attempt": args.run_attempt, "image": args.image,
        "digest": args.digest,
        "deployment_image": args.image.rsplit(":", 1)[0] + "@" + args.digest,
        "builder_image": args.builder_image,
        "dockerfile": dockerfile.as_posix(),
        "dockerfile_sha256": hashlib.sha256(dockerfile.read_bytes()).hexdigest(),
        "validation_scope": "Image published; K8s pull, GPU and model/media E2E not yet validated",
    }
    write_json(args.output, manifest)
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as summary:
            summary.write(f"### Published image\n\n`{args.image}`\n\n")
            summary.write(f"Digest: `{args.digest}`\n\nSource commit: `{args.source_commit}`\n\n")
            summary.write(f"Deploy by digest: `{manifest['deployment_image']}`\n\n")
            summary.write(manifest["validation_scope"] + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    resolve = sub.add_parser("resolve-base")
    resolve.add_argument("--lock", type=Path, required=True)
    resolve.add_argument("--github-output", type=Path, required=True)
    verify = sub.add_parser("verify-locks")
    verify.add_argument("--directory", type=Path, required=True)
    verify.add_argument("--github-output", type=Path, required=True)
    record = sub.add_parser("record-image")
    for name in ("image", "digest", "source-commit", "run-id", "run-attempt", "dockerfile"):
        record.add_argument("--" + name, required=True)
    record.add_argument("--builder-image", default="")
    record.add_argument("--output", type=Path, required=True)
    record.add_argument("--summary", type=Path)
    args = parser.parse_args()
    if args.command == "resolve-base":
        resolve_base(args.lock, args.github_output)
    elif args.command == "verify-locks":
        verify_locks(args.directory, args.github_output)
    else:
        record_image(args)


if __name__ == "__main__":
    main()
