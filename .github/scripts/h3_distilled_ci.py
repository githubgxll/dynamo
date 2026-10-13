#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate and record the H3 distilled image on the existing Dingo Runner.

This helper performs no registry access and never prints credentials. The normal
Dingo framework matrix remains independent of this adapter-only build profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROFILE = Path("container/h3-distilled")
EXPECTED_REPOSITORY = "registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm"
EXPECTED_BASE = EXPECTED_REPOSITORY + "@sha256:aace31a2806c2c18535cb0bc6a81cc940e0901b4f02f380d2d880c1f34f965c2"
EXPECTED_TAG_PREFIX = "v0.30.0-ubuntu2404-distilled"
EXPECTED_FRAMEWORK_COMMIT = "958c2ce95cd361fd5a6770e3023eac4de968f674"
SHA_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
REPORTS = {
    "runtime-source-check.json": "PASS_INSTALLED_CODE_ONLY",
    "assets-check.json": "PASS_ASSETS_ONLY",
    "bundle-check.json": "PASS_DISTILLED_CPU_ONLY",
}
COPIED_INPUTS = {
    "image.lock.json": PROFILE / "image.lock.json",
    "assets.lock.json": PROFILE / "assets.lock.json",
    "Dockerfile": PROFILE / "Dockerfile",
    "dingo-images.json": Path(".github/dingo-images.json"),
}


def read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_sha(environ: dict[str, str]) -> str:
    value = environ.get("GITHUB_SHA", "")
    if not SHA_PATTERN.fullmatch(value):
        raise ValueError("GITHUB_SHA must be a complete lowercase 40-character commit SHA")
    return value


def prepare_values(root: Path, environ: dict[str, str]) -> dict[str, str]:
    config = read_object(root / ".github/dingo-images.json")
    lock = read_object(root / PROFILE / "image.lock.json")
    if lock.get("schema") != "h3-distilled-image/1":
        raise ValueError("Unsupported distilled image lock schema")
    registry = config.get("registry")
    namespace = config.get("namespace")
    if registry != "registry.hd-04.alayanew.com:8443" or namespace != "openclaw":
        raise ValueError("Distilled publishing is restricted to the reviewed hd04/openclaw target")
    images = config.get("images")
    if not isinstance(images, list):
        raise ValueError("The group image configuration must contain images")
    vllm = [item for item in images if isinstance(item, dict) and item.get("framework") == "vllm"]
    if len(vllm) != 1 or vllm[0].get("repository") != "ai-dingo-vllm":
        raise ValueError("Expected exactly the existing ai-dingo-vllm repository")
    if config.get("platform") != "linux/amd64" or lock.get("platform") != "linux/amd64":
        raise ValueError("Only the reviewed linux/amd64 target is supported")
    if config.get("commit_sha_length") != 12:
        raise ValueError("The reviewed group tag convention requires a 12-character SHA suffix")
    if lock.get("publication_repository") != EXPECTED_REPOSITORY:
        raise ValueError("Image lock publication repository differs from the approved target")
    if lock.get("base_image") != EXPECTED_BASE:
        raise ValueError("Distilled image must inherit the successful Omni 0.30 digest")
    if lock.get("tag_prefix") != EXPECTED_TAG_PREFIX:
        raise ValueError("The image tag must include the reviewed distilled prefix")
    if lock.get("recipe_base_commit") != EXPECTED_FRAMEWORK_COMMIT:
        raise ValueError("Installed framework provenance must remain the successful 0.30 commit")
    return {
        "registry": registry,
        "image": EXPECTED_REPOSITORY + ":" + EXPECTED_TAG_PREFIX + "-" + source_sha(environ)[:12],
        "base_image": EXPECTED_BASE,
        "platform": "linux/amd64",
    }


def prepare(root: Path, environ: dict[str, str], output: Path) -> dict[str, str]:
    values = prepare_values(root, environ)
    with output.open("a", encoding="utf-8", newline="\n") as handle:
        for key, value in values.items():
            handle.write(f"{key}={value}\n")
    return values


def copy_inputs(root: Path, directory: Path) -> None:
    """Attach exact committed recipe inputs to the already-exported reports."""
    if not directory.is_dir():
        raise ValueError("Build evidence directory does not exist; the export must succeed first")
    for name, relative in COPIED_INPUTS.items():
        target = directory / name
        if target.exists() and (target.is_symlink() or not target.is_file()):
            raise ValueError(f"Unsafe evidence destination: {name}")
        # Report-producing stages may already include these inputs. Reject any
        # mismatch instead of replacing potentially inconsistent build evidence.
        if target.is_file() and target.read_bytes() != (root / relative).read_bytes():
            raise ValueError(f"Exported input differs from checkout: {name}")
        shutil.copyfile(root / relative, target)


def validate_evidence(root: Path, environ: dict[str, str], directory: Path) -> dict[str, dict]:
    prepare_values(root, environ)
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("Expected a real exported evidence directory")
    reports = {}
    for name, status in REPORTS.items():
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing regular evidence file: {name}")
        report = read_object(path)
        if report.get("status") != status:
            raise ValueError(f"Unsuccessful {name}; expected {status}")
        reports[name] = report
    bundle = reports["bundle-check.json"]
    if bundle.get("source_commit") != source_sha(environ):
        raise ValueError("Bundle evidence source_commit differs from this build's GITHUB_SHA")
    expected_bundle = {
        "base_image": EXPECTED_BASE,
        "framework_source_commit": EXPECTED_FRAMEWORK_COMMIT,
        "environment_unchanged": True,
        "asset_lock_sha256": sha256_file(root / PROFILE / "assets.lock.json"),
        "image_lock_sha256": sha256_file(root / PROFILE / "image.lock.json"),
    }
    for name, expected in expected_bundle.items():
        if bundle.get(name) != expected:
            raise ValueError(f"Bundle evidence does not confirm reviewed {name}")
    if reports["assets-check.json"].get("lock_sha256") != expected_bundle["asset_lock_sha256"]:
        raise ValueError("Asset verification used a different asset lock")
    for name, relative in COPIED_INPUTS.items():
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing regular recipe input: {name}")
        if path.read_bytes() != (root / relative).read_bytes():
            raise ValueError(f"Evidence recipe differs from checkout: {name}")
    return reports


def record(root: Path, environ: dict[str, str], image: str, digest: str, directory: Path) -> dict:
    values = prepare_values(root, environ)
    reports = validate_evidence(root, environ, directory)
    if image != values["image"]:
        raise ValueError("Published image differs from the approved target and current source SHA")
    if not DIGEST_PATTERN.fullmatch(digest):
        raise ValueError("Published digest must be sha256 followed by 64 lowercase hex characters")
    for name in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT"):
        if not re.fullmatch(r"[1-9][0-9]*", environ.get(name, "")):
            raise ValueError(f"{name} must identify an actual workflow run")
    fingerprints = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlinks are not permitted in publication evidence")
        if path.is_file() and path.name not in {"image.json", "deployment-image.txt"}:
            fingerprints[path.relative_to(directory).as_posix()] = sha256_file(path)
    manifest = {
        "schema": "h3-distilled-publication/1",
        "source_commit": source_sha(environ),
        "run_id": environ["GITHUB_RUN_ID"],
        "run_attempt": environ["GITHUB_RUN_ATTEMPT"],
        "image": image,
        "digest": digest,
        "deployment_image": EXPECTED_REPOSITORY + "@" + digest,
        "base_image": values["base_image"],
        "framework_source_commit": EXPECTED_FRAMEWORK_COMMIT,
        "platform": values["platform"],
        "assets_lock_sha256": sha256_file(root / PROFILE / "assets.lock.json"),
        "image_lock_sha256": sha256_file(root / PROFILE / "image.lock.json"),
        "report_statuses": {name: report["status"] for name, report in reports.items()},
        "evidence_sha256": fingerprints,
        "validation_scope": "Image published after CPU-only source and asset checks; Harbor pull, H100 kernels, Base identity and model/media E2E are not validated",
    }
    manifest_path = directory / "image.json"
    if manifest_path.exists():
        if manifest_path.is_symlink() or read_object(manifest_path) != manifest:
            raise ValueError("Refusing to replace different publication evidence")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (directory / "deployment-image.txt").write_text(manifest["deployment_image"] + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--github-output", type=Path, required=True)
    p = sub.add_parser("validate-evidence")
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--copy-inputs", action="store_true")
    p = sub.add_parser("record")
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--digest", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(ROOT, os.environ, args.github_output)
    elif args.command == "validate-evidence":
        if args.copy_inputs:
            copy_inputs(ROOT, args.directory)
        validate_evidence(ROOT, os.environ, args.directory)
        result = {"status": "PASS_PREPUBLISH_EVIDENCE_ONLY"}
    else:
        result = record(ROOT, os.environ, args.image, args.digest, args.directory)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
