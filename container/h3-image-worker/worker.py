#!/usr/bin/env python3
"""Validate the frozen local model assets and start the native Images API."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile

HERE = Path(__file__).resolve().parent
MODEL_ROOT = PurePosixPath("/opt/models/minimax-h3-image")
OUTPUT_ROOT = Path("/work/h3-image-worker")
IMAGE_REPOSITORY = "MiniMaxAI/MiniMax-H3-Image"
BASE_REPOSITORY = "MiniMaxAI/MiniMax-H3"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Invalid model-relative asset path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or ":" in value or str(path) != value:
        raise ValueError("Unsafe model-relative asset path: " + value)
    return path


def required_assets(manifest):
    if manifest.get("schema_version") != 1:
        raise ValueError("Expected the frozen schema-1 asset lock")
    files = {}
    models = manifest.get("models", [])
    if {model.get("id") for model in models} != {IMAGE_REPOSITORY, BASE_REPOSITORY}:
        raise ValueError("The image and base-encoder source locks are required")
    for model in models:
        for entry in model["files"]:
            if entry.get("required") is not True:
                continue
            name = str(relative_path(entry["path"]))
            if model["id"] == BASE_REPOSITORY and not name.startswith("text_encoder/"):
                raise ValueError("Base assets must belong to text_encoder/")
            if name in files:
                raise ValueError("Duplicate required asset: " + name)
            size = entry.get("bytes")
            sha256 = entry.get("sha256")
            git_sha1 = entry.get("git_blob_sha1")
            if type(size) is not int or size < 0:
                raise ValueError("Invalid asset byte count: " + name)
            digest, length = (sha256, 64) if sha256 else (git_sha1, 40)
            if not isinstance(digest, str) or len(digest) != length or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Missing or invalid asset content digest: " + name)
            files[name] = {"bytes": size, "sha256": sha256, "git_blob_sha1": git_sha1,
                           "repository": model["id"], "revision": model["revision"]}
    if not files:
        raise ValueError("The required asset selection is empty")
    return files


def asset_path(root, name):
    path = root.joinpath(*relative_path(name).parts)
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Asset resolves outside the model root: " + name)
    if not path.is_file():
        raise ValueError("Required model asset is missing or unreadable: " + name)
    return path


def verify_selection(manifest, source_roots):
    selection = required_assets(manifest)
    verified = {}
    for name, expected in sorted(selection.items()):
        root = source_roots[expected["repository"]]
        path = asset_path(root, name)
        if path.stat().st_size != expected["bytes"]:
            raise ValueError("Asset byte count differs: " + name)
        algorithm = "sha256" if expected["sha256"] else "sha1"
        digest = hashlib.new(algorithm)
        if algorithm == "sha1":
            digest.update(f"blob {expected['bytes']}\0".encode("ascii"))
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        wanted = expected["sha256"] or expected["git_blob_sha1"]
        if digest.hexdigest() != wanted:
            raise ValueError("Asset content digest differs: " + name)
        verified[name] = {"bytes": expected["bytes"], "algorithm": algorithm,
                          "digest": digest.hexdigest(), "source_path": str(path.resolve()),
                          "source_root": str(root)}
    for name in selection:
        if not name.endswith(".index.json"):
            continue
        index = read_json(verified[name]["source_path"])
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Weight index is missing its weight_map: " + name)
        parent = PurePosixPath(name).parent
        for shard in weight_map.values():
            shard_name = str(parent / relative_path(shard))
            if shard_name not in selection:
                raise ValueError("Weight index references an unlocked shard: " + shard_name)
            if selection[shard_name]["repository"] != selection[name]["repository"]:
                raise ValueError("Weight index crosses locked source roots: " + shard_name)
    return {"schema": "h3-image-worker-model-validation/1", "status": "PASS_MODEL_ASSETS",
            "verified_utc": datetime.now(timezone.utc).isoformat(),
            "source_roots": {name: str(root) for name, root in sorted(source_roots.items())},
            "verification": "Full SHA256 for weight payloads and Git blob SHA1 for ordinary files",
            "required_files": len(verified), "verified_bytes": sum(item["bytes"] for item in verified.values()),
            "files": verified,
            "sources": [{"repository": model["id"], "revision": model["revision"]}
                        for model in manifest["models"]]}


def verify_assets(model_root, manifest):
    root = Path(model_root).resolve()
    if not root.is_dir():
        raise ValueError("Model root directory is missing: " + str(root))
    report = verify_selection(manifest, {IMAGE_REPOSITORY: root, BASE_REPOSITORY: root})
    report["model_root"] = str(root)
    return report


def verify_sources(shared_root, manifest):
    discovery = discover_models(shared_root)
    roots = {name: Path(value) for name, value in discovery["source_roots"].items()}
    report = verify_selection(manifest, roots)
    report["shared_root"] = discovery["shared_root"]
    return report


def view_identity(report):
    payload = {"schema": "h3-image-worker-model-view/1", "source_roots": report["source_roots"],
               "files": report["files"]}
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return identity, {**payload, "layout_sha256": identity}


def validate_view(view, expected):
    if view.is_symlink() or not view.is_dir():
        raise ValueError("Model view is not an owned directory: " + str(view))
    marker = view / "view-manifest.json"
    if marker.is_symlink() or not marker.is_file() or read_json(marker) != expected:
        raise ValueError("Existing model view ownership or manifest differs")
    files = expected["files"]
    wanted = set(files) | {"view-manifest.json"}
    actual = set()
    for path in view.rglob("*"):
        name = path.relative_to(view).as_posix()
        if path.is_symlink():
            if name not in files or os.readlink(path) != files[name]["source_path"]:
                raise ValueError("Model view has an unowned or altered link: " + name)
            if path.resolve(strict=True) != Path(files[name]["source_path"]):
                raise ValueError("Model view source link changed: " + name)
            actual.add(name)
        elif path.is_file():
            if name != "view-manifest.json":
                raise ValueError("Model view contains an unowned regular file: " + name)
            actual.add(name)
        elif not path.is_dir():
            raise ValueError("Model view contains an unsupported entry: " + name)
    if actual != wanted:
        raise ValueError("Existing model view file selection differs")


def prepare_view(report, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base = output / "model-view"
    if base.is_symlink():
        raise ValueError("Model view parent must be an owned local directory")
    base.mkdir(exist_ok=True)
    identity, expected = view_identity(report)
    view = base / identity
    if view.exists() or view.is_symlink():
        validate_view(view, expected)
        return view.resolve()
    with tempfile.TemporaryDirectory(prefix=".building-", dir=base) as temporary:
        building = Path(temporary)
        for name, item in expected["files"].items():
            link = building.joinpath(*relative_path(name).parts)
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(item["source_path"])
        write_json(building / "view-manifest.json", expected)
        validate_view(building, expected)
        try:
            building.rename(view)
        except OSError:
            if not view.is_dir() or view.is_symlink():
                raise
            validate_view(view, expected)
    return view.resolve()


def build_argv(profiles, profile_id, model_root=MODEL_ROOT, port=None):
    matches = [profile for profile in profiles["profiles"] if profile["id"] == profile_id]
    if len(matches) != 1:
        raise ValueError("Unknown or ambiguous worker profile: " + profile_id)
    arguments = list(matches[0]["argv"])
    if arguments[:3] != ["vllm", "serve", profiles["model_root"]]:
        raise ValueError("Unexpected native CLI entry point")
    root = os.fspath(model_root)
    if not Path(root).is_absolute() and not PurePosixPath(root).is_absolute():
        raise ValueError("The model root must be an absolute local path")
    arguments[2] = root
    if port is not None:
        if type(port) is not int or not 1 <= port <= 65535 or arguments.count("--port") != 1:
            raise ValueError("The service port must be between 1 and 65535")
        arguments[arguments.index("--port") + 1] = str(port)
    return arguments


def discover_models(shared_root):
    root = Path(shared_root).resolve()
    candidates = []
    for parent in (root, root / "generation", root / "H3_Image_models", root / "H3_Image_models/generation"):
        image = parent / "MiniMax-H3-Image"
        encoder = parent / "MiniMax-H3/text_encoder"
        if image.is_dir() and (image / "model_index.json").is_file() and encoder.is_dir():
            candidates.append((image.resolve(), encoder.resolve()))
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise ValueError(f"Expected one flat or generation model layout; found {len(candidates)}")
    image, encoder = candidates[0]
    if not image.is_relative_to(root) or not encoder.is_relative_to(root):
        raise ValueError("Discovered model directories resolve outside the shared root")
    return {"schema": "h3-image-worker-discovery/1", "status": "LAYOUT_FOUND_CONTENT_NOT_VERIFIED",
            "shared_root": str(root),
            "source_roots": {IMAGE_REPOSITORY: str(image), BASE_REPOSITORY: str(encoder.parent)},
            "read_only_mounts": [{"source": str(root), "target": "/source-models"}],
            "next_step": "Use --shared-root against this read-only mount to verify and assemble an owned local view"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="h100-4-tp4")
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--model-root", type=Path)
    sources.add_argument("--shared-root", type=Path)
    parser.add_argument("--port", type=int)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--verify-assets-only", action="store_true")
    parser.add_argument("--discover-models", type=Path)
    args = parser.parse_args()
    if args.discover_models is not None:
        print(json.dumps(discover_models(args.discover_models), indent=2))
        return
    profiles = read_json(HERE / "profiles.json")
    # Validate the requested profile before reading large weight payloads.
    build_argv(profiles, args.profile, MODEL_ROOT, args.port)
    manifest = read_json(HERE / "assets.lock.json")
    if args.shared_root is not None:
        report = verify_sources(args.shared_root, manifest)
        model_root = prepare_view(report, args.output_dir)
        report["model_root"] = str(model_root)
        report["view_manifest"] = str(model_root / "view-manifest.json")
    else:
        model_root = args.model_root or MODEL_ROOT
        report = verify_assets(model_root, manifest)
    arguments = build_argv(profiles, args.profile, model_root, args.port)
    report["profile"] = args.profile
    report["argv"] = arguments
    write_json(args.output_dir / "model-validation.json", report)
    print(f"MODEL_ASSETS_VERIFIED files={report['required_files']} bytes={report['verified_bytes']}", flush=True)
    if args.verify_assets_only:
        return
    environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    # The installed vLLM CLI is the framework's normal --omni entry point.
    command = [sys.executable, "-P", "-m", "vllm.entrypoints.cli.main", *arguments[1:]]
    os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()
