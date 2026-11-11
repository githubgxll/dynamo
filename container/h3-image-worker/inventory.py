#!/usr/bin/env python3
"""Read installed metadata in a fresh process before optional imports."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata as metadata
import io
import json
from pathlib import Path, PurePosixPath
import re


GENERATED_SOURCE = "vllm_omni/_version.py"
CRITICAL_SOURCE_PATHS = (
    "vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3_image.py",
    "vllm_omni/diffusion/models/minimax_h3/image_vae.py",
    "vllm_omni/diffusion/models/minimax_h3/encoder.py",
    "vllm_omni/diffusion/models/minimax_h3/minimax_h3_blocks.py",
    "vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py",
    "vllm_omni/engine/arg_utils.py",
    "vllm_omni/entrypoints/cli/serve.py",
    "vllm_omni/entrypoints/openai/api_server.py",
    "vllm_omni/entrypoints/openai/protocol/images.py",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical_name(name):
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Installed distribution has no valid Name metadata")
    return re.sub(r"[-_.]+", "-", name).lower()


def distribution_record(distribution):
    name = canonical_name(distribution.metadata["Name"])
    directory = getattr(distribution, "_path", None)
    if directory is None:
        raise ValueError("Cannot locate installed metadata directory for " + name)
    directory = Path(directory).resolve()
    metadata_text = distribution.read_text("METADATA")
    if metadata_text is None:
        metadata_text = distribution.read_text("PKG-INFO")
    if metadata_text is None:
        metadata_text = distribution.read_text("")
    if not metadata_text:
        raise ValueError("Missing or empty METADATA/PKG-INFO/legacy egg-info in " + str(directory))
    record = distribution.read_text("RECORD")
    direct_url = distribution.read_text("direct_url.json")
    return {
        "canonical_name": name,
        "metadata_directory": str(directory),
        "version": distribution.version,
        "metadata_sha256": digest(metadata_text.encode()),
        "record_sha256": None if record is None else digest(record.encode()),
        "direct_url_sha256": None if direct_url is None else digest(direct_url.encode()),
    }


def installed_source_inventory(omni, expected):
    root = Path(omni.locate_file("vllm_omni")).resolve()
    if not root.is_dir():
        raise ValueError("Installed Omni package directory is missing: " + str(root))
    record = omni.read_text("RECORD")
    if record is None:
        raise ValueError("Active Omni distribution has no RECORD")
    recorded = {}
    for row in csv.reader(io.StringIO(record)):
        if len(row) != 3:
            raise ValueError("Malformed Omni RECORD row: " + repr(row))
        name, recorded_hash, recorded_size = row
        if not name.startswith("vllm_omni/") or not name.endswith(".py"):
            continue
        relative = PurePosixPath(name)
        if ("\\" in name or ":" in name or relative.is_absolute()
                or ".." in relative.parts or str(relative) != name):
            raise ValueError("Unsafe Omni source RECORD path: " + name)
        if name in recorded:
            raise ValueError("Duplicate Omni source RECORD row: " + name)
        path = Path(omni.locate_file(name)).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Omni source escapes package directory: " + name)
        if not path.is_file():
            raise ValueError("Omni RECORD source file is missing: " + name)
        payload = path.read_bytes()
        encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode().rstrip("=")
        if recorded_hash != "sha256=" + encoded:
            raise ValueError("Omni source RECORD hash differs: " + name)
        if not recorded_size.isdecimal() or int(recorded_size) != len(payload):
            raise ValueError("Omni source RECORD size differs: " + name)
        recorded[name] = digest(payload)
    actual = {"vllm_omni/" + path.relative_to(root).as_posix()
              for path in root.rglob("*.py") if path.is_file()}
    missing = sorted(set(recorded) - actual)
    unrecorded = sorted(actual - set(recorded))
    if missing or unrecorded:
        raise ValueError("Omni filesystem/RECORD source mismatch: " + json.dumps({
            "missing_record_sources": missing, "unexpected_installed_sources": unrecorded,
        }))
    unknown = sorted(set(recorded) - set(expected) - {GENERATED_SOURCE})
    if unknown:
        raise ValueError("Installed Omni source absent from COPY manifest: " + json.dumps(unknown))
    absent_critical = sorted(set(CRITICAL_SOURCE_PATHS) - set(recorded))
    if absent_critical:
        raise ValueError("Required installed Omni source is missing: " + json.dumps(absent_critical))
    generated = {name: value for name, value in recorded.items() if name == GENERATED_SOURCE}
    projected = {name: value for name, value in recorded.items() if name != GENERATED_SOURCE}
    coverage = {
        "status": "PASS", "copied_source_files": len(expected),
        "installed_record_source_files": len(recorded), "recorded_source_files": len(recorded),
        "parent_projected_source_files": len(projected),
        "missing_record_sources": [], "unexpected_installed_sources": [],
        "critical_paths_verified": list(CRITICAL_SOURCE_PATHS),
        "generated_source_files": sorted(generated),
        "omitted_from_wheel": sorted(set(expected) - set(recorded)),
    }
    return root, projected, generated, coverage


def collect(manifest):
    instances = {}
    for distribution in metadata.distributions():
        record = distribution_record(distribution)
        key = record["metadata_directory"]
        if key in instances and instances[key] != record:
            raise ValueError("Conflicting metadata at the same physical directory: " + key)
        instances[key] = record
    packages = {}
    for name in sorted({record["canonical_name"] for record in instances.values()}):
        active = distribution_record(metadata.distribution(name))
        if instances.get(active["metadata_directory"]) != active:
            raise ValueError("Selected distribution is absent from metadata inventory: " + name)
        packages[name] = active
    omni = metadata.distribution("vllm-omni")
    expected = json.loads(Path(manifest).read_text(encoding="utf-8"))["files"]
    root, files, generated, coverage = installed_source_inventory(omni, expected)
    return {"packages": packages,
            "distributions": sorted(instances.values(), key=lambda item: (item["canonical_name"], item["metadata_directory"])),
            "source_sha256": dict(sorted(files.items())),
            "generated_source_sha256": dict(sorted(generated.items())),
            "source_coverage": coverage, "omni_root": str(root),
            "omni_version": omni.version, "omni_direct_url": omni.read_text("direct_url.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.source_manifest), sort_keys=True))
