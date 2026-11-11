#!/usr/bin/env python3
"""Read installed metadata in a fresh process before optional imports."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import re


def digest(data):
    return hashlib.sha256(data).hexdigest()


def collect(manifest):
    packages = {}
    for distribution in metadata.distributions():
        name = re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower()
        if name in packages:
            raise ValueError("Duplicate installed distribution: " + name)
        packages[name] = {
            "version": distribution.version,
            "metadata_sha256": digest((distribution.read_text("METADATA") or "").encode()),
            "record_sha256": digest((distribution.read_text("RECORD") or "").encode()),
            "direct_url_sha256": digest((distribution.read_text("direct_url.json") or "").encode()),
        }
    omni = metadata.distribution("vllm-omni")
    expected = json.loads(Path(manifest).read_text(encoding="utf-8"))["files"]
    files = {}
    for name in expected:
        path = Path(omni.locate_file(name))
        if not path.is_file():
            raise ValueError("Parent package source is missing: " + name)
        files[name] = digest(path.read_bytes())
    return {"packages": dict(sorted(packages.items())), "source_sha256": dict(sorted(files.items())),
            "omni_root": str(Path(omni.locate_file("vllm_omni")).resolve()),
            "omni_version": omni.version, "omni_direct_url": omni.read_text("direct_url.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.source_manifest), sort_keys=True))
