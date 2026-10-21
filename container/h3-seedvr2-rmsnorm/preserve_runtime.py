#!/usr/bin/env python3
"""Compare unchanged parent metadata, Omni source and inherited legal/H3 files."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
INFO = HERE / "build-info"
TARGET = "vllm_omni/diffusion/models/seedvr2/nadit.py"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory():
    packages = []
    for dist in md.distributions():
        packages.append({"name": dist.metadata["Name"], "version": dist.version,
                         "metadata": dist.read_text("METADATA"),
                         "direct_url": dist.read_text("direct_url.json")})
    packages.sort(key=lambda row: (row["name"].lower(), row["version"], row["metadata"] or ""))
    distribution = md.distribution("vllm-omni")
    source = Path(distribution.locate_file("vllm_omni")).resolve(strict=True)
    files = {}
    for file in sorted(source.rglob("*.py")):
        relative = file.relative_to(source).as_posix()
        if "vllm_omni/" + relative != TARGET:
            files["omni/" + relative] = digest(file)
    # Includes the old immutable build reports; those must never be overwritten.
    for name in ("/legal", "/opt/h3-experimental", "/opt/dynamo/build-info"):
        root = Path(name)
        if not root.is_dir():
            raise RuntimeError("Required inherited directory missing: " + name)
        for file in sorted(root.rglob("*")):
            if file.is_file() and "__pycache__" not in file.parts and file.suffix not in {".pyc", ".pyo"}:
                files[file.as_posix()] = digest(file)
    dpkg = subprocess.run(["dpkg-query", "-W", "-f=${Package}\t${Version}\t${Architecture}\n"],
                          check=True, capture_output=True, text=True).stdout
    return {"packages": packages, "files": files, "dpkg_packages": sorted(dpkg.splitlines())}


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["capture", "verify"])
    args = parser.parse_args()
    lock = json.loads((HERE / "inputs/source-lock.json").read_text(encoding="utf-8"))
    actual = inventory()
    if args.action == "capture":
        if os.environ.get("H3_EXPERIMENTAL_SOURCE_COMMIT") != lock["parent_source_commit"]:
            raise RuntimeError("Parent image source marker does not match the successful image")
        for name, expected in lock["versions"].items():
            if md.version(name) != expected:
                raise RuntimeError("Unexpected parent package version: " + name)
        source = Path(md.distribution("vllm-omni").locate_file(TARGET))
        if digest(source) != lock["source_before_sha256"]:
            raise RuntimeError("Parent nadit.py differs from the reviewed source")
        write(INFO / "parent-snapshot.json", actual)
    else:
        before = json.loads((INFO / "parent-snapshot.json").read_text(encoding="utf-8"))
        if actual != before:
            changes = [key for key in before if actual.get(key) != before[key]]
            raise RuntimeError("Unexpected changes outside the RMSNorm patch: " + ", ".join(changes))
        if os.environ.get("H3_RMSNORM_PARENT_IMAGE") != lock["parent_image"]:
            raise RuntimeError("Wrong immediate parent image identity")
        if not re.fullmatch(r"[0-9a-f]{40}", os.environ.get("H3_EXPERIMENTAL_SOURCE_COMMIT", "")):
            raise RuntimeError("Invalid new source commit marker")
        write(INFO / "preservation.json", {
            "status": "PASS_PRESERVED_PARENT_CONTENT_CPU_ONLY", "euid": os.geteuid(),
            "source_commit": os.environ["H3_EXPERIMENTAL_SOURCE_COMMIT"],
            "parent_image": lock["parent_image"], "snapshot_sha256": digest(INFO / "parent-snapshot.json"),
            "distribution_count": len(actual["packages"]), "unchanged_file_count": len(actual["files"]),
            "scope": "Unchanged installed METADATA/direct_url, dpkg inventory, all other Omni Python source, inherited H3 scripts/reports and legal files. Native library identity additionally rests on inherited layers; no GPU/media claim."
        })


if __name__ == "__main__":
    main()
