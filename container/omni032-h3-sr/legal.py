# SPDX-License-Identifier: Apache-2.0
"""Refresh Python attribution and gate changed components with existing policy.

The immutable parent's OS/Rust/native notices remain historical inherited
attribution. This supplement lists the current full Python environment and
validates every added/changed Python distribution, without reclassifying an
unchanged vendor baseline or weakening the existing license policy.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import subprocess
import sys

from packaging.utils import canonicalize_name

HERE = Path(__file__).resolve().parent
LEGAL = Path("/legal/omni032-upgrade")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_delta(rows, inventory):
    baseline = {(canonicalize_name(p["canonical_name"]), p["version"])
                for p in inventory["distributions"]}
    return [row for row in rows if (canonicalize_name(row["name"]), row["version"]) not in baseline]


def manifest(directory):
    return {p.relative_to(directory).as_posix(): sha256(p) for p in sorted(directory.rglob("*")) if p.is_file()}


def manifest_digest(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build(output):
    if os.geteuid() != 0 or LEGAL.exists():
        raise ValueError("Create-once root attribution action required")
    inputs = output / "environment"
    before = json.loads((inputs / "parent-inventory.json").read_text())
    paths = sorted({str(Path(getattr(dist, "_path")).parent.resolve()) for dist in md.distributions()
                    if getattr(dist, "_path", None) is not None})
    command = [sys.executable, "-B", "-m", "compliance.generators", "--ecosystem", "python",
               "--output-dir", str(LEGAL), "--policy", str(HERE / "compliance/policy/licenses.toml")]
    for path in paths:
        command.extend(["--site-packages", path])
    env = dict(os.environ, PYTHONPATH=str(HERE), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    if result.returncode:
        print(result.stdout, flush=True)
        raise RuntimeError("Current Python attribution generation failed")
    (LEGAL / "generation.txt").write_text(result.stdout, encoding="utf-8")
    with (LEGAL / "osrb-deps.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    delta = select_delta(rows, before)
    versions = {(canonicalize_name(row["name"]), row["version"]) for row in delta}
    if not {("vllm", "0.32.0"), ("vllm-omni", "0.32.0")}.issubset(versions):
        raise ValueError("License delta did not include both upgraded framework releases")
    with (LEGAL / "changed-python-deps.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(delta)
    check = [sys.executable, "-B", "-m", "compliance.policy.validate", "--policy",
             str(HERE / "compliance/policy/licenses.toml"), "--image", "vllm-runtime",
             "--input", str(LEGAL / "changed-python-deps.csv")]
    result = subprocess.run(check, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    (LEGAL / "policy-validation.txt").write_text(result.stdout, encoding="utf-8")
    print(result.stdout, flush=True)
    if result.returncode:
        raise RuntimeError("Changed package licenses failed the inherited policy; no policy relaxation applied")
    (LEGAL / "README.txt").write_text(
        "Omni 0.32 upgrade attribution supplement\n"
        "osrb-deps.csv/osrb.cdx.json and NOTICES-Python.txt describe the current full Python environment.\n"
        "changed-python-deps.csv is the added/version-changed Python delta relative to the verified b315 parent; "
        "it passed the unchanged vllm-runtime license policy.\n"
        "Older /legal Python inventories describe the inherited image and are superseded for Python by this directory. "
        "The parent's unchanged OS/Rust/native attributions are retained.\n"
        "No source archive or new legal approval is claimed by this metadata/policy check.\n", encoding="utf-8")
    files = manifest(LEGAL)
    receipt = {"schema": "omni032-license-supplement/1", "status": "PASS", "legal_directory": str(LEGAL),
               "source_commit": os.environ["OMNI032_H3_SR_SOURCE_COMMIT"], "python_components": len(rows),
               "changed_python_components": len(delta), "policy_sha256": sha256(HERE / "compliance/policy/licenses.toml"),
               "files": files, "legal_manifest_sha256": manifest_digest(files),
               "scope": "Full current Python notices plus strict changed-component policy; inherited non-Python attribution retained"}
    (output / "legal-validation.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def validate(inputs):
    receipt = json.loads((inputs / "legal-validation.json").read_text())
    current = manifest(LEGAL)
    if (receipt.get("status") != "PASS" or receipt.get("files") != current
            or receipt.get("legal_manifest_sha256") != manifest_digest(current)
            or receipt.get("source_commit") != os.environ["OMNI032_H3_SR_SOURCE_COMMIT"]
            or receipt.get("policy_sha256") != sha256(HERE / "compliance/policy/licenses.toml")):
        raise ValueError("Packaged license supplement differs from build validation")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
