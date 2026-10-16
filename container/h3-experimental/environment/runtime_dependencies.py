# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare one dependency solve from installed Dynamo/vLLM and released Omni.

The base image supplies the compiled GPU stack. Its installed metadata is the
source for constraints and dependency roots; no GPU is imported or required.
The snapshots are evidence of the installation, not a transitive input lock.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import platform
from pathlib import Path
import sys

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version


DEFAULT_ROOTS = ("ai-dingo[video-gateway]", "ai-dingo-runtime", "vllm", "gpu-memory-service")
# Reviewed released wheels need their shared Python APIs solved jointly with
# the inherited vLLM defaults. An unknown future release retains the base pins.
JOINT_API_OMNI_VERSIONS = {"0.31.0rc1"}


def snapshot(distributions=None, *, phase):
    records = []
    for dist in md.distributions() if distributions is None else distributions:
        name = dist.metadata["Name"]
        if not name:
            raise ValueError("Installed distribution is missing Name")
        records.append({
            "name": name,
            "canonical_name": canonicalize_name(name, validate=True),
            "version": str(Version(dist.version)),
            "metadata_path": str(getattr(dist, "_path", "")),
            "location": str(dist.locate_file("")),
            "requires_dist": list(dist.requires or []),
            "license_expression": dist.metadata.get("License-Expression"),
            "declared_license": dist.metadata.get("License"),
        })
    return {
        "schema_version": 1,
        "phase": phase,
        "python_version": platform.python_version(),
        "sys_path": list(sys.path),
        "distributions": sorted(records, key=lambda record: (
            record["canonical_name"], record["metadata_path"], record["version"]
        )),
    }


def index_records(inventory):
    records = {}
    for record in inventory["distributions"]:
        records.setdefault(record["canonical_name"], []).append(record)
    return records


def unique_version(records, name):
    versions = {record["version"] for record in records.get(name, [])}
    if not versions:
        raise ValueError(f"Required installed distribution is missing: {name}")
    if len(versions) != 1:
        raise ValueError(f"Conflicting installed versions for {name}: {sorted(versions)}")
    return next(iter(versions))


def protected_constraints(inventory, configured_names, *, omni_version):
    records = index_records(inventory)
    names = {canonicalize_name(name) for name in configured_names}
    # Freeze every NVIDIA library in this image, not just a handwritten subset.
    names.update(name for name in records if name.startswith("nvidia-"))
    if omni_version in JOINT_API_OMNI_VERSIONS:
        # Shared Python APIs must satisfy both frameworks' requirements. The
        # installed GPU stack remains fixed, including upstream NCCL overrides.
        names.difference_update({"transformers", "tokenizers"})
    for name in ("vllm", "torch"):
        unique_version(records, name)
        names.add(name)
    return [f"{name}=={unique_version(records, name)}" for name in sorted(names)
            if name in records]


def runtime_requirements(inventory, *, roots=DEFAULT_ROOTS, environment=None):
    """Collect installed default requirements, preserving requested extras.

    Extra-only roots such as ai-dingo[vllm] are deliberately not requested:
    their backend pins describe a different vLLM version. All default vLLM and
    vLLM-Omni dependencies are resolved together by the caller instead.
    """
    records = index_records(inventory)
    marker_environment = default_environment()
    if environment:
        marker_environment.update(environment)
    marker_environment["extra"] = ""
    lines = []
    for project in roots:
        root = Requirement(project)
        name = canonicalize_name(root.name)
        active_extras = {"", *root.extras}
        version = unique_version(records, name)
        # Repeated discovery of one metadata directory is kept in the snapshot;
        # conflicting requirements for a critical root must not be hidden.
        requirement_sets = {tuple(sorted(record["requires_dist"])) for record in records[name]}
        if len(requirement_sets) != 1:
            raise ValueError(f"Conflicting installed requirements for {name}")
        lines.append(f"# Active requirements from installed {name}=={version}; extras={sorted(root.extras)}")
        for raw in sorted(next(iter(requirement_sets))):
            requirement = Requirement(raw)
            if requirement.marker and not any(requirement.marker.evaluate({**marker_environment, "extra": extra}) for extra in active_extras):
                continue
            # The root extra is already evaluated on the actual build platform.
            # Do not leave an extra marker for pip to re-evaluate as empty.
            if requirement.marker and "extra" in str(requirement.marker):
                requirement.marker = None
            lines.append(str(requirement))
    return lines


def verify_protected(inventory, constraints):
    records = index_records(inventory)
    for line in constraints:
        name, expected = line.split("==", 1)
        actual = unique_version(records, canonicalize_name(name))
        if actual != expected:
            raise ValueError(f"Protected package changed: {name}: {expected} -> {actual}")


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("before", "after"))
    parser.add_argument("--directory", type=Path, default=Path("/opt/dynamo/build-info"))
    parser.add_argument("--protected-packages", type=Path)
    parser.add_argument("--omni-version")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    inventory = snapshot(phase=args.phase)
    write_json(args.directory / f"runtime-{args.phase}-omni.json", inventory)
    constraints_path = args.directory / "protected-before-omni.txt"
    if args.phase == "before":
        if not args.protected_packages or not args.omni_version:
            parser.error("before requires --protected-packages and --omni-version")
        names = [line.split("#", 1)[0].strip()
                 for line in args.protected_packages.read_text(encoding="utf-8").splitlines()]
        constraints = protected_constraints(
            inventory, filter(None, names), omni_version=args.omni_version
        )
        constraints_path.write_text("\n".join(constraints) + "\n", encoding="utf-8")
        (args.directory / "dynamo-vllm-default-requirements.txt").write_text(
            "\n".join(runtime_requirements(inventory)) + "\n", encoding="utf-8"
        )
    else:
        verify_protected(inventory, constraints_path.read_text(encoding="utf-8").splitlines())


if __name__ == "__main__":
    main()
