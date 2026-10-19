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
import subprocess
import sys

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version


DEFAULT_ROOTS = ("ai-dingo[video-gateway]", "ai-dingo-runtime", "vllm", "gpu-memory-service")
# Reviewed released wheels need their shared Python APIs solved jointly with
# the inherited vLLM defaults. An unknown future release retains the base pins.
JOINT_API_OMNI_VERSIONS = {"0.31.0rc1"}
# Introduced with SDK 1.44: it has no release compatible with the inherited
# LMCache API<=1.40 bound. Older OTLP exporters supply their implementation
# without this distribution. Only remove it after solving when no installed
# package requires it; never delete the shared opentelemetry namespace.
RESOLVER_MANAGED_TELEMETRY = frozenset({"opentelemetry-exporter-otlp-common"})


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


def inherited_telemetry_requirements(inventory):
    """Keep inherited monitoring capabilities in the joint solve.

    uv install does not solve unrelated installed packages. A retained exporter
    can otherwise require a newer SDK than the requested graph selects. Use
    names, not old version pins or old dependency edges, so the resolver can
    select a mutually compatible family. The reviewed new-family internal
    split is resolver-managed, not a mandatory monitoring capability.
    """
    records = index_records(inventory)
    names = sorted(name for name in records if name.startswith("opentelemetry-"))
    for name in names:
        unique_version(records, name)
    return [name for name in names if name not in RESOLVER_MANAGED_TELEMETRY]


def protected_runtime_requirements(inventory, constraints, *, environment=None):
    """Solve the dependencies of every package whose installed version is fixed.

    A constraint such as lmcache==0.5.5 does not make uv inspect LMCache's
    dependencies unless it is reached by the requested graph. Read its installed
    metadata, like all other protected components, without requesting a new
    wheel for vendor-built packages. Mutable packages' old metadata must not be
    frozen this way; their new metadata is resolved by uv.
    """
    constraints = list(constraints)
    verify_protected(inventory, constraints)
    return runtime_requirements(inventory, roots=constraints, environment=environment)


def telemetry_reconciliation_plan(inventory):
    """Only prune reviewed internal splits with no remaining metadata consumers.

    Include even inactive extras/platform edges conservatively: never remove a
    package named by any installed consumer. The final audit still decides
    compatibility when a split has to remain.
    """
    records = index_records(inventory)
    candidates = {}
    for name in sorted(RESOLVER_MANAGED_TELEMETRY & records.keys()):
        version = unique_version(records, name)
        consumers = sorted({f"{owner}=={item['version']}: {raw}"
                            for owner, items in records.items() for item in items
                            for raw in item["requires_dist"]
                            if canonicalize_name(Requirement(raw).name) == name})
        candidates[name] = {"version": version, "consumers": consumers,
                            "action": "retain-required" if consumers else "remove-orphan"}
    return {"schema": "h3-telemetry-reconciliation/1", "candidates": candidates,
            "remove": [name for name, item in candidates.items() if not item["consumers"]],
            "scope": "Exact reviewed internal split only; preserve SDK and public exporters; all dependency checks still required"}


def reconcile_telemetry(inventory, directory):
    plan = telemetry_reconciliation_plan(inventory)
    plan["status"] = "PLANNED"
    path = directory / "telemetry-reconciliation.json"
    write_json(path, plan)
    if plan["remove"]:
        # uv uses wheel RECORD to remove only this distribution's owned files.
        result = subprocess.run(["uv", "pip", "uninstall", "--python", sys.executable, *plan["remove"]],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        (directory / "telemetry-reconciliation.txt").write_text(result.stdout, encoding="utf-8")
        print(result.stdout, flush=True)
        if result.returncode:
            plan["status"] = "FAIL"
            write_json(path, plan)
            raise RuntimeError("Cannot reconcile reviewed orphan telemetry implementation")
    plan["status"] = "COMPLETE"
    write_json(path, plan)


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
    parser.add_argument("phase", choices=("before", "reconcile-telemetry", "after"))
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
        protected_requirements = protected_runtime_requirements(inventory, constraints)
        (args.directory / "protected-runtime-requirements.txt").write_text(
            "\n".join(protected_requirements) + "\n", encoding="utf-8"
        )
        write_json(args.directory / "protected-runtime-solve-inputs.json", {
            "schema": "h3-protected-runtime-solve/1",
            "source_inventory": "runtime-before-omni.json",
            "protected_roots": constraints,
            "requirements": protected_requirements,
            "strategy": "Include active default Requires-Dist from every exact-version protected component in the joint solve; preserve requested dependency extras and the reviewed NCCL override",
        })
        telemetry = inherited_telemetry_requirements(inventory)
        (args.directory / "inherited-telemetry-requirements.txt").write_text(
            "\n".join(telemetry) + "\n", encoding="utf-8"
        )
        records = index_records(inventory)
        write_json(args.directory / "telemetry-solve-inputs.json", {
            "strategy": "Resolve inherited monitoring capabilities together with full Omni/service and protected-runtime requirements (including installed LMCache bounds); the reviewed SDK1.44+ internal otlp-common split is not a mandatory root and is pruned after solving only if no installed metadata consumer requires it; no metadata exemptions",
            "roots": telemetry,
            "installed_before": {name: unique_version(records, name) for name in telemetry},
            "resolver_managed_internal_packages": {
                name: unique_version(records, name) for name in sorted(RESOLVER_MANAGED_TELEMETRY)
                if name in records
            },
        })
    elif args.phase == "reconcile-telemetry":
        reconcile_telemetry(inventory, args.directory)
    else:
        verify_protected(inventory, constraints_path.read_text(encoding="utf-8").splitlines())


if __name__ == "__main__":
    main()
