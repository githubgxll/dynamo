# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record installed package versions and H3 source hashes without a GPU.

This is an inventory of the built environment, not a transitive input lock or
proof of GPU inference. Do not collect environment variables or credentials.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import platform
import re
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version


def validate_framework_dependencies() -> None:
    """Compatibility entry point; main additionally audits every distribution."""
    errors = []
    for project in ("vllm", "vllm-omni"):
        for raw in md.requires(project) or []:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            try:
                installed = md.version(requirement.name)
            except md.PackageNotFoundError:
                errors.append(f"{project}: missing {requirement}")
                continue
            if requirement.specifier and not requirement.specifier.contains(installed, prereleases=True):
                errors.append(f"{project}: {requirement}, installed {installed}")
    if errors:
        raise RuntimeError("Framework dependency check failed:\n" + "\n".join(errors))


def audit_dependencies(distributions=None, *, python_version=None, environment=None):
    """Audit installed metadata, propagating extras explicitly requested by edges.

    Installed metadata does not record which top-level extras were selected.
    Start with each installed package's base requirements, then recursively
    activate extras requested by active requirements. Do not activate every
    optional development/backend extra merely because it is advertised.
    """
    python_version = python_version or platform.python_version()
    marker_environment = default_environment()
    marker_environment.update({
        "python_full_version": python_version,
        "python_version": ".".join(python_version.split(".")[:2]),
    })
    if environment:
        marker_environment.update(environment)
    report = {
        "python": python_version,
        "status": "FAIL",
        "packages": {},
        "active_extras": {},
        "checked_requirements": 0,
        "known_metadata_exceptions": [],
        "errors": [],
        "validation_scope": (
            "Installed metadata requirements and referenced extras; the exact "
            "KVBM/NIXL metadata exception is not proof of ABI or GPU compatibility"
        ),
    }
    records = {}
    for dist in md.distributions() if distributions is None else distributions:
        try:
            name = dist.metadata["Name"]
            if not name:
                raise ValueError("missing Name")
            key = canonicalize_name(name, validate=True)
            version = str(Version(dist.version))
            if key in records:
                raise ValueError(f"duplicate installed distribution: {key}")
            requirements = [Requirement(raw) for raw in dist.requires or []]
            requires_python = dist.metadata.get("Requires-Python")
            if requires_python and not SpecifierSet(requires_python).contains(
                python_version, prereleases=True
            ):
                report["errors"].append(
                    f"{name}=={version}: Requires-Python {requires_python}, "
                    f"running {python_version}"
                )
            records[key] = (name, version, requirements)
            report["packages"][name] = version
        except Exception as exc:
            report["errors"].append(f"Invalid installed metadata: {exc}")

    active_extras = {name: set() for name in records}

    def active(requirement, extras):
        if not requirement.marker:
            return True
        return any(
            requirement.marker.evaluate({**marker_environment, "extra": extra})
            for extra in {"", *extras}
        )

    # A package may be visited before its consumer requests an extra. Iterate
    # until every recursive extra reference has propagated, including cycles.
    try:
        changed = True
        while changed:
            changed = False
            for key, (_, _, requirements) in records.items():
                for requirement in requirements:
                    target = canonicalize_name(requirement.name)
                    if target in active_extras and active(requirement, active_extras[key]):
                        before = len(active_extras[target])
                        active_extras[target].update(
                            canonicalize_name(extra) for extra in requirement.extras
                        )
                        changed |= before != len(active_extras[target])

        for key, (name, version, requirements) in records.items():
            for requirement in requirements:
                if not active(requirement, active_extras[key]):
                    continue
                report["checked_requirements"] += 1
                target = canonicalize_name(requirement.name)
                if target not in records:
                    report["errors"].append(f"{name}=={version}: missing {requirement}")
                    continue
                installed = records[target][1]
                if requirement.specifier.contains(installed, prereleases=True):
                    continue
                detail = f"{name}=={version}: requires {requirement}, installed {installed}"
                known_exception = (
                    key == "kvbm" and version == "1.3.0"
                    and target == "nixl" and installed == "1.3.1"
                    and requirement.extras == {"cu13"}
                    and str(requirement.specifier) == "==1.0.1"
                    and requirement.marker is None and requirement.url is None
                )
                report["known_metadata_exceptions" if known_exception else "errors"].append(detail)
    except Exception as exc:
        report["errors"].append(f"Dependency marker evaluation failed: {exc}")
    report["active_extras"] = {
        key: sorted(extras) for key, extras in sorted(active_extras.items()) if extras
    }
    report["packages"] = dict(sorted(report["packages"].items()))
    report["errors"] = sorted(set(report["errors"]))
    report["known_metadata_exceptions"] = sorted(set(report["known_metadata_exceptions"]))
    finalize_status(report)
    return report


def finalize_status(report):
    report["status"] = (
        "FAIL" if report["errors"] else
        "PASS_WITH_KNOWN_METADATA_EXCEPTION" if report["known_metadata_exceptions"] else
        "PASS"
    )


def review_uv_check(report, exit_code, output):
    """A uv failure must be explained by exactly the reviewed metadata mismatch."""
    report["uv_pip_check_exit_code"] = exit_code
    if exit_code not in (0, 1):
        report["errors"].append(f"uv pip check failed as a tool (exit {exit_code})")
    elif exit_code == 0 and report["known_metadata_exceptions"]:
        report["errors"].append(
            "uv reports a clean environment but the metadata audit found the "
            "KVBM/NIXL exception; verify both checks used the same interpreter"
        )
    elif exit_code == 1:
        # Unknown diagnostic text is intentionally fail-closed. A changed uv
        # output format must be reviewed, never mistaken for the known issue.
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        diagnostic = (
            "The package `kvbm` requires `nixl[cu13]==1.0.1`, "
            "but `1.3.1` is installed"
        )
        expected = {
            "Found 1 incompatibility",
            diagnostic,
        }
        meaningful = [
            line for line in lines
            if not re.fullmatch(r"Checked \d+ packages? in .+", line)
            and not re.fullmatch(r"Using Python .+ environment at: .+", line)
        ]
        if (len(report["known_metadata_exceptions"]) != 1
                or len(meaningful) != 2 or set(meaningful) != expected):
            report["errors"].append(
                "uv pip check failure is not exactly the known KVBM/NIXL diagnostic; "
                "review uv-pip-check.txt"
            )
    finalize_status(report)


def main() -> None:
    directory = Path("/opt/dynamo/build-info")
    directory.mkdir(parents=True, exist_ok=True)
    report = audit_dependencies()
    try:
        status = int((directory / "uv-pip-check.exit-code").read_text().strip())
        output = (directory / "uv-pip-check.txt").read_text(encoding="utf-8")
        review_uv_check(report, status, output)
    except Exception as exc:
        report["errors"].append(f"Cannot read complete uv dependency evidence: {exc}")
    for project in ("vllm", "vllm-omni"):
        if project not in {canonicalize_name(name) for name in report["packages"]}:
            report["errors"].append(f"Required framework missing: {project}")
    finalize_status(report)
    (directory / "dependency-audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if report["errors"]:
        raise RuntimeError("Runtime dependency audit failed:\n" + "\n".join(report["errors"]))
    packages = report["packages"]
    omni = md.distribution("vllm-omni")
    hashes = {}
    for name in (
        "vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py",
        "vllm_omni/diffusion/models/minimax_h3/vae.py",
        "vllm_omni/diffusion/models/minimax_h3/quality_policy.py",
        "vllm_omni/diffusion/models/minimax_h3/time_request.py",
    ):
        source = Path(omni.locate_file(name))
        if not source.is_file():
            raise RuntimeError(f"Required H3 source missing: {name}")
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "packages": packages,
        "dependency_audit_status": report["status"],
        "known_metadata_exceptions": report["known_metadata_exceptions"],
        "h3_source_sha256": hashes,
        "validation_scope": "CPU build inventory; GPU/media E2E still required",
    }
    (directory / "runtime-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (directory / "installed-versions.txt").write_text(
        "".join(f"{name}=={version}\n" for name, version in packages.items()),
        encoding="utf-8",
    )
    print("Runtime package inventory and H3 source hashes recorded")


if __name__ == "__main__":
    main()
