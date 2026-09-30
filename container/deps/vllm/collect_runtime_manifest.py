# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit installed defaults and recursively requested extras; not GPU validation."""
from __future__ import annotations
import hashlib
import importlib.metadata as md
import importlib.util
import json
import os
import platform
import re
from pathlib import Path
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

BUILD_INFO_DIRECTORY = Path("/opt/dynamo/build-info")
BASE_DIGEST = "sha256:439c19d48db36401abc914b9842d060fe610a54bc3ac29bb02505f0b69af1baf"
BASE_TAG = "v0.30.0-ubuntu2404"
NCCL_NAME, NCCL_VERSION, TORCH_VERSION = "nvidia-nccl-cu13", "2.30.7", "2.13.0+cu130"
SHADOWABLE_MODULES = {
    "cryptography": "cryptography", "pyjwt": "jwt", "six": "six", "oauthlib": "oauthlib",
}


def upstream_nccl_evidence(base_reference, overrides_text, protected_text):
    """Require exact pinned base, upstream override, and pre-Omni snapshot."""
    errors = []
    reference = base_reference.removeprefix("docker.io/")
    if reference not in {f"vllm/vllm-openai:{BASE_TAG}@{BASE_DIGEST}", f"vllm/vllm-openai@{BASE_DIGEST}"}:
        errors.append("Runtime base is not the reviewed digest-pinned official vLLM 0.30 image")

    def pins(text):
        result = {}
        for raw in text.splitlines():
            raw = raw.split("#", 1)[0].strip()
            if raw:
                requirement = Requirement(raw)
                result.setdefault(canonicalize_name(requirement.name), []).append(requirement)
        return result

    def exact_pin(requirements, name, version):
        values = requirements.get(name, [])
        return (len(values) == 1 and str(values[0].specifier) == f"=={version}"
                and not values[0].extras and values[0].marker is None and values[0].url is None)

    try:
        if not exact_pin(pins(overrides_text), NCCL_NAME, NCCL_VERSION):
            errors.append("Official uv overrides lack the unique exact NCCL 2.30.7 pin")
        protected = pins(protected_text)
        for name, version in (("torch", TORCH_VERSION), (NCCL_NAME, NCCL_VERSION)):
            if not exact_pin(protected, name, version):
                errors.append(f"Protected pre-Omni snapshot lacks unique {name}=={version}")
    except Exception as exc:
        errors.append(f"Cannot parse NCCL provenance evidence: {exc}")
    return {"verified": not errors, "base_reference": base_reference,
            "override_file": "/etc/uv-overrides.txt",
            "overrides_sha256": hashlib.sha256(overrides_text.encode()).hexdigest(),
            "protected_before_omni_sha256": hashlib.sha256(protected_text.encode()).hexdigest(),
            "errors": errors, "scope": "One upstream Torch/NCCL metadata override; no ABI or GPU claim"}


def read_upstream_nccl_evidence(directory):
    try:
        return upstream_nccl_evidence(os.environ.get("DYNAMO_VLLM_BASE_IMAGE", ""),
            Path("/etc/uv-overrides.txt").read_text(encoding="utf-8"),
            (directory / "protected-before-omni.txt").read_text(encoding="utf-8"))
    except Exception as exc:
        return {"verified": False, "errors": [f"Cannot read NCCL provenance evidence: {exc}"]}


def resolved_path(path):
    return str(Path(path).resolve()).replace("\\", "/")


def metadata_path(dist):
    # importlib's actual .dist-info/.egg-info path, not just the package name.
    value = getattr(dist, "_path", None)
    return resolved_path(value) if value is not None else None


def import_origin(module):
    spec = importlib.util.find_spec(module)
    if spec is None or not spec.origin or spec.origin in {"built-in", "frozen"}:
        raise ValueError(f"Cannot resolve import origin for {module}")
    return resolved_path(spec.origin)


def select_distributions(distributions, report, effective_distribution, origin_resolver):
    """Deduplicate realpaths; verify the effective wheel over Ubuntu metadata.

    Ubuntu can expose both egg-info and dist-info for one system version.
    Metadata count alone does not tell us which package Python imports.
    """
    groups = {}
    for dist in distributions:
        try:
            name = dist.metadata["Name"]
            if not name:
                raise ValueError("missing Name")
            key = canonicalize_name(name, validate=True)
            record = {"name": name, "version": str(Version(dist.version)),
                      "metadata_path": metadata_path(dist), "distribution": dist,
                      "requirements": [Requirement(raw) for raw in dist.requires or []]}
            groups.setdefault(key, []).append(record)
        except Exception as exc:
            report["errors"].append(f"Invalid installed metadata: {exc}")
    records = {}
    for key, candidates in groups.items():
        unique = []
        for candidate in candidates:
            repeated = next((old for old in unique if candidate["metadata_path"] is not None
                             and old["metadata_path"] == candidate["metadata_path"]), None)
            if repeated is None:
                unique.append(candidate)
            elif (candidate["version"], candidate["requirements"], candidate["distribution"].metadata.get("Requires-Python")) != (
                    repeated["version"], repeated["requirements"], repeated["distribution"].metadata.get("Requires-Python")):
                report["errors"].append(f"Inconsistent metadata read at {candidate['metadata_path']}")
        chosen, import_file = unique[0], None
        if len(unique) > 1:
            try:
                if key not in SHADOWABLE_MODULES:
                    raise ValueError("not a reviewed Ubuntu system/wheel shadow")
                if not all(item["metadata_path"] for item in unique):
                    raise ValueError("metadata path unavailable")
                major_minor = ".".join(report["python"].split(".")[:2])
                wheel_root = f"/usr/local/lib/python{major_minor}/dist-packages"
                system_root = "/usr/lib/python3/dist-packages"
                wheel = [item for item in unique if item["metadata_path"].rsplit("/", 1)[0] == wheel_root]
                system = [item for item in unique if item["metadata_path"].rsplit("/", 1)[0] == system_root]
                if len(wheel) != 1 or not system or len(wheel) + len(system) != len(unique):
                    raise ValueError("unexpected metadata roots")
                if not wheel[0]["metadata_path"].endswith(".dist-info"):
                    raise ValueError("effective local metadata is not a wheel dist-info")
                # Permit one Ubuntu record or its egg-info/dist-info pair, all
                # describing the same inactive system version. Two wheels or
                # competing system versions still indicate an ambiguous install.
                system_kinds = [Path(item["metadata_path"]).suffix for item in system]
                if (len(system_kinds) != len(set(system_kinds))
                        or not set(system_kinds).issubset({".egg-info", ".dist-info"})
                        or len({item["version"] for item in system}) != 1):
                    raise ValueError("ambiguous Ubuntu system metadata versions or formats")
                selected, chosen = effective_distribution(key), wheel[0]
                if metadata_path(selected) != chosen["metadata_path"] or str(Version(selected.version)) != chosen["version"]:
                    raise ValueError("effective metadata is not the wheel distribution")
                import_file = origin_resolver(SHADOWABLE_MODULES[key])
                wheel_files = {resolved_path(chosen["distribution"].locate_file(file))
                               for file in chosen["distribution"].files or []}
                if not import_file.startswith(wheel_root + "/") or import_file not in wheel_files:
                    raise ValueError("actual import origin does not belong to the effective wheel")
                report["shadowed_distributions"].append({"name": key,
                    "effective_version": chosen["version"], "effective_metadata_path": chosen["metadata_path"],
                    "import_origin": import_file, "shadowed_version": system[0]["version"],
                    "shadowed_metadata_path": system[0]["metadata_path"] if len(system) == 1 else None,
                    "shadowed_metadata_paths": sorted(item["metadata_path"] for item in system),
                    "reason": "Verified Ubuntu system metadata shadowed by the imported local wheel"})
            except Exception as exc:
                report["errors"].append(f"Invalid installed metadata: duplicate installed distribution: {key}: {exc}")
        records[key] = chosen
        report["distribution_records"][key] = {"effective_version": chosen["version"],
            "effective_metadata_path": chosen["metadata_path"], "import_origin": import_file,
            "discovered": [{"version": item["version"], "metadata_path": item["metadata_path"]} for item in unique],
            "same_realpath_duplicates": len(candidates) - len(unique)}
    return records


def is_nccl_requirement(requirement, *, uv_diagnostic=False):
    markers = {'platform_system == "Linux"'}
    if uv_diagnostic:
        # uv normalizes the wheel's platform_system marker to sys_platform.
        markers.add('sys_platform == "linux"')
    return (canonicalize_name(requirement.name) == NCCL_NAME and str(requirement.specifier) == "==2.29.7"
            and not requirement.extras and requirement.url is None and str(requirement.marker) in markers)


def finalize_status(report):
    report["status"] = ("FAIL" if report["errors"] else "PASS_WITH_UPSTREAM_NCCL_OVERRIDE"
                        if report["known_metadata_exceptions"] else "PASS")


def audit_dependencies(distributions=None, *, python_version=None, environment=None,
                       nccl_evidence=None, effective_distribution=None, origin_resolver=None):
    python_version = python_version or platform.python_version()
    marker_environment = default_environment()
    marker_environment.update({"python_full_version": python_version,
                               "python_version": ".".join(python_version.split(".")[:2])})
    marker_environment.update(environment or {})
    evidence = nccl_evidence or {"verified": False, "errors": ["NCCL source evidence not supplied"]}
    report = {"python": python_version, "marker_environment": marker_environment, "status": "FAIL",
        "packages": {}, "distribution_records": {}, "shadowed_distributions": [], "active_extras": {},
        "frameworks": {}, "checked_requirements": 0, "known_metadata_exceptions": [], "errors": [],
        "upstream_nccl_evidence": evidence,
        "validation_scope": "Installed default requirements and recursively referenced extras; GPU/media E2E still required"}
    records = select_distributions(list(md.distributions() if distributions is None else distributions), report,
                                   effective_distribution or md.distribution, origin_resolver or import_origin)
    for key, record in records.items():
        report["packages"][record["name"]] = record["version"]
        record["metadata_errors"] = []
        try:
            requires_python = record["distribution"].metadata.get("Requires-Python")
            if requires_python and not SpecifierSet(requires_python).contains(python_version, prereleases=True):
                record["metadata_errors"].append(f"{record['name']}=={record['version']}: Requires-Python {requires_python}, running {python_version}")
        except Exception as exc:
            record["metadata_errors"].append(f"Invalid installed metadata: {key}: {exc}")

    def active(requirement, extras):
        return not requirement.marker or any(requirement.marker.evaluate({**marker_environment, "extra": extra})
                                              for extra in {"", *extras})

    def closure(roots):
        extras = {key: set() for key in roots if key in records}
        changed = True
        while changed:
            changed = False
            for key in list(extras):
                for requirement in records[key]["requirements"]:
                    if not active(requirement, extras[key]):
                        continue
                    target = canonicalize_name(requirement.name)
                    if target not in records:
                        continue
                    if target not in extras:
                        extras[target], changed = set(), True
                    previous = len(extras[target])
                    extras[target].update(canonicalize_name(extra) for extra in requirement.extras)
                    changed |= len(extras[target]) != previous
        return extras

    def check_edge(key, requirement):
        record, target = records[key], canonicalize_name(requirement.name)
        if target not in records:
            return "error", f"{record['name']}=={record['version']}: missing {requirement}"
        installed = records[target]["version"]
        if requirement.url:
            return "error", f"{record['name']}: cannot verify direct URL requirement {requirement} from version metadata alone"
        if requirement.specifier.contains(installed, prereleases=True):
            return "pass", None
        detail = f"{record['name']}=={record['version']}: requires {requirement}, installed {installed}"
        if (key == "torch" and record["version"] == TORCH_VERSION and installed == NCCL_VERSION
                and is_nccl_requirement(requirement) and marker_environment.get("platform_system") == "Linux"
                and marker_environment.get("sys_platform") == "linux" and marker_environment.get("platform_machine") == "x86_64"
                and records.get("vllm", {}).get("version") == "0.30.0" and evidence.get("verified") is True):
            return "override", detail
        return "error", detail

    def check_closure(extras):
        result = {"errors": [], "known_metadata_exceptions": [], "checked_requirements": 0,
                  "packages": sorted(extras), "active_extras": {k: sorted(v) for k, v in sorted(extras.items()) if v}}
        for key in extras:
            result["errors"].extend(records[key]["metadata_errors"])
            for requirement in records[key]["requirements"]:
                if active(requirement, extras[key]):
                    result["checked_requirements"] += 1
                    kind, detail = check_edge(key, requirement)
                    if kind != "pass":
                        result["known_metadata_exceptions" if kind == "override" else "errors"].append(detail)
        finalize_status(result)
        return result

    try:
        global_result = check_closure(closure(records))
        for field in ("errors", "known_metadata_exceptions"):
            report[field].extend(global_result[field])
        for field in ("checked_requirements", "active_extras"):
            report[field] = global_result[field]
        for project in ("vllm", "vllm-omni"):
            if project not in records:
                report["frameworks"][project] = {"status": "MISSING"}
                continue
            direct = {"requirements": [], "errors": [], "known_metadata_exceptions": []}
            for requirement in records[project]["requirements"]:
                if active(requirement, set()):
                    direct["requirements"].append(str(requirement))
                    kind, detail = check_edge(project, requirement)
                    # No direct vLLM or Omni default requirement can use a waiver.
                    if kind != "pass":
                        direct["errors"].append(detail)
            finalize_status(direct)
            tree = check_closure(closure([project]))
            report["frameworks"][project] = {"version": records[project]["version"], "status": tree["status"],
                "default_requirements": direct, "recursive_dependency_closure": tree}
    except Exception as exc:
        report["errors"].append(f"Dependency marker evaluation failed: {exc}")
    report["packages"] = dict(sorted(report["packages"].items()))
    report["errors"] = sorted(set(report["errors"]))
    report["known_metadata_exceptions"] = sorted(set(report["known_metadata_exceptions"]))
    finalize_status(report)
    return report


def validate_framework_dependencies():
    """Compatibility entry point uses the same audit as the image gate."""
    report = audit_dependencies(nccl_evidence=read_upstream_nccl_evidence(BUILD_INFO_DIRECTORY))
    for project, result in report["frameworks"].items():
        if result["status"] == "MISSING":
            report["errors"].append(f"Required framework missing: {project}")
    if report["errors"]:
        raise RuntimeError("Framework dependency check failed:\n" + "\n".join(report["errors"]))


def review_uv_check(report, exit_code, output):
    """A uv failure must be exactly the proven upstream NCCL mismatch."""
    report["uv_pip_check_exit_code"] = exit_code
    if exit_code not in (0, 1):
        report["errors"].append(f"uv pip check failed as a tool (exit {exit_code})")
    elif exit_code == 0 and report["known_metadata_exceptions"]:
        report["errors"].append("uv reports clean metadata but the audit found an NCCL override; verify both checks use the same interpreter")
    elif exit_code == 1:
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        meaningful = [line for line in lines if not re.fullmatch(r"Checked \d+ packages? in .+", line)
                      and not re.fullmatch(r"Using Python .+ environment at: .+", line)]
        valid = False
        if (len(report["known_metadata_exceptions"]) == 1 and report["upstream_nccl_evidence"].get("verified") is True
                and len(meaningful) == 2 and meaningful[0] == "Found 1 incompatibility"):
            match = re.fullmatch(r"The package `torch` requires `([^`]+)`, but `2\.30\.7` is installed", meaningful[1])
            try:
                valid = bool(match and is_nccl_requirement(Requirement(match[1]), uv_diagnostic=True))
            except Exception:
                valid = False
        if not valid:
            report["errors"].append("uv pip check failure is not exactly the verified upstream Torch/NCCL diagnostic; review uv-pip-check.txt")
    finalize_status(report)


def main():
    directory = BUILD_INFO_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    report = audit_dependencies(nccl_evidence=read_upstream_nccl_evidence(directory))
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
    (directory / "dependency-audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # Failed Docker RUN layers cannot export build-info. Put actionable paths,
    # versions and source validation errors in the log, never arbitrary env.
    summary = {
        "status": report["status"],
        "frameworks": {
            name: {"status": scope["status"],
                   "default_requirements": scope.get("default_requirements", {}).get("status", "MISSING"),
                   "recursive_dependency_closure": scope.get("recursive_dependency_closure", {}).get("status", "MISSING")}
            for name, scope in report["frameworks"].items()
        },
        "upstream_nccl_evidence": {
            "verified": report["upstream_nccl_evidence"].get("verified", False),
            "errors": report["upstream_nccl_evidence"].get("errors", []),
        },
        "duplicate_distribution_paths": {
            name: record for name, record in report["distribution_records"].items()
            if len(record["discovered"]) > 1 or record["same_realpath_duplicates"]
        },
        "shadowed_distributions": report["shadowed_distributions"],
        "known_metadata_exceptions": report["known_metadata_exceptions"],
        "errors": report["errors"],
    }
    print("Runtime dependency audit summary:\n" + json.dumps(summary, indent=2), flush=True)
    if report["errors"]:
        raise RuntimeError("Runtime dependency audit failed:\n" + "\n".join(report["errors"]))
    packages, omni, hashes = report["packages"], md.distribution("vllm-omni"), {}
    for name in ("vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py",
                 "vllm_omni/diffusion/models/minimax_h3/vae.py",
                 "vllm_omni/diffusion/models/minimax_h3/quality_policy.py",
                 "vllm_omni/diffusion/models/minimax_h3/time_request.py"):
        source = Path(omni.locate_file(name))
        if not source.is_file():
            raise RuntimeError(f"Required H3 source missing: {name}")
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = {"python": platform.python_version(), "machine": platform.machine(), "packages": packages,
                "dependency_audit_status": report["status"], "known_metadata_exceptions": report["known_metadata_exceptions"],
                "frameworks": report["frameworks"], "upstream_nccl_evidence": report["upstream_nccl_evidence"],
                "h3_source_sha256": hashes, "validation_scope": "CPU build inventory; GPU/media E2E still required"}
    (directory / "runtime-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (directory / "installed-versions.txt").write_text("".join(f"{name}=={version}\n" for name, version in packages.items()), encoding="utf-8")
    print("Runtime inventory, separate vLLM/Omni dependency closures and H3 source hashes recorded")


if __name__ == "__main__":
    main()
