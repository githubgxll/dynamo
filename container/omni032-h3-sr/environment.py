"""Release-paired Omni 0.32 dependency upgrade; no old-version CLI entrypoints."""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata as md
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

HERE = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def legacy_module(name):
    choices = (HERE / "legacy-environment", HERE.parent / "h3-experimental" / "environment")
    folder = next((path for path in choices if (path / (name + ".py")).is_file()), None)
    if folder is None:
        raise RuntimeError(f"Missing read-only legacy pure helper: {name}")
    spec = importlib.util.spec_from_file_location("omni032_legacy_" + name, folder / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dependency_auditor(target=False):
    module = legacy_module("dependency_audit")
    if target:
        # Reuse the generic parser/closure/duplicate-origin logic, not the
        # inherited CLI nor its old framework version requirements. The base
        # provenance remains the actual inherited 0.31 image, deliberately.
        module.VLLM_VERSION = module.OMNI_VERSION = "0.32.0"
        module.EXPECTED_FRAMEWORK_VERSIONS = {"vllm": "0.32.0", "vllm-omni": "0.32.0"}
    return module


def pip_records(inventory, python_minor="3.12"):
    """Never turn Ubuntu apt distributions into PyPI installation roots."""
    prefix = f"/usr/local/lib/python{python_minor}/dist-packages"
    records = {}
    for item in inventory["distributions"]:
        if item["location"].rstrip("/") != prefix:
            continue
        name = canonicalize_name(item["name"])
        if name in records and records[name] != item:
            raise ValueError(f"Multiple effective pip distributions: {name}")
        records[name] = item
    if not records:
        raise RuntimeError("No inherited Python 3.12 pip installation found")
    return records


def native_names(records, policy):
    result = set(policy["native_names"]) & records.keys()
    result.update(name for name in records if any(name.startswith(prefix) for prefix in policy["native_prefixes"]))
    return sorted(name for name in result if not any(name.startswith(prefix) for prefix in policy["mutable_native_prefixes"]))


def orphan_plan(records, policy):
    result = {}
    for name in policy["reviewed_orphans"]:
        if name not in records:
            continue
        consumers = sorted({owner for owner, row in records.items() for raw in row["requires_dist"]
                            if canonicalize_name(Requirement(raw).name) == name})
        result[name] = {"version": records[name]["version"], "consumers": consumers,
                        "action": "retain-required" if consumers else "remove-orphan"}
    return result


def solve_plan(inventory, policy):
    records = pip_records(inventory, policy["python_minor"])
    orphans = orphan_plan(records, policy)
    omitted = {name for name, item in orphans.items() if item["action"] == "remove-orphan"}
    roots = set(records) - omitted
    roots.update(policy["target_versions"])
    for name, extras in policy["root_extras"].items():
        roots.discard(name)
        roots.add(name + "[" + ",".join(extras) + "]")
    # Freeze compatible inherited versions, not their old mutable dependency
    # edges. Only the reviewed release changes below replace these pins. This
    # makes metadata-only resolution match the installed-preferred Linux solve
    # and prevents an unrelated API/tooling upgrade during a framework repair.
    constraints = {name: row["version"] for name, row in records.items() if name not in omitted}
    constraints.update(policy["preserved_versions"])
    constraints.update(policy["target_versions"])
    return {"roots": sorted(roots), "constraints": [f"{name}=={version}" for name, version in sorted(constraints.items())],
            "protected_names": native_names(records, policy), "retained_pip_names": sorted(set(records) - omitted),
            "inherited_telemetry": sorted(name for name in records if name.startswith("opentelemetry-") and name not in omitted),
            "reviewed_orphans": orphans, "omitted_orphans": sorted(omitted),
            "scope": "All effective inherited pip capabilities plus release targets; unpinned roots replace old mutable Requires-Dist edges. Ubuntu system metadata is audited, never reinstalled from PyPI."}


def protected_file_manifest(names, distributions=None):
    """Record actual native and service bytes, not unchanged version labels."""
    lookup = md.distribution if distributions is None else distributions.__getitem__
    result = {}
    for name in names:
        dist = lookup(name)
        files = {}
        for entry in dist.files or []:
            value = str(entry).replace("\\", "/")
            filename = Path(value).name
            service = name in {"ai-dingo", "ai-dingo-runtime", "gpu-memory-service"}
            native = ".so" in filename or filename.endswith((".pyd", ".dll", ".dylib", ".a"))
            metadata = filename in {"METADATA", "WHEEL", "RECORD", "direct_url.json"}
            if not (native or metadata or (service and not value.endswith((".pyc", ".pyo")))):
                continue
            path = Path(dist.locate_file(entry))
            if not path.is_file():
                raise RuntimeError(f"Protected installed file is missing: {name}: {path}")
            files[str(path)] = {"sha256": sha256(path), "symlink": os.readlink(path) if path.is_symlink() else None}
        service_root = {"ai-dingo": "dingo", "ai-dingo-runtime": "dynamo", "gpu-memory-service": "gpu_memory_service"}.get(name)
        if service_root:
            root = Path(dist.locate_file(service_root))
            if not root.is_dir():
                raise RuntimeError(f"Protected service source tree missing: {root}")
            for path in root.rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}:
                    files[str(path)] = {"sha256": sha256(path), "symlink": os.readlink(path) if path.is_symlink() else None}
        if not files:
            raise RuntimeError(f"Protected distribution has no verifiable files: {name}")
        result[name] = {"version": dist.version, "files": files}
    return result


def nccl_evidence(policy, before, auditor):
    records = pip_records(before, policy["python_minor"])
    protected = "".join(f"{name}=={records[name]['version']}\n" for name in ("torch", "nvidia-nccl-cu13"))
    actual_reference = os.environ.get("DYNAMO_VLLM_BASE_IMAGE", "")
    if actual_reference != policy["inherited_vllm_base"]:
        raise RuntimeError("Inherited vLLM base reference differs from the source-pinned parent")
    override = Path("/etc/uv-overrides.txt").read_text(encoding="utf-8")
    evidence = auditor.upstream_nccl_evidence(actual_reference, override, protected)
    if not evidence["verified"]:
        raise RuntimeError("Inherited Torch/NCCL provenance failed: " + json.dumps(evidence))
    evidence["inherited_from_parent"] = policy["parent_image"]
    evidence["override_text"] = override
    evidence["scope"] = "Same protected inherited Torch/NCCL bytes and original upstream metadata override; not a claim that a new 0.32 base was inherited or that GPU execution passed."
    return evidence


def require_versions(packages, versions):
    normalized = {canonicalize_name(name): version for name, version in packages.items()}
    errors = [f"{name}: expected {version}, installed {normalized.get(name)}" for name, version in versions.items()
              if normalized.get(name) != version]
    if errors:
        raise RuntimeError("Exact environment identity mismatch: " + "; ".join(errors))


def prepare(output, policy):
    if platform.system() != "Linux" or platform.machine() != "x86_64" or platform.python_version_tuple()[:2] != tuple(policy["python_minor"].split(".")):
        raise RuntimeError("Upgrade targets the inherited Linux amd64 Python 3.12 runtime only")
    if (output / "parent-inventory.json").exists():
        raise RuntimeError("Refusing to overwrite frozen parent evidence")
    before = legacy_module("runtime_dependencies").snapshot(phase="omni032-before")
    records = pip_records(before, policy["python_minor"])
    require_versions({name: item["version"] for name, item in records.items()}, policy["parent_versions"])
    auditor = dependency_auditor()
    evidence = nccl_evidence(policy, before, auditor)
    report = auditor.audit_dependencies(nccl_evidence=evidence, root_extras=policy["root_extras"])
    auditor.require_framework_versions(report)
    write_json(output / "parent-runtime-audit.json", report)
    if report["errors"]:
        raise RuntimeError("Parent dependency audit failed: " + "; ".join(report["errors"]))
    plan = solve_plan(before, policy)
    native = protected_file_manifest(plan["protected_names"])
    write_json(output / "environment-input-policy.json", policy)
    write_json(output / "solve-plan.json", plan)
    write_json(output / "nccl-provenance.json", evidence)
    # Keep frozen inputs distinct from audit receipts when output == inputs.
    write_json(output / "parent-native-manifest.json", native)
    (output / "roots.in").write_text("\n".join(plan["roots"]) + "\n", encoding="utf-8")
    (output / "constraints.txt").write_text("\n".join(plan["constraints"]) + "\n", encoding="utf-8")
    (output / "reviewed-overrides.txt").write_text("nvidia-nccl-cu13==2.30.7\n", encoding="utf-8")
    # This file is the preparation-complete marker. Never let a failed native
    # snapshot cause the installer to consume partially prepared inputs.
    write_json(output / "parent-inventory.json", before)
    return {"status": "PARENT_AND_SOLVE_INPUTS_PREPARED", "roots": len(plan["roots"]), "protected_distributions": len(plan["protected_names"])}


def wheel_evidence(inputs, artifact, *, dist=None):
    path = inputs / "wheels" / artifact["filename"]
    if path.is_symlink() or path.parent.is_symlink() or not path.is_file() or path.resolve() != path:
        raise RuntimeError(f"Retained release wheel is not a regular local artifact: {path}")
    if path.stat().st_size != artifact["size_bytes"] or sha256(path) != artifact["sha256"]:
        raise RuntimeError(f"Retained release wheel content differs: {path.name}")
    if dist is not None:
        origin = json.loads(dist.read_text("direct_url.json") or "{}")
        if origin.get("url") != path.as_uri():
            raise RuntimeError(f"Actual installed wheel origin differs: {artifact['filename']}")
        hashes = origin.get("archive_info", {}).get("hashes", {})
        direct_hash = origin.get("archive_info", {}).get("hash", "")
        if hashes.get("sha256", artifact["sha256"]) != artifact["sha256"] or (direct_hash and direct_hash != "sha256=" + artifact["sha256"]):
            raise RuntimeError("Installer supplied a conflicting wheel SHA256")
    return {"wheel_path": str(path), "wheel_uri": path.as_uri(), "sha256": artifact["sha256"], "size_bytes": artifact["size_bytes"], "url": artifact["url"]}


def download(output, policy):
    wheels = output / "wheels"
    wheels.mkdir(exist_ok=True)
    receipts = {}
    for name, artifact in policy["wheel_artifacts"].items():
        path = wheels / artifact["filename"]
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".partial")
            for attempt in range(1, 5):
                try:
                    request = urllib.request.Request(artifact["url"], headers={"User-Agent": "H3-Omni032-release-wheel/1"})
                    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as target:
                        for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
                            target.write(block)
                    if temporary.stat().st_size != artifact["size_bytes"] or sha256(temporary) != artifact["sha256"]:
                        raise RuntimeError("Downloaded release wheel failed size/SHA256 verification")
                    temporary.replace(path)
                    break
                except (OSError, urllib.error.URLError) as error:
                    print(f"Release wheel {name}: attempt {attempt}/4 failed: {type(error).__name__}", flush=True)
                    if attempt == 4:
                        raise
                    time.sleep(min(attempt * 2, 6))
            else:
                raise RuntimeError("Release wheel download did not complete")
        receipts[name] = wheel_evidence(output, artifact)
    write_json(output / "download-receipts.json", receipts)
    return {"status": "RELEASE_WHEELS_SHA256_VERIFIED", "artifacts": receipts}


def reconcile(output, inputs, policy):
    inventory = legacy_module("runtime_dependencies").snapshot(phase="omni032-reconcile")
    records = pip_records(inventory, policy["python_minor"])
    plan = orphan_plan(records, policy)
    write_json(output / "orphan-reconciliation.json", plan)
    for name, item in plan.items():
        if item["action"] == "remove-orphan":
            subprocess.run(["uv", "pip", "uninstall", "--python", sys.executable, name], check=True)
    return {"status": "REVIEWED_INTERNAL_ORPHANS_RECONCILED", "plan": plan}


def verify_provider(policy):
    versions = {name: md.version(name) for name in ("opencv-python", "opencv-python-headless")}
    require_versions(versions, {name: policy["preserved_versions"][name] for name in versions})
    dist = md.distribution("opencv-python-headless")
    origin = importlib.util.find_spec("cv2")
    owned = Path(dist.locate_file("cv2/__init__.py")).resolve()
    if origin is None or not origin.origin or Path(origin.origin).resolve() != owned:
        raise RuntimeError("cv2 import origin is not the verified headless provider")
    hashes = {}
    for entry in dist.files or []:
        name = str(entry).replace("\\", "/")
        if not name.startswith("cv2/") or entry.hash is None:
            continue
        if entry.hash.mode != "sha256":
            raise RuntimeError("Unexpected cv2 RECORD algorithm")
        actual = sha256(dist.locate_file(entry))
        if base64.urlsafe_b64encode(bytes.fromhex(actual)).rstrip(b"=").decode() != entry.hash.value:
            raise RuntimeError(f"Actual cv2 bytes differ from headless RECORD: {entry}")
        hashes[name] = actual
    if "cv2/__init__.py" not in hashes or not any(".so" in name for name in hashes):
        raise RuntimeError("Incomplete headless cv2 RECORD")
    return {"status": "PASS_SAME_VERSION_SHARED_NAMESPACE", "versions": versions, "origin": str(owned), "files_sha256": hashes}


def core_record_evidence(dist=None, origin=None):
    """The upgraded compiled core must still match its installed wheel RECORD."""
    dist = md.distribution("vllm") if dist is None else dist
    origin = importlib.util.find_spec("vllm") if origin is None else origin
    owned = Path(dist.locate_file("vllm/__init__.py")).resolve()
    if origin is None or not origin.origin or Path(origin.origin).resolve() != owned:
        raise RuntimeError("Actual vLLM import does not belong to the released core wheel")
    files = {}
    for entry in dist.files or []:
        if entry.hash is None:
            continue
        if entry.hash.mode != "sha256":
            raise RuntimeError("Unexpected vLLM RECORD algorithm")
        actual = sha256(dist.locate_file(entry))
        if base64.urlsafe_b64encode(bytes.fromhex(actual)).rstrip(b"=").decode() != entry.hash.value:
            raise RuntimeError(f"Released vLLM installed file differs from RECORD: {entry}")
        files[str(entry)] = actual
    if "vllm/__init__.py" not in files or not any(".so" in name for name in files):
        raise RuntimeError("Released vLLM RECORD lacks package or compiled-core evidence")
    return {"status": "PASS", "origin": str(owned), "files_sha256": files}


def helper_artifact_evidence():
    """Bind changed helpers to immutable inputs and verify installed RECORDs."""
    evidence = {}
    for raw in (HERE / "requirements.lock").read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        requirement = Requirement(raw)
        expected_url, expected_hash = requirement.url.rsplit("#sha256=", 1)
        dist = md.distribution(requirement.name)
        origin = json.loads(dist.read_text("direct_url.json") or "{}")
        if origin.get("url") != expected_url:
            raise RuntimeError(f"Changed helper origin differs from its immutable release input: {requirement.name}")
        archive = origin.get("archive_info", {})
        declared = archive.get("hashes", {}).get("sha256")
        old_hash = archive.get("hash")
        if (declared and declared != expected_hash) or (old_hash and old_hash != "sha256=" + expected_hash):
            raise RuntimeError(f"Changed helper installer hash conflicts: {requirement.name}")
        checked = 0
        for entry in dist.files or []:
            if entry.hash is None:
                continue
            if entry.hash.mode != "sha256":
                raise RuntimeError(f"Unexpected helper RECORD algorithm: {requirement.name}")
            actual = sha256(dist.locate_file(entry))
            if base64.urlsafe_b64encode(bytes.fromhex(actual)).rstrip(b"=").decode() != entry.hash.value:
                raise RuntimeError(f"Changed helper installed file differs from RECORD: {requirement.name}: {entry}")
            checked += 1
        if not checked:
            raise RuntimeError(f"Changed helper has no verifiable RECORD files: {requirement.name}")
        evidence[canonicalize_name(requirement.name)] = {"version": dist.version, "url": expected_url, "input_sha256": expected_hash,
            "installer_sha256_receipt_present": bool(declared or old_hash), "verified_record_files": checked,
            "scope": "Hash-fragment enforced immutable install input, exact installed origin and actual RECORD content; helper wheel archive is not independently retained."}
    return evidence


def run_uv_check(output, auditor, report):
    with tempfile.TemporaryDirectory(prefix="omni032-uv-") as cache:
        # Check installed metadata only. Inherited UV_CONFIG_FILE/UV_PYTHON,
        # cache and index settings must not select another environment or
        # access a root-owned path from the final UID1000 validation step.
        env = {key: value for key, value in os.environ.items() if not key.upper().startswith("UV_")}
        command = ["uv", "--no-config", "--no-cache", "--offline", "--no-python-downloads",
                   "pip", "check", "--python", sys.executable]
        try:
            result = subprocess.run(command, env=env, cwd=cache, timeout=120,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            result = subprocess.CompletedProcess(command, 2, f"uv metadata check failed as a tool: {type(error).__name__}\n")
    (output / "uv-pip-check.txt").write_text(result.stdout, encoding="utf-8")
    (output / "uv-pip-check.exit-code").write_text(str(result.returncode) + "\n", encoding="utf-8")
    write_json(output / "uv-pip-check-context.json", {"python": sys.executable, "euid": os.geteuid(), "command": command,
        "cache": "disabled_private_temporary_workdir", "environment_policy": "Child-only UV_* removal; no config/cache/network/Python downloads", "exit_code": result.returncode})
    print(f"=== uv-pip-check.txt (exit={result.returncode}) ===\n{result.stdout}=== end uv-pip-check.txt ===", flush=True)
    auditor.review_uv_check(report, result.returncode, result.stdout)


def audit(output, inputs, policy):
    if platform.system() != "Linux" or platform.machine() != "x86_64" or platform.python_version_tuple()[:2] != tuple(policy["python_minor"].split(".")):
        raise RuntimeError("Installed-image audit must run in the actual Linux amd64 Python 3.12 runtime")
    before = read_json(inputs / "parent-inventory.json")
    if read_json(inputs / "environment-input-policy.json") != policy:
        raise RuntimeError("Frozen environment policy differs from the active recipe")
    inventory = legacy_module("runtime_dependencies").snapshot(phase="omni032-after")
    write_json(output / "installed-environment.json", inventory)
    auditor = dependency_auditor(target=True)
    evidence = nccl_evidence(policy, before, auditor)
    if evidence != read_json(inputs / "nccl-provenance.json"):
        raise RuntimeError("Inherited NCCL provenance changed during framework upgrade")
    report = auditor.audit_dependencies(nccl_evidence=evidence, root_extras=policy["root_extras"])
    auditor.require_framework_versions(report)
    packages = {canonicalize_name(name): version for name, version in report["packages"].items()}
    required = {**policy["target_versions"], **policy["preserved_versions"]}
    independent = {}
    checks = {
        "required-versions": lambda: require_versions(packages, required),
        "native-preservation": lambda: compare_native(inputs),
        "opencv-provider": lambda: verify_provider(policy),
        "release-core-record": core_record_evidence,
        "release-helper-artifacts": helper_artifact_evidence,
        "release-wheel-origins": lambda: {name: wheel_evidence(inputs, artifact, dist=md.distribution(name)) for name, artifact in policy["wheel_artifacts"].items()},
    }
    for name, check in checks.items():
        try:
            value = check()
            independent[name] = {"status": "PASS", "evidence": value}
        except Exception as error:
            independent[name] = {"status": "FAIL", "error": str(error)}
            report["errors"].append(f"{name}: {error}")
        write_json(output / (name + ".json"), independent[name])
    plan = read_json(inputs / "solve-plan.json")
    for name in plan["retained_pip_names"]:
        if name not in packages:
            report["errors"].append(f"Inherited runtime capability disappeared: {name}")
    for name in policy["forbidden_components"]:
        if name in packages:
            report["errors"].append(f"Unexpected disabled optional component installed: {name}")
    run_uv_check(output, auditor, report)
    auditor.finalize_status(report)
    write_json(output / "dependency-audit.json", report)
    delta = dependency_delta(before, inventory)
    write_json(output / "dependency-delta.json", delta)
    summary = {"status": report["status"], "euid": os.geteuid(), "frameworks": {name: {"version": item.get("version"), "status": item["status"]} for name, item in report["frameworks"].items()},
               "checks": {name: value["status"] for name, value in independent.items()}, "known_metadata_exceptions": report["known_metadata_exceptions"],
               "preserved_monitoring_names": plan["inherited_telemetry"], "package_delta": delta, "errors": report["errors"], "scope": "Actual installed dependency/default/extras/origin/native checks only; GPU and generated media acceptance remain separate."}
    write_json(output / "environment-check.json", summary)
    if report["errors"]:
        print(json.dumps(summary, indent=2), flush=True)
        raise RuntimeError("Omni 0.32 installed environment audit failed")
    return summary


def compare_native(inputs):
    before = read_json(inputs / "parent-native-manifest.json")
    if not isinstance(before, dict) or not before or any(
            not isinstance(item, dict) or not isinstance(item.get("version"), str)
            or not isinstance(item.get("files"), dict) or not item["files"] for item in before.values()):
        raise RuntimeError("Invalid parent native snapshot; expected original package/file manifests, not an audit receipt")
    current = protected_file_manifest(sorted(before))
    changed = [name for name in before if before[name] != current[name]]
    if changed:
        raise RuntimeError("Inherited native/service files or versions changed: " + ", ".join(changed))
    return {"protected_distributions": len(before), "protected_file_count": sum(len(item["files"]) for item in before.values())}


def dependency_delta(before, after):
    def versions(inventory):
        return {name: item["version"] for name, item in pip_records(inventory).items()}
    initial, final = versions(before), versions(after)
    return [{"name": name, "before": initial.get(name), "after": final.get(name)} for name in sorted(set(initial) | set(final)) if initial.get(name) != final.get(name)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "download", "reconcile", "audit"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs = (args.inputs or args.output_dir).resolve()
    policy = read_json(HERE / "environment-policy.json")
    if args.action == "prepare":
        result = prepare(output, policy)
    elif args.action == "download":
        result = download(output, policy)
    elif args.action == "reconcile":
        result = reconcile(output, inputs, policy)
    else:
        result = audit(output, inputs, policy)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
