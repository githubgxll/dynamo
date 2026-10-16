"""Strict CPU-side dependency audit for the explicitly reviewed H3 overlay."""
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

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

import dependency_audit as audit
from runtime_dependencies import snapshot, verify_protected

HERE = Path(__file__).resolve().parent


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_policy():
    return json.loads((HERE / "gateway-dependency-policy.json").read_text(encoding="utf-8"))


def reference_evidence(policy, *, gateway_sha):
    errors = []
    if gateway_sha != policy["gateway_sha"]:
        errors.append("DYNAMO_GATEWAY_SOURCE_SHA does not match the reviewed gateway commit")
    hashes = {}
    for name, expected in policy["gateway_reference_sha256"].items():
        path = HERE / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        hashes[name] = actual
        if actual != expected:
            errors.append(f"Pinned gateway reference differs: {name}")
    return {"verified": not errors, "gateway_sha": gateway_sha,
            "reference_sha256": hashes, "errors": errors,
            "scope": "Pinned colleague service source and requirement evidence; no GPU or service E2E claim"}


def check_override_file(text):
    values = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            values.append(str(Requirement(line)).lower())
    if values != ["nvidia-nccl-cu13==2.30.7"]:
        raise RuntimeError("Inherited /etc/uv-overrides.txt is not the unique reviewed NCCL override")


def prepare(policy, directory):
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("This environment installer targets Linux x86_64 only")
    # Core identities must already be furnished by the pinned gateway runtime.
    names = ("ai-dingo", "ai-dingo-runtime", "gpu-memory-service", "vllm", "pynvvideocodec")
    versions = {name: md.version(name) for name in names}
    for name, actual in versions.items():
        if actual != policy["required_versions"][name]:
            raise RuntimeError(f"Wrong base: {name}=={actual}")
    overrides = Path("/etc/uv-overrides.txt").read_text(encoding="utf-8")
    check_override_file(overrides)
    evidence = audit.upstream_nccl_evidence(
        os.environ.get("DYNAMO_VLLM_BASE_IMAGE", ""), overrides,
        f"torch=={md.version('torch')}\nnvidia-nccl-cu13=={md.version('nvidia-nccl-cu13')}\n")
    gateway = reference_evidence(policy, gateway_sha=os.environ.get("DYNAMO_GATEWAY_SOURCE_SHA", ""))
    if not evidence["verified"] or not gateway["verified"]:
        raise RuntimeError(json.dumps({"nccl": evidence, "gateway": gateway}, indent=2))
    (directory / "h3-reviewed-overrides.txt").write_text(
        "nvidia-nccl-cu13==2.30.7\n", encoding="utf-8")
    (directory / "h3-shared-constraints.txt").write_text(
        "\n".join(policy["solver_constraints"]) + "\n", encoding="utf-8")
    (directory / "h3-backend-helpers.txt").write_text(
        "\n".join(policy["additional_backend_requirements"]) + "\n", encoding="utf-8")
    write_json(directory / "environment-input-policy.json", policy)
    write_json(directory / "environment-input-evidence.json", {"gateway": gateway, "nccl": evidence})


def cv2_provider_evidence(policy):
    """Check the real imported module and all hashed cv2 files against headless RECORD."""
    required = policy["required_versions"]
    versions = {name: md.version(name) for name in ("opencv-python", "opencv-python-headless")}
    for name, version in versions.items():
        if version != required[name]:
            raise RuntimeError(f"OpenCV distribution identity drift: {name}=={version}")
    dist = md.distribution("opencv-python-headless")
    owned = Path(dist.locate_file("cv2/__init__.py")).resolve()
    spec = importlib.util.find_spec("cv2")
    if spec is None or not spec.origin or Path(spec.origin).resolve() != owned:
        raise RuntimeError("Actual cv2 import is not the headless provider's package")
    hashes = {}
    for record in dist.files or []:
        if not str(record).replace("\\", "/").startswith("cv2/") or record.hash is None:
            continue
        if record.hash.mode != "sha256":
            raise RuntimeError(f"Unexpected cv2 RECORD hash algorithm: {record}")
        path = Path(dist.locate_file(record))
        actual = hashlib.sha256(path.read_bytes()).digest()
        encoded = base64.urlsafe_b64encode(actual).rstrip(b"=").decode("ascii")
        if encoded != record.hash.value:
            raise RuntimeError(f"Headless cv2 file no longer matches its RECORD: {record}")
        hashes[str(record)] = actual.hex()
    if "cv2/__init__.py" not in hashes or not any(name.endswith(".so") for name in hashes):
        raise RuntimeError("Headless RECORD does not cover the cv2 package and native extension")
    import cv2
    if cv2.__version__ != "4.13.0":
        raise RuntimeError(f"cv2 native version differs: {cv2.__version__}")
    return {"status": "PASS_SAME_VERSION_SHARED_NAMESPACE", "distributions": versions,
            "import_origin": str(owned), "native_version": cv2.__version__,
            "provider": "opencv-python-headless", "verified_cv2_file_sha256": hashes,
            "note": "Two upstream required distributions share cv2; last-installed headless is verified. Never uninstall one alone at runtime."}


def check_versions(report, policy):
    packages = {canonicalize_name(name): version for name, version in report["packages"].items()}
    for name, version in policy["required_versions"].items():
        if packages.get(name) != version:
            report["errors"].append(f"Required profile version mismatch: {name}: expected {version}, installed {packages.get(name)}")
    for name in policy["disabled_optional_components"]:
        if name in packages:
            report["errors"].append(f"Experimental profile must not contain optional {name}")
    audit.finalize_status(report)


def wheel_origin_evidence(policy):
    """The version label alone cannot prove the released wheel bytes were used."""
    dist = md.distribution("vllm-omni")
    raw = dist.read_text("direct_url.json")
    if not raw:
        raise RuntimeError("Omni is missing direct_url.json for the checksum-pinned released wheel")
    origin = json.loads(raw)
    url, expected_hash = policy["omni_wheel"].split("#sha256=", 1)
    archive = origin.get("archive_info", {})
    actual_hash = archive.get("hashes", {}).get("sha256")
    if actual_hash is None and archive.get("hash", "").startswith("sha256="):
        actual_hash = archive["hash"].split("=", 1)[1]
    if origin.get("url") != url or actual_hash != expected_hash:
        raise RuntimeError("Omni direct_url evidence does not match the pinned released wheel and SHA256")
    return {"url": url, "sha256": actual_hash,
            "scope": "Installer direct-url receipt for exact released wheel; Ref8 source patch is checked separately"}


def check_backend_helpers(report, policy):
    packages = {canonicalize_name(name): version for name, version in report["packages"].items()}
    for text in policy["additional_backend_requirements"]:
        requirement = Requirement(text)
        version = packages.get(canonicalize_name(requirement.name))
        if version is None or not requirement.specifier.contains(version, prereleases=True):
            report["errors"].append(f"Gateway backend helper requirement is unsatisfied: {text}, installed {version}")
    audit.finalize_status(report)


def write_license_inventory(directory, inputs, inventory):
    before = json.loads((inputs / "runtime-before-omni.json").read_text(encoding="utf-8"))
    def by_name(data):
        result = {}
        for item in data["distributions"]:
            result.setdefault(item["canonical_name"], set()).add(item["version"])
        return result
    initial, final = by_name(before), by_name(inventory)
    delta = [{"name": name, "before": sorted(initial.get(name, set())),
              "after": sorted(final.get(name, set()))}
             for name in sorted(set(initial) | set(final)) if initial.get(name) != final.get(name)]
    records = [{key: item.get(key) for key in
                ("name", "version", "metadata_path", "license_expression", "declared_license")}
               for item in inventory["distributions"]]
    bundled = {}
    for package in ("av", "opencv-python", "opencv-python-headless", "soxr", "torchcodec", "pynvvideocodec"):
        dist = md.distribution(package)
        bundled[package] = [str(path) for path in dist.files or []
                            if ".so" in str(path) or "license" in str(path).lower()
                            or "copying" in str(path).lower()]
    write_json(directory / "dependency-license-inventory.json", {
        "status": "METADATA_INVENTORY_ONLY_NOT_LEGAL_CLEARANCE", "packages": records,
        "python_dependency_delta_from_inherited_vllm_plus_dingo": delta, "media_bundled_files": bundled,
        "scope": "Declared Python licenses and package delta only; does not establish bundled codec or OS license compliance. Framework policy validation is a separate build gate."})
    result = subprocess.run(["dpkg-query", "-W", "-f=${Package}\t${Version}\n"],
                            capture_output=True, text=True, check=True)
    (directory / "dpkg-packages.txt").write_text(result.stdout, encoding="utf-8")
    copyright_file = Path("/usr/share/doc/ffmpeg/copyright")
    if copyright_file.is_file():
        (directory / "ffmpeg-copyright.txt").write_bytes(copyright_file.read_bytes())


def run_audit(policy, directory, inputs):
    protected_text = (inputs / "protected-before-omni.txt").read_text(encoding="utf-8")
    nccl = audit.upstream_nccl_evidence(os.environ.get("DYNAMO_VLLM_BASE_IMAGE", ""),
        Path("/etc/uv-overrides.txt").read_text(encoding="utf-8"), protected_text)
    gateway = reference_evidence(policy, gateway_sha=os.environ.get("DYNAMO_GATEWAY_SOURCE_SHA", ""))
    report = audit.audit_dependencies(nccl_evidence=nccl, root_extras=policy["enabled_root_extras"])
    if not nccl["verified"] or not gateway["verified"]:
        report["errors"].extend(nccl["errors"] + gateway["errors"])
    audit.require_framework_versions(report)
    check_versions(report, policy)
    check_backend_helpers(report, policy)
    try:
        verify_protected(snapshot(phase="final"), protected_text.splitlines())
        write_json(directory / "omni-wheel-origin.json", wheel_origin_evidence(policy))
        opencv = cv2_provider_evidence(policy)
    except Exception as exc:
        report["errors"].append(str(exc))
        opencv = {"status": "FAIL", "error": str(exc)}
    result = subprocess.run(["uv", "pip", "check", "--python", sys.executable],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env={**os.environ, "NO_COLOR": "1"}, check=False)
    (directory / "uv-pip-check.txt").write_text(result.stdout, encoding="utf-8")
    (directory / "uv-pip-check.exit-code").write_text(str(result.returncode) + "\n", encoding="utf-8")
    audit.review_uv_check(report, result.returncode, result.stdout)
    write_json(directory / "opencv-provider.json", opencv)
    write_json(directory / "dependency-audit.json", report)
    final_inventory = snapshot(phase="final")
    write_json(directory / "installed-environment.json", final_inventory)
    try:
        write_license_inventory(directory, inputs, final_inventory)
    except Exception as exc:
        report["errors"].append(f"Cannot record complete dependency/media inventory: {exc}")
        audit.finalize_status(report)
        write_json(directory / "dependency-audit.json", report)
    status = "FAIL" if report["errors"] else "PASS_EXPERIMENTAL_ENVIRONMENT_CPU_ONLY"
    summary = {"status": status, "gateway_sha": policy["gateway_sha"],
        "required_versions": policy["required_versions"],
        "dependency_scope": policy["dependency_scope"],
        "additional_backend_requirements": policy["additional_backend_requirements"],
        "dependency_status": report["status"],
        "known_metadata_exceptions": report["known_metadata_exceptions"],
        "opencv_provider_status": opencv["status"], "errors": report["errors"],
        "euid": os.geteuid(), "scope": "Installed dependencies and import provenance only; media smoke and GPU acceptance are separate gates"}
    write_json(directory / "environment-check.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if report["errors"]:
        raise RuntimeError("H3 experimental dependency audit failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    inputs = Path(os.environ.get("H3_BUILD_INFO", "/opt/dynamo/build-info"))
    directory = args.output_dir or inputs
    directory.mkdir(parents=True, exist_ok=True)
    policy = read_policy()
    if args.prepare_only:
        prepare(policy, directory)
    else:
        run_audit(policy, directory, inputs)


if __name__ == "__main__":
    main()
