# SPDX-License-Identifier: Apache-2.0
"""Preserve every installed package and existing repair while applying SR deltas."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
INFO = HERE / "build-info"
PARENT_DIR = Path("/opt/h3-hsdp-offload")
PARENT = "registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm@sha256:45b3ec519faec85ac4a40ff8766e033704bc92623f97fda16610efdc85e71e0f"
PARENT_COMMIT = "430c62be0e38179535b4a6a1aa38dc67e8367d00"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parent_runtime():
    name = "_sr_opt_parent_runtime"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, PARENT_DIR / "runtime.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def contract(*, require_default_observations=False):
    parent = parent_runtime()
    c14, lite, lock = parent.contract()
    source = read(HERE / "inputs/source-lock.json")
    if source["parent_image"] != PARENT or source["parent_source_commit"] != PARENT_COMMIT:
        raise ValueError("Wrong selected parent")
    if os.environ.get("H3_HSDP_SOURCE_COMMIT") != PARENT_COMMIT or os.environ.get("SEEDVR2_OPT_PARENT_IMAGE") != PARENT:
        raise ValueError("Unexpected inherited/derived recipe identity")
    if not re.fullmatch(r"[a-f0-9]{40}", os.environ.get("SEEDVR2_OPT_SOURCE_COMMIT", "")):
        raise ValueError("Missing source commit")
    if require_default_observations and (os.environ.get("SEEDVR2_C14_PROFILE") or os.environ.get("SEEDVR2_METRICS_DIR")):
        raise ValueError("Image cannot enable experimental observations by default")
    return parent, c14, lite, lock


def snapshot():
    parent = parent_runtime()
    value = parent.snapshot_environment()
    # Include the entire HSDP implementation and receipts, not just C14/lite.
    value["inherited_hsdp_files_sha256"] = {
        p.relative_to(PARENT_DIR).as_posix(): digest(p.read_bytes())
        for p in PARENT_DIR.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")}
    return value


def verify_installed():
    parent, c14, lite, _ = contract()
    from source_patches import CHANGES, ADDITIONS, expected_sources
    expected = expected_sources(parent.source_expected())
    dist, root = lite.installed_root("vllm-omni")
    actual = {p.relative_to(root).as_posix(): digest(lite.safe_file(root, p.relative_to(root).as_posix()).read_bytes())
              for p in (root / "vllm_omni").rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")}
    lite.compare_manifest(actual, expected, "Entire Omni parent plus exact SR delta")
    record = c14.check_record(lite, dist, actual)
    receipt = read(INFO / "application.json")
    if digest(record.read_bytes()) != receipt["omni_record_after_sha256"] or digest((INFO / "omni-RECORD.before.csv").read_bytes()) != receipt["omni_record_before_sha256"]:
        raise ValueError("Omni RECORD proof changed")
    # This changes neither gateway sources nor its generated service identity.
    _, dingo_root = lite.installed_root("ai-dingo")
    gateway = lite.gateway_sources(dingo_root, read(parent.LITE / "gateway.lock.json"), patched=True)
    if gateway["generated_version_sha256"] != read(parent.C14 / "build-info/source-before.json")["gateway"]["generated_version_sha256"]:
        raise ValueError("Gateway source identity changed")
    return {"status": "PASS", "parent_image": PARENT, "source_sha256": actual,
            "source_manifest_sha256": digest(json.dumps(actual, sort_keys=True, separators=(",", ":")).encode()),
            "changed_sources": sorted(CHANGES), "added_sources": sorted(ADDITIONS),
            "rmsnorm_preserved_sha256": actual[c14.NADIT], "gateway_commit": c14.GATEWAY}


def apply(output):
    if os.geteuid() != 0 or output.resolve() != INFO.resolve() or INFO.exists():
        raise ValueError("Apply is a create-once root build operation")
    parent, c14, lite, _ = contract(require_default_observations=True)
    parent_proof = parent.verify_installed()
    parent_report = read(PARENT_DIR / "build-info/runtime-validation.json")
    if parent_report.get("status") != "PASS_CPU_ONLY" or parent_report.get("source_commit") != PARENT_COMMIT:
        raise ValueError("Parent lacks its source-bound CPU gate")
    baseline = snapshot()
    from source_patches import CHANGES, ADDITIONS, expected_sources, locked_changes
    expected_sources(parent.source_expected())
    dist, root = lite.installed_root("vllm-omni")
    changes = locked_changes(root)
    record = lite.distribution_record(dist)
    before_record = record.read_bytes()
    after_record = before_record
    additions = {}
    for name, (before, after) in changes.items():
        after_record = lite.update_record(after_record, name, before, after)
    for name, local in ADDITIONS.items():
        if (root / name).exists():
            raise ValueError("Cannot overwrite existing helper")
        body = (HERE / local).read_bytes()
        compile(body, name, "exec")
        additions[name] = body
        after_record = c14.add_record(lite, after_record, name, body)
    output.mkdir()
    dump(output / "parent-snapshot.json", baseline)
    dump(output / "source-before.json", parent_proof)
    (output / "omni-RECORD.before.csv").write_bytes(before_record)
    for name, (before, after) in changes.items():
        target = output / "parent-sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(before)
        lite.atomic_write(lite.safe_file(root, name), after)
        for cached in (root / name).parent.joinpath("__pycache__").glob(Path(name).stem + ".*.pyc"):
            cached.unlink()
    for name, body in additions.items():
        with (root / name).open("xb") as stream:
            stream.write(body)
        (root / name).chmod(0o644)
    lite.atomic_write(record, after_record)
    receipt = {"schema": "seedvr2-optimization-application/1", "status": "PASS", "parent_image": PARENT,
               "source_commit": os.environ["SEEDVR2_OPT_SOURCE_COMMIT"], "changed_sources": sorted(CHANGES), "added_sources": sorted(ADDITIONS),
               "omni_record_before_sha256": digest(before_record), "omni_record_after_sha256": digest(after_record), "gpu_validation_status": "NOT_RUN"}
    dump(output / "application.json", receipt)
    dump(output / "source-verification.json", verify_installed())
    c14.assert_snapshot(baseline, snapshot(), output, "after-apply")
    return receipt


def validate(output):
    if os.geteuid() != 1000:
        raise ValueError("Validate final installed runtime as UID1000")
    parent, c14, lite, _ = contract(require_default_observations=True)
    output.mkdir(parents=True, exist_ok=True)
    baseline = read(INFO / "parent-snapshot.json")
    c14.assert_snapshot(baseline, snapshot(), output, "before-validation")
    proof = verify_installed()
    dump(output / "source-verification.json", proof)
    _, root = lite.installed_root("vllm-omni")
    for relative in proof["changed_sources"] + proof["added_sources"] + [parent.TARGET, parent.HELPER, parent.PIPELINE]:
        module = importlib.import_module(relative[:-3].replace("/", "."))
        if Path(module.__file__).resolve() != (root / relative).resolve():
            raise ValueError("Shadowed installed module: " + relative)
    native = c14.load_module("_sr_opt_torch_contract", PARENT_DIR / "torch_contract.py")
    dump(output / "torch-contract.json", native.verify())
    norm = c14.rmsnorm(output / "rmsnorm", lite, proof)
    dump(output / "rmsnorm/runtime-validation.json", norm)
    for name in ("audit_environment.py", "media_smoke.py", "service_smoke.py"):
        lite.run_script(parent.LITE / "environment" / name, ["--output-dir", str(output / "environment")], output)
    for name, script in (("c14", parent.C14 / "cpu_smoke.py"), ("hsdp", PARENT_DIR / "cpu_smoke.py")):
        logs = output / name
        logs.mkdir(parents=True, exist_ok=True)
        lite.run_script(script, ["--output-dir", str(logs)], logs)
    lite.run_script(HERE / "cpu_smoke.py", ["--output-dir", str(output)], output)
    regression = read(output / "regression.json")
    if regression.get("status") != "PASS" or regression.get("tests_run", 0) < 1 or any(regression.get(k) != 0 for k in ("failures", "errors", "skipped")):
        raise ValueError("Actual image CPU regression gate incomplete")
    c14.assert_snapshot(baseline, snapshot(), output, "after-validation")
    if verify_installed() != proof:
        raise ValueError("Validation changed installed sources")
    dump(output / "preservation.json", {"status": "PASS", "packages": len(baseline["packages"]), "dpkg": len(baseline["dpkg"]),
         "scope": "All package METADATA/direct_url, dpkg, inherited legal/build evidence and entire HSDP implementation unchanged"})
    result = {"schema": "seedvr2-optimization-runtime/1", "status": "PASS_CPU_ONLY", "euid": 1000,
              "parent_image": PARENT, "source_commit": os.environ["SEEDVR2_OPT_SOURCE_COMMIT"], "gpu_validation_status": "NOT_RUN",
              "source_manifest_sha256": proof["source_manifest_sha256"], "retained_packages_unchanged": True,
              "metrics_schema": "sr-timing-memory/1", "scope": "Installed imports, preservation, numerical/CPU/media regressions; SP8/timing perturbation/quality NOT_RUN"}
    dump(output / "runtime-validation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("apply", "validate"))
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(apply(args.output_dir) if args.action == "apply" else validate(args.output_dir), indent=2))
    except Exception as error:
        dump(args.output_dir / "failure.json", {"status": "FAIL", "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
