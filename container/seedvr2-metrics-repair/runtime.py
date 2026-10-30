# SPDX-License-Identifier: Apache-2.0
"""Repair observations while preserving all prior runtime and media behavior."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
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
PARENT_DIR = Path("/opt/seedvr2-optimization")
PARENT = "registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm@sha256:9547a5e924383dcaba4cdb8cc4fc9c5b5b3b80c24a5290edbd64e4cf96992d69"
PARENT_COMMIT = "bd3f34bd37d6b82c52a5af0c2a7aa2349285a0bc"
PARENT_MANIFEST = "387886a6f99ac514aa6977419100e28c9b5ad18061a0225ddf91c7de3bdf5ceb"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def parent_runtime():
    return load("_sr_metrics_parent_runtime", PARENT_DIR / "runtime.py")


def contract(*, require_default_observations=False):
    parent = parent_runtime()
    hsdp, c14, lite, _ = parent.contract(require_default_observations=require_default_observations)
    source = read(HERE / "inputs/source-lock.json")
    if source["parent_image"] != PARENT or source["parent_source_commit"] != PARENT_COMMIT or source["parent_source_manifest_sha256"] != PARENT_MANIFEST:
        raise ValueError("Wrong immutable repair parent")
    if os.environ.get("SEEDVR2_OPT_SOURCE_COMMIT") != PARENT_COMMIT or os.environ.get("SEEDVR2_METRICS_REPAIR_PARENT_IMAGE") != PARENT:
        raise ValueError("Unexpected inherited/repair image identity")
    if not re.fullmatch(r"[0-9a-f]{40}", os.environ.get("SEEDVR2_METRICS_REPAIR_SOURCE_COMMIT", "")):
        raise ValueError("Missing metrics repair source commit")
    return parent, hsdp, c14, lite


def parent_sources():
    value = read(HERE / "inputs/parent-source-manifest.json")
    files = value["source_sha256"]
    if value.get("parent_image") != PARENT or value.get("source_commit") != PARENT_COMMIT or value.get("source_manifest_sha256") != PARENT_MANIFEST or digest(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()) != PARENT_MANIFEST:
        raise ValueError("Archived complete parent source manifest differs")
    return files


@contextmanager
def inherited_source_verifier():
    # The historical verifier deliberately uses its own source_patches module.
    # Bind only for this create-once parent validation, then restore the child.
    inherited = load("_sr_metrics_parent_source_patches", PARENT_DIR / "source_patches.py")
    old = sys.modules.get("source_patches")
    try:
        sys.modules["source_patches"] = inherited
        yield
    finally:
        if old is None:
            sys.modules.pop("source_patches", None)
        else:
            sys.modules["source_patches"] = old


def snapshot():
    value = parent_runtime().snapshot()
    value["inherited_sr_optimization_files_sha256"] = {
        p.relative_to(PARENT_DIR).as_posix(): digest(p.read_bytes()) for p in PARENT_DIR.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")}
    return value


def verify_installed():
    parent, hsdp, c14, lite = contract()
    from source_patches import CHANGES, expected_sources
    expected = expected_sources(parent_sources())
    dist, root = lite.installed_root("vllm-omni")
    actual = {p.relative_to(root).as_posix(): digest(lite.safe_file(root, p.relative_to(root).as_posix()).read_bytes())
              for p in (root / "vllm_omni").rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")}
    lite.compare_manifest(actual, expected, "Entire 9547 parent plus exact metrics repair")
    record = c14.check_record(lite, dist, actual)
    receipt = read(INFO / "application.json")
    if digest(record.read_bytes()) != receipt["omni_record_after_sha256"] or digest((INFO / "omni-RECORD.before.csv").read_bytes()) != receipt["omni_record_before_sha256"]:
        raise ValueError("Repair RECORD/application identity differs")
    _, dingo_root = lite.installed_root("ai-dingo")
    gateway = lite.gateway_sources(dingo_root, read(hsdp.LITE / "gateway.lock.json"), patched=True)
    if gateway["generated_version_sha256"] != read(hsdp.C14 / "build-info/source-before.json")["gateway"]["generated_version_sha256"]:
        raise ValueError("Inherited Gateway identity changed")
    return {"status": "PASS", "parent_image": PARENT, "source_sha256": actual,
        "source_manifest_sha256": digest(json.dumps(actual, sort_keys=True, separators=(",", ":")).encode()),
        "source_install_root": str(root), "changed_sources": sorted(CHANGES), "added_sources": [],
        "rmsnorm_preserved_sha256": actual[c14.NADIT], "gateway_commit": c14.GATEWAY}


def apply(output):
    if os.geteuid() != 0 or output.resolve() != INFO.resolve() or INFO.exists():
        raise ValueError("Create-once root build operation required")
    parent, hsdp, c14, lite = contract(require_default_observations=True)
    with inherited_source_verifier():
        before_proof = parent.verify_installed()
    if before_proof["source_sha256"] != parent_sources():
        raise ValueError("Actual parent differs from archived successful manifest")
    report = read(PARENT_DIR / "build-info/runtime-validation.json")
    if report.get("status") != "PASS_CPU_ONLY" or report.get("source_commit") != PARENT_COMMIT or report.get("source_manifest_sha256") != PARENT_MANIFEST:
        raise ValueError("Missing source-bound successful parent runtime gate")
    baseline = snapshot()
    from source_patches import locked_changes
    dist, root = lite.installed_root("vllm-omni")
    changes = locked_changes(root)
    record = lite.distribution_record(dist)
    before_record = record.read_bytes()
    after_record = before_record
    for name, (before, after) in changes.items():
        after_record = lite.update_record(after_record, name, before, after)
    output.mkdir()
    dump(output / "parent-snapshot.json", baseline)
    dump(output / "source-before.json", before_proof)
    (output / "omni-RECORD.before.csv").write_bytes(before_record)
    for name, (before, after) in changes.items():
        archived = output / "parent-sources" / name
        archived.parent.mkdir(parents=True, exist_ok=True)
        archived.write_bytes(before)
        lite.atomic_write(lite.safe_file(root, name), after)
        for cached in (root / name).parent.joinpath("__pycache__").glob(Path(name).stem + ".*.pyc"):
            cached.unlink()
    lite.atomic_write(record, after_record)
    result = {"schema": "seedvr2-metrics-repair-application/1", "status": "PASS", "parent_image": PARENT,
        "source_commit": os.environ["SEEDVR2_METRICS_REPAIR_SOURCE_COMMIT"], "changed_sources": sorted(changes), "added_sources": [],
        "omni_record_before_sha256": digest(before_record), "omni_record_after_sha256": digest(after_record), "gpu_validation_status": "NOT_RUN"}
    dump(output / "application.json", result)
    dump(output / "source-verification.json", verify_installed())
    c14.assert_snapshot(baseline, snapshot(), output, "after-apply")
    return result


def validate(output):
    if os.geteuid() != 1000:
        raise ValueError("Final installed validation must run as UID1000")
    parent, hsdp, c14, lite = contract(require_default_observations=True)
    output.mkdir(parents=True, exist_ok=True)
    baseline = read(INFO / "parent-snapshot.json")
    c14.assert_snapshot(baseline, snapshot(), output, "before-validation")
    proof = verify_installed()
    dump(output / "source-verification.json", proof)
    _, root = lite.installed_root("vllm-omni")
    for relative in proof["changed_sources"] + [hsdp.TARGET, hsdp.HELPER, hsdp.PIPELINE]:
        module = importlib.import_module(relative[:-3].replace("/", "."))
        if Path(module.__file__).resolve() != (root / relative).resolve():
            raise ValueError("Installed repair module shadowed: " + relative)
    native = c14.load_module("_sr_metrics_torch_contract", hsdp.HERE / "torch_contract.py")
    dump(output / "torch-contract.json", native.verify())
    norm = c14.rmsnorm(output / "rmsnorm", lite, proof)
    dump(output / "rmsnorm/runtime-validation.json", norm)
    for name in ("audit_environment.py", "media_smoke.py", "service_smoke.py"):
        lite.run_script(hsdp.LITE / "environment" / name, ["--output-dir", str(output / "environment")], output)
    for name, script in (("c14", hsdp.C14 / "cpu_smoke.py"), ("hsdp", hsdp.HERE / "cpu_smoke.py"),
                         ("media", HERE / "inherited_media_smoke.py")):
        logs = output / name
        logs.mkdir(parents=True, exist_ok=True)
        lite.run_script(script, ["--output-dir", str(logs)], logs)
    lite.run_script(HERE / "cpu_smoke.py", ["--output-dir", str(output)], output)
    regression = read(output / "regression.json")
    if regression.get("status") != "PASS" or regression.get("tests_run", 0) < 1 or any(regression.get(k) != 0 for k in ("failures", "errors", "skipped")):
        raise ValueError("Required CPU repair regression gate incomplete")
    c14.assert_snapshot(baseline, snapshot(), output, "after-validation")
    if verify_installed() != proof:
        raise ValueError("Validation changed installed sources")
    dump(output / "preservation.json", {"status": "PASS", "packages": len(baseline["packages"]), "dpkg": len(baseline["dpkg"]),
        "scope": "All METADATA/direct_url, OS packages, legal, H3/C14/optimization code and old release evidence preserved"})
    result = {"schema": "seedvr2-metrics-repair-runtime/1", "status": "PASS_CPU_ONLY", "euid": 1000,
        "parent_image": PARENT, "source_commit": os.environ["SEEDVR2_METRICS_REPAIR_SOURCE_COMMIT"],
        "source_manifest_sha256": proof["source_manifest_sha256"], "retained_packages_unchanged": True,
        "metrics_schema": "sr-timing-memory/2", "gpu_validation_status": "NOT_RUN",
        "scope": "Actual CPU tensor, installed modules, media and synthetic CUDA-control regressions; actual CUDA/SP8 metrics revalidation required"}
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
