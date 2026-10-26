# SPDX-License-Identifier: Apache-2.0
"""Apply/verify exact C14 source delta over the published Omni/Gateway lite image."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
INFO = HERE / "build-info"
LITE = Path("/opt/omni-gateway-lite")
PARENT = "registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm@sha256:9382949d2789b23cdfe4ea71d68c7973592de28f209082bd7907c1d09fe008c9"
PARENT_COMMIT = "2362a3241c03a286b98e51b8003a51f494fab4f1"
GATEWAY = "9ccee2f56d0122be94d7436ae2f5fec96d15ea9b"
HELPER = "vllm_omni/diffusion/models/seedvr2/c14_capture.py"
NADIT = "vllm_omni/diffusion/models/seedvr2/nadit.py"
NORM_SHA = "03fd3a0c199305ce3004643f8d1510183b06731099a72bc447c2bd589c1a6f01"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def lite_runtime():
    return load_module("_c14_inherited_lite_runtime", LITE / "runtime.py")


def contract(lite, lock):
    lite.validate_contract(lock, before=False)
    if os.environ.get("H3_C14_PARENT_IMAGE") != PARENT or os.environ.get("H3_EXPERIMENTAL_SOURCE_COMMIT") != PARENT_COMMIT:
        raise ValueError("C14 requires the exact successful lite parent and original source marker")
    if not re.fullmatch(r"[0-9a-f]{40}", os.environ.get("H3_C14_SOURCE_COMMIT", "")):
        raise ValueError("Missing C14 recipe commit")
    source = read(HERE / "inputs/source-lock.json")
    if source["parent_image"] != PARENT or source["parent_source_commit"] != PARENT_COMMIT or source["gateway_commit"] != GATEWAY:
        raise ValueError("C14 source locks disagree")


def preservation_lock(lock):
    return {**lock, "preserved_trees": [*lock["preserved_trees"], str(LITE)]}


def json_probe(script, *arguments):
    """Read JSON in a fresh interpreter with the same user/startup environment."""
    result = subprocess.run([sys.executable, "-B", str(script), *arguments],
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding="utf-8", errors="strict", timeout=120)
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise ValueError("Snapshot probe did not return an object")
    return value


def snapshot_environment():
    # Native imports (notably setuptools via Torch) can append bundled vendor
    # metadata to sys.path. Inventory the unchanged installation in a clean
    # process, just as the inherited parent validator's subprocess gates do.
    # No package is filtered; real metadata/direct_url/OS/file changes still fail.
    return json_probe(HERE / "runtime.py", "snapshot")


def snapshot_difference(expected, actual):
    sections = {}
    for key in sorted(set(expected) | set(actual)):
        before, after = expected.get(key), actual.get(key)
        if before == after and (key in expected) == (key in actual):
            continue
        if key == "packages" and isinstance(before, list) and isinstance(after, list):
            def identities(rows):
                return Counter(json.dumps({"name": row["name"], "version": row["version"],
                    "metadata_sha256": digest(json.dumps(row.get("metadata")).encode()),
                    "direct_url_sha256": digest(json.dumps(row.get("direct_url")).encode())}, sort_keys=True)
                    for row in rows)
            old, new = identities(before), identities(after)
            sections[key] = {"added": [json.loads(item) for item in sorted((new - old).elements())],
                "removed": [json.loads(item) for item in sorted((old - new).elements())]}
        elif isinstance(before, dict) and isinstance(after, dict):
            sections[key] = {"added": sorted(set(after) - set(before)),
                "removed": sorted(set(before) - set(after)),
                "changed": sorted(name for name in before.keys() & after.keys() if before[name] != after[name])}
        elif isinstance(before, list) and isinstance(after, list):
            old, new = Counter(before), Counter(after)
            sections[key] = {"added": sorted((new - old).elements()), "removed": sorted((old - new).elements())}
        else:
            sections[key] = {"changed": True}
    return sections


def assert_snapshot(expected, actual, output, phase):
    if actual == expected:
        return
    changes = snapshot_difference(expected, actual)
    report = {"status": "FAIL", "phase": phase, "changes": changes,
        "inspection": "Fresh interpreter; complete inherited inventory, no package exceptions"}
    dump(output / ("environment-diff-" + phase + ".json"), report)
    # Log names/counts, never raw package metadata or credential-bearing URLs.
    summary = {key: {kind: len(value) if isinstance(value, list) else value for kind, value in delta.items()}
               for key, delta in changes.items()}
    preview = {key: {kind: [(item["name"] + "==" + item["version"] if isinstance(item, dict) else item)
                            for item in value[:10]] if isinstance(value, list) else value
                     for kind, value in delta.items()} for key, delta in changes.items()}
    print(json.dumps({"status": "FAIL_ENVIRONMENT_PRESERVATION", "phase": phase,
        "sections": summary, "first_changed_items": preview}), flush=True)
    raise ValueError("Inherited environment changed; review environment-diff-" + phase + ".json")


def source_expected(lite, lock):
    wheel = Path("/opt/dynamo/build-info/omni-wheels") / lock["omni_wheel_filename"]
    if digest(wheel.read_bytes()) != lock["omni_wheel_sha256"]:
        raise ValueError("Original Omni wheel changed")
    with zipfile.ZipFile(wheel) as archive:
        names = [x.filename for x in archive.infolist() if not x.is_dir() and x.filename.startswith("vllm_omni/")]
        if not names or len(names) != len(set(names)):
            raise ValueError("Invalid original wheel package")
        expected = {name: digest(archive.read(name)) for name in names}
    expected[NADIT] = NORM_SHA
    patches = read(HERE / "source-patches.lock.json")
    for relative, hashes in patches["files"].items():
        if expected.get(relative) != hashes["before_sha256"]:
            raise ValueError("C14 patch does not derive from the pinned installed parent: " + relative)
        expected[relative] = hashes["after_sha256"]
    if HELPER in expected:
        raise ValueError("New capture helper collides with parent package")
    helper = (HERE / "c14_capture.py").read_bytes()
    if digest(helper) != patches["helper_sha256"]:
        raise ValueError("Capture helper changed since source lock")
    expected[HELPER] = digest(helper)
    return expected


def check_record(lite, dist, expected):
    record = lite.distribution_record(dist)
    rows = list(csv.reader(io.StringIO(record.read_text(encoding="utf-8"), newline="")))
    mapping = {}
    for row in rows:
        if len(row) != 3 or row[0] in mapping:
            raise ValueError("Duplicate or invalid installed RECORD")
        mapping[row[0]] = row
    root = Path(dist.locate_file("")).resolve()
    for name in expected:
        body = lite.safe_file(root, name).read_bytes()
        if mapping.get(name, [None])[1:] != [lite.record_digest(body), str(len(body))]:
            raise ValueError("Installed RECORD differs from exact source: " + name)
    return record


def verify_installed():
    """Read-only identity gate also used by the SR-only CLI before a run."""
    lite = lite_runtime()
    lock, gateway = read(LITE / "runtime.lock.json"), read(LITE / "gateway.lock.json")
    lite.validate_inputs(lock, gateway)
    contract(lite, lock)
    expected = source_expected(lite, lock)
    dist, root = lite.installed_root("vllm-omni")
    actual = {p.relative_to(root).as_posix(): digest(lite.safe_file(root, p.relative_to(root).as_posix()).read_bytes())
        for p in (root / "vllm_omni").rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")}
    lite.compare_manifest(actual, expected, "Complete Omni + exact C14 delta")
    record = check_record(lite, dist, actual)
    _, dingo_root = lite.installed_root("ai-dingo")
    dingo = lite.gateway_sources(dingo_root, gateway, patched=True)
    baseline = read(INFO / "application.json")
    if baseline.get("status") != "PASS" or digest(record.read_bytes()) != baseline["omni_record_after_sha256"]:
        raise ValueError("C14 application receipt/RECORD differs")
    if digest((INFO / "omni-RECORD.before.csv").read_bytes()) != baseline["omni_record_before_sha256"]:
        raise ValueError("Parent RECORD provenance changed")
    if dingo["generated_version_sha256"] != read(INFO / "source-before.json")["gateway"]["generated_version_sha256"]:
        raise ValueError("Generated Dingo wheel versions changed")
    return {"status": "PASS", "parent_image": PARENT, "gateway_commit": GATEWAY,
        "seedvr2_source_sha256": actual, "package_file_count": len(actual),
        "source_manifest_sha256": digest(json.dumps(actual, sort_keys=True, separators=(",", ":")).encode()),
        "changed_sources": sorted(read(HERE / "source-patches.lock.json")["files"]), "added_sources": [HELPER],
        "rmsnorm_preserved_sha256": actual[NADIT], "gateway_source_count": dingo["tracked_python_count"]}


def add_record(lite, data, name, body):
    rows = list(csv.reader(io.StringIO(data.decode("utf-8"), newline="")))
    if any(row and row[0] == name for row in rows):
        raise ValueError("Capture helper already belongs to installed RECORD")
    rows.append([name, lite.record_digest(body), str(len(body))])
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(rows)
    return output.getvalue().encode("utf-8")


def apply(output):
    if os.geteuid() != 0 or output.resolve() != INFO.resolve() or INFO.exists():
        raise ValueError("Apply is a create-once root Docker layer action")
    if os.environ.get("SEEDVR2_C14_PROFILE"):
        raise ValueError("Image must not embed an active diagnostic profile")
    lite = lite_runtime()
    lock, gateway = read(LITE / "runtime.lock.json"), read(LITE / "gateway.lock.json")
    lite.validate_inputs(lock, gateway)
    contract(lite, lock)
    before_sources = lite.source_proofs(lock, gateway, patched=True)
    snapshot = snapshot_environment()
    dist, root = lite.installed_root("vllm-omni")
    record = lite.distribution_record(dist)
    record_before = record.read_bytes()
    from source_patches import locked_changes
    changes = locked_changes(root)
    expected = source_expected(lite, lock)
    helper = (HERE / "c14_capture.py").read_bytes()
    compile(helper, HELPER, "exec")
    if (root / HELPER).exists():
        raise ValueError("Capture helper already exists")
    record_after = record_before
    for relative, before, after in changes:
        record_after = lite.update_record(record_after, relative, before, after)
    record_after = add_record(lite, record_after, HELPER, helper)
    output.mkdir()
    dump(output / "parent-snapshot.json", snapshot)
    dump(output / "source-before.json", before_sources)
    (output / "omni-RECORD.before.csv").write_bytes(record_before)
    for relative, before, after in changes:
        target = lite.safe_file(root, relative)
        lite.atomic_write(target, after)
        for cached in (target.parent / "__pycache__").glob(target.stem + ".*.pyc"):
            cached.unlink()
    with (root / HELPER).open("xb") as stream:
        stream.write(helper)
    (root / HELPER).chmod(0o644)
    lite.atomic_write(record, record_after)
    receipt = {"schema": "seedvr2-c14-application/1", "status": "PASS", "parent_image": PARENT,
        "source_commit": os.environ["H3_C14_SOURCE_COMMIT"], "omni_record_before_sha256": digest(record_before),
        "omni_record_after_sha256": digest(record_after), "changed_sources": [x[0] for x in changes],
        "added_sources": [HELPER], "scope": "Exact default-off instrumentation; C14 quality is not yet determined"}
    dump(output / "application.json", receipt)
    proof = verify_installed()
    assert_snapshot(snapshot, snapshot_environment(), output, "after-apply")
    dump(output / "source-verification.json", proof)
    return receipt


def rmsnorm(output, lite, proof):
    # Old validator binds its pre-C14 RECORD and whole source tree. Preserve that
    # receipt intact; reuse its unchanged numerical cases after the new complete
    # source/RECORD check above instead of weakening the old validator.
    sys.path.insert(0, str(LITE / "rmsnorm"))
    try:
        numerical = load_module("_c14_rmsnorm_numeric", LITE / "rmsnorm/validate_runtime.py")
        norm = importlib.import_module("vllm_omni.diffusion.models.seedvr2.nadit")
        _, root = lite.installed_root("vllm-omni")
        actual_path = Path(norm.__file__).resolve()
        if actual_path != (root / NADIT).resolve() or digest(actual_path.read_bytes()) != NORM_SHA:
            raise ValueError("Actual installed RMSNorm import is shadowed or unpatched")
        import torch
        cases, all_300 = numerical.numeric_cases(norm.RMSNorm, torch, "cpu")
        output.mkdir(parents=True, exist_ok=True)
        numerical.run_upstream_tests(output, numerical.lock_data())
        result = {"schema": "seedvr2-c14-rmsnorm/1", "status": "PASS", "device": "cpu", "euid": os.geteuid(),
            "module_path": str(actual_path), "module_sha256": NORM_SHA,
            "upstream_commit": "0708f05e2e27f16fca8b48e7c5a085277d1cafda", "six_case_count": len(cases),
            "six_cases": cases, "all_300": all_300, "upstream_pytest_case_count": 6, "inventory_unchanged": True,
            "parent_image": PARENT, "source_commit": os.environ["H3_C14_SOURCE_COMMIT"],
            "source_manifest_sha256": proof["source_manifest_sha256"]}
        dump(output / "runtime-validation.json", result)
        return result
    finally:
        sys.path.pop(0)


def validate(output):
    if os.geteuid() != 1000:
        raise ValueError("Run final image validation as inherited UID1000")
    if os.environ.get("SEEDVR2_C14_PROFILE"):
        raise ValueError("Default image cannot enable diagnostics")
    output.mkdir(parents=True, exist_ok=True)
    lite = lite_runtime()
    lock = read(LITE / "runtime.lock.json")
    baseline = read(INFO / "parent-snapshot.json")
    assert_snapshot(baseline, snapshot_environment(), output, "before-validation")
    search_path_before = list(sys.path)
    proof = verify_installed()
    dump(output / "source-verification.json", proof)
    # Actual installed imports; static extraction of functions is not accepted here.
    _, root = lite.installed_root("vllm-omni")
    for relative in [*read(HERE / "source-patches.lock.json")["files"], HELPER]:
        module = importlib.import_module(relative[:-3].replace("/", "."))
        if Path(module.__file__).resolve() != (root / relative).resolve():
            raise ValueError("Patched module import is shadowed: " + relative)
    norm = rmsnorm(output / "rmsnorm", lite, proof)
    gateway = importlib.import_module("dingo.video_gateway.artifact_store")
    regression = load_module("_c14_gateway_regression", LITE / "gateway_regression.py")
    dump(output / "gateway-regression.json", regression.check_capacity(gateway))
    for name in ("audit_environment.py", "media_smoke.py", "service_smoke.py"):
        lite.run_script(LITE / "environment" / name, ["--output-dir", str(output / "environment")], output)
    # The image has Torch/safetensors. Require those fixtures here; local Windows
    # tests deliberately cannot stand in for these actual installed CPU checks.
    lite.run_script(HERE / "cpu_smoke.py", ["--output-dir", str(output / "diagnostics")], output)
    diagnostics = read(output / "diagnostics/diagnostics-regression.json")
    if diagnostics.get("status") != "PASS" or diagnostics.get("tests_run", 0) < 3 or any(diagnostics.get(k) != 0 for k in ("failures", "errors", "skipped")):
        raise ValueError("Actual CPU diagnostic regression suite incomplete")
    dump(output / "diagnostics-regression.json", diagnostics)
    dump(output / "inventory-context.json", {"mode": "fresh-interpreter-same-user-and-environment",
        "validation_sys_path_before": search_path_before, "validation_sys_path_after": list(sys.path),
        "scope": "Import search-path changes are recorded separately from the strict installed environment snapshot"})
    assert_snapshot(baseline, snapshot_environment(), output, "after-validation")
    source_after = verify_installed()
    dump(output / "source-verification-after.json", source_after)
    if source_after != proof:
        raise ValueError("Installed source proof changed; compare source-verification.json and source-verification-after.json")
    dump(output / "preservation.json", {"status": "PASS", "packages": len(baseline["packages"]),
        "dpkg": len(baseline["dpkg"]), "inherited_evidence_files": len(baseline["inherited_files_sha256"]),
        "scope": "Metadata/direct_url, dpkg, legal and all inherited lite/build receipts unchanged"})
    result = {"schema": "seedvr2-c14-runtime/1", "status": "PASS", "euid": 1000, "parent_image": PARENT,
        "source_commit": os.environ["H3_C14_SOURCE_COMMIT"], "gateway_commit": GATEWAY,
        "retained_packages_unchanged": True, "adapters_present": False, "ref8_patch_present": False,
        "diagnostics_default_enabled": False, "numerical_case_count": norm["six_case_count"],
        "source_manifest_sha256": proof["source_manifest_sha256"],
        "scope": "Actual installed CPU/source/dependency gates; no SP8/C14/video quality claim"}
    dump(output / "runtime-validation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("apply", "validate", "snapshot"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.action == "snapshot":
        lite = lite_runtime()
        lock = preservation_lock(read(LITE / "runtime.lock.json"))
        print(json.dumps(lite.snapshot(lock)))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required for apply/validate")
    try:
        print(json.dumps(apply(args.output_dir) if args.action == "apply" else validate(args.output_dir), indent=2))
    except Exception as error:
        dump(args.output_dir / "failure.json", {"status": "FAIL", "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
