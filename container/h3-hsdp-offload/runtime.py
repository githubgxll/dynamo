# SPDX-License-Identifier: Apache-2.0
"""Source-only HSDP offload candidate over the exact successful C14 image."""
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
C14 = Path("/opt/seedvr2-c14")
LITE = Path("/opt/omni-gateway-lite")
PARENT = "registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm@sha256:87365073cbdf95fa17ca352a24a93cb74fde144a47c96135e91011d1a03df72e"
PARENT_COMMIT = "0b11a41b083facf74d3387696a72893d0d3e40e1"
TARGET = "vllm_omni/diffusion/offloader/sequential_backend.py"
HELPER = "vllm_omni/diffusion/offloader/hsdp_migration.py"
PIPELINE = "vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def c14_runtime():
    name = "_hsdp_parent_c14_runtime"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, C14 / "runtime.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def contract():
    c14 = c14_runtime()
    lite = c14.lite_runtime()
    lock = read(LITE / "runtime.lock.json")
    c14.contract(lite, lock)
    if os.environ.get("H3_C14_SOURCE_COMMIT") != PARENT_COMMIT:
        raise ValueError("Not the successful C14 source commit")
    if os.environ.get("H3_HSDP_PARENT_IMAGE") != PARENT:
        raise ValueError("Wrong HSDP parent image")
    if not re.fullmatch(r"[a-f0-9]{40}", os.environ.get("H3_HSDP_SOURCE_COMMIT", "")):
        raise ValueError("Missing HSDP recipe commit")
    source = read(HERE / "inputs/source-lock.json")
    if source["parent_image"] != PARENT or source["parent_source_commit"] != PARENT_COMMIT:
        raise ValueError("HSDP source lock differs from selected parent")
    if os.environ.get("SEEDVR2_C14_PROFILE"):
        raise ValueError("Image must not embed enabled diagnostics")
    return c14, lite, lock


def snapshot_environment():
    return c14_runtime().json_probe(HERE / "runtime.py", "snapshot")


def source_expected():
    c14, lite, lock = contract()
    expected = c14.source_expected(lite, lock)
    patch = read(HERE / "source-patches.lock.json")
    if expected.get(TARGET) != patch["before_sha256"] or HELPER in expected:
        raise ValueError("Patch must extend exact successful C14 sources")
    if patch["pipeline"]["target"] != PIPELINE or expected.get(PIPELINE) != patch["pipeline"]["before_sha256"]:
        raise ValueError("H3 pipeline patch must extend the original parent source")
    if digest((HERE / "hsdp_migration.py").read_bytes()) != patch["helper_sha256"]:
        raise ValueError("Helper bytes changed")
    expected[TARGET] = patch["after_sha256"]
    expected[HELPER] = patch["helper_sha256"]
    expected[PIPELINE] = patch["pipeline"]["after_sha256"]
    return expected


def verify_installed():
    c14, lite, lock = contract()
    expected = source_expected()
    dist, root = lite.installed_root("vllm-omni")
    actual = {p.relative_to(root).as_posix(): digest(lite.safe_file(root, p.relative_to(root).as_posix()).read_bytes())
              for p in (root / "vllm_omni").rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")}
    lite.compare_manifest(actual, expected, "Entire inherited Omni + exact HSDP delta")
    record = c14.check_record(lite, dist, actual)
    receipt = read(INFO / "application.json")
    if receipt.get("status") != "PASS" or digest(record.read_bytes()) != receipt["omni_record_after_sha256"]:
        raise ValueError("HSDP application receipt/RECORD differs")
    if receipt.get("changed_sources") != [TARGET, PIPELINE] or receipt.get("added_sources") != [HELPER]:
        raise ValueError("Application receipt is missing the exact migration and DiT entry changes")
    if digest((INFO / "omni-RECORD.before.csv").read_bytes()) != receipt["omni_record_before_sha256"]:
        raise ValueError("Parent RECORD evidence changed")
    _, dingo_root = lite.installed_root("ai-dingo")
    gateway = lite.gateway_sources(dingo_root, read(LITE / "gateway.lock.json"), patched=True)
    if gateway["generated_version_sha256"] != read(C14 / "build-info/source-before.json")["gateway"]["generated_version_sha256"]:
        raise ValueError("Inherited Dingo build identity changed")
    return {"status": "PASS", "parent_image": PARENT, "package_file_count": len(actual),
            "source_sha256": actual, "source_manifest_sha256": digest(json.dumps(actual, sort_keys=True, separators=(",", ":")).encode()),
            "changed_sources": [TARGET, PIPELINE], "added_sources": [HELPER], "gateway_commit": c14.GATEWAY,
            "dit_entry_status": "PASS",
            "rmsnorm_preserved_sha256": actual[c14.NADIT]}


def archive_parent(output, root):
    import torch
    files = {}
    for name in (TARGET, "vllm_omni/diffusion/distributed/hsdp.py", "vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py"):
        files[name] = (root / name).read_bytes()
    torch_root = Path(torch.__file__).resolve().parent
    for file in (torch_root / "distributed/fsdp").rglob("*.py"):
        files["torch/" + file.relative_to(torch_root).as_posix()] = file.read_bytes()
    files["torch/nn/modules/module.py"] = (torch_root / "nn/modules/module.py").read_bytes()
    for name, body in files.items():
        target = output / "parent-sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    dump(output / "parent-sources.json", {"status": "PASS", "parent_image": PARENT,
         "torch_version": torch.__version__, "torch_git_version": torch.version.git_version,
         "sha256": {name: digest(body) for name, body in files.items()}})


def apply(output):
    if os.geteuid() != 0 or output.resolve() != INFO.resolve() or INFO.exists():
        raise ValueError("Apply is a create-once root Docker action")
    c14, lite, lock = contract()
    parent = c14.verify_installed()
    from torch_contract import verify
    native = verify()
    snapshot = snapshot_environment()
    dist, root = lite.installed_root("vllm-omni")
    from source_patches import locked_changes, locked_pipeline_change
    relative, before, after, helper = locked_changes(root)
    pipeline_relative, pipeline_before, pipeline_after = locked_pipeline_change(root)
    source_expected()
    compile(helper, HELPER, "exec")
    if (root / HELPER).exists():
        raise ValueError("New helper would overwrite installed source")
    record = lite.distribution_record(dist)
    record_before = record.read_bytes()
    record_after = lite.update_record(record_before, relative, before, after)
    record_after = lite.update_record(record_after, pipeline_relative, pipeline_before, pipeline_after)
    record_after = c14.add_record(lite, record_after, HELPER, helper)
    output.mkdir()
    dump(output / "parent-snapshot.json", snapshot)
    dump(output / "source-before.json", parent)
    dump(output / "torch-contract.json", native)
    archive_parent(output, root)
    (output / "omni-RECORD.before.csv").write_bytes(record_before)
    for name, body in ((relative, after), (pipeline_relative, pipeline_after)):
        lite.atomic_write(lite.safe_file(root, name), body)
        for cached in (root / name).parent.joinpath("__pycache__").glob(Path(name).stem + ".*.pyc"):
            cached.unlink()
    with (root / HELPER).open("xb") as stream:
        stream.write(helper)
    (root / HELPER).chmod(0o644)
    lite.atomic_write(record, record_after)
    receipt = {"schema": "h3-hsdp-offload-application/1", "status": "PASS", "parent_image": PARENT,
        "source_commit": os.environ["H3_HSDP_SOURCE_COMMIT"], "omni_record_before_sha256": digest(record_before),
        "omni_record_after_sha256": digest(record_after), "changed_sources": [relative, pipeline_relative], "added_sources": [HELPER],
        "dit_entry_status": "PASS",
        "gpu_validation_status": "NOT_RUN"}
    dump(output / "application.json", receipt)
    dump(output / "source-verification.json", verify_installed())
    c14.assert_snapshot(snapshot, snapshot_environment(), output, "after-apply")
    return receipt


def validate(output):
    if os.geteuid() != 1000:
        raise ValueError("Final runtime validation must run as inherited UID1000")
    output.mkdir(parents=True, exist_ok=True)
    c14, lite, lock = contract()
    baseline = read(INFO / "parent-snapshot.json")
    c14.assert_snapshot(baseline, snapshot_environment(), output, "before-validation")
    proof = verify_installed()
    dump(output / "source-verification.json", proof)
    from torch_contract import verify
    dump(output / "torch-contract.json", verify())
    _, root = lite.installed_root("vllm-omni")
    for name in (TARGET, HELPER, PIPELINE):
        module = importlib.import_module(name[:-3].replace("/", "."))
        if Path(module.__file__).resolve() != (root / name).resolve():
            raise ValueError("Patched installed module is shadowed: " + name)
    norm = c14.rmsnorm(output / "rmsnorm", lite, proof)
    # The six cases are inherited unchanged; the observation belongs to this
    # candidate, while the old case-contract identity remains explicit.
    norm.update(schema="h3-hsdp-offload-rmsnorm/1",
                inherited_case_parent_image=norm["parent_image"],
                inherited_case_source_commit=norm["source_commit"],
                parent_image=PARENT, source_commit=os.environ["H3_HSDP_SOURCE_COMMIT"])
    dump(output / "rmsnorm/runtime-validation.json", norm)
    gateway = importlib.import_module("dingo.video_gateway.artifact_store")
    regression = c14.load_module("_hsdp_gateway_regression", LITE / "gateway_regression.py")
    dump(output / "gateway-regression.json", regression.check_capacity(gateway))
    for name in ("audit_environment.py", "media_smoke.py", "service_smoke.py"):
        lite.run_script(LITE / "environment" / name, ["--output-dir", str(output / "environment")], output)
    lite.run_script(C14 / "cpu_smoke.py", ["--output-dir", str(output / "c14")], output)
    lite.run_script(HERE / "cpu_smoke.py", ["--output-dir", str(output)], output)
    tests = read(output / "regression.json")
    if tests.get("status") != "PASS" or tests.get("tests_run", 0) <= 0 or any(tests.get(k) != 0 for k in ("failures", "errors", "skipped")):
        raise ValueError("CPU regression suite incomplete")
    c14.assert_snapshot(baseline, snapshot_environment(), output, "after-validation")
    if verify_installed() != proof:
        raise ValueError("Validation changed installed sources")
    dump(output / "preservation.json", {"status": "PASS", "packages": len(baseline["packages"]),
        "dpkg": len(baseline["dpkg"]), "inherited_evidence_files": len(baseline["inherited_files_sha256"]),
        "scope": "All package metadata/direct_url, dpkg, legal, lite and C14 files unchanged"})
    result = {"schema": "h3-hsdp-offload-runtime/1", "status": "PASS_CPU_ONLY", "euid": 1000,
        "parent_image": PARENT, "source_commit": os.environ["H3_HSDP_SOURCE_COMMIT"], "gpu_validation_status": "NOT_RUN",
        "source_manifest_sha256": proof["source_manifest_sha256"], "retained_packages_unchanged": True,
        "dit_entry_status": "PASS",
        "scope": "Installed imports/native method identity/CPU regressions; real HSDP roundtrip and H3 media NOT_RUN"}
    dump(output / "runtime-validation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("apply", "validate", "snapshot"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.action == "snapshot":
        c14 = c14_runtime()
        lite = c14.lite_runtime()
        lock = c14.preservation_lock(read(LITE / "runtime.lock.json"))
        lock["preserved_trees"] = [*lock["preserved_trees"], str(C14)]
        print(json.dumps(lite.snapshot(lock)))
        return
    if args.output_dir is None:
        parser.error("--output-dir required")
    try:
        print(json.dumps(apply(args.output_dir) if args.action == "apply" else validate(args.output_dir), indent=2))
    except Exception as error:
        dump(args.output_dir / "failure.json", {"status": "FAIL", "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
