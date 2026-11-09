#!/usr/bin/env python3
"""Apply an exact, copy-only production delta to the verified c88 runtime."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib
import importlib.metadata as md
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
INFO = HERE / "build-info"
SOURCE_ENV = "H3_SR_PRODUCTION_SOURCE_COMMIT"
PARENT_RUNTIME = Path("/opt/omni032-h3-sr/runtime.py")
PARENT_RUNTIME_SHA = "9ccdd8344e6b32f284b39d3b8147aaf3c3691e32b71ba194d6f6901db9f75f6b"
PATCHERS = ("peak_patch", "sr_timing_patch", "window_headers_patch", "duration_patch")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def map_sha(mapping):
    return sha(json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def policy():
    value = read(HERE / "inputs/source-lock.json")
    if (value["schema"] != "h3-sr-production-source-lock/1"
            or os.environ.get("H3_SR_PRODUCTION_PARENT_IMAGE") != value["parent_image"]
            or not re.fullmatch(r"[0-9a-f]{40}", os.environ.get(SOURCE_ENV, ""))):
        raise ValueError("Production source markers or parent policy differ")
    return value


def parent_module():
    if PARENT_RUNTIME.is_symlink() or sha(PARENT_RUNTIME.read_bytes()) != PARENT_RUNTIME_SHA:
        raise ValueError("Published parent verifier differs")
    return load("_h3sr_frozen_parent_runtime", PARENT_RUNTIME)


def metadata_snapshot():
    """Default discovery in a fresh interpreter, without importing engines."""
    discovered = []
    packages = []
    for dist in md.distributions():
        name = dist.metadata.get("Name")
        if name:
            packages.append((name.lower().replace("_", "-"), dist.version))
            discovered.append({"name": name, "version": dist.version,
                "location": str(dist.locate_file("")), "metadata_path": str(getattr(dist, "_path", ""))})
    return {"schema": "h3sr-clean-metadata/1", "status": "PASS", "python": sys.executable,
            "euid": os.geteuid() if hasattr(os, "geteuid") else None,
            "cwd": os.getcwd(), "sys_path": list(sys.path), "packages": sorted(packages), "discovered": discovered}


def fresh_process(action, directory, *, timeout=240):
    """No -I/-S/path filtering: retain the real site, .pth, environment and cwd."""
    directory = Path(directory)
    command = [sys.executable, "-B", str(HERE / "runtime.py"), action, "--output-dir", str(directory)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        def as_text(value):
            return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
        detail = as_text(getattr(error, "stdout", None)) + as_text(getattr(error, "stderr", None))
        (directory / "fresh-process.log").write_text(detail + "\n" + type(error).__name__, encoding="utf-8")
        if detail:
            print(detail[-14000:], flush=True)
        raise RuntimeError("Fresh metadata process failed: " + type(error).__name__) from error
    log = result.stdout + result.stderr
    (directory / "fresh-process.log").write_text(log, encoding="utf-8")
    if result.returncode:
        print(log[-14000:], flush=True)
        raise RuntimeError("Fresh metadata process failed (exit " + str(result.returncode) + ")")
    context = read(directory / "clean-metadata-context.json")
    euid = os.geteuid() if hasattr(os, "geteuid") else None
    if (context.get("schema") != "h3sr-clean-metadata/1" or context.get("status") != "PASS"
            or context.get("python") != sys.executable or context.get("cwd") != os.getcwd()
            or context.get("euid") != euid or not isinstance(context.get("packages"), list)):
        raise ValueError("Fresh metadata process identity or report differs")
    return context


def inventory():
    with tempfile.TemporaryDirectory(prefix="h3sr-metadata-") as temporary:
        report = fresh_process("metadata-inventory", temporary)
    # Preserve duplicate rows; true startup-visible duplicate distributions are
    # still rejected by the unchanged dependency auditor in its fresh process.
    return [tuple(item) for item in report["packages"]]


def source_map(root, package):
    folder = root / package
    return {p.relative_to(root).as_posix(): sha(p.read_bytes()) for p in sorted(folder.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")}


def recipe_map():
    return {p.relative_to(HERE).as_posix(): sha(p.read_bytes()) for p in sorted(HERE.rglob("*"))
            if p.is_file() and not p.is_relative_to(INFO) and "__pycache__" not in p.parts
            and p.suffix not in (".pyc", ".pyo")}


def startup_sources():
    directory = HERE.parent / "h3-sr-startup"
    if not directory.is_dir() or any(path.is_symlink() for path in directory.rglob("*")):
        raise ValueError("Frozen startup bundle is missing or contains a symlink")
    return source_map(HERE.parent, directory.name)


def distribution(name):
    dist = md.distribution(name)
    root = Path(dist.locate_file("")).resolve()
    records = [Path(dist.locate_file(p)).resolve() for p in dist.files or [] if str(p).endswith(".dist-info/RECORD")]
    if len(records) != 1:
        raise ValueError("Expected one package RECORD: " + name)
    return dist, root, records[0]


def target(root, name):
    relative = PurePosixPath(name)
    path = root.joinpath(*relative.parts)
    if (not name.endswith(".py") or ".." in relative.parts or relative.is_absolute()
            or "\\" in name or ":" in name
            or path.is_symlink() or not path.resolve().is_relative_to(root.resolve())):
        raise ValueError("Only exact regular Python source deltas are permitted: " + name)
    return path


def update_record(data, changes):
    rows = list(csv.reader(io.StringIO(data.decode("utf-8"))))
    if any(len(row) != 3 for row in rows) or len({row[0] for row in rows}) != len(rows):
        raise ValueError("Package RECORD has malformed or duplicate entries")
    by_name = {row[0]: row for row in rows}
    for name, contents in changes.items():
        encoded = base64.urlsafe_b64encode(hashlib.sha256(contents).digest()).rstrip(b"=").decode()
        by_name[name] = [name, "sha256=" + encoded, str(len(contents))]
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(by_name[name] for name in sorted(by_name))
    return output.getvalue().encode()


def native_expected():
    return read(HERE / "inputs/expected-source-manifest.json")


def plan_native(root, parent_sources):
    """Compose exact locked transforms before touching the installed tree."""
    changes = {}
    for module_name in PATCHERS:
        patcher = load("_production_" + module_name, HERE / (module_name + ".py"))
        for relative in patcher.CHANGES:
            if relative in changes:
                contents = changes[relative]
            else:
                contents = target(root, relative).read_bytes()
                if sha(contents) != parent_sources.get(relative):
                    raise ValueError("Patch input is not part of the published parent: " + relative)
            changes[relative] = patcher.transform(relative, contents)
            compile(changes[relative], relative, "exec")
    after = {**parent_sources, **{name: sha(data) for name, data in changes.items()}}
    return changes, {"source_sha256": after, "source_manifest_sha256": map_sha(after)}


def apply():
    if os.geteuid() != 0 or (INFO / "application.json").exists():
        raise ValueError("Apply is a create-once root build action")
    value = policy()
    startup = startup_sources()
    if startup != value["startup_sources"]:
        raise ValueError("Installed startup bundle differs from the frozen source lock")
    parent = parent_module()
    before = parent.verify_installed()
    if (before["source_commit"] != value["parent_source_commit"]
            or before["source_manifest_sha256"] != value["parent_source_manifest_sha256"]):
        raise ValueError("Actual installed parent does not match c88 publication")
    parent_sources = read(HERE / "inputs/parent-source-manifest.json")["source_sha256"]
    if before["source_sha256"] != parent_sources:
        raise ValueError("Parent complete source inventory differs")
    packages = inventory()
    preserved = parent.preserved_snapshot()
    _, omni_root, omni_record = distribution("vllm-omni")
    _, dingo_root, dingo_record = distribution("ai-dingo")
    omni_changes, planned = plan_native(omni_root, parent_sources)
    expected = native_expected()
    if planned != expected:
        raise ValueError("Actual planned production delta differs from the frozen complete manifest")
    dingo_changes = {}
    for relative, entry in value["dingo_changes"].items():
        path = target(dingo_root, relative)
        original = sha(path.read_bytes()) if path.exists() else None
        if original != entry["before_sha256"]:
            raise ValueError("Installed Gateway source differs from reviewed parent: " + relative)
        contents = (HERE / "dingo-overlay" / relative).read_bytes()
        if sha(contents) != entry["after_sha256"]:
            raise ValueError("Frozen Gateway source payload differs: " + relative)
        compile(contents, relative, "exec")
        dingo_changes[relative] = contents
    dump(INFO / "parent-verification.json", before)
    dump(INFO / "package-inventory.json", packages)
    dump(INFO / "preserved-parent-sources.json", preserved)
    for name, contents in omni_changes.items():
        target(omni_root, name).write_bytes(contents)
    for name, contents in dingo_changes.items():
        path = target(dingo_root, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    omni_record.write_bytes(update_record(omni_record.read_bytes(), omni_changes))
    dingo_record.write_bytes(update_record(dingo_record.read_bytes(), dingo_changes))
    receipt = {"schema": "h3sr-production-application/1", "status": "PASS",
               "source_commit": os.environ[SOURCE_ENV], "parent_image": value["parent_image"],
               "source_manifest_sha256": expected["source_manifest_sha256"],
               "omni_changes": {k: sha(v) for k, v in omni_changes.items()},
               "dingo_changes": {k: sha(v) for k, v in dingo_changes.items()},
               "recipe_sources": recipe_map(),
               "startup_sources": startup,
               "omni_record_sha256": sha(omni_record.read_bytes()), "dingo_record_sha256": sha(dingo_record.read_bytes())}
    dump(INFO / "application.json", receipt)
    return verify()


def verify():
    value = policy()
    receipt = read(INFO / "application.json")
    if receipt["source_commit"] != os.environ[SOURCE_ENV] or receipt["parent_image"] != value["parent_image"]:
        raise ValueError("Installed application identity differs")
    if recipe_map() != receipt["recipe_sources"]:
        raise ValueError("Production helper/profile/recipe sources changed")
    if startup_sources() != receipt["startup_sources"] or receipt["startup_sources"] != value["startup_sources"]:
        raise ValueError("Installed startup bundle changed")
    if Path("/usr/local/lib/python3.12/dist-packages/h3_sr_production.pth").read_bytes() != b"/opt/h3-sr\n":
        raise ValueError("Spawn import path for the independent production modules differs")
    expected = native_expected()
    _, root, record = distribution("vllm-omni")
    if source_map(root, "vllm_omni") != expected["source_sha256"]:
        raise ValueError("Complete installed Omni source changed")
    if sha(record.read_bytes()) != receipt["omni_record_sha256"]:
        raise ValueError("Omni RECORD changed")
    _, dingo_root, dingo_record = distribution("ai-dingo")
    preserved = read(INFO / "preserved-parent-sources.json")
    for name, digest in receipt["dingo_changes"].items():
        preserved[name] = digest
    if parent_module().preserved_snapshot() != preserved:
        raise ValueError("Gateway or historical parent recipe source changed outside approved delta")
    if sha(dingo_record.read_bytes()) != receipt["dingo_record_sha256"] or inventory() != [tuple(p) for p in read(INFO / "package-inventory.json")]:
        raise ValueError("Installed package/RECORD inventory changed")
    for name, version in value["versions"].items():
        if md.version(name) != version:
            raise ValueError("Native/runtime version changed: " + name)
    return {"schema": "h3sr-production-source-proof/1", "status": "PASS",
            "source_commit": os.environ[SOURCE_ENV], "parent_image": value["parent_image"],
            "source_install_root": str(root), "source_sha256": expected["source_sha256"],
            "source_manifest_sha256": expected["source_manifest_sha256"],
            "changed_sources": sorted(receipt["omni_changes"]), "dingo_changes": receipt["dingo_changes"],
            "recipe_manifest_sha256": map_sha(receipt["recipe_sources"]),
            "startup_manifest_sha256": map_sha(receipt["startup_sources"]),
            "package_inventory_status": "UNCHANGED", "gpu_status": "NOT_RUN"}


def dependency_checks(output):
    # Reuse the actual parent's metadata auditor and verified NCCL exception,
    # without pretending approved new Python source bytes equal the old tree.
    env = load("_h3sr_parent_environment", Path("/opt/omni032-h3-sr/environment.py"))
    inputs = Path("/opt/omni032-h3-sr/build-info/environment")
    policy = read("/opt/omni032-h3-sr/environment-policy.json")
    auditor = env.dependency_auditor(target=True)
    before = read(inputs / "parent-inventory.json")
    evidence = env.nccl_evidence(policy, before, auditor)
    if evidence != read(inputs / "nccl-provenance.json"):
        raise ValueError("Inherited verified NCCL metadata exception changed")
    report = auditor.audit_dependencies(nccl_evidence=evidence, root_extras=policy["root_extras"])
    auditor.require_framework_versions(report)
    env.run_uv_check(output, auditor, report)
    auditor.finalize_status(report)
    dump(output / "dependency-audit.json", report)
    if report["errors"]:
        raise ValueError("Actual installed dependency audit failed: " + json.dumps(report["errors"]))
    return report["status"]


def isolated_dependency_checks(output):
    directory = Path(tempfile.mkdtemp(prefix="metadata-audit-", dir=output))
    context = fresh_process("audit-dependencies", directory)
    report = read(directory / "dependency-audit.json")
    if report.get("errors") or not isinstance(report.get("status"), str):
        raise ValueError("Fresh installed dependency audit failed")
    dump(output / "metadata-isolation.json", {"status": "PASS", "evidence_directory": directory.name,
        "parent_sys_path_after_imports": list(sys.path), "fresh_context": context,
        "policy": "Same interpreter/UID/site/.pth/environment/cwd; no distribution filtering or new exceptions"})
    return report["status"]


def isolated_verify(output, phase):
    directory = Path(tempfile.mkdtemp(prefix="source-" + phase + "-", dir=output))
    fresh_process("verify-fresh", directory)
    return read(directory / "source-verification.json")


def validate(output):
    if os.name != "posix" or os.geteuid() != 1000 or sys.version_info[:2] != (3, 12):
        raise ValueError("Actual Linux Python3.12 UID1000 validation required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    proof = isolated_verify(output, "initial")
    dump(output / "source-verification.json", proof)
    # Import every altered source from its actual installed package; prevent
    # fixture, copied /opt code or a user PYTHONPATH from shadowing the runtime.
    for relative in proof["changed_sources"]:
        module = importlib.import_module(relative[:-3].replace("/", "."))
        if Path(module.__file__).resolve() != (Path(proof["source_install_root"]) / relative).resolve():
            raise ValueError("Installed source is shadowed: " + relative)
    for relative in proof["dingo_changes"]:
        module = importlib.import_module(relative[:-3].replace("/", "."))
        _, root, _ = distribution("ai-dingo")
        if Path(module.__file__).resolve() != (root / relative).resolve():
            raise ValueError("Gateway source is shadowed: " + relative)
    for name in ("service.profile", "service.worker_extension", "service.ownership", "service.control", "service.launcher"):
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() != (HERE / (name.replace(".", "/") + ".py")).resolve():
            raise ValueError("Production runtime module is shadowed: " + name)
    peak = load("_actual_peak_policy", HERE / "peak_patch.py")
    dump(output / "actual-peak-api.json", peak.validate_installed_runtime())
    cli = load("_production_actual_native_cli", HERE / "cli_smoke.py")
    dump(output / "native-cli.json", cli.run())
    dependencies = isolated_dependency_checks(output)
    # The Torch contract imports a heavy runtime. Keep it in the import/test
    # process, never in the clean package-metadata audit process.
    native = load("_h3sr_actual_torch_contract", Path("/opt/h3-hsdp-offload/torch_contract.py"))
    dump(output / "torch-contract.json", native.verify())
    result = subprocess.run([sys.executable, "-B", str(HERE / "cpu_smoke.py"), "--output-dir", str(output)],
                            capture_output=True, text=True, timeout=600,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    (output / "cpu-tests.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        print((result.stdout + result.stderr)[-14000:], flush=True)
        raise RuntimeError("Actual-image production CPU tests failed")
    tests = read(output / "regression.json")
    if tests["tests"] < 1 or any(tests[k] for k in ("errors", "failures", "skipped")):
        raise ValueError("Incomplete production CPU regression run")
    if isolated_verify(output, "final") != proof:
        raise ValueError("Validation changed installed sources")
    value = {"schema": "h3sr-production-runtime/1", "status": "PASS_CPU_ONLY", "euid": 1000,
             "source_commit": os.environ[SOURCE_ENV], "parent_image": policy()["parent_image"],
             "source_manifest_sha256": proof["source_manifest_sha256"], "tests": tests,
             "dependency_status": dependencies, "errors": [], "gpu_status": "NOT_RUN",
             "scope": "Actual installed Python sources/imports, package metadata, native Torch method identity and CPU contracts; no model/GPU/media/production acceptance"}
    dump(output / "runtime-validation.json", value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("apply", "verify", "validate", "verify-fresh", "metadata-inventory", "audit-dependencies"))
    parser.add_argument("--output-dir", type=Path, default=INFO / "validation")
    args = parser.parse_args()
    if args.action in ("metadata-inventory", "audit-dependencies", "verify-fresh"):
        args.output_dir.mkdir(parents=True, exist_ok=True)
        context = args.output_dir / "clean-metadata-context.json"
        if context.exists():
            raise ValueError("Fresh metadata evidence must not reuse an existing report")
        dump(context, metadata_snapshot())
        if args.action == "audit-dependencies":
            print(json.dumps({"status": dependency_checks(args.output_dir)}))
        elif args.action == "verify-fresh":
            dump(args.output_dir / "source-verification.json", verify())
        return
    result = apply() if args.action == "apply" else validate(args.output_dir) if args.action == "validate" else verify()
    if args.action == "verify":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        dump(args.output_dir / "source-verification.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
