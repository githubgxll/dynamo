#!/usr/bin/env python3
"""Install two exact SeedVR2 source deltas while retaining the real H3/SR verifier."""
from __future__ import annotations

import argparse
import base64
import copy
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
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
INFO = HERE / "build-info"
BASE = Path("/opt/h3-sr")
PARENT_IMAGE = "registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm@sha256:26d3767f090fa2fecb2d5a94cf5b326657dd2fef891a13ded41476056f99fe18"
PARENT_IMAGE_ID = "sha256:7b0f8c28d87fe914b1c1061c732746de7171862e6c6da3c6223164b1d9d5fd87"
PARENT_COMMIT = "e5cf166562b4479112832ad628f3affe6a2e0782"
BASE_RUNTIME_SHA256 = "a08e09c2e3da843ecd41e0b83be3d539d8b2aa4bcf62af37be67f39b5e0752e6"
SOURCE_ENV = "SEEDVR2_BF16_SOURCE_COMMIT"
REL = "vllm_omni/diffusion/models/seedvr2/pipeline_seedvr2.py"
HELPER = "vllm_omni/diffusion/models/seedvr2/bf16_runtime.py"
PARENT_FIVE = {
    REL: "bc25563a73707078626826cd665cf2eb7612a7212950a57c2bba32a3f9210658",
    "vllm_omni/diffusion/models/seedvr2/nadit.py": "03fd3a0c199305ce3004643f8d1510183b06731099a72bc447c2bd589c1a6f01",
    "vllm_omni/diffusion/models/seedvr2/vae.py": "06a2ffa2df5aaac83c1fab78086d8429e2ff02e0c676a3115e793c76ce28a1f3",
    "vllm_omni/diffusion/models/seedvr2/config.py": "0debf02936e3f84f737747fd50b4e1448d6294c0ffee11a14a0d83450f6228ea",
    "vllm_omni/diffusion/ipc.py": "eda648854822bc739e678c165b6193b3c2c26273c73bd00283b829a27f214781",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def map_sha(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object: " + str(path))
    return value


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    path.write_bytes(data)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load locked file: " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def inherited():
    path = BASE / "runtime.py"
    if path.is_symlink() or sha(path.read_bytes()) != BASE_RUNTIME_SHA256:
        raise ValueError("Actual inherited H3/SR verifier changed")
    return load("_seedvr2_unchanged_h3sr_verifier", path)


def policy():
    value = read(HERE / "inputs/release-lock.json")
    if (value.get("schema") != "seedvr2-bf16-release-lock/1" or value.get("parent_image") != PARENT_IMAGE
            or value.get("parent_image_id") != PARENT_IMAGE_ID or value.get("platform") != "linux/amd64"
            or value.get("parent_runtime_path") != (BASE / "runtime.py").as_posix()
            or value.get("parent_runtime_sha256") != BASE_RUNTIME_SHA256
            or value.get("parent_source_commit") != PARENT_COMMIT
            or value.get("pipeline_relative_path") != REL or value.get("helper_relative_path") != HELPER
            or value.get("helper_sha256") != sha((HERE / "core/bf16_runtime.py").read_bytes())
            or {k.replace(".", "/") + ".py": v for k, v in value.get("diagnostic_source_sha256", {}).items()} != PARENT_FIVE):
        raise ValueError("Release lock does not identify the sole reviewed 26d parent")
    return value


def identity(commit=None):
    source = commit if commit is not None else os.environ.get(SOURCE_ENV, "")
    if not re.fullmatch(r"[0-9a-f]{40}", source) or source == PARENT_COMMIT:
        raise ValueError("A new exact 40-character child source commit is required")
    return {"source_commit": source, "parent_image": PARENT_IMAGE, "parent_image_id": PARENT_IMAGE_ID}


def manifest(path):
    value = read(path)
    sources = value.get("source_sha256")
    if (not isinstance(sources, dict) or not sources
            or any(not isinstance(k, str) or not re.fullmatch(r"[0-9a-f]{64}", v) for k, v in sources.items())
            or value.get("source_manifest_sha256") != map_sha(sources)):
        raise ValueError("Complete source manifest or canonical hash differs: " + str(path))
    return value


def target(root, name):
    relative = PurePosixPath(name)
    path = root.joinpath(*relative.parts)
    if (name not in (REL, HELPER) or ".." in relative.parts or relative.is_absolute()
            or path.is_symlink() or not path.resolve().is_relative_to(root.resolve())):
        raise ValueError("Only the two reviewed SeedVR2 sources can be written: " + name)
    return path


def update_record(data, changes):
    """Replace one existing row and insert one new row; retain other raw bytes."""
    if set(changes) != {REL, HELPER}:
        raise ValueError("RECORD permits precisely pipeline plus new BF16 helper")
    lines = data.decode("utf-8").splitlines(keepends=True)
    reader = csv.reader(io.StringIO(data.decode("utf-8"), newline=""))
    entries, start, names = [], 0, set()
    for row in reader:
        raw = "".join(lines[start:reader.line_num]).encode("utf-8")
        start = reader.line_num
        if len(row) != 3 or row[0] in names:
            raise ValueError("Malformed or duplicate installed RECORD entries")
        names.add(row[0])
        entries.append((row, raw))
    if REL not in names or HELPER in names:
        raise ValueError("RECORD must contain original pipeline and no preexisting helper")
    output = bytearray()
    for row, raw in entries:
        if row[0] != REL:
            output.extend(raw)
            continue
        terminator = "\r\n" if raw.endswith(b"\r\n") else "\n"
        for name in (REL, HELPER):
            value = changes[name]
            digest = base64.urlsafe_b64encode(hashlib.sha256(value).digest()).rstrip(b"=").decode()
            stream = io.StringIO(newline="")
            csv.writer(stream, lineterminator=terminator).writerow([name, "sha256=" + digest, str(len(value))])
            output.extend(stream.getvalue().encode())
    return bytes(output)


def projected_lock(original, parent_sources, preserved):
    value = copy.deepcopy(original)
    if value.get("schema") != "h3-sr-production-source-lock/1":
        raise ValueError("Unexpected inherited source-lock schema")
    dingo = {k: v for k, v in preserved.items() if k.startswith("dingo/")}
    if not dingo:
        raise ValueError("Current preserved Dingo source evidence is empty")
    value.update(parent_image=PARENT_IMAGE, parent_image_id=PARENT_IMAGE_ID,
                 parent_source_commit=PARENT_COMMIT, parent_source_manifest_sha256=map_sha(parent_sources),
                 parent_dingo_sources=dingo, parent_dingo_sources_sha256=map_sha(dingo), dingo_changes={})
    return value


def checked_plan(root, parent_sources):
    for name, digest in PARENT_FIVE.items():
        if parent_sources.get(name) != digest or sha((root / name).read_bytes()) != digest:
            raise ValueError("Actual diagnostic parent source differs: " + name)
    if (root / HELPER).exists():
        raise ValueError("New BF16 helper already exists in parent")
    patcher = load("_seedvr2_locked_source_patch", HERE / "core/source_patch.py")
    if (patcher.PIPELINE_RELATIVE_PATH != REL
            or patcher.HELPER_MODULE.replace(".", "/") + ".py" != HELPER):
        raise ValueError("Core patcher source paths differ from the two-source allowlist")
    patched, proof = patcher.apply_source(target(root, REL).read_bytes())
    helper = (HERE / "core/bf16_runtime.py").read_bytes()
    compile(helper, HELPER, "exec")
    changes = {REL: patched, HELPER: helper}
    sources = {**parent_sources, **{k: sha(v) for k, v in changes.items()}}
    return changes, {"source_sha256": sources, "source_manifest_sha256": map_sha(sources)}, proof


def command_run(command, output, *, env=None, timeout=600):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                            timeout=timeout, env=env)
    output.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        print((result.stdout + result.stderr)[-14000:], flush=True)
        raise RuntimeError("Validation subprocess failed (exit " + str(result.returncode) + "): " + str(output))
    return result


def clean_inventory(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    command_run([sys.executable, "-B", str(HERE / "runtime.py"), "_inventory", "--output-dir", str(directory)],
                directory / "process.log", timeout=240)
    report = read(directory / "metadata.json")
    if (report.get("python") != sys.executable or report.get("cwd") != os.getcwd()
            or report.get("euid") != (os.geteuid() if hasattr(os, "geteuid") else None)):
        raise ValueError("Clean metadata process identity differs")
    return report


def base_fresh_verify(directory):
    inherited()  # Hash check before starting the unchanged command.
    directory = Path(directory)
    command_run([sys.executable, "-B", str(BASE / "runtime.py"), "verify", "--output-dir", str(directory)],
                directory / "process.log", timeout=600)
    return read(directory / "source-verification.json")


def copy_evidence(source, destination):
    source, destination = Path(source), Path(destination)
    if source.is_symlink() or any(p.is_symlink() for p in source.rglob("*")):
        raise ValueError("Inherited evidence contains a symlink")
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))


def recipe_sources():
    return {p.relative_to(HERE).as_posix(): sha(p.read_bytes()) for p in sorted(HERE.rglob("*"))
            if p.is_file() and not p.is_relative_to(INFO) and "__pycache__" not in p.parts
            and p.suffix not in (".pyc", ".pyo")}


def base_evidence_sources():
    return {p.relative_to(INFO).as_posix(): sha(p.read_bytes()) for p in sorted(INFO.rglob("*"))
            if p.is_file() and p.relative_to(INFO).parts[0].startswith("base-")
            and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")}


def apply(commit):
    if not hasattr(os, "geteuid") or os.geteuid() != 0 or INFO.exists():
        raise ValueError("Apply is a create-once root action; preserve any failed build evidence")
    release = policy()
    child = identity(commit)
    base = inherited()
    # No child environment identity has been set yet. This is the actual 26d
    # verifier under its real inherited source-lock / receipt / environment.
    original_env = {k: os.environ.get(k) for k in ("H3_SR_PRODUCTION_SOURCE_COMMIT", "H3_SR_PRODUCTION_PARENT_IMAGE")}
    before = base_fresh_verify(INFO / "base-verification")
    parent_manifest = manifest(HERE / "inputs/parent-source-manifest.json")
    if (before.get("source_commit") != PARENT_COMMIT or before.get("source_sha256") != parent_manifest["source_sha256"]
            or before.get("source_manifest_sha256") != parent_manifest["source_manifest_sha256"]):
        raise ValueError("Actual verified 26d source inventory differs from release parent lock")
    old_lock = read(BASE / "inputs/source-lock.json")
    if old_lock.get("versions") != release.get("versions"):
        raise ValueError("Release version lock differs from actual inherited policy")
    old_receipt = read(BASE / "build-info/application.json")
    preserved = base.parent_module().preserved_snapshot()
    startup = base.startup_sources()
    if startup != old_receipt["startup_sources"] or startup != old_lock["startup_sources"]:
        raise ValueError("Original startup evidence differs")
    _, omni_root, omni_record = base.distribution("vllm-omni")
    _, _, dingo_record = base.distribution("ai-dingo")
    changes, planned, patch_proof = checked_plan(omni_root, parent_manifest["source_sha256"])
    expected = manifest(HERE / "inputs/expected-source-manifest.json")
    if planned["source_sha256"] != expected["source_sha256"] or planned["source_manifest_sha256"] != expected["source_manifest_sha256"]:
        raise ValueError("The complete planned child source map differs from frozen expected sources")
    if (release.get("parent_source_manifest_sha256") != parent_manifest["source_manifest_sha256"]
            or release.get("expected_source_manifest_sha256") != expected["source_manifest_sha256"]
            or release.get("pipeline_patched_sha256") != sha(changes[REL])):
        raise ValueError("Release identity differs from parent/child source manifests")
    old_record = omni_record.read_bytes()
    new_record = update_record(old_record, changes)
    metadata = clean_inventory(INFO / "base-metadata-process")
    # Preserve originals before any target or inherited verifier input changes.
    copy_evidence(BASE / "inputs", INFO / "base-inputs")
    copy_evidence(BASE / "build-info", INFO / "base-build-info")
    (INFO / "base-runtime.py").write_bytes((BASE / "runtime.py").read_bytes())
    (INFO / "base-omni-RECORD").write_bytes(old_record)
    (INFO / "base-dingo-RECORD").write_bytes(dingo_record.read_bytes())
    for relative in PARENT_FIVE:
        path = INFO / "base-diagnostic-sources" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((omni_root / relative).read_bytes())
    dump(INFO / "base-metadata.json", metadata)
    dump(INFO / "base-preserved-current.json", preserved)
    dump(INFO / "base-startup-sources.json", startup)
    dump(INFO / "base-identity.json", original_env)
    dump(INFO / "patch-proof.json", patch_proof)
    for name, contents in changes.items():
        target(omni_root, name).write_bytes(contents)
    omni_record.write_bytes(new_record)
    dump(BASE / "inputs/source-lock.json", projected_lock(old_lock, parent_manifest["source_sha256"], preserved))
    (BASE / "inputs/parent-source-manifest.json").write_bytes((HERE / "inputs/parent-source-manifest.json").read_bytes())
    (BASE / "inputs/expected-source-manifest.json").write_bytes((HERE / "inputs/expected-source-manifest.json").read_bytes())
    dump(BASE / "build-info/preserved-parent-sources.json", preserved)
    receipt = {**old_receipt, "source_commit": child["source_commit"], "parent_image": PARENT_IMAGE,
               "source_manifest_sha256": expected["source_manifest_sha256"], "dingo_changes": {},
               "omni_changes": {k: sha(v) for k, v in changes.items()},
               "omni_record_sha256": sha(new_record), "recipe_sources": base.recipe_map(),
               "startup_sources": startup,
               "delta_lineage": {"schema": "seedvr2-bf16-inherited-verifier-projection/1",
                   "method": "Verified 26d plus exactly two source deltas; historical c88 apply algorithms NOT replayed",
                   **child, "base_application_sha256": sha((INFO / "base-build-info/application.json").read_bytes()),
                   "base_verifier_sha256": BASE_RUNTIME_SHA256,
                   "base_source_manifest_sha256": parent_manifest["source_manifest_sha256"],
                   "original_parent_image": old_receipt["parent_image"]}}
    dump(BASE / "build-info/application.json", receipt)
    os.environ.update({SOURCE_ENV: child["source_commit"], "SEEDVR2_BF16_PARENT_IMAGE": PARENT_IMAGE,
                       "H3_SR_PRODUCTION_SOURCE_COMMIT": child["source_commit"], "H3_SR_PRODUCTION_PARENT_IMAGE": PARENT_IMAGE})
    # Exercise the unchanged verifier itself in this process and in a fresh one.
    current = base.verify()
    if base_fresh_verify(INFO / "projected-verification") != current:
        raise ValueError("Fresh inherited verification differs from in-process result")
    after = clean_inventory(INFO / "child-metadata-process")
    checker = load("_seedvr2_clean_inventory", HERE / "inventory.py")
    checker.require_preserved(metadata, after, record_path=str(omni_record), record_sha256=sha(new_record))
    dump(INFO / "child-metadata.json", after)
    application = {"schema": "seedvr2-bf16-application/1", "status": "PASS", **child,
                   "source_manifest_sha256": expected["source_manifest_sha256"],
                   "omni_changes": {k: sha(v) for k, v in changes.items()},
                   "omni_record_path": str(omni_record), "omni_record_sha256": sha(new_record),
                   "dingo_record_sha256": sha(dingo_record.read_bytes()), "inventory_preserved": True,
                   "base_verifier_sha256": BASE_RUNTIME_SHA256, "recipe_sources": recipe_sources(),
                   "base_evidence_sources": base_evidence_sources(),
                   "inherited_receipt_sha256": sha((BASE / "build-info/application.json").read_bytes()),
                   "gpu_status": "NOT_RUN"}
    dump(INFO / "application.json", application)
    return verify(INFO / "apply-final-verification")


def verify(output):
    policy()
    child = identity()
    if (os.environ.get("SEEDVR2_BF16_PARENT_IMAGE") != PARENT_IMAGE
            or os.environ.get("H3_SR_PRODUCTION_SOURCE_COMMIT") != child["source_commit"]
            or os.environ.get("H3_SR_PRODUCTION_PARENT_IMAGE") != PARENT_IMAGE):
        raise ValueError("Child and inherited verifier environment identities differ")
    receipt = read(INFO / "application.json")
    if any(receipt.get(k) != v for k, v in child.items()) or receipt.get("recipe_sources") != recipe_sources():
        raise ValueError("Installed production recipe or identity changed")
    if receipt.get("base_evidence_sources") != base_evidence_sources():
        raise ValueError("Archived original parent evidence changed")
    if sha((BASE / "build-info/application.json").read_bytes()) != receipt["inherited_receipt_sha256"]:
        raise ValueError("Projected inherited application receipt changed")
    proof = base_fresh_verify(Path(output) / "inherited")
    expected = manifest(HERE / "inputs/expected-source-manifest.json")
    if proof.get("source_sha256") != expected["source_sha256"] or proof.get("source_manifest_sha256") != expected["source_manifest_sha256"]:
        raise ValueError("Unchanged inherited verifier did not prove complete expected child sources")
    base = inherited()
    _, _, record = base.distribution("vllm-omni")
    _, _, dingo_record = base.distribution("ai-dingo")
    if (record.read_bytes() != update_record((INFO / "base-omni-RECORD").read_bytes(),
            {name: (Path(proof["source_install_root"]) / name).read_bytes() for name in (REL, HELPER)})
            or dingo_record.read_bytes() != (INFO / "base-dingo-RECORD").read_bytes()):
        raise ValueError("An unrelated installed RECORD row or Dingo RECORD changed")
    actual = clean_inventory(Path(output) / "metadata")
    checker = load("_seedvr2_clean_inventory", HERE / "inventory.py")
    checker.require_preserved(read(INFO / "base-metadata.json"), actual,
                             record_path=str(record), record_sha256=receipt["omni_record_sha256"])
    if actual["files"] != read(INFO / "child-metadata.json")["files"]:
        raise ValueError("Child metadata evidence changed")
    result = {"schema": "seedvr2-bf16-source-proof/1", "status": "PASS", **child,
              "source_install_root": proof["source_install_root"], "source_sha256": expected["source_sha256"],
              "source_manifest_sha256": expected["source_manifest_sha256"],
              "inventory_preserved": True, "inherited_verifier_status": proof["status"], "gpu_status": "NOT_RUN"}
    dump(Path(output) / "source-verification.json", result)
    return result


def worker(action, directory, *, timeout=900):
    directory = Path(directory)
    command_run([sys.executable, "-B", str(HERE / "runtime.py"), action, "--output-dir", str(directory)],
                directory / "process.log", timeout=timeout)
    return read(directory / "result.json")


def validate_profiles():
    profiles = load("_seedvr2_actual_profile", BASE / "service/profile.py")
    configure = load("_seedvr2_actual_configure", Path("/opt/h3-sr-startup/configure.py"))
    results, cleaned_sr_environment = [], None
    for variant in ("fl2va", "ref2va"):
        path = Path("/opt/h3-sr-startup/profiles") / (variant + "-sr.json")
        original = read(path)
        configure.validate_profile(original, variant)
        profiles.validate(original)
        derived = copy.deepcopy(original)
        derived["sr"]["stage_environment"]["SEEDVR2_DIT_BF16"] = "1"
        configure.validate_profile(derived, variant)
        profiles.validate(derived)
        inherited_env = {**os.environ, "SEEDVR2_DIT_BF16": "0", "SEEDVR2_TEST_SENTINEL": "1"}
        for role in ("h3", "sr", "gateway"):
            function = profiles.gateway_environment if role == "gateway" else profiles.environment
            positional = (derived,) if role == "gateway" else (derived, role)
            original_positional = (original,) if role == "gateway" else (original, role)
            cleaned = function(*original_positional, profile_path=path, epoch="bf16-cpu-profile-check", root=Path("/work/h3-sr"), inherited=inherited_env)
            if "SEEDVR2_TEST_SENTINEL" in cleaned or "SEEDVR2_DIT_BF16" in cleaned:
                raise ValueError("Actual original launcher did not clear inherited SEEDVR2 precision flags")
            if role == "sr":
                cleaned_sr_environment = cleaned
            env = function(*positional, profile_path=path, epoch="bf16-cpu-profile-check", root=Path("/work/h3-sr"), inherited=inherited_env)
            if ("SEEDVR2_TEST_SENTINEL" in env or env.get("SEEDVR2_DIT_BF16") != ("1" if role == "sr" else None)
                    or env.get("H3_SR_PRODUCTION_SOURCE_COMMIT") != os.environ["H3_SR_PRODUCTION_SOURCE_COMMIT"]
                    or env.get("H3_SR_PRODUCTION_PARENT_IMAGE") != PARENT_IMAGE):
                raise ValueError("Actual launcher environment lost role precision isolation or verifier identity")
            results.append({"profile": variant + "-sr", "role": role,
                            "dit_bf16": env.get("SEEDVR2_DIT_BF16"), "inherited_precision_flags_cleared": True,
                            "status": "PASS"})
        sr_argv, h3_argv = profiles.argv(derived, "sr"), profiles.argv(derived, "h3")
        if configure.flag(sr_argv, "--dtype") != "float16" or configure.flag(h3_argv, "--dtype") != "bfloat16":
            raise ValueError("SR CLI/global dtype or H3 BF16 changed")
    # Loading only the installed helper with the launcher's cleaned flag absent
    # proves its built-in default. No torch/model import is necessary.
    base = inherited()
    _, root, _ = base.distribution("vllm-omni")
    if cleaned_sr_environment is None:
        raise ValueError("Actual cleaned SR environment was not produced")
    env = cleaned_sr_environment
    code = "import importlib.util; s=importlib.util.spec_from_file_location('helper',%r); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); assert m.configuration()['enabled'] is True" % str(root / HELPER)
    with tempfile.TemporaryDirectory(prefix="bf16-default-check-") as temporary:
        command_run([sys.executable, "-B", "-c", code], Path(temporary) / "process.log", env=env, timeout=120)
    return {"status": "PASS_ACTUAL_LAUNCHER_PROFILE_ENVIRONMENT", "profiles": results,
            "cleaned_helper_default_enabled": True, "startup_files_changed": False, "gpu_status": "NOT_RUN"}


def validate(output):
    if os.name != "posix" or os.geteuid() != 1000 or sys.version_info[:2] != (3, 12):
        raise ValueError("Actual Linux Python3.12 UID1000 validation required; do not install substitute Torch")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    proof = worker("_verify", output / "initial-verification")
    dump(output / "source-verification.json", proof)
    dependencies = worker("_dependencies", output / "dependencies")
    native = worker("_native", output / "native-torch")
    cli = worker("_cli", output / "native-cli", timeout=1200)
    imports = worker("_imports", output / "installed-imports")
    profiles = worker("_profiles", output / "profile-environments")
    if (dependencies.get("status") not in ("PASS", "PASS_WITH_UPSTREAM_NCCL_OVERRIDE")
            or native.get("status") != "PASS" or cli.get("status") != "PASS_ACTUAL_INSTALLED_PARSER_ONLY"):
        raise ValueError("Inherited native, dependency or real parser validation did not pass")
    helper = Path(proof["source_install_root"]) / HELPER
    cpu_path = output / "precision-cpu-checks.json"
    command_run([sys.executable, "-B", str(HERE / "tests/cpu_selfcheck.py"), "--runtime", str(helper), "--output", str(cpu_path)],
                output / "precision-cpu-checks.log", timeout=600)
    cpu = read(cpu_path)
    if cpu.get("status") != "PASS_REAL_CPU_TORCH_HELPERS_ONLY" or not cpu.get("checks"):
        raise ValueError("Real installed Torch CPU helper checks were not completed")
    tests = worker("_tests", output / "tests")
    if tests.get("tests", 0) < 1 or any(tests.get(k, -1) != 0 for k in ("errors", "failures", "skipped")):
        raise ValueError("Required standard-library regression tests failed or skipped")
    if worker("_verify", output / "final-verification") != proof:
        raise ValueError("Validation changed installed sources or complete metadata")
    result = {"schema": "seedvr2-bf16-runtime/1", "status": "PASS_CPU_ONLY", **identity(),
              "euid": 1000, "inventory_preserved": True, "errors": [], "gpu_status": "NOT_RUN",
              "source_manifest_sha256": proof["source_manifest_sha256"], "precision_cpu_checks": cpu,
              "tests": tests, "dependency_status": dependencies["status"], "cli_status": "PASS",
              "cli_detail_status": cli["status"],
              "native_torch_status": native["status"], "installed_imports": imports, "profile_environment": profiles,
              "scope": "Actual parent/child installed sources, metadata, original verifiers, native contracts and CPU helpers; GPU/media/quality/performance NOT_RUN"}
    dump(output / "runtime-validation.json", result)
    return result


def internal(action, output):
    output.mkdir(parents=True, exist_ok=True)
    if action == "_inventory":
        module = load("_seedvr2_clean_inventory", HERE / "inventory.py")
        dump(output / "metadata.json", module.snapshot())
        return
    if action == "_verify":
        result = verify(output)
    elif action == "_dependencies":
        result = {"status": inherited().isolated_dependency_checks(output)}
    elif action == "_native":
        result = load("_seedvr2_native_torch_contract", Path("/opt/h3-hsdp-offload/torch_contract.py")).verify()
    elif action == "_cli":
        result = load("_seedvr2_inherited_cli_smoke", BASE / "cli_smoke.py").run()
    elif action == "_profiles":
        result = validate_profiles()
    elif action == "_imports":
        _, root, _ = inherited().distribution("vllm-omni")  # Before heavy imports alter sys.path.
        origins = {}
        for relative in (HELPER, REL):
            module = importlib.import_module(relative[:-3].replace("/", "."))
            origin = Path(module.__file__).resolve()
            if origin != (root / relative).resolve():
                raise ValueError("Actual installed helper/pipeline import was shadowed: " + relative)
            origins[relative] = str(origin)
        result = {"status": "PASS_INSTALLED_PACKAGE_IMPORT_ORIGINS", "origins": origins, "gpu_status": "NOT_RUN"}
    elif action == "_tests":
        suite = unittest.defaultTestLoader.discover(str(HERE / "tests"), pattern="test_*.py")
        result_value = unittest.TextTestRunner(verbosity=2).run(suite)
        result = {"tests": result_value.testsRun, "errors": len(result_value.errors),
                  "failures": len(result_value.failures), "skipped": len(result_value.skipped)}
        dump(output / "result.json", result)
        if not result_value.wasSuccessful() or result_value.skipped:
            raise ValueError("Standard-library runtime/core regression failed or skipped")
    else:
        raise ValueError("Unknown internal action")
    dump(output / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("apply", "verify", "validate", "_inventory", "_verify", "_dependencies", "_native", "_cli", "_profiles", "_imports", "_tests"))
    parser.add_argument("--source-commit")
    parser.add_argument("--output-dir", type=Path, default=INFO / "validation")
    args = parser.parse_args()
    if args.action.startswith("_"):
        internal(args.action, args.output_dir)
        return
    result = apply(args.source_commit) if args.action == "apply" else validate(args.output_dir) if args.action == "validate" else verify(args.output_dir)
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
