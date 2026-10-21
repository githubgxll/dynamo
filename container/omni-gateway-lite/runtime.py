# SPDX-License-Identifier: Apache-2.0
"""Offline source-only Gateway/SeedVR2 update over the complete framework image."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib
import importlib.metadata as md
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
INFO = HERE / "build-info"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def atomic_write(path, data):
    mode = path.stat().st_mode & 0o777
    fd, temporary = tempfile.mkstemp(prefix=".lite-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def record_digest(data):
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")


def update_record(data, target, before, after):
    rows = list(csv.reader(io.StringIO(data.decode("utf-8"), newline="")))
    matches = [row for row in rows if row and row[0] == target]
    if len(matches) != 1 or len(matches[0]) != 3:
        raise ValueError("Expected exactly one owned source RECORD row: " + target)
    if matches[0][1:] != [record_digest(before), str(len(before))]:
        raise ValueError("Installed source does not match original RECORD: " + target)
    matches[0][1:] = [record_digest(after), str(len(after))]
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(rows)
    return output.getvalue().encode("utf-8")


def safe_file(root, relative):
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts or "\\" in relative:
        raise ValueError("Unsafe relative source path: " + relative)
    path = root / relative_path
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Source path escapes installed distribution: " + relative)
    if not path.is_file():
        raise ValueError("Missing source: " + relative)
    return path


def compare_manifest(actual, expected, label):
    if actual != expected:
        changed = sorted(name for name in set(actual) | set(expected) if actual.get(name) != expected.get(name))
        raise ValueError(label + " source mismatch: " + ", ".join(changed[:20]))


def validate_inputs(lock, gateway):
    if lock.get("schema") != "omni-gateway-lite-runtime-lock/1" or gateway.get("schema") != "omni-gateway-lite-gateway-lock/1":
        raise ValueError("Unexpected lite runtime/source lock schema")
    for a, b in (("gateway_commit", "current_commit"), ("gateway_before_commit", "previous_commit"),
                 ("parent_source_commit", "parent_source_commit")):
        if lock[a] != gateway[b]:
            raise ValueError("Runtime and Gateway source locks disagree at " + a)
    for field in ("vendored_sha256", "reference_sha256"):
        for relative, expected in gateway[field].items():
            if sha256(safe_file(HERE / "gateway", relative).read_bytes()) != expected:
                raise ValueError("Vendored Gateway bytes changed: " + relative)
    before, after = gateway["source_before_sha256"], gateway["source_after_sha256"]
    changes = {name for name in set(before) | set(after) if before.get(name) != after.get(name)}
    if changes != {gateway["changed_runtime_file"]}:
        raise ValueError("Gateway source overlay must change exactly one reviewed runtime file")
    latest = {name: value for name, value in after.items() if name.startswith("dingo/video_gateway/")}
    compare_manifest(latest, gateway["latest_gateway_python_sha256"], "Latest upstream Gateway")


def installed_root(distribution):
    dist = md.distribution(distribution)
    return dist, Path(dist.locate_file("")).resolve()


def distribution_record(dist):
    entries = [item for item in dist.files or [] if str(item).endswith(".dist-info/RECORD")]
    if len(entries) != 1:
        raise ValueError("Expected one distribution RECORD")
    return safe_file(Path(dist.locate_file("")).resolve(), str(entries[0]))


def gateway_sources(root, lock, *, patched):
    expected = lock["source_after_sha256" if patched else "source_before_sha256"]
    generated = set(lock["allowed_generated_version_paths"])
    actual = {}
    generated_actual = {}
    for path in sorted((root / "dingo").rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        source = safe_file(root, relative)
        target = generated_actual if relative in generated else actual
        target[relative] = sha256(source.read_bytes())
    compare_manifest(actual, expected, "Gateway/Dingo")
    if set(generated_actual) != generated:
        raise ValueError("Generated Dingo version-file set differs from the reviewed Hatch build")
    return {"status": "PASS", "tracked_python_count": len(actual), "source_sha256": actual,
            "generated_version_sha256": generated_actual,
            "scope": "Latest Gateway subtree plus the successful Omni031 integration adaptations; generated version files retained separately"}


def omni_sources(root, wheel, lock, *, patched):
    if sha256(wheel.read_bytes()) != lock["omni_wheel_sha256"]:
        raise ValueError("Retained original Omni wheel SHA256 differs")
    with zipfile.ZipFile(wheel) as archive:
        names = [item.filename for item in archive.infolist()
                 if not item.is_dir() and item.filename.startswith("vllm_omni/")]
        if len(names) != len(set(names)) or not names:
            raise ValueError("Invalid duplicate/empty Omni wheel package")
        expected = {name: sha256(archive.read(name)) for name in names}
    if lock["omni_patch_path"] not in expected or expected.get(lock["native_lora_path"]) != lock["native_lora_sha256"]:
        raise ValueError("Original Omni wheel does not contain reviewed SeedVR2 and native H3 source")
    if patched:
        expected[lock["omni_patch_path"]] = lock["omni_patch_after_sha256"]
    actual = {}
    for path in sorted((root / "vllm_omni").rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo"):
            relative = path.relative_to(root).as_posix()
            actual[relative] = sha256(safe_file(root, relative).read_bytes())
    compare_manifest(actual, expected, "Complete Omni wheel")
    return {"status": "PASS", "package_file_count": len(actual),
            "python_file_count": sum(name.endswith(".py") for name in actual),
            "original_wheel_sha256": lock["omni_wheel_sha256"],
            "native_lora_sha256": actual[lock["native_lora_path"]],
            "source_sha256": actual, "only_allowed_delta": lock["omni_patch_path"] if patched else None}


def source_proofs(lock, gateway, *, patched):
    _, dingo_root = installed_root("ai-dingo")
    _, omni_root = installed_root("vllm-omni")
    wheel = Path("/opt/dynamo/build-info/omni-wheels") / lock["omni_wheel_filename"]
    return {"status": "PASS", "gateway": gateway_sources(dingo_root, gateway, patched=patched),
            "omni": omni_sources(omni_root, wheel, lock, patched=patched)}


def snapshot(lock):
    packages = [{"name": dist.metadata["Name"], "version": dist.version,
                 "metadata": dist.read_text("METADATA"), "direct_url": dist.read_text("direct_url.json")}
                for dist in md.distributions()]
    packages.sort(key=lambda row: (row["name"].lower(), row["version"], row["metadata"] or ""))
    files = {}
    for name in lock["preserved_trees"]:
        directory = Path(name)
        if not directory.is_dir():
            raise ValueError("Missing inherited evidence: " + name)
        for path in sorted(directory.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo"):
                files[path.as_posix()] = sha256(path.read_bytes())
    dpkg = subprocess.run(["dpkg-query", "-W", "-f=${Package}\t${Version}\t${Architecture}\n"],
                          check=True, capture_output=True, text=True).stdout.splitlines()
    return {"packages": packages, "inherited_files_sha256": files, "dpkg": sorted(dpkg)}


def validate_contract(lock, *, before):
    if os.environ.get("H3_LITE_PARENT_IMAGE") != lock["parent_image"]:
        raise ValueError("Wrong framework parent identity")
    expected_gateway = lock["gateway_before_commit"] if before else lock["gateway_commit"]
    if os.environ.get("DYNAMO_GATEWAY_SOURCE_SHA") != expected_gateway:
        raise ValueError("Wrong Gateway source marker")
    if os.environ.get("DYNAMO_COMMIT_SHA") != lock["parent_source_commit"]:
        raise ValueError("Inherited Dynamo wheel source commit marker differs")
    if os.environ.get("H3_BUILD_INFO", "/opt/dynamo/build-info") != "/opt/dynamo/build-info":
        raise ValueError("H3_BUILD_INFO must preserve inherited audit inputs")
    for name, expected in lock["versions"].items():
        if md.version(name) != expected:
            raise ValueError("Unexpected package version: " + name)
    for name in lock["forbidden_paths"]:
        if Path(name).exists():
            raise ValueError("Wrong parent contains omitted experimental assets/patches: " + name)
    if not before and not re.fullmatch(r"[0-9a-f]{40}", os.environ.get("H3_EXPERIMENTAL_SOURCE_COMMIT", "")):
        raise ValueError("Missing new recipe source commit")


def gateway_patch(gateway, output):
    dist, root = installed_root("ai-dingo")
    path = safe_file(root, gateway["changed_runtime_file"])
    before = path.read_bytes()
    after = (HERE / "gateway" / "after" / "artifact_store.py").read_bytes()
    if sha256(before) != gateway["before_sha256"] or sha256(after) != gateway["after_sha256"]:
        raise ValueError("Gateway before/after source identity differs")
    compile(after, str(path), "exec")
    record = distribution_record(dist)
    record_before = record.read_bytes()
    record_after = update_record(record_before, gateway["changed_runtime_file"], before, after)
    output.mkdir(parents=True, exist_ok=True)
    (output / "dingo-RECORD.before.csv").write_bytes(record_before)
    atomic_write(path, after)
    atomic_write(record, record_after)
    for cached in (path.parent / "__pycache__").glob("artifact_store.*.pyc"):
        cached.unlink()
    result = {"status": "PASS", "module_path": str(path), "record_path": str(record),
              "before_sha256": sha256(before), "after_sha256": sha256(after),
              "record_before_sha256": sha256(record_before), "record_after_sha256": sha256(record_after),
              "gateway_commit": gateway["current_commit"]}
    write_json(output / "gateway-patch.json", result)
    return result


def run_script(script, arguments, output):
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               DO_NOT_TRACK="1", VLLM_NO_USAGE_STATS="1")
    with tempfile.TemporaryDirectory(prefix=".lite-check-", dir=output) as temporary:
        env.update(TMPDIR=temporary, TMP=temporary, TEMP=temporary)
        result = subprocess.run([sys.executable, "-B", str(script), *arguments], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", timeout=600, check=False)
    (output / (script.stem + ".log")).write_text(result.stdout, encoding="utf-8")
    print(result.stdout, flush=True)
    if result.returncode:
        raise RuntimeError(f"{script.name} failed (exit {result.returncode})")


def apply(output):
    lock, gateway = read_json(HERE / "runtime.lock.json"), read_json(HERE / "gateway.lock.json")
    validate_inputs(lock, gateway)
    if os.geteuid() != 0 or output.resolve() != INFO.resolve():
        raise ValueError("Apply is a root build-layer action with fixed immutable receipt location")
    validate_contract(lock, before=True)
    proof_before = source_proofs(lock, gateway, patched=False)
    baseline = snapshot(lock)
    write_json(output / "parent-snapshot.json", baseline)
    write_json(output / "source-before.json", proof_before)
    receipt = gateway_patch(gateway, output)
    run_script(HERE / "rmsnorm/apply_patch.py", ["--apply-installed", "--report", str(output / "rmsnorm/patch-application.json")], output)
    proof_after = source_proofs(lock, gateway, patched=True)
    if snapshot(lock) != baseline:
        raise ValueError("Source-only patch changed package metadata, inherited evidence or OS inventory")
    write_json(output / "source-after.json", proof_after)
    result = {"schema": "omni-gateway-lite-application/1", "status": "PASS",
              "parent_image": lock["parent_image"], "parent_source_commit": lock["parent_source_commit"],
              "gateway_commit": gateway["current_commit"], "gateway_patch": receipt,
              "package_inventory_unchanged": True, "scope": "Source patch only; runtime CPU/GPU acceptance separate"}
    write_json(output / "application.json", result)
    return result


def validate(output):
    output = output.resolve()
    lock, gateway = read_json(HERE / "runtime.lock.json"), read_json(HERE / "gateway.lock.json")
    validate_inputs(lock, gateway)
    if os.geteuid() != 1000:
        raise ValueError("Final runtime validation must run as the inherited dynamo UID1000")
    validate_contract(lock, before=False)
    output.mkdir(parents=True, exist_ok=True)
    application = read_json(INFO / "application.json")
    if application.get("status") != "PASS" or application.get("parent_image") != lock["parent_image"] or application.get("gateway_commit") != gateway["current_commit"]:
        raise ValueError("Source patch application receipt differs")
    before = read_json(INFO / "parent-snapshot.json")
    if snapshot(lock) != before:
        raise ValueError("Inherited package metadata, legal evidence or OS inventory changed")
    proof = source_proofs(lock, gateway, patched=True)
    generated_before = read_json(INFO / "source-before.json")["gateway"]["generated_version_sha256"]
    if proof["gateway"]["generated_version_sha256"] != generated_before:
        raise ValueError("Generated Dingo build version files changed")
    receipt = read_json(INFO / "gateway-patch.json")
    dist, root = installed_root("ai-dingo")
    record = distribution_record(dist)
    if sha256(record.read_bytes()) != receipt["record_after_sha256"] or sha256((INFO / "dingo-RECORD.before.csv").read_bytes()) != receipt["record_before_sha256"]:
        raise ValueError("Dingo installed RECORD or original RECORD evidence changed")
    module = importlib.import_module("dingo.video_gateway.artifact_store")
    if Path(module.__file__).resolve() != safe_file(root, gateway["changed_runtime_file"]).resolve():
        raise ValueError("Actual Gateway import is shadowed")
    from gateway_regression import check_capacity
    gateway_regression = check_capacity(module)
    write_json(output / "gateway-regression.json", gateway_regression)
    write_json(output / "source-verification.json", proof)
    run_script(HERE / "rmsnorm/validate_runtime.py", ["--output-dir", str(output / "rmsnorm"), "--device", "cpu",
               "--application-report", str(INFO / "rmsnorm/patch-application.json")], output)
    for name in ("audit_environment.py", "media_smoke.py", "service_smoke.py"):
        run_script(HERE / "environment" / name, ["--output-dir", str(output / "environment")], output)
    if snapshot(lock) != before:
        raise ValueError("Runtime checks changed inherited package metadata or evidence")
    if source_proofs(lock, gateway, patched=True) != proof or sha256(record.read_bytes()) != receipt["record_after_sha256"]:
        raise ValueError("Runtime checks changed validated Omni/Gateway sources or Dingo RECORD")
    write_json(output / "preservation.json", {"status": "PASS", "package_count": len(before["packages"]),
               "inherited_file_count": len(before["inherited_files_sha256"]), "dpkg_count": len(before["dpkg"]),
               "scope": "Unchanged package metadata/direct_url, dpkg and inherited legal/build evidence before and after all checks"})
    result = {"schema": "omni-gateway-lite-runtime/1", "status": "PASS", "euid": os.geteuid(),
              "parent_image": lock["parent_image"], "gateway_commit": lock["gateway_commit"],
              "source_commit": os.environ["H3_EXPERIMENTAL_SOURCE_COMMIT"],
              "retained_packages_unchanged": True, "adapters_present": False, "ref8_patch_present": False,
              "gateway_regression_case_count": gateway_regression["case_count"],
              "scope": "Real installed CPU imports, complete Omni source, exact Gateway source and RMSNorm regressions; no GPU or generated-video quality claim"}
    write_json(output / "runtime-validation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("apply", "validate"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = apply(args.output_dir) if args.action == "apply" else validate(args.output_dir)
        print(json.dumps(result, indent=2))
    except Exception as error:
        write_json(args.output_dir / ("application.json" if args.action == "apply" else "runtime-validation.json"),
                   {"status": "FAIL", "error": str(error), "error_type": type(error).__name__})
        raise


if __name__ == "__main__":
    main()
