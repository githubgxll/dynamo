# SPDX-License-Identifier: Apache-2.0
"""Verify a released framework upgrade and apply only reviewed source deltas.

The inherited image is a functional baseline, not an Omni 0.32 installation.
Historical build reports are preserved as history and are never relabelled as
proof for this release. CUDA/model acceptance remains a separate K8S gate.
"""
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
import zipfile

HERE = Path(__file__).resolve().parent
INFO = HERE / "build-info"
WHEEL_NAME = "vllm_omni-0.32.0-py3-none-any.whl"
SOURCE_ENV = "OMNI032_H3_SR_SOURCE_COMMIT"
OLD_DIRS = ("omni-gateway-lite", "h3-hsdp-offload", "seedvr2-c14",
            "seedvr2-optimization", "seedvr2-metrics-repair")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def map_digest(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def policy():
    value = read(HERE / "inputs/source-lock.json")
    if value.get("versions", {}).get("vllm") != "0.32.0" or value["versions"].get("vllm-omni") != "0.32.0":
        raise ValueError("This recipe is specific to the reviewed formal 0.32 pair")
    commit = os.environ.get(SOURCE_ENV, "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Missing full upgrade source commit")
    if os.environ.get("OMNI032_H3_SR_PARENT_IMAGE") != value["parent_image"]:
        raise ValueError("Unexpected upgrade parent")
    return value


def safe(root, relative):
    name = PurePosixPath(relative)
    if name.is_absolute() or ".." in name.parts or "\\" in relative or ":" in relative:
        raise ValueError("Unsafe source path: " + relative)
    result = root.joinpath(*name.parts)
    if result.is_symlink() or not result.resolve().is_relative_to(root.resolve()):
        raise ValueError("Source escaped installed root: " + relative)
    return result


def installed():
    dist = md.distribution("vllm-omni")
    root = Path(dist.locate_file("")).resolve()
    records = [Path(dist.locate_file(p)).resolve() for p in dist.files or []
               if str(p).endswith(".dist-info/RECORD")]
    if len(records) != 1:
        raise ValueError("Expected one installed Omni RECORD")
    return dist, root, records[0]


def sources(root):
    return {p.relative_to(root).as_posix(): digest(safe(root, p.relative_to(root).as_posix()).read_bytes())
            for p in sorted((root / "vllm_omni").rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")}


def wheel_sources():
    wheel = INFO / "environment/wheels" / WHEEL_NAME
    if digest(wheel.read_bytes()) != policy()["omni_wheel_sha256"]:
        raise ValueError("Released Omni wheel differs from source lock")
    with zipfile.ZipFile(wheel) as archive:
        values = {name: archive.read(name) for name in archive.namelist()
                  if name.startswith("vllm_omni/") and not name.endswith("/")}
    if not values:
        raise ValueError("Released wheel contains no Omni sources")
    for name in values:
        safe(Path("/tmp/omni032-safe-root"), name)
    return values


def preserved_snapshot():
    # Independent of importlib's transient vendored sys.path entries. Only the
    # installed service source and historical evidence are compared here;
    # environment.py separately owns metadata/native ABI preservation.
    root = Path(md.distribution("ai-dingo").locate_file("")).resolve()
    dingo = root / "dingo"
    if not dingo.is_dir():
        raise ValueError("Inherited installed Dingo tree is absent")
    result = {"dingo/" + p.relative_to(dingo).as_posix(): digest(p.read_bytes())
              for p in sorted(dingo.rglob("*")) if p.is_file()
              and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")}
    for directory in OLD_DIRS:
        folder = HERE.parent / directory
        if not folder.is_dir():
            raise ValueError("Required inherited repair evidence is absent: " + directory)
        for path in sorted(folder.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo"):
                result["opt/" + directory + "/" + path.relative_to(folder).as_posix()] = digest(path.read_bytes())
    return result


def parent(output):
    if os.geteuid() != 0 or output.resolve() != INFO.resolve() or output.exists():
        raise ValueError("Parent verification is a create-once root build action")
    value = policy()
    code = ("import sys,json; sys.path.insert(0,'/opt/seedvr2-metrics-repair'); "
            "import runtime; print(json.dumps(runtime.verify_installed()))")
    result = subprocess.run([sys.executable, "-B", "-c", code], check=False,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode:
        raise RuntimeError("Actual inherited runtime proof failed: " + result.stderr[-8000:])
    proof = json.loads(result.stdout)
    archive = read(HERE / "inputs/parent-source-manifest.json")
    if (proof.get("status") != "PASS" or proof.get("source_sha256") != archive["source_sha256"]
            or map_digest(proof["source_sha256"]) != value["parent_source_manifest_sha256"]):
        raise ValueError("Actual inherited Omni sources differ from published b315")
    output.mkdir()
    dump(output / "parent-source-verification.json", proof)
    dump(output / "preserved-before.json", preserved_snapshot())
    dump(output / "parent-verified.json", {"status": "PASS", "parent_image": value["parent_image"],
         "source_commit": os.environ[SOURCE_ENV], "scope": "Actual published parent source identity before installation"})


def prepare_changes(root, pristine):
    h3 = load("_omni032_h3_source", HERE / "h3_patches.py")
    sr = load("_omni032_sr_source", HERE / "sr_patches.py")
    changes = sr.locked_changes(root)
    for name in h3.CHANGES:
        before = safe(root, name).read_bytes()
        changes[name] = (before, h3.transform(name, before))
    additions = sr.additions()
    helper = (HERE.parent / "h3-hsdp-offload/hsdp_migration.py").read_bytes()
    if digest(helper) != h3.HELPER_SHA256:
        raise ValueError("Inherited reviewed HSDP helper changed")
    additions[h3.HELPER] = helper
    if set(changes) & set(additions):
        raise ValueError("Duplicate source ownership in the migration")
    expected = {name: digest(data) for name, data in pristine.items()}
    for name, (before, after) in changes.items():
        if expected.get(name) != digest(before):
            raise ValueError("Change does not originate in the released wheel: " + name)
        compile(after, name, "exec")
        expected[name] = digest(after)
    for name, data in additions.items():
        if name in expected or safe(root, name).exists():
            raise ValueError("Helper addition would overwrite existing source: " + name)
        compile(data, name, "exec")
        expected[name] = digest(data)
    return changes, additions, expected


def record_update(data, changes, additions):
    rows = list(csv.reader(io.StringIO(data.decode())))
    counts = {}
    for row in rows:
        if len(row) != 3:
            raise ValueError("Malformed wheel RECORD")
        counts[row[0]] = counts.get(row[0], 0) + 1
    for name in set(changes) | set(additions):
        if counts.get(name, 0) != int(name in changes):
            raise ValueError("Unexpected RECORD source ownership: " + name)
    for row in rows:
        if row[0] in changes:
            before, after = changes[row[0]]
            old = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(before).digest()).rstrip(b"=").decode()
            if row[1:] != [old, str(len(before))]:
                raise ValueError("Released RECORD hash differs before patch: " + row[0])
            row[1:] = ["sha256=" + base64.urlsafe_b64encode(hashlib.sha256(after).digest()).rstrip(b"=").decode(), str(len(after))]
    for name, data in additions.items():
        rows.append([name, "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode(), str(len(data))])
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerows(rows)
    return stream.getvalue().encode()


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".omni032-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    try:
        temporary.chmod(path.stat().st_mode & 0o777 if path.exists() else 0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def apply(output):
    if os.geteuid() != 0 or output.resolve() != INFO.resolve() or (INFO / "application.json").exists():
        raise ValueError("Patch application is a create-once root build action")
    value = policy()
    if read(INFO / "parent-verified.json").get("source_commit") != os.environ[SOURCE_ENV]:
        raise ValueError("Missing actual parent verification for this source commit")
    dist, root, record = installed()
    if dist.version != "0.32.0" or md.version("vllm") != "0.32.0":
        raise ValueError("Install the full released framework pair before applying source patches")
    pristine = wheel_sources()
    archived = read(HERE / "inputs/parent-source-manifest.json")["source_sha256"]
    actual = sources(root)
    removed = {}
    # pip only removes files present in the old RECORD. Remove a stale helper
    # only when its exact published parent hash is archived and target release
    # does not own it. Unknown or modified files always block the build.
    for name in sorted(set(actual) - set(pristine)):
        if archived.get(name) != actual[name]:
            raise ValueError("Unknown file survived the released installation: " + name)
        removed[name] = actual[name]
        safe(root, name).unlink()
    released = {name: digest(data) for name, data in pristine.items()}
    if sources(root) != released:
        raise ValueError("Actual fresh Omni installation differs from the released wheel")
    changes, additions, expected = prepare_changes(root, pristine)
    before_record = record.read_bytes()
    after_record = record_update(before_record, changes, additions)
    dump(INFO / "released-source-manifest.json", released)
    (INFO / "omni-RECORD.before.csv").write_bytes(before_record)
    for name, (_, after) in changes.items():
        atomic_write(safe(root, name), after)
    for name, data in additions.items():
        atomic_write(safe(root, name), data)
    atomic_write(record, after_record)
    if sources(root) != expected or preserved_snapshot() != read(INFO / "preserved-before.json"):
        raise ValueError("Patch application changed unreviewed framework or inherited service sources")
    application = {"schema": "omni032-h3-sr-application/1", "status": "PASS", "parent_image": value["parent_image"],
        "source_commit": os.environ[SOURCE_ENV], "changes": sorted(changes), "additions": sorted(additions),
        "removed_known_stale_sources": removed, "source_sha256": expected,
        "source_manifest_sha256": map_digest(expected), "omni_record_after_sha256": digest(after_record),
        "gpu_validation_status": "NOT_RUN"}
    dump(INFO / "application.json", application)
    proof = verify_installed()
    dump(INFO / "source-verification.json", proof)
    return {k: v for k, v in application.items() if k != "source_sha256"}


def verify_installed():
    value = policy()
    dist, root, record = installed()
    receipt = read(INFO / "application.json")
    actual = sources(root)
    sr = load("_omni032_sr_verified", HERE / "sr_patches.py")
    expected = sr.expected_sources({name: digest(data) for name, data in wheel_sources().items()})
    h3_lock = read(HERE / "h3-patches.lock.json")
    for name, entry in h3_lock["changes"].items():
        if expected.get(name) != entry["before_sha256"]:
            raise ValueError("H3 patch source does not belong to this released wheel: " + name)
        expected[name] = entry["after_sha256"]
    for name, value_hash in h3_lock["unchanged_reference_sha256"].items():
        if expected.get(name) != value_hash:
            raise ValueError("Unchanged native HSDP source differs: " + name)
    for name, value_hash in h3_lock["additions"].items():
        if name in expected:
            raise ValueError("H3 helper collides with released source: " + name)
        expected[name] = value_hash
    if dist.version != "0.32.0" or md.version("vllm") != "0.32.0":
        raise ValueError("The actual installed framework pair changed")
    if (actual != expected or receipt.get("source_commit") != os.environ[SOURCE_ENV] or receipt.get("source_sha256") != actual
            or receipt.get("source_manifest_sha256") != map_digest(actual)
            or receipt.get("omni_record_after_sha256") != digest(record.read_bytes())):
        raise ValueError("Installed source identity or wheel RECORD changed")
    if preserved_snapshot() != read(INFO / "preserved-before.json"):
        raise ValueError("Inherited Dingo or historical repair evidence changed")
    return {"schema": "omni032-h3-sr-source-proof/1", "status": "PASS", "parent_image": value["parent_image"],
        "source_commit": os.environ[SOURCE_ENV], "source_install_root": str(root), "source_sha256": actual,
        "source_manifest_sha256": map_digest(actual), "changed_sources": receipt["changes"],
        "added_sources": receipt["additions"], "gateway_commit": value["gateway_commit"],
        "upstream_omni_commit": value["omni_release_commit"], "metrics_schema": "sr-timing-memory/2"}


def rmsnorm(output):
    directory = HERE.parent / "omni-gateway-lite/rmsnorm"
    sys.path.insert(0, str(directory))
    try:
        numerical = load("_omni032_norm_numeric", directory / "validate_runtime.py")
        norm = importlib.import_module("vllm_omni.diffusion.models.seedvr2.nadit")
        source = "vllm_omni/diffusion/models/seedvr2/nadit.py"
        expected = read(HERE / "sr-patches.lock.json")["absorbed"][source]["sha256"]
        if digest(Path(norm.__file__).read_bytes()) != expected:
            raise ValueError("Actual absorbed upstream RMSNorm import differs")
        import torch
        output.mkdir(parents=True, exist_ok=True)
        cases, all_300 = numerical.numeric_cases(norm.RMSNorm, torch, "cpu")
        numerical.run_upstream_tests(output, numerical.lock_data())
        if len(cases) != 6:
            raise ValueError("All six RMSNorm precision/affine cases are required")
        result = {"status": "PASS", "six_case_count": 6, "six_cases": cases, "all_300": all_300,
            "module_sha256": expected, "source_commit": os.environ[SOURCE_ENV], "device": "cpu",
            "scope": "Absorbed official PR8767; actual installed 0.32 RMSNorm, no repeat patch"}
        dump(output / "runtime-validation.json", result)
        return result
    finally:
        sys.path.pop(0)


def run(script, arguments, directory):
    directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-B", str(script), *map(str, arguments)]
    with (directory / (script.stem + ".log")).open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        detail = (directory / (script.stem + ".log")).read_text(encoding="utf-8", errors="replace")[-20000:]
        print(detail, flush=True)
        raise RuntimeError(script.name + " failed: " + str(result.returncode))


def validate(output):
    if os.geteuid() != 1000:
        raise ValueError("Final runtime validation must use inherited UID1000")
    value = policy()
    output.mkdir(parents=True, exist_ok=True)
    proof = verify_installed()
    dump(output / "source-verification.json", proof)
    for name in proof["changed_sources"] + proof["added_sources"]:
        module = importlib.import_module(name[:-3].replace("/", "."))
        if Path(module.__file__).resolve() != (Path(proof["source_install_root"]) / name).resolve():
            raise ValueError("Installed module is shadowed: " + name)
    native = load("_omni032_torch_contract", HERE.parent / "h3-hsdp-offload/torch_contract.py")
    dump(output / "torch-contract.json", native.verify())
    rmsnorm(output / "rmsnorm")
    run(HERE / "environment.py", ["audit", "--output-dir", output / "environment", "--inputs", INFO / "environment"], output)
    for name in ("media_smoke.py", "service_smoke.py"):
        run(HERE / "legacy-environment" / name, ["--output-dir", output / "environment"], output)
    run(HERE / "native_cli_smoke.py", ["--output-dir", output / "native-cli"], output)
    cli = read(output / "native-cli/native-cli.json")
    if (cli.get("status") != "PASS" or cli.get("source_commit") != os.environ[SOURCE_ENV]
            or any(cli.get(key) is not False for key in ("dispatch_executed", "engine_started", "model_loaded"))
            or cli.get("unknown_options_rejected") is not True
            or not re.fullmatch(r"[0-9a-f]{64}", cli.get("argv_contract_sha256", ""))
            or cli["argv_contract_sha256"] != digest((HERE / "native-cli-argv.json").read_bytes())
            or not isinstance(cli.get("profiles"), list) or len(cli["profiles"]) < 2
            or any(p.get("status") != "PASS" for p in cli["profiles"])):
        raise ValueError("Complete native CLI contract did not pass without engine/model execution")
    run(HERE / "cpu_smoke.py", ["--output-dir", output / "cpu"], output)
    regression = read(output / "cpu/regression.json")
    if (regression.get("status") != "PASS" or regression.get("tests_run", 0) < 1
            or any(regression.get(k) != 0 for k in ("errors", "failures", "skipped"))):
        raise ValueError("Required actual-image CPU regressions are incomplete")
    legal = load("_omni032_legal_validation", HERE / "legal.py").validate(INFO)
    dump(output / "legal-validation.json", legal)
    if verify_installed() != proof:
        raise ValueError("Runtime validation mutated installed sources")
    packages = {key: md.version(key) for key in dict.fromkeys((*value["versions"], "diffusers", "transformers", "tokenizers", "torchvision", "torchaudio", "triton"))}
    result = {"schema": "omni032-h3-sr-runtime/1", "status": "PASS_CPU_ONLY", "euid": os.geteuid(),
        "source_commit": os.environ[SOURCE_ENV], "parent_image": value["parent_image"],
        "source_manifest_sha256": proof["source_manifest_sha256"], "gateway_commit": value["gateway_commit"],
        "versions": packages, "runtime_cohort": {"python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "cuda": importlib.import_module("torch").version.cuda, "packages": packages}, "metrics_schema": "sr-timing-memory/2",
        "legal_status": legal["status"], "legal_manifest_sha256": legal["legal_manifest_sha256"],
        "native_cli_status": cli["status"], "native_cli_argv_contract_sha256": cli["argv_contract_sha256"],
        "gpu_validation_status": "NOT_RUN", "scope": "Actual package/import/native-source/CPU/media gates; no H100 inference or quality claim"}
    dump(output / "runtime-validation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("parent", "prepare", "apply", "verify", "validate"))
    parser.add_argument("--output-dir", type=Path, default=INFO)
    args = parser.parse_args()
    actions = {"parent": parent, "prepare": parent, "apply": apply, "validate": validate}
    result = verify_installed() if args.action == "verify" else actions[args.action](args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
