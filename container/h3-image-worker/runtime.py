#!/usr/bin/env python3
"""Apply two source-bound repairs and qualify the inherited runtime on CPU."""
from __future__ import annotations

import argparse
import base64
import copy
import csv
import hashlib
import importlib
import importlib.metadata as metadata
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs"
BUILD = HERE / "build-info"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def hash_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def probe():
    result = subprocess.run([sys.executable, "-I", "-B", str(HERE / "inventory.py"),
                             "--source-manifest", str(INPUTS / "parent-source-manifest.json")],
                            capture_output=True, text=True, encoding="utf-8", timeout=180)
    if result.returncode:
        raise RuntimeError("Installed inventory probe failed (exit " + str(result.returncode) + ")\n"
                           + result.stdout[-6000:] + result.stderr[-12000:])
    return json.loads(result.stdout)


def expected_sources(patched):
    files = dict(read(INPUTS / "parent-source-manifest.json")["files"])
    if patched:
        files.update({item["path"]: item["patched_sha256"] for item in read(HERE / "patches.lock.json")["patches"]})
    return files


def snapshot():
    inventory = probe()
    expected = expected_sources(False)
    if any(expected.get(name) != digest for name, digest in inventory["source_sha256"].items()):
        raise ValueError("Installed parent source differs from its fully verified source layer")
    dump(BUILD / "parent-inventory.json", inventory)
    return {"status": "PASS_PARENT_SOURCE", "files": len(inventory["source_sha256"])}


def apply():
    before = read(BUILD / "parent-inventory.json")
    if probe() != before:
        raise ValueError("Parent changed before repairs were applied")
    distribution = metadata.distribution("vllm-omni")
    patches = read(HERE / "patches.lock.json")["patches"]
    payloads = {}
    for patch in patches:
        path = Path(distribution.locate_file(patch["path"]))
        payload = (HERE / patch["payload"]).read_bytes()
        if hashlib.sha256(path.read_bytes()).hexdigest() != patch["parent_sha256"]:
            raise ValueError("Patch parent differs: " + patch["path"])
        if hashlib.sha256(payload).hexdigest() != patch["patched_sha256"]:
            raise ValueError("Patch payload differs: " + patch["path"])
        payloads[patch["path"]] = (path, payload)
    records = [item for item in distribution.files if str(item).endswith(".dist-info/RECORD")]
    if len(records) != 1:
        raise ValueError("One installed Omni RECORD is required")
    record_path = Path(distribution.locate_file(records[0]))
    rows = list(csv.reader(io.StringIO(record_path.read_text(encoding="utf-8"))))
    if any(sum(row[0] == name for row in rows) != 1 for name in payloads):
        raise ValueError("Both repaired modules must occur exactly once in RECORD")
    for name, (path, payload) in payloads.items():
        path.write_bytes(payload)
        encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode().rstrip("=")
        for row in rows:
            if row[0] == name:
                row[1:] = ["sha256=" + encoded, str(len(payload))]
    with record_path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)
    after = probe()
    expected_packages = dict(before["packages"])
    expected_omni = dict(expected_packages["vllm-omni"])
    expected_omni["record_sha256"] = after["packages"]["vllm-omni"]["record_sha256"]
    expected_packages["vllm-omni"] = expected_omni
    expected_distributions = [dict(item) for item in before["distributions"]]
    for item in expected_distributions:
        if item["metadata_directory"] == expected_omni["metadata_directory"]:
            item["record_sha256"] = expected_omni["record_sha256"]
    patched_sources = expected_sources(True)
    projected = {name: patched_sources[name] for name in before["source_sha256"]}
    if (after["packages"] != expected_packages or after["distributions"] != expected_distributions
            or after["source_sha256"] != projected
            or after["source_coverage"] != before["source_coverage"]
            or after["generated_source_sha256"] != before["generated_source_sha256"]):
        raise ValueError("Unexpected package or source change during the two repairs")
    dump(BUILD / "expected-inventory.json", after)
    dump(BUILD / "patch-application.json", {"status": "PASS", "patches": patches,
         "source_manifest_sha256": hash_value(after["source_sha256"]),
         "record_update": "Only repaired module hash and size rows changed"})
    return {"status": "PASS_TWO_SOURCE_REPAIRS", "files": len(patches)}


def run_script(script, arguments, output, name):
    result = subprocess.run([sys.executable, "-P", "-B", str(script), *arguments],
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    (output / name).write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        print(result.stdout + result.stderr, flush=True)
        raise RuntimeError(f"{script.name} failed with exit {result.returncode}")


def check_cli():
    serve = importlib.import_module("vllm_omni.entrypoints.cli.serve")
    serve._ensure_vllm_platform()
    parser = serve.TrackingArgumentParser()
    subcommands = parser.add_subparsers(dest="subparser")
    command = next(item for item in serve.cmd_init() if item.name == "serve")
    command.subparser_init(subcommands)
    engine = importlib.import_module("vllm_omni.engine.arg_utils").OmniEngineArgs
    results = []
    profiles = read(HERE / "profiles.json")
    # EngineArgs.__post_init__ resolves model/tokenizer paths in HF offline mode.
    # This parameter-contract gate has no weights: use a real empty local directory
    # only for dataclass construction, after asserting the actual CLI model mapping.
    with tempfile.TemporaryDirectory(prefix="h3-image-cli-contract-") as fixture:
        for profile in profiles["profiles"]:
            namespace = parser.parse_args(profile["argv"][1:])
            if getattr(namespace, "model_tag", None) != profiles["model_root"]:
                raise ValueError("CLI positional model differs from the frozen path")
            namespace.model = namespace.model_tag  # Same precedence as OmniServeCommand.cmd.
            if getattr(namespace, "tokenizer", None) is not None:
                raise ValueError("The frozen diffusion profile must not resolve a tokenizer")
            fixture_namespace = copy.deepcopy(namespace)
            fixture_namespace.model = fixture
            arguments = engine.from_cli_args(fixture_namespace)
            if namespace.model != profiles["model_root"] or namespace.model_tag != profiles["model_root"]:
                raise ValueError("CLI fixture mutated the original positional model namespace")
            if arguments.model != fixture or getattr(arguments, "tokenizer", None) is not None:
                raise ValueError("CLI dataclass did not retain its isolated local model fixture")
            for name, expected in (("num_gpus", profile["gpus"]), ("tensor_parallel_size", profile["dit_tp"]),
                                   ("text_encoder_tp_size", profile["te_tp"]), ("ulysses_degree", profile["sp"])):
                if getattr(arguments, name) != expected:
                    raise ValueError("CLI/dataclass dropped " + name + " for " + profile["id"])
            for name in ("diffusion_offload_config", "additional_config", "enable_multithread_weight_load"):
                if getattr(arguments, name) != getattr(namespace, name):
                    raise ValueError("CLI/dataclass changed " + name)
            if namespace.enable_multithread_weight_load is not False:
                raise ValueError("The frozen profile must disable parallel weight loading")
            results.append({"profile": profile["id"], "status": "PASS", "gpus": profile["gpus"],
                            "dit_tp": profile["dit_tp"], "te_tp": profile["te_tp"]})
    return results


def check_lmcache_native(output):
    """Strict CPU loader check of the active LMCache distribution and locked ELF payloads."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "STARTED_NATIVE_CHECK", "files": [], "imports": [],
              "gpu_validation": "NOT_RUN", "driver_error_skips": False}
    dump(output / "native-validation.json", report)
    try:
        lock = read(HERE / "native.lock.json")
        if (lock.get("schema") != "h3-image-worker-native-artifacts/1"
                or lock.get("distribution") != "lmcache"
                or lock.get("gpu_entrypoint") != "lmcache.cuda_ops"):
            raise ValueError("Unknown LMCache native artifact contract")
        items = lock.get("files", [])
        if len(items) != 4 or len({item["path"] for item in items}) != 4:
            raise ValueError("Exactly four distinct locked LMCache native artifacts are required")
        prefixes = {"cuda_ops", "lmcache_fs", "lmcache_native", "lmcache_redis"}
        if {Path(item["path"]).name.split(".")[0] for item in items} != prefixes:
            raise ValueError("Unknown LMCache native artifact name")
        distribution = metadata.distribution("lmcache")
        if distribution.version != lock["version"]:
            raise ValueError("Active LMCache distribution version differs from native.lock")
        package_root = Path(distribution.locate_file("lmcache")).resolve()
        recorded_files = {str(item).replace("\\", "/") for item in distribution.files or ()}
        installed_native = {name for name in recorded_files if name.startswith("lmcache/") and name.endswith(".so")}
        locked = {item["path"] for item in items}
        if installed_native != locked:
            raise ValueError("Installed LMCache native selection differs from native.lock: "
                             + json.dumps({"missing": sorted(locked - installed_native),
                                           "unknown": sorted(installed_native - locked)}))
        filesystem_native = {"lmcache/" + path.relative_to(package_root).as_posix()
                             for path in package_root.rglob("*.so")}
        if filesystem_native != locked:
            raise ValueError("Actual LMCache native files differ from native.lock: "
                             + json.dumps({"missing": sorted(locked - filesystem_native),
                                           "unknown": sorted(filesystem_native - locked)}))
        record_text = distribution.read_text("RECORD")
        if not isinstance(record_text, str):
            raise ValueError("Active LMCache distribution RECORD is absent")
        rows = list(csv.reader(io.StringIO(record_text)))
        expected_paths = {}
        for item in items:
            name = item["path"]
            relative = Path(name)
            if (relative.is_absolute() or ".." in relative.parts or "\\" in name
                    or not name.startswith("lmcache/") or not name.endswith(".so")):
                raise ValueError("Unsafe native artifact path: " + name)
            wanted = item.get("sha256")
            size = item.get("bytes")
            if (not isinstance(wanted, str) or len(wanted) != 64
                    or any(c not in "0123456789abcdef" for c in wanted)
                    or type(size) is not int or size <= 0):
                raise ValueError("Invalid native artifact identity: " + name)
            path = Path(distribution.locate_file(name)).resolve()
            if not path.is_relative_to(package_root) or not path.is_file():
                raise ValueError("Native artifact is outside the active distribution: " + name)
            if path.stat().st_size != size:
                raise ValueError("Native artifact byte count differs: " + name)
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(8 * 1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != wanted:
                raise ValueError("Native artifact SHA256 differs: " + name)
            matched = [row for row in rows if row and row[0] == name]
            encoded = base64.urlsafe_b64encode(digest.digest()).decode("ascii").rstrip("=")
            if len(matched) != 1 or matched[0][1:] != ["sha256=" + encoded, str(size)]:
                raise ValueError("Native artifact RECORD hash/size row differs: " + name)
            expected_paths[name] = path
            report["files"].append({"path": name, "resolved_path": str(path),
                                    "sha256": digest.hexdigest(), "bytes": size,
                                    "record_verified": True})
        # Load Torch first for libtorch dependencies, without requiring a CUDA driver.
        # Any actual loader/driver/import error below is fatal; no skip allowlist.
        for name, distribution_name in (("torch", "torch"), ("lmcache", "lmcache")):
            module = importlib.import_module(name)
            active = metadata.distribution(distribution_name)
            active_root = Path(active.locate_file(name)).resolve()
            origin = Path(module.__file__).resolve()
            spec_origin = getattr(getattr(module, "__spec__", None), "origin", None)
            if not spec_origin or Path(spec_origin).resolve() != origin:
                raise ValueError("Import __file__/__spec__.origin differs: " + name)
            if not origin.is_relative_to(active_root):
                raise ValueError("Import did not use the active distribution: " + name)
            report["imports"].append({"module": name, "origin": str(origin)})
        for name in ("lmcache.lmcache_native", "lmcache.cuda_ops"):
            basename = name.split(".")[-1] + "."
            matches = [path for relative, path in expected_paths.items()
                       if Path(relative).name.startswith(basename)]
            if len(matches) != 1:
                raise ValueError("Native import has no unique locked origin: " + name)
            module = importlib.import_module(name)
            origin = Path(module.__file__).resolve()
            spec_origin = getattr(getattr(module, "__spec__", None), "origin", None)
            if not spec_origin or Path(spec_origin).resolve() != origin or origin != matches[0]:
                raise ValueError("Native import did not use the active locked artifact: " + name)
            report["imports"].append({"module": name, "origin": str(matches[0]), "locked_origin_verified": True})
        report.update(status="PASS_STRICT_NATIVE_CPU_LOADER",
                      distribution="lmcache", version=distribution.version,
                      distribution_root=str(package_root),
                      native_lock_sha256=hashlib.sha256((HERE / "native.lock.json").read_bytes()).hexdigest())
    except Exception as exc:
        report.update(status="FAILED_NATIVE_CHECK", error_type=type(exc).__name__,
                      error=str(exc), traceback=traceback.format_exc())
        dump(output / "native-validation.json", report)
        (output / "native-failure.txt").write_text(report["traceback"], encoding="utf-8")
        raise
    dump(output / "native-validation.json", report)
    return report



def validate(output):
    output.mkdir(parents=True, exist_ok=True)
    value = read(INPUTS / "release-lock.json")
    current = probe()
    if os.geteuid() != 1000 or current != read(BUILD / "expected-inventory.json"):
        raise ValueError("Expected UID1000 and exact repaired parent inventory")
    versions = {name: metadata.version(name) for name in value["versions"]}
    if versions != value["versions"]:
        raise ValueError("Protected package versions differ: " + json.dumps(versions))
    for tool in ([sys.executable, "-m", "pip", "check"], ["uv", "--no-cache", "pip", "check", "--python", sys.executable]):
        result = subprocess.run(tool, capture_output=True, text=True, encoding="utf-8", errors="replace")
        name = "uv-pip-check.txt" if tool[0] == "uv" else "pip-check.txt"
        (output / name).write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            print(result.stdout + result.stderr, flush=True)
            raise RuntimeError("Strict installed dependency check failed: " + name)
    for name in ("torch", "vllm", "vllm_omni"):
        module = importlib.import_module(name)
        distribution_name = "vllm-omni" if name == "vllm_omni" else name.split(".")[0]
        distribution = metadata.distribution(distribution_name)
        root = Path(distribution.locate_file(name.split(".")[0])).resolve()
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise ValueError("Import did not use the active distribution: " + name)
    run_script(Path(__file__), ["native-check", "--output-dir", str(output)], output, "native-checks.txt")
    native = read(output / "native-validation.json")
    if native.get("status") != "PASS_STRICT_NATIVE_CPU_LOADER":
        raise ValueError("Strict native CPU loader check did not pass")
    import torch
    if torch.__version__ != value["versions"]["torch"] or torch.version.cuda != "13.0":
        raise ValueError("Inherited Torch/CUDA ABI version differs")
    cli = check_cli()
    dump(output / "cli-validation.json", {"status": "PASS", "profiles": cli,
          "model_path_isolation": "Original positional model asserted; empty temporary local directory used only for dataclass path resolution",
          "scope": "Actual parser and engine dataclass only; no weights, serve.validate, model loading or GPU"})
    run_script(HERE / "cpu_checks.py", ["--output-dir", str(output)], output, "cpu-checks.txt")
    checks = read(output / "cpu-checks.json")
    if (checks.get("status") != "PASS_CPU_RUNTIME_ONLY" or checks.get("tests_run", 0) < 10
            or any(checks.get(name) for name in ("failures", "errors", "skipped"))):
        raise ValueError("Installed-module CPU checks are incomplete")
    run_script(Path(__file__), ["local-tests", "--output-dir", str(output)], output, "worker-tests.txt")
    # Imports may add vendored sys.path entries. Probe the real installation freshly.
    if probe() != current:
        raise ValueError("The final installed runtime changed during CPU qualification")
    commit = os.environ["H3_IMAGE_WORKER_SOURCE_COMMIT"]
    proof = {"status": "PASS", "source_commit": commit, "parent_image": value["parent_image"],
             "source_sha256": current["source_sha256"],
             "generated_source_sha256": current["generated_source_sha256"],
             "source_coverage": current["source_coverage"],
             "source_manifest_sha256": hash_value(current["source_sha256"]),
             "parent_copied_git_head": value["parent_source_git_head"],
             "omni_version": current["omni_version"], "omni_root": current["omni_root"],
             "omni_direct_url": current["omni_direct_url"], "patches": read(HERE / "patches.lock.json")}
    dump(output / "source-verification.json", proof)
    dump(output / "installed-inventory.json", current)
    components = []
    for distribution in metadata.distributions():
        name = distribution.metadata["Name"]
        components.append({"type": "library", "name": name, "version": distribution.version,
                           "purl": "pkg:pypi/" + name.lower().replace("_", "-") + "@" + distribution.version,
                           "properties": [{"name": "declared-license", "value": distribution.metadata.get("License-Expression")
                                           or distribution.metadata.get("License") or "UNKNOWN"}]})
    dump(output / "packages.cdx.json", {"bomFormat": "CycloneDX", "specVersion": "1.5", "version": 1,
                                       "components": sorted(components, key=lambda item: item["name"].lower())})
    report = {"status": "PASS_CPU_ONLY", "source_commit": commit,
              "parent_image": value["parent_image"], "parent_image_id": value["parent_image_id"],
              "euid": os.geteuid(), "inventory_preserved": True,
              "preservation_scope": "All versions/METADATA/direct_url/native packages retained; two source files and their RECORD rows repaired",
              "source_manifest_sha256": proof["source_manifest_sha256"], "versions": versions,
              "cpu_checks": checks, "cli_profiles": cli, "native_validation": native,
              "errors": [], "gpu_validation": "NOT_RUN"}
    dump(output / "runtime-validation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("snapshot", "apply", "validate", "local-tests", "native-check"))
    parser.add_argument("--output-dir", type=Path, default=BUILD / "validation")
    args = parser.parse_args()
    if args.action == "native-check":
        print(json.dumps(check_lmcache_native(args.output_dir), indent=2))
        return
    if args.action == "local-tests":
        result = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", str(HERE / "tests")],
                                text=True, encoding="utf-8")
        raise SystemExit(result.returncode)
    result = {"snapshot": snapshot, "apply": apply}.get(args.action)
    report = result() if result else validate(args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
