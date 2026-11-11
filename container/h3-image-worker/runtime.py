#!/usr/bin/env python3
"""Apply two source-bound repairs and qualify the inherited runtime on CPU."""
from __future__ import annotations

import argparse
import base64
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
                            capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(result.stdout)


def expected_sources(patched):
    files = dict(read(INPUTS / "parent-source-manifest.json")["files"])
    if patched:
        files.update({item["path"]: item["patched_sha256"] for item in read(HERE / "patches.lock.json")["patches"]})
    return files


def snapshot():
    inventory = probe()
    if inventory["source_sha256"] != expected_sources(False):
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
    if after["packages"] != expected_packages or after["source_sha256"] != expected_sources(True):
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
    for profile in read(HERE / "profiles.json")["profiles"]:
        namespace = parser.parse_args(profile["argv"][1:])
        arguments = engine.from_cli_args(namespace)
        for name, expected in (("num_gpus", profile["gpus"]), ("tensor_parallel_size", profile["dit_tp"]),
                               ("text_encoder_tp_size", profile["te_tp"]), ("ulysses_degree", profile["sp"])):
            if getattr(arguments, name) != expected:
                raise ValueError("CLI/dataclass dropped " + name + " for " + profile["id"])
        for name in ("diffusion_offload_config", "additional_config", "disable_multithread_weight_load"):
            if getattr(arguments, name) != getattr(namespace, name):
                raise ValueError("CLI/dataclass changed " + name)
        results.append({"profile": profile["id"], "status": "PASS", "gpus": profile["gpus"],
                        "dit_tp": profile["dit_tp"], "te_tp": profile["te_tp"]})
    return results


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
    for name in ("torch", "vllm", "vllm_omni", "lmcache.c_ops"):
        importlib.import_module(name)
    import torch
    if torch.__version__ != value["versions"]["torch"] or torch.version.cuda != "13.0":
        raise ValueError("Inherited Torch/CUDA ABI version differs")
    cli = check_cli()
    dump(output / "cli-validation.json", {"status": "PASS", "profiles": cli,
          "scope": "Actual parser and engine dataclass only; no GPU model loading"})
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
              "cpu_checks": checks, "cli_profiles": cli, "errors": [], "gpu_validation": "NOT_RUN"}
    dump(output / "runtime-validation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("snapshot", "apply", "validate", "local-tests"))
    parser.add_argument("--output-dir", type=Path, default=BUILD / "validation")
    args = parser.parse_args()
    if args.action == "local-tests":
        result = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", str(HERE / "tests")],
                                text=True, encoding="utf-8")
        raise SystemExit(result.returncode)
    result = {"snapshot": snapshot, "apply": apply}.get(args.action)
    report = result() if result else validate(args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
