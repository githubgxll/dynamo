#!/usr/bin/env python3
"""Offline candidate gates and exact-image publication for the adapter-free recipe."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = Path("delivery/seedvr2-c14/source-lock.json")
BRANCH = "Dingorouter-H3-vllm-omni-jirx"
REPOSITORY = "registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm"
PARENT = REPOSITORY + "@sha256:9382949d2789b23cdfe4ea71d68c7973592de28f209082bd7907c1d09fe008c9"
PARENT_COMMIT = "2362a3241c03a286b98e51b8003a51f494fab4f1"
PARENT_IMAGE_ID = "sha256:a7ec5a64808d76b0104ea078e462ee466d51f364057790db5b7640ef1cc2e5d7"
FRAMEWORK_COMMIT = "db65d1ec869b6deee02debe7a58e108de7c927ef"
GATEWAY_BEFORE = "9ccee2f56d0122be94d7436ae2f5fec96d15ea9b"
GATEWAY = "9ccee2f56d0122be94d7436ae2f5fec96d15ea9b"
UPSTREAM = "0708f05e2e27f16fca8b48e7c5a085277d1cafda"
BEFORE = "03fd3a0c199305ce3004643f8d1510183b06731099a72bc447c2bd589c1a6f01"
AFTER = "03fd3a0c199305ce3004643f8d1510183b06731099a72bc447c2bd589c1a6f01"
BUILD_INFO = "/opt/seedvr2-c14/build-info"
CONFIG_FIELDS = ("User", "Entrypoint", "Cmd", "WorkingDir", "ExposedPorts", "Volumes", "Healthcheck", "StopSignal", "Shell", "OnBuild")

INVENTORY_PROBE = r'''
import hashlib, importlib.metadata as m, json, os, pathlib, subprocess
dist = m.distribution("vllm-omni")
source = pathlib.Path(dist.locate_file("vllm_omni/diffusion/models/seedvr2/nadit.py")).resolve()
packages = sorted((d.metadata["Name"], d.version) for d in m.distributions())
dpkg = subprocess.run(["dpkg-query", "-W", "-f=${binary:Package}\t${Version}\n"], check=True, capture_output=True, text=True)
print(json.dumps({"packages": packages, "dpkg": sorted(dpkg.stdout.splitlines()),
 "module_path": str(source), "module_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
 "source_commit": os.environ.get("H3_C14_SOURCE_COMMIT"),
 "lite_source_commit": os.environ.get("H3_EXPERIMENTAL_SOURCE_COMMIT"),
 "dynamo_commit": os.environ.get("DYNAMO_COMMIT_SHA"),
 "gateway_commit": os.environ.get("DYNAMO_GATEWAY_SOURCE_SHA"),
 "adapters_present": pathlib.Path("/opt/h3-adapters").exists(), "euid": os.geteuid()}))
'''


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(environ):
    commit = environ.get("GITHUB_SHA", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Expected full GITHUB_SHA")
    if environ.get("GITHUB_REF_NAME") != BRANCH or environ.get("GITHUB_EVENT_NAME") not in {"push", "workflow_dispatch"}:
        raise ValueError("Use push or dispatch on the existing H3 branch")
    for key in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT"):
        if not re.fullmatch(r"[1-9][0-9]*", environ.get(key, "")):
            raise ValueError("Invalid Actions identity: " + key)
    return commit


def validate_lock(lock):
    expected = {"schema": "seedvr2-c14-source-lock/1", "parent_image": PARENT, "parent_source_commit": PARENT_COMMIT,
                "parent_image_id": PARENT_IMAGE_ID, "recipe_base_commit": PARENT_COMMIT,
                "omni_source_commit": "6dd0d1f9310f7598b773c796b434c8000c2816ec",
                "build_recipe": "seedvr2-c14", "tag_prefix": "omni031rc1-gateway-lite-seedvr2-c14",
                "gateway_commit": GATEWAY, "upstream_commit": UPSTREAM,
                "remote_branch": BRANCH, "repository": "https://github.com/githubgxll/dynamo.git",
                "harbor_repository": REPOSITORY, "source_before_sha256": BEFORE, "source_after_sha256": AFTER,
                "versions": {"vllm": "0.31.0", "vllm-omni": "0.31.0rc1", "torch": "2.13.0+cu130", "cache-dit": "1.5.0"},
                "platform": "linux/amd64"}
    for key, value in expected.items():
        if lock.get(key) != value:
            raise ValueError("Unreviewed source lock field: " + key)


def write_outputs(path, values):
    if path:
        with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
            for key, value in values.items():
                if "\n" in str(value) or "\r" in str(value):
                    raise ValueError("Multiline Actions output is forbidden")
                stream.write(f"{key}={value}\n")


def execute(command, log=None):
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    output = result.stdout + result.stderr
    if log:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        Path(log).write_text(output, encoding="utf-8", newline="\n")
    if result.returncode:
        print(output[-16000:], file=sys.stderr)
        raise RuntimeError(f"{command[0]} exited {result.returncode}")
    return result.stdout


def docker(*args, log=None):
    return execute(["docker", *args], log)


def inspect(image):
    values = json.loads(docker("image", "inspect", image))
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError("Expected exactly one image")
    return values[0]


def env_map(config):
    values = {}
    for entry in config.get("Env", []):
        key, separator, value = entry.partition("=")
        if not separator or key in values:
            raise ValueError("Malformed image environment")
        values[key] = value
    return values


def compare_images(parent, child, commit):
    if parent.get("Id") != PARENT_IMAGE_ID:
        raise ValueError("Pulled parent image ID differs from the published lite release")
    for item in (parent, child):
        if item.get("Os") != "linux" or item.get("Architecture") != "amd64":
            raise ValueError("Expected linux/amd64")
    layers = parent.get("RootFS", {}).get("Layers", [])
    new_layers = child.get("RootFS", {}).get("Layers", [])
    if not layers or new_layers[:len(layers)] != layers or len(new_layers) <= len(layers):
        raise ValueError("Candidate must extend the complete immutable published lite parent")
    before, after = parent["Config"], child["Config"]
    for field in CONFIG_FIELDS:
        if before.get(field) != after.get(field):
            raise ValueError("Inherited config changed: " + field)
    old_env, new_env = env_map(before), env_map(after)
    if (old_env.get("DYNAMO_COMMIT_SHA") != FRAMEWORK_COMMIT
            or old_env.get("DYNAMO_GATEWAY_SOURCE_SHA") != GATEWAY
            or old_env.get("H3_EXPERIMENTAL_SOURCE_COMMIT") != PARENT_COMMIT):
        raise ValueError("Unexpected published lite source markers")
    allowed = {"H3_C14_SOURCE_COMMIT": commit, "H3_C14_PARENT_IMAGE": PARENT}
    if new_env != {**old_env, **allowed}:
        raise ValueError("Unexpected environment mutation")
    old_labels, new_labels = before.get("Labels") or {}, after.get("Labels") or {}
    mutable_labels = {"org.opencontainers.image.title", "org.opencontainers.image.revision", "org.opencontainers.image.base.name"}
    for key, value in old_labels.items():
        if key not in mutable_labels and new_labels.get(key) != value:
            raise ValueError("Inherited label changed: " + key)
    if new_labels.get("org.opencontainers.image.revision") != commit or new_labels.get("org.opencontainers.image.base.name") != PARENT:
        raise ValueError("Wrong image revision/base label")
    return {"status": "PASS", "parent_image_id": parent["Id"], "candidate_image_id": child["Id"],
            "inherited_layers": len(layers), "added_layers": len(new_layers) - len(layers),
            "parent_size_bytes": parent.get("Size"), "candidate_size_bytes": child.get("Size"),
            "allowed_identity_environment_changes": allowed}


def inventory(image):
    return json.loads(docker("run", "--rm", "--network", "none", "--read-only", "--user", "1000:0",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=2g", "--entrypoint", "python3", image, "-B", "-c", INVENTORY_PROBE))


def compare_inventories(parent, child, commit):
    for key in ("packages", "dpkg"):
        if not parent.get(key) or parent[key] != child.get(key):
            raise ValueError("Dependency inventory changed: " + key)
    if parent.get("module_sha256") != BEFORE or child.get("module_sha256") != AFTER or parent.get("module_path") != child.get("module_path"):
        raise ValueError("Installed SeedVR2 source identity differs")
    if parent.get("gateway_commit") != GATEWAY_BEFORE or child.get("gateway_commit") != GATEWAY:
        raise ValueError("Gateway source markers differ")
    if child.get("source_commit") != commit:
        raise ValueError("Wrong final source commit")
    for item in (parent, child):
        if item.get("dynamo_commit") != FRAMEWORK_COMMIT or item.get("lite_source_commit") != PARENT_COMMIT or item.get("euid") != 1000 or item.get("adapters_present") is not False:
            raise ValueError("Wrong native source, UID or inherited distilled assets")
    return {"status": "PASS", "python_distribution_count": len(parent["packages"]),
            "dpkg_package_count": len(parent["dpkg"]), "adapters_present": False}


def copy_from_image(image, source, destination):
    cid = docker("create", "--entrypoint", "python3", image, "-c", "pass").strip()
    try:
        destination.mkdir(parents=True, exist_ok=False)
        docker("cp", cid + ":" + source.rstrip("/") + "/.", str(destination))
    finally:
        docker("rm", cid)


def run_final_validation(image, directory):
    # Evidence lives in the disposable writable container layer, not tmpfs;
    # docker cp runs after the process exits. No source directory is mounted.
    cid = docker("create", "--network", "none", "--user", "1000:0", "--env", "TMPDIR=/tmp",
        "--env", "XDG_CACHE_HOME=/tmp/lite-cache", "--env", "HF_HOME=/tmp/lite-cache/hf",
        "--env", "TORCHINDUCTOR_CACHE_DIR=/tmp/lite-cache/inductor", "--env", "TRITON_CACHE_DIR=/tmp/lite-cache/triton",
        "--entrypoint", "python3", image, "-B", "/opt/seedvr2-c14/runtime.py", "validate",
        "--output-dir", "/tmp/c14-final").strip()
    try:
        failed = None
        try:
            docker("start", "--attach", cid, log=directory / "final-container.log")
        except RuntimeError as exc:
            failed = exc
        state = json.loads(docker("container", "inspect", cid))[0]["State"]
        target = directory / "final"
        target.mkdir(parents=True, exist_ok=False)
        try:
            docker("cp", cid + ":/tmp/c14-final/.", str(target))
        except RuntimeError:
            if not failed and state.get("ExitCode") == 0:
                raise
        dump(directory / "final-container-state.json", {"exit_code": state.get("ExitCode"), "oom_killed": state.get("OOMKilled")})
        if failed or state.get("ExitCode") != 0 or state.get("OOMKilled"):
            raise RuntimeError("Offline nonroot final image validation failed") from failed
    finally:
        docker("rm", cid)


def validate_reports(directory, commit):
    report = read(directory / "runtime-validation.json")
    expected = {"schema": "seedvr2-c14-runtime/1", "status": "PASS", "euid": 1000,
                "gateway_commit": GATEWAY, "parent_image": PARENT, "source_commit": commit,
                "retained_packages_unchanged": True, "adapters_present": False, "ref8_patch_present": False,
                "diagnostics_default_enabled": False}
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError("Final runtime gate failed: " + key)
    for name in ("source-verification.json", "preservation.json"):
        result = read(directory / name)
        if result.get("status") != "PASS" or result.get("errors"):
            raise ValueError("Missing/failed core gate: " + name)
    for name, expected_status in {
        "environment/environment-check.json": "PASS_EXPERIMENTAL_ENVIRONMENT_CPU_ONLY",
        "environment/media-smoke.json": "PASS_REAL_CPU_MEDIA_SMOKE",
        "environment/service-smoke.json": "PASS_SERVICE_IMPORTS_AND_HELP_ONLY",
    }.items():
        result = read(directory / name)
        if result.get("status") != expected_status or result.get("errors"):
            raise ValueError("Missing/failed retained capability gate: " + name)
    gateway = read(directory / "gateway-regression.json")
    if gateway.get("status") != "PASS" or gateway.get("case_count") != 9:
        raise ValueError("Gateway capacity regression gate is incomplete")
    diagnostics = read(directory / "diagnostics-regression.json")
    if (diagnostics.get("status") != "PASS" or type(diagnostics.get("tests_run")) is not int
            or diagnostics["tests_run"] <= 0
            or any(diagnostics.get(key) != 0 for key in ("failures", "errors", "skipped"))):
        raise ValueError("C14 diagnostic CPU regression gate is incomplete")
    norm = read(directory / "rmsnorm/runtime-validation.json")
    for key, value in {"status": "PASS", "device": "cpu", "euid": 1000,
                       "parent_image": PARENT, "source_commit": commit,
                       "module_sha256": AFTER, "upstream_commit": UPSTREAM,
                       "six_case_count": 6, "upstream_pytest_case_count": 6, "inventory_unchanged": True}.items():
        if norm.get(key) != value:
            raise ValueError("RMSNorm gate failed: " + key)
    source_manifest = read(directory / "source-verification.json").get("source_manifest_sha256")
    if (not isinstance(source_manifest, str) or not re.fullmatch(r"[0-9a-f]{64}", source_manifest)
            or norm.get("source_manifest_sha256") != source_manifest
            or report.get("source_manifest_sha256") != source_manifest):
        raise ValueError("Runtime and numerical evidence are not bound to the same installed source manifest")
    wanted = {(dtype, affine) for dtype in ("torch.float16", "torch.bfloat16", "torch.float32") for affine in (False, True)}
    cases = norm.get("six_cases", [])
    if len(cases) != 6 or {(x.get("dtype"), x.get("affine")) for x in cases} != wanted:
        raise ValueError("Incomplete six precision/affine combinations")
    if any(x.get("finite") is not True or x.get("allclose") is not True or x.get("device") != "cpu" for x in cases):
        raise ValueError("Numerical comparison failed")
    if norm.get("all_300", {}).get("passed") is not True or norm.get("all_300", {}).get("old_formula_all_zero") is not True:
        raise ValueError("Missing FP16 zero-output regression")
    module = norm.get("module_path", "")
    if not module or "/opt/seedvr2-c14/" in module or "/opt/omni-gateway-lite/" in module:
        raise ValueError("Not the installed Omni module")
    return report


def candidate_tag(commit, environ):
    return "seedvr2-c14-ci:" + commit[:12] + "-" + environ["GITHUB_RUN_ID"] + "-" + environ["GITHUB_RUN_ATTEMPT"]


def final_tag(commit):
    return REPOSITORY + ":omni031rc1-gateway-lite-seedvr2-c14-" + commit[:12]


def export_recipe(directory):
    # Delivery exports the exact committed delta plus sealed reused inputs.
    execute([sys.executable, "-B", str(ROOT / "delivery/seedvr2-c14/ops.py"),
             "export-recipe", "--directory", str(directory / "recipe")],
            directory / "recipe-export.txt")


def prepare(environ, directory, output=None):
    commit = identity(environ)
    validate_lock(read(ROOT / LOCK_PATH))
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Use a fresh evidence directory for this run/attempt")
    directory.mkdir(parents=True, exist_ok=True)
    execute([sys.executable, "-B", str(ROOT / "delivery/seedvr2-c14/ops.py"), "check-seal", "--committed"], directory / "input-seal-check.txt")
    values = {"registry": REPOSITORY.split("/", 1)[0], "parent_image": PARENT,
              "candidate_image": candidate_tag(commit, environ), "final_image": final_tag(commit),
              "dockerfile": "container/seedvr2-c14/Dockerfile"}
    dump(directory / "inputs.json", {"schema": "seedvr2-c14-ci-inputs/1", "source_commit": commit,
        "run_id": environ["GITHUB_RUN_ID"], "run_attempt": environ["GITHUB_RUN_ATTEMPT"],
        "source_lock_sha256": sha256(ROOT / LOCK_PATH), **values})
    export_recipe(directory)
    write_outputs(output, values)


def checked_inputs(environ, directory):
    commit = identity(environ)
    inputs = read(directory / "inputs.json")
    expected = {"source_commit": commit, "parent_image": PARENT, "run_id": environ["GITHUB_RUN_ID"],
                "run_attempt": environ["GITHUB_RUN_ATTEMPT"], "source_lock_sha256": sha256(ROOT / LOCK_PATH),
                "candidate_image": candidate_tag(commit, environ), "final_image": final_tag(commit)}
    if any(inputs.get(key) != value for key, value in expected.items()):
        raise ValueError("Inputs do not belong to this reviewed run")
    return inputs


def verify_candidate(environ, directory):
    inputs = checked_inputs(environ, directory)
    commit, tag = inputs["source_commit"], inputs["candidate_image"]
    parent, candidate = inspect(PARENT), inspect(tag)
    dump(directory / "inheritance.json", compare_images(parent, candidate, commit))
    before, after = inventory(PARENT), inventory(candidate["Id"])
    dump(directory / "parent-inventory.json", before)
    dump(directory / "candidate-inventory.json", after)
    dump(directory / "package-preservation.json", compare_inventories(before, after, commit))
    # All later probes target immutable local ID, even if another process retags.
    copy_from_image(candidate["Id"], BUILD_INFO, directory / "build")
    validate_reports(directory / "build", commit)
    run_final_validation(candidate["Id"], directory)
    validate_reports(directory / "final", commit)
    dump(directory / "candidate-verified.json", {"status": "PASS_CANDIDATE_CPU_ONLY", "image_id": candidate["Id"],
         "source_commit": commit, "candidate_image": tag, "parent_image": PARENT})


def publication_digest(output):
    found = set(re.findall(r"(?m)^\S+: digest: (sha256:[0-9a-f]{64}) size: [0-9]+\s*$", output))
    if len(found) != 1:
        raise ValueError("This push did not return one unambiguous digest")
    return found.pop()


def publish(environ, directory, output=None):
    inputs = checked_inputs(environ, directory)
    verified = read(directory / "candidate-verified.json")
    commit, tag = inputs["source_commit"], inputs["candidate_image"]
    if (verified.get("status") != "PASS_CANDIDATE_CPU_ONLY" or verified.get("source_commit") != commit
            or verified.get("candidate_image") != tag or verified.get("parent_image") != PARENT):
        raise ValueError("This candidate has not passed")
    image_id = verified["image_id"]
    if inspect(tag)["Id"] != image_id:
        raise ValueError("Candidate tag changed after validation")
    validate_reports(directory / "build", commit)
    validate_reports(directory / "final", commit)
    docker("tag", image_id, inputs["final_image"])
    docker("push", inputs["final_image"], log=directory / "registry-push.txt")
    digest = publication_digest((directory / "registry-push.txt").read_text(encoding="utf-8"))
    reference = REPOSITORY + "@" + digest
    docker("pull", "--platform", "linux/amd64", reference, log=directory / "registry-pull.txt")
    pulled = inspect(reference)
    if pulled["Id"] != image_id:
        raise ValueError("Registry pull differs from the exact tested image")
    compare_images(inspect(PARENT), pulled, commit)
    record = {"schema": "seedvr2-c14-publication/1", "status": "PASS_PUBLISHED_AND_PULLED_CPU_VALIDATED",
        "source_commit": commit, "branch": BRANCH, "image": inputs["final_image"], "image_id": image_id,
        "deployment_image": reference, "digest": digest, "parent_image": PARENT, "parent_source_commit": PARENT_COMMIT,
        "gateway_commit": GATEWAY, "upstream_commit": UPSTREAM, "distilled_adapters": False,
        "source_before_sha256": BEFORE, "source_after_sha256": AFTER,
        "run_id": environ["GITHUB_RUN_ID"], "run_attempt": environ["GITHUB_RUN_ATTEMPT"],
        "actions_url": "https://github.com/githubgxll/dynamo/actions/runs/" + environ["GITHUB_RUN_ID"],
        "not_run": ["GPU inference", "Gateway external-service E2E", "same-input video A/B", "human quality review"]}
    record["evidence_sha256"] = {p.relative_to(directory).as_posix(): sha256(p) for p in sorted(directory.rglob("*"))
                                if p.is_file() and p.name not in {"image.json", "deployment-image.txt"}}
    dump(directory / "image.json", record)
    (directory / "deployment-image.txt").write_text(reference + "\n", encoding="utf-8", newline="\n")
    write_outputs(output, {"digest": digest, "deployment_image": reference})
    print(json.dumps(record, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify-candidate", "publish"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.command == "prepare":
        prepare(os.environ, directory, args.github_output)
    elif args.command == "verify-candidate":
        verify_candidate(os.environ, directory)
    else:
        publish(os.environ, directory, args.github_output)


if __name__ == "__main__":
    main()
