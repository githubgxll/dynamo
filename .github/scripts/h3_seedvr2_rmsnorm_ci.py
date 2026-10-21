#!/usr/bin/env python3
"""Build-publication gates for the digest-pinned SeedVR2 RMSNorm derivative.

No dependency installation, framework rendering, remote branch merge or model
download. Publication tags the exact local image that passed the offline gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = Path("delivery/h3-seedvr2-rmsnorm/source-lock.json")
BRANCH = "Dingorouter-H3-vllm-omni-jirx"
REPOSITORY = "registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm"
PARENT = REPOSITORY + "@sha256:112e34379489f05c737079212abb9fa587676fd82650dd1c1af96b32f46dc9a7"
PARENT_COMMIT = "db65d1ec869b6deee02debe7a58e108de7c927ef"
UPSTREAM = "0708f05e2e27f16fca8b48e7c5a085277d1cafda"
BEFORE = "3c782ad3588f7f544a9cb14fe8258a9eb4d921881deee8501460ceaf0bf1ad63"
AFTER = "03fd3a0c199305ce3004643f8d1510183b06731099a72bc447c2bd589c1a6f01"
SHA40 = re.compile(r"[0-9a-f]{40}\Z")
DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
NUMBER = re.compile(r"[1-9][0-9]*\Z")
BUILD_INFO = "/opt/h3-seedvr2-rmsnorm/build-info"
CONFIG_FIELDS = ("User", "Entrypoint", "Cmd", "WorkingDir", "ExposedPorts", "Volumes", "Healthcheck", "StopSignal", "Shell", "OnBuild")
RETAINED_REPORTS = {
    "environment-check.json": "PASS_EXPERIMENTAL_ENVIRONMENT_CPU_ONLY",
    "media-smoke.json": "PASS_REAL_CPU_MEDIA_SMOKE",
    "service-smoke.json": "PASS_SERVICE_IMPORTS_AND_HELP_ONLY",
    "assets-report.json": "PASS_ASSETS_ONLY",
    "profiles-check.json": "PASS_INSTALLED_PROFILES_CPU_ONLY",
    "ref8-contract.json": "PASS_INSTALLED_CONTRACT_CPU_ONLY",
}

# This probe is intentionally stdlib-only: it must work with the unchanged
# parent and child environments and cannot hide dependency drift by installing.
INVENTORY_PROBE = r'''
import hashlib, importlib.metadata as m, json, os, pathlib, subprocess
dist = m.distribution("vllm-omni")
source = pathlib.Path(dist.locate_file("vllm_omni/diffusion/models/seedvr2/nadit.py")).resolve()
packages = sorted((d.metadata["Name"], d.version) for d in m.distributions())
dpkg = subprocess.run(["dpkg-query", "-W", "-f=${binary:Package}\t${Version}\n"], check=True, capture_output=True, text=True)
print(json.dumps({"packages": packages, "dpkg": sorted(dpkg.stdout.splitlines()),
 "module_path": str(source), "module_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
 "source_commit": os.environ.get("H3_EXPERIMENTAL_SOURCE_COMMIT"),
 "gateway_commit": os.environ.get("DYNAMO_GATEWAY_SOURCE_SHA"), "euid": os.geteuid()}))
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
    if not SHA40.fullmatch(commit):
        raise ValueError("GITHUB_SHA must be a full lowercase commit")
    if environ.get("GITHUB_REF_NAME") != BRANCH:
        raise ValueError("Only the existing designated H3 experimental branch is allowed")
    if environ.get("GITHUB_EVENT_NAME") not in {"push", "workflow_dispatch"}:
        raise ValueError("Unsupported triggering event")
    for key in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT"):
        if not NUMBER.fullmatch(environ.get(key, "")):
            raise ValueError(f"Missing or invalid {key}")
    return commit


def validate_lock(lock):
    expected = {
        "schema": "h3-seedvr2-rmsnorm-source-lock/1", "parent_image": PARENT,
        "parent_source_commit": PARENT_COMMIT, "remote_branch": BRANCH,
        "repository": "https://github.com/githubgxll/dynamo.git",
        "harbor_repository": REPOSITORY, "upstream_commit": UPSTREAM,
        "source_before_sha256": BEFORE, "source_after_sha256": AFTER,
        "upstream_test_sha256": "56e39d8422fa0e89a201820e6de8e34ef20e92600cdedf54e4cd38c0e1c5374a",
        "omni_source_commit": "6dd0d1f9310f7598b773c796b434c8000c2816ec",
        "versions": {"vllm": "0.31.0", "vllm-omni": "0.31.0rc1", "torch": "2.13.0+cu130", "cache-dit": "1.5.0"},
        "platform": "linux/amd64",
    }
    for key, value in expected.items():
        if lock.get(key) != value:
            raise ValueError("Unreviewed source-lock value: " + key)


def write_outputs(path, values):
    if path:
        with Path(path).open("a", encoding="utf-8", newline="\n") as out:
            for name, value in values.items():
                if "\n" in str(value) or "\r" in str(value):
                    raise ValueError("Multiline GitHub output is forbidden")
                out.write(f"{name}={value}\n")


def execute(arguments, log=None):
    result = subprocess.run(arguments, cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, check=False)
    combined = result.stdout + result.stderr
    if log:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        Path(log).write_text(combined, encoding="utf-8", newline="\n")
    if result.returncode:
        if combined:
            print(combined[-16000:], file=sys.stderr)
        raise RuntimeError(f"{arguments[0]} {arguments[1] if len(arguments) > 1 else ''} exited {result.returncode}")
    return result.stdout


def docker(*args, log=None):
    return execute(["docker", *args], log=log)


def inspect(image):
    values = json.loads(docker("image", "inspect", image))
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError("Expected exactly one inspected image")
    return values[0]


def env_map(config):
    result = {}
    for entry in config.get("Env", []):
        key, separator, value = entry.partition("=")
        if not separator or key in result:
            raise ValueError("Malformed or duplicate image environment key")
        result[key] = value
    return result


def compare_images(parent, child, commit):
    for image in (parent, child):
        if image.get("Os") != "linux" or image.get("Architecture") != "amd64":
            raise ValueError("Expected linux/amd64 image")
    parent_layers = parent.get("RootFS", {}).get("Layers", [])
    child_layers = child.get("RootFS", {}).get("Layers", [])
    if not parent_layers or child_layers[:len(parent_layers)] != parent_layers or len(child_layers) <= len(parent_layers):
        raise ValueError("Candidate does not extend the complete pinned parent layer chain")
    p, c = parent["Config"], child["Config"]
    for field in CONFIG_FIELDS:
        if p.get(field) != c.get(field):
            raise ValueError("Inherited image config changed: " + field)
    before, after = env_map(p), env_map(c)
    if before.get("H3_EXPERIMENTAL_SOURCE_COMMIT") != PARENT_COMMIT:
        raise ValueError("Pinned parent reports an unexpected H3 source marker")
    allowed = {"H3_EXPERIMENTAL_SOURCE_COMMIT": commit,
               "H3_RMSNORM_PARENT_IMAGE": PARENT,
               "H3_RMSNORM_UPSTREAM_COMMIT": UPSTREAM}
    expected = {**before, **allowed}
    if after != expected:
        names = sorted(key for key in set(expected) | set(after) if expected.get(key) != after.get(key))
        raise ValueError("Unexpected image environment change: " + ", ".join(names))
    plabels, clabels = p.get("Labels") or {}, c.get("Labels") or {}
    for key, value in plabels.items():
        if key not in {"org.opencontainers.image.title", "org.opencontainers.image.revision", "org.opencontainers.image.base.name"} and clabels.get(key) != value:
            raise ValueError("Inherited custom label changed: " + key)
    if clabels.get("org.opencontainers.image.revision") != commit or clabels.get("org.opencontainers.image.base.name") != PARENT:
        raise ValueError("Candidate OCI revision/base identity is incorrect")
    return {"status": "PASS_PARENT_LAYER_AND_CONFIG_INHERITANCE", "parent_image_id": parent["Id"],
            "candidate_image_id": child["Id"], "inherited_layers": len(parent_layers),
            "added_layers": len(child_layers) - len(parent_layers), "preserved_config_fields": list(CONFIG_FIELDS),
            "allowed_identity_environment_changes": allowed}


def inventory(image):
    return json.loads(docker("run", "--rm", "--network", "none", "--read-only", "--user", "1000:0",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=2g", "--entrypoint", "python3", image, "-B", "-c", INVENTORY_PROBE))


def compare_inventories(parent, child, commit):
    for key in ("packages", "dpkg"):
        if not parent.get(key) or parent[key] != child.get(key):
            raise ValueError("Installed dependency inventory changed: " + key)
    if parent.get("module_sha256") != BEFORE or child.get("module_sha256") != AFTER:
        raise ValueError("Actual installed SeedVR2 module hashes do not match the reviewed patch")
    if parent.get("module_path") != child.get("module_path"):
        raise ValueError("Installed module location changed")
    if parent.get("source_commit") != PARENT_COMMIT or child.get("source_commit") != commit:
        raise ValueError("Container source identities differ from the build input")
    if parent.get("gateway_commit") != "e8c9195672f761ec09344a641730844c9e6c9e99" or child.get("gateway_commit") != parent.get("gateway_commit"):
        raise ValueError("Gateway source identity changed")
    if parent.get("euid") != 1000 or child.get("euid") != 1000:
        raise ValueError("Inventory probes did not run as UID 1000")
    return {"status": "PASS_UNCHANGED_PACKAGE_VERSIONS", "python_distribution_count": len(parent["packages"]),
            "dpkg_package_count": len(parent["dpkg"]), "scope": "Installed versions and module identity; retained capability tests are separate"}


def copy_from_image(image, source, destination):
    cid = docker("create", "--entrypoint", "python3", image, "-c", "pass").strip()
    try:
        destination.mkdir(parents=True, exist_ok=False)
        docker("cp", cid + ":" + source.rstrip("/") + "/.", str(destination))
    finally:
        docker("rm", cid)


def run_final_validation(image, directory):
    # A dedicated container exports evidence even when numerical/import checks
    # fail. Output uses its disposable writable layer, NOT tmpfs (which vanishes
    # when the process exits). The source image remains immutable. No checked-out
    # source directory is mounted into this process.
    cid = docker("create", "--network", "none", "--user", "1000:0",
                 "--env", "TMPDIR=/tmp", "--env", "XDG_CACHE_HOME=/tmp/srnorm-cache",
                 "--env", "HF_HOME=/tmp/srnorm-cache/hf", "--env", "UV_CACHE_DIR=/tmp/srnorm-cache/uv",
                 "--env", "TORCHINDUCTOR_CACHE_DIR=/tmp/srnorm-cache/inductor",
                 "--env", "TRITON_CACHE_DIR=/tmp/srnorm-cache/triton", "--entrypoint", "python3", image,
                 "-B", "/opt/h3-seedvr2-rmsnorm/validate_runtime.py",
                 "--output-dir", "/tmp/srnorm-final", "--device", "cpu").strip()
    try:
        execution_error = None
        try:
            docker("start", "--attach", cid, log=directory / "final-container.log")
        except RuntimeError as exc:
            execution_error = exc
        state = json.loads(docker("container", "inspect", cid))[0]["State"]
        target = directory / "final"
        target.mkdir(parents=True, exist_ok=False)
        try:
            docker("cp", cid + ":/tmp/srnorm-final/.", str(target))
        except RuntimeError:
            if not execution_error and state.get("ExitCode") == 0:
                raise
        dump(directory / "final-container-state.json", {"exit_code": state.get("ExitCode"), "oom_killed": state.get("OOMKilled")})
        if execution_error or state.get("ExitCode") != 0 or state.get("OOMKilled"):
            raise RuntimeError("Offline nonroot final-image validation failed; see container log") from execution_error
    finally:
        docker("rm", cid)


def validate_runtime_report(report, commit=None):
    if report.get("status") != "PASS" or report.get("device") != "cpu" or report.get("module_sha256") != AFTER:
        raise ValueError("Final image runtime identity/numerics did not pass")
    if report.get("six_case_count") != 6 or len(report.get("six_cases", [])) != 6:
        raise ValueError("The six upstream precision/affine combinations were not all executed")
    expected = {(dtype, affine) for dtype in ("torch.float16", "torch.bfloat16", "torch.float32") for affine in (False, True)}
    actual = {(item.get("dtype"), item.get("affine")) for item in report["six_cases"]}
    if actual != expected or any(item.get("finite") is not True or item.get("allclose") is not True or item.get("device") != "cpu" for item in report["six_cases"]):
        raise ValueError("Numerical case coverage/results are incomplete")
    if report.get("upstream_pytest_case_count") != 6 or report.get("inventory_unchanged") is not True or report.get("euid") != 1000:
        raise ValueError("Actual upstream tests, nonroot identity or unchanged dependency checks are missing")
    if commit is not None and report.get("source_commit") != commit:
        raise ValueError("Runtime evidence belongs to a different source commit")
    if not report.get("all_300", {}).get("passed"):
        raise ValueError("The full-300 FP16 silent-zero regression did not pass")
    if not report.get("module_path") or "/opt/h3-seedvr2-rmsnorm/" in report["module_path"]:
        raise ValueError("Tests imported a vendored source copy instead of the installed module")


def validate_build_evidence(directory, commit=None):
    preservation = read(directory / "preservation.json")
    if preservation.get("status") != "PASS_PRESERVED_PARENT_CONTENT_CPU_ONLY":
        raise ValueError("Parent content preservation check is missing or failed")
    for name, status in RETAINED_REPORTS.items():
        report = read(directory / "retained" / name)
        if report.get("status") != status or report.get("errors"):
            raise ValueError("Retained capability check missing or failed: " + name)
    validate_runtime_report(read(directory / "runtime-validation.json"), commit)
    receipt = read(directory / "patch-application.json")
    if receipt.get("status") != "PASS" or receipt.get("before_sha256") != BEFORE or receipt.get("after_sha256") != AFTER or receipt.get("upstream_commit") != UPSTREAM:
        raise ValueError("Missing or incorrect exact patch application report")


def prepare(root, environ, directory, output=None):
    commit = identity(environ)
    lock = read(root / LOCK_PATH)
    validate_lock(lock)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Evidence directory is not empty; use this run/attempt's own path")
    directory.mkdir(parents=True, exist_ok=True)
    execute([sys.executable, "-B", str(root / "delivery/h3-seedvr2-rmsnorm/ops.py"), "check-seal", "--committed"], log=directory / "input-seal-check.txt")
    final_image = REPOSITORY + ":omni031rc1-gateway-distilled-ref8shift12-srnormfix-" + commit[:12]
    candidate = "h3-srnorm-ci:" + commit[:12] + "-" + environ["GITHUB_RUN_ID"] + "-" + environ["GITHUB_RUN_ATTEMPT"]
    values = {"registry": REPOSITORY.split("/", 1)[0], "parent_image": PARENT,
              "candidate_image": candidate, "final_image": final_image, "platform": "linux/amd64",
              "evidence_directory": str(directory), "dockerfile": "container/h3-seedvr2-rmsnorm/Dockerfile"}
    dump(directory / "inputs.json", {"schema": "h3-seedvr2-rmsnorm-ci-inputs/1", "source_commit": commit,
        "branch": BRANCH, "parent_source_commit": PARENT_COMMIT, "upstream_commit": UPSTREAM,
        "run_id": environ["GITHUB_RUN_ID"], "run_attempt": environ["GITHUB_RUN_ATTEMPT"],
        "source_lock_sha256": sha256(root / LOCK_PATH), **values})
    seal_path = Path("delivery/h3-seedvr2-rmsnorm/input-seal.json")
    sealed = read(root / seal_path)
    recipe_paths = {seal_path, LOCK_PATH}
    for name, change in sealed["delta"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe sealed recipe path")
        if change["status"] != "D":
            recipe_paths.add(relative)
    for relative in sorted(recipe_paths):
        target = directory / "recipe" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    write_outputs(output, values)
    return values


def checked_inputs(directory, environ):
    inputs = read(directory / "inputs.json")
    commit = identity(environ)
    if inputs.get("source_commit") != commit or inputs.get("parent_image") != PARENT or inputs.get("run_id") != environ["GITHUB_RUN_ID"] or inputs.get("run_attempt") != environ["GITHUB_RUN_ATTEMPT"]:
        raise ValueError("Evidence belongs to a different source, parent or Actions attempt")
    if inputs.get("source_lock_sha256") != sha256(ROOT / LOCK_PATH):
        raise ValueError("Source lock changed during CI")
    expected_candidate = "h3-srnorm-ci:" + commit[:12] + "-" + environ["GITHUB_RUN_ID"] + "-" + environ["GITHUB_RUN_ATTEMPT"]
    expected_final = REPOSITORY + ":omni031rc1-gateway-distilled-ref8shift12-srnormfix-" + commit[:12]
    if inputs.get("candidate_image") != expected_candidate or inputs.get("final_image") != expected_final:
        raise ValueError("Candidate/publication tag does not match this sealed run")
    return inputs


def verify_candidate(directory, environ):
    inputs = checked_inputs(directory, environ)
    commit, image = inputs["source_commit"], inputs["candidate_image"]
    parent, candidate = inspect(PARENT), inspect(image)
    inheritance = compare_images(parent, candidate, commit)
    dump(directory / "inheritance.json", inheritance)
    parent_inventory, candidate_inventory = inventory(PARENT), inventory(image)
    dump(directory / "parent-inventory.json", parent_inventory)
    dump(directory / "candidate-inventory.json", candidate_inventory)
    dump(directory / "package-preservation.json", compare_inventories(parent_inventory, candidate_inventory, commit))
    copy_from_image(image, BUILD_INFO, directory / "build")
    validate_build_evidence(directory / "build", commit)
    run_final_validation(image, directory)
    final_report = read(directory / "final/runtime-validation.json")
    validate_runtime_report(final_report, commit)
    if final_report["module_path"] != candidate_inventory["module_path"]:
        raise ValueError("Final import differs from the actual installed distribution module")
    dump(directory / "candidate-verified.json", {"status": "PASS_CANDIDATE_CPU_ONLY", "image_id": candidate["Id"],
        "source_commit": commit, "parent_image": PARENT, "candidate_image": image,
        "scope": "Actual final-image offline UID1000 import/numerics plus inherited CPU gates; no GPU/media quality claim"})


def publication_digest(push_log):
    # Multiple manifests can refer to the same local config ID, so RepoDigests
    # is not a reliable selector on a persistent Runner. Use THIS push receipt.
    matches = set(re.findall(r"(?m)^\S+: digest: (sha256:[0-9a-f]{64}) size: [0-9]+\s*$", push_log))
    if len(matches) != 1:
        raise ValueError("Cannot select an unambiguous digest from this docker push receipt")
    return matches.pop()


def publish(directory, environ, output=None):
    inputs = checked_inputs(directory, environ)
    verified = read(directory / "candidate-verified.json")
    image, final_image = inputs["candidate_image"], inputs["final_image"]
    if verified.get("status") != "PASS_CANDIDATE_CPU_ONLY" or verified.get("source_commit") != inputs["source_commit"] or verified.get("candidate_image") != image:
        raise ValueError("This exact candidate was not verified")
    image_id = verified["image_id"]
    if inspect(image)["Id"] != image_id:
        raise ValueError("Local image tag changed after validation; refusing publication")
    validate_build_evidence(directory / "build", inputs["source_commit"])
    validate_runtime_report(read(directory / "final/runtime-validation.json"), inputs["source_commit"])
    # Tag the immutable local image ID, not a tag that another process may move.
    docker("tag", image_id, final_image)
    docker("push", final_image, log=directory / "registry-push.txt")
    published = inspect(final_image)
    if published["Id"] != image_id:
        raise ValueError("Published local tag no longer identifies the checked image")
    digest = publication_digest((directory / "registry-push.txt").read_text(encoding="utf-8"))
    reference = REPOSITORY + "@" + digest
    docker("pull", "--platform", "linux/amd64", reference, log=directory / "registry-pull.txt")
    pulled = inspect(reference)
    if pulled["Id"] != image_id:
        raise ValueError("Immutable registry pull does not match the tested image ID")
    compare_images(inspect(PARENT), pulled, inputs["source_commit"])
    record = {"schema": "h3-seedvr2-rmsnorm-publication/1", "status": "PASS_PUBLISHED_AND_PULLED_CPU_VALIDATED",
        "source_commit": inputs["source_commit"], "branch": BRANCH, "image": final_image,
        "deployment_image": reference, "digest": digest, "image_id": image_id,
        "parent_image": PARENT, "parent_source_commit": PARENT_COMMIT, "upstream_commit": UPSTREAM,
        "source_before_sha256": BEFORE, "source_after_sha256": AFTER,
        "run_id": environ["GITHUB_RUN_ID"], "run_attempt": environ["GITHUB_RUN_ATTEMPT"],
        "actions_url": "https://github.com/githubgxll/dynamo/actions/runs/" + environ["GITHUB_RUN_ID"],
        "not_run": ["CUDA precision tests", "H100 inference", "fixed-input 1088p/1440p raw/final media comparison", "human artifact/flicker review"]}
    record["evidence_sha256"] = {p.relative_to(directory).as_posix(): sha256(p)
        for p in sorted(directory.rglob("*")) if p.is_file() and p.name not in {"image.json", "deployment-image.txt"}}
    dump(directory / "image.json", record)
    (directory / "deployment-image.txt").write_text(reference + "\n", encoding="utf-8", newline="\n")
    write_outputs(output, {"digest": digest, "deployment_image": reference})
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify-candidate", "publish"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.command == "prepare":
        prepare(ROOT, os.environ, directory, args.github_output)
    elif args.command == "verify-candidate":
        verify_candidate(directory, os.environ)
    else:
        publish(directory, os.environ, args.github_output)


if __name__ == "__main__":
    main()
