#!/usr/bin/env python3
"""One final image build, exact-image offline validation, one final publication."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BRANCH = "Dingorouter-H3-vllm-omni-jirx"
REPOSITORY = "registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm"
PARENT = REPOSITORY + "@sha256:b3150072a8b917e643ceae5fd2b278e736df09fc76397b749b3ca4531cbe8bf3"
PARENT_ID = "sha256:4c762dc81077fc52594e874f366324d48993c2c201de49e3223b4b0a93f30b62"
BASE_COMMIT = "aacb23c7a7443efcf178d2375a00b686a4d96bda"
BUILD_INFO = "/opt/omni032-h3-sr/build-info"
RUNTIME = "/opt/omni032-h3-sr/runtime.py"
EXPECTED_MANIFEST = ROOT / "delivery/omni032-h3-sr/expected-source-manifest.json"
EXPECTED_MANIFEST_SHA256 = "6de9b4fe3db09914d4d497e888e2554265843e36af717d5e0e19d1f940109c69"
INHERITED_BASE = "vllm/vllm-openai:v0.31.0-ubuntu2404@sha256:3fb023cc987cc7c2bca7ca14c1e0aff5ae06bd9431bf07773731ce7e8a682460"
NCCL_DIAGNOSTIC = 'torch==2.13.0+cu130: requires nvidia-nccl-cu13==2.29.7; platform_system == "Linux", installed 2.30.7'
# These describe the new producer's two inherited native pins, not a claim
# that a historical full-environment text file was reused byte-for-byte.
NATIVE_PIN_SNAPSHOT = "torch==2.13.0+cu130\nnvidia-nccl-cu13==2.30.7\n"
NCCL_OVERRIDE_TEXT = "nvidia-nccl-cu13==2.30.7\n"
CONFIG_FIELDS = ("User", "Entrypoint", "Cmd", "WorkingDir", "ExposedPorts", "Volumes", "Healthcheck", "StopSignal", "Shell", "OnBuild")
spec = importlib.util.spec_from_file_location("_omni032_submission", ROOT / "delivery/omni032-h3-sr/ops.py")
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)
read, dump, hash_value = ops.read, ops.dump, ops.hash_value


def execute(arguments, log=None):
    result = subprocess.run(arguments, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if log is not None:
        log = Path(log)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError("Command failed: " + arguments[0] + "\n" + result.stdout[-8000:] + result.stderr[-8000:])
    return result.stdout


def docker(*arguments, log=None):
    return execute(["docker", *arguments], log=log)


def inspect(image):
    result = json.loads(docker("image", "inspect", image))
    if len(result) != 1:
        raise ValueError("Expected one exact image")
    return result[0]


def identity(environ):
    commit = environ.get("GITHUB_SHA", "")
    if (not re.fullmatch(r"[0-9a-f]{40}", commit)
            or environ.get("GITHUB_REF_NAME") != BRANCH
            or environ.get("GITHUB_EVENT_NAME") != "push"
            or environ.get("GITHUB_RUN_ATTEMPT") != "1"
            or not re.fullmatch(r"[1-9][0-9]*", environ.get("GITHUB_RUN_ID", ""))):
        raise ValueError("Only the first push-triggered run on the designated branch may build")
    return commit


def validate_lock(value):
    expected = {"repository": "https://github.com/githubgxll/dynamo.git", "remote_branch": BRANCH,
                "harbor_repository": REPOSITORY, "parent_image": PARENT,
                "parent_source_commit": BASE_COMMIT, "recipe_base_commit": BASE_COMMIT,
                "working_tree_base_commit": BASE_COMMIT, "platform": "linux/amd64",
                "omni_release_commit": "5917730a4b02aedb44d374173cdc1b10484ccd52"}
    for key, item in expected.items():
        if value.get(key) != item:
            raise ValueError("Unexpected reviewed source lock: " + key)
    if value.get("runtime_base_image", PARENT) != PARENT:
        raise ValueError("Actual parent must remain the reviewed functional image")
    for name, version in {"vllm": "0.32.0", "vllm-omni": "0.32.0", "torch": "2.13.0+cu130", "cache-dit": "1.5.0"}.items():
        if value.get("versions", {}).get(name) != version:
            raise ValueError("Unreviewed framework dependency target: " + name)
    if not re.fullmatch(r"omni032[a-z0-9.-]*", value.get("tag_prefix", "")):
        raise ValueError("Unsafe or ambiguous final tag prefix")


def prepare(directory, github_output=None):
    commit = identity(os.environ)
    value, _, recipe = ops.check_seal(committed=True)
    validate_lock(value)
    message = ops.git("log", "-1", "--format=%B")
    if not ops.valid_request_message(message):
        raise ValueError("A unique upgrade commit marker is required")
    if ops.git("rev-parse", "HEAD") != commit:
        raise ValueError("Checkout is not the requested immutable commit")
    changed = ops.git("diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD").splitlines()
    if not any(name not in ops.AUXILIARY for name in changed):
        raise ValueError("Empty/approval-only request commits cannot trigger another image build")
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "preparation.json").exists():
        raise ValueError("This job preparation was already requested")
    docker("pull", PARENT, log=directory / "parent-pull.txt")
    parent = inspect(PARENT)
    if parent.get("Id") != PARENT_ID or parent.get("Os") != "linux" or parent.get("Architecture") != "amd64":
        raise ValueError("Actual parent differs from the successful functional image")
    dump(directory / "parent-inspect.json", parent)
    tag = REPOSITORY + ":" + value["tag_prefix"] + "-" + commit[:12]
    record = {"schema": "omni032-ci-preparation/1", "status": "PASS", "commit": commit,
              "recipe_sha256": recipe["recipe_sha256"], "reviewed_tree": recipe["tree"],
              "parent_image": PARENT, "image": tag, "run_id": os.environ["GITHUB_RUN_ID"], "run_attempt": 1}
    dump(directory / "preparation.json", record)
    if github_output:
        with Path(github_output).open("a", encoding="utf-8", newline="\n") as stream:
            for key, item in {"image": tag, "base_image": PARENT, "registry": REPOSITORY.split("/")[0]}.items():
                if "\n" in item or "\r" in item:
                    raise ValueError("Unsafe Actions output")
                stream.write(f"{key}={item}\n")
    return record


def compare_config(parent, child, commit):
    if parent.get("Id") != PARENT_ID:
        raise ValueError("Unreviewed parent config")
    for item in (parent, child):
        if item.get("Os") != "linux" or item.get("Architecture") != "amd64":
            raise ValueError("Expected linux/amd64 image")
    for field in CONFIG_FIELDS:
        if parent["Config"].get(field) != child["Config"].get(field):
            raise ValueError("Inherited service config changed: " + field)
    old_env = dict(item.split("=", 1) for item in parent["Config"].get("Env", []))
    new_env = dict(item.split("=", 1) for item in child["Config"].get("Env", []))
    if any(new_env.get(key) != value for key, value in old_env.items()):
        raise ValueError("Inherited environment/identity changed")
    labels = child["Config"].get("Labels") or {}
    if labels.get("org.opencontainers.image.revision") != commit or labels.get("org.opencontainers.image.base.name") != PARENT:
        raise ValueError("Candidate labels do not identify the source/parent")
    parent_layers, child_layers = parent.get("RootFS", {}).get("Layers", []), child.get("RootFS", {}).get("Layers", [])
    if not parent_layers or child_layers[:len(parent_layers)] != parent_layers or len(child_layers) <= len(parent_layers):
        raise ValueError("Candidate does not extend the complete functional parent")
    return {"status": "PASS", "image_id": child["Id"], "inherited_layers": len(parent_layers),
            "added_layers": len(child_layers) - len(parent_layers)}


def expected_sources():
    value = read(EXPECTED_MANIFEST)
    files = value.get("source_sha256")
    if (value.get("schema") != "omni032-expected-source-manifest/1"
            or value.get("omni_wheel_sha256") != "242b94e64b359d3cd6d1ebf1b9d6b43ae854a4f97c780fd5e2121dab06cb68e6"
            or not isinstance(files, dict) or len(files) != 2037
            or hash_value(files) != EXPECTED_MANIFEST_SHA256
            or value.get("source_manifest_sha256") != EXPECTED_MANIFEST_SHA256):
        raise ValueError("Frozen expected-source manifest differs from the reviewed release plus patches")
    return files


def validate_dependency_report(report):
    if report.get("errors") != []:
        raise ValueError("Installed dependency audit has errors or omits its error inventory")
    known = report.get("known_metadata_exceptions")
    if report.get("status") == "PASS" and known == []:
        return
    evidence = report.get("upstream_nccl_evidence", {})
    if (report.get("status") != "PASS_WITH_UPSTREAM_NCCL_OVERRIDE" or known != [NCCL_DIAGNOSTIC]
            or evidence.get("verified") is not True or evidence.get("errors") != []
            or evidence.get("inherited_from_parent") != PARENT or evidence.get("base_reference") != INHERITED_BASE
            or evidence.get("override_file") != "/etc/uv-overrides.txt"
            or evidence.get("overrides_sha256") != hashlib.sha256(NCCL_OVERRIDE_TEXT.encode()).hexdigest()
            or evidence.get("protected_before_omni_sha256") != hashlib.sha256(NATIVE_PIN_SNAPSHOT.encode()).hexdigest()):
        raise ValueError("Dependency pass is not the single verified inherited Torch/NCCL metadata override")


def validate_extra_gates(directory, commit, runtime, files):
    norm = read(directory / "rmsnorm/runtime-validation.json")
    cases = norm.get("six_cases", [])
    expected_cases = {(dtype, affine) for dtype in ("torch.float16", "torch.bfloat16", "torch.float32") for affine in (False, True)}
    if (norm.get("status") != "PASS" or norm.get("source_commit") != commit or norm.get("device") != "cpu"
            or norm.get("six_case_count") != 6 or len(cases) != 6
            or {(item.get("dtype"), item.get("affine")) for item in cases} != expected_cases
            or any(item.get("finite") is not True or item.get("allclose") is not True for item in cases)
            or norm.get("all_300", {}).get("passed") is not True
            or norm.get("module_sha256") != files["vllm_omni/diffusion/models/seedvr2/nadit.py"]):
        raise ValueError("All six actual installed RMSNorm numerical cases are required")
    native = read(directory / "torch-contract.json")
    torch_policy = read(ROOT / "container/h3-hsdp-offload/torch-contract.json")
    methods = {item["module"] + "." + item["class"] + "." + item["method"]: item["ast_sha256"] for item in torch_policy["methods"]}
    if (native.get("status") != "PASS" or native.get("torch_version") != "2.13.0+cu130"
            or native.get("schema") != torch_policy["schema"] or native.get("fingerprint") != torch_policy["fingerprint"]
            or native.get("methods") != methods):
        raise ValueError("Native Torch/FSDP conversion identity gate is incomplete")
    legal = read(directory / "legal-validation.json")
    legal_files = legal.get("files")
    policy_sha = hashlib.sha256((ROOT / "container/compliance/policy/licenses.toml").read_bytes()).hexdigest()
    if (legal.get("status") != "PASS" or legal.get("source_commit") != commit
            or not isinstance(legal_files, dict) or not legal_files
            or legal.get("legal_manifest_sha256") != hash_value(legal_files)
            or legal.get("legal_manifest_sha256") != runtime["legal_manifest_sha256"]
            or legal.get("policy_sha256") != policy_sha or legal.get("changed_python_components", 0) < 2):
        raise ValueError("Current Python legal receipt is not bound to this source/policy/manifest")
    cli = read(directory / "native-cli/native-cli.json")
    argv = ROOT / "container/omni032-h3-sr/native-cli-argv.json"
    argv_contract = read(argv)
    expected_argv = {item["name"]: item["argv_sha256"] for item in argv_contract["profiles"]}
    profiles = cli.get("profiles", [])
    if (cli.get("schema") != "omni032-native-cli-validation/1" or cli.get("status") != "PASS"
            or cli.get("source_commit") != commit or cli.get("euid") != 1000
            or any(cli.get(key) is not False for key in ("dispatch_executed", "engine_started", "model_loaded"))
            or cli.get("unknown_options_rejected") is not True
            or cli.get("argv_contract_sha256") != hashlib.sha256(argv.read_bytes()).hexdigest()
            or {item.get("name") for item in profiles} != set(expected_argv)
            or len(profiles) != 3 or any(item.get("status") != "PASS" or item.get("argv_sha256") != expected_argv.get(item.get("name")) for item in profiles)
            or cli.get("parser_sources_sha256") != argv_contract["parser_sources_sha256"]
            or any(files.get(name) != sha for name, sha in argv_contract["parser_sources_sha256"].items())):
        raise ValueError("The actual native CLI/parser/configuration smoke gate is required")


def validate_reports(directory, commit):
    runtime = read(directory / "runtime-validation.json")
    for key, expected in {"status": "PASS_CPU_ONLY", "source_commit": commit, "euid": 1000,
                          "gpu_validation_status": "NOT_RUN", "legal_status": "PASS"}.items():
        if runtime.get(key) != expected:
            raise ValueError("Runtime report failed: " + key)
    if not re.fullmatch(r"[0-9a-f]{64}", runtime.get("legal_manifest_sha256", "")):
        raise ValueError("Missing upgraded Python license/SBOM manifest identity")
    proof = read(directory / "source-verification.json")
    files = proof.get("source_sha256")
    if (proof.get("status") != "PASS" or proof.get("errors") or not isinstance(files, dict) or not files
            or any(not isinstance(name, str) or not re.fullmatch(r"[0-9a-f]{64}", str(value)) for name, value in files.items())):
        raise ValueError("Missing complete installed-source verification")
    manifest = hash_value(files)
    if proof.get("source_manifest_sha256") != manifest or runtime.get("source_manifest_sha256") != manifest:
        raise ValueError("Runtime/source report manifests disagree")
    if files != expected_sources():
        raise ValueError("Actual installed source map differs from the complete frozen expected source map")
    dependency = read(directory / "environment/dependency-audit.json")
    validate_dependency_report(dependency)
    regression = read(directory / "cpu/regression.json")
    if (regression.get("status") != "PASS" or type(regression.get("tests_run")) is not int
            or regression["tests_run"] < 1 or any(regression.get(key) != 0 for key in ("failures", "errors", "skipped"))):
        raise ValueError("Required image regressions incomplete")
    validate_extra_gates(directory, commit, runtime, files)
    return {"status": "PASS", "source_manifest_sha256": manifest, "sources": len(files),
            "legal_manifest_sha256": runtime["legal_manifest_sha256"]}


def copy_reports(image, destination):
    destination.mkdir(parents=True, exist_ok=True)
    container = docker("create", "--entrypoint", "python3", image, "-c", "pass").strip()
    try:
        docker("cp", container + ":" + BUILD_INFO + "/.", str(destination))
    finally:
        docker("rm", "-v", container)


def validate(directory):
    commit = identity(os.environ)
    preparation = read(directory / "preparation.json")
    if preparation.get("commit") != commit:
        raise ValueError("Wrong preparation source commit")
    child = inspect(preparation["image"])
    config = compare_config(read(directory / "parent-inspect.json"), child, commit)
    image_id = child["Id"]
    copy_reports(image_id, directory / "build")
    built = validate_reports(directory / "build/validation", commit)
    final = (directory / "final").resolve()
    final.mkdir(parents=True, exist_ok=True)
    final.chmod(0o777)
    # Run against the immutable imageID, never a movable tag. No dependency
    # downloads, root permission workaround or additional image build occurs.
    output = docker("run", "--rm", "--network", "none", "--user", "1000:0",
                    "--tmpfs", "/tmp:rw,nosuid,nodev,size=4g", "-e", "UV_NO_CACHE=1",
                    "--mount", f"type=bind,source={final},target=/evidence",
                    "--entrypoint", "python3", image_id, "-B", RUNTIME, "validate", "--output-dir", "/evidence",
                    log=directory / "final-runtime.txt")
    (directory / "final-runtime.txt").write_text(output, encoding="utf-8")
    checked = validate_reports(final, commit)
    if checked != built:
        raise ValueError("Final actual-image validation differs from build-time evidence")
    result = {"schema": "omni032-ci-candidate/1", "status": "PASS_CPU_ONLY", "commit": commit,
              "image_id": image_id, "image": preparation["image"], "recipe_sha256": preparation["recipe_sha256"],
              "source_manifest_sha256": checked["source_manifest_sha256"], "config": config, "gpu_validation_status": "NOT_RUN"}
    dump(directory / "candidate.json", result)
    return result


def digest_from_push(output):
    values = set(re.findall(r"digest:\s*(sha256:[0-9a-f]{64})", output))
    if len(values) != 1:
        raise ValueError("Push output must identify exactly one publication digest")
    return values.pop()


def publish(directory):
    commit = identity(os.environ)
    candidate = read(directory / "candidate.json")
    if candidate.get("status") != "PASS_CPU_ONLY" or candidate.get("commit") != commit:
        raise ValueError("No validated exact-image candidate")
    if inspect(candidate["image"]).get("Id") != candidate["image_id"]:
        raise ValueError("Final tag changed after validation")
    checked = validate_reports(directory / "final", commit)
    if checked["source_manifest_sha256"] != candidate["source_manifest_sha256"]:
        raise ValueError("Final source evidence changed")
    # Publication request is create-once even if the push response is lost.
    ops.claim_request(directory / "publication-request.json", {"status": "REQUESTED", "commit": commit,
                      "image_id": candidate["image_id"], "image": candidate["image"], "request_count": 1})
    output = docker("push", candidate["image"], log=directory / "push.txt")
    (directory / "push.txt").write_text(output, encoding="utf-8")
    digest = digest_from_push(output)
    reference = REPOSITORY + "@" + digest
    docker("pull", reference, log=directory / "published-digest-pull.txt")
    pulled = inspect(reference)
    if pulled.get("Id") != candidate["image_id"]:
        raise ValueError("Published digest is not the exact validated imageID")
    receipt = {**candidate, "schema": "omni032-image-publication/1", "status": "PUBLISHED_CPU_VALIDATED_GPU_PENDING",
               "published_image": reference, "digest": digest,
               "run_url": f"https://github.com/githubgxll/dynamo/actions/runs/{os.environ['GITHUB_RUN_ID']}",
               "push_count": 1, "gpu_validation_status": "NOT_RUN"}
    dump(directory / "publication.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "validate", "publish"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    result = prepare(args.directory, args.github_output) if args.action == "prepare" else validate(args.directory) if args.action == "validate" else publish(args.directory)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
