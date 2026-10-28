#!/usr/bin/env python3
"""Build evidence and exact-image publication for two independent native SR stacks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = Path("delivery/sr-native/source-lock.json")
BRANCH = "Dingorouter-H3-vllm-omni-jirx"
REPOSITORY = "registry.hd-04.alayanew.com:8443/openclaw/vllm-omni"
BACKENDS = ("seedvr2", "flashvsr-plus")
BASES = {
    "seedvr2": "nvidia/cuda@sha256:cc55d151af1e8e083f3210af753a5cfbcbc5455421531eb0459887026bb4699f",
    "flashvsr-plus": "nvidia/cuda@sha256:cb239b67719dfa32ec6b525b54c1b78559bebd51a47249dd702f6c5429372154",
}
TAG_PREFIXES = {"seedvr2": "sr-native-seedvr2-3b7b", "flashvsr-plus": "sr-native-flashvsr-plus-base-tc4"}
TORCH = {"seedvr2": "2.3.0+cu121", "flashvsr-plus": "2.7.1+cu126"}


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object: " + str(path))
    return value


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(command, log=None, timeout=1800):
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=timeout, check=False)
    output = result.stdout + result.stderr
    if log:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        Path(log).write_text(output, encoding="utf-8", newline="\n")
    if result.returncode:
        print(output[-12000:], file=sys.stderr)
        raise RuntimeError(f"{command[0]} exited {result.returncode}; inspect the saved command log")
    return result.stdout


def docker(*args, log=None):
    return execute(["docker", *args], log=log)


def inspect(image):
    values = json.loads(docker("image", "inspect", image))
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError("Expected exactly one image")
    return values[0]


def write_outputs(path, values):
    if path:
        with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
            for key, value in values.items():
                if "\n" in str(value) or "\r" in str(value):
                    raise ValueError("Multiline Actions output forbidden")
                stream.write(f"{key}={value}\n")


def identity(environ):
    commit = environ.get("GITHUB_SHA", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Expected full GITHUB_SHA")
    if environ.get("GITHUB_REF_NAME") != BRANCH or environ.get("GITHUB_EVENT_NAME") not in {"push", "workflow_dispatch"}:
        raise ValueError("Use the designated existing branch")
    for name in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT"):
        if not re.fullmatch(r"[1-9][0-9]*", environ.get(name, "")):
            raise ValueError("Invalid Actions identity: " + name)
    return commit


def validate_lock(lock):
    for key, expected in {"schema": "sr-native-source-lock/1", "remote_branch": BRANCH,
                          "repository": "https://github.com/githubgxll/dynamo.git",
                          "harbor_repository": REPOSITORY, "build_recipe": "sr-native"}.items():
        if lock.get(key) != expected:
            raise ValueError("Unexpected source lock: " + key)
    if set(lock.get("models", {})) != set(BACKENDS):
        raise ValueError("Exactly the two reviewed native backends are required")
    for backend in BACKENDS:
        model = lock["models"][backend]
        for key, expected in {"base_image": BASES[backend], "dockerfile": f"container/sr-native/{backend}/Dockerfile",
                              "tag_prefix": TAG_PREFIXES[backend]}.items():
            if model.get(key) != expected:
                raise ValueError("Unreviewed backend identity: " + backend + "/" + key)


def expected_inputs(backend, environ):
    if backend not in BACKENDS:
        raise ValueError("Unknown backend")
    commit = identity(environ)
    lock = read(ROOT / LOCK_PATH)
    validate_lock(lock)
    run = environ["GITHUB_RUN_ID"] + "-" + environ["GITHUB_RUN_ATTEMPT"]
    return {"schema": "sr-native-ci-inputs/1", "backend": backend, "source_commit": commit,
            "run_id": environ["GITHUB_RUN_ID"], "run_attempt": environ["GITHUB_RUN_ATTEMPT"],
            "source_lock_sha256": sha256(ROOT / LOCK_PATH), "registry": REPOSITORY.split("/", 1)[0],
            "base_image": BASES[backend], "dockerfile": lock["models"][backend]["dockerfile"],
            "candidate_image": f"sr-native-{backend}-ci:{commit[:12]}-{run}",
            "final_image": f"{REPOSITORY}:{TAG_PREFIXES[backend]}-{commit[:12]}-{run}"}


def prepare(backend, environ, directory, output=None):
    values = expected_inputs(backend, environ)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Use a fresh evidence directory")
    directory.mkdir(parents=True, exist_ok=True)
    execute([sys.executable, "-B", str(ROOT / "delivery/sr-native/ops.py"), "check-seal", "--committed"],
            directory / "input-seal-check.txt")
    execute([sys.executable, "-B", str(ROOT / "delivery/sr-native/ops.py"), "export-recipe",
             "--directory", str(directory / "recipe")], directory / "recipe-export.txt")
    dump(directory / "inputs.json", values)
    write_outputs(output, values)


def checked_inputs(backend, environ, directory):
    expected = expected_inputs(backend, environ)
    if read(directory / "inputs.json") != expected:
        raise ValueError("Evidence belongs to another source, backend or Actions attempt")
    return expected


def pull_with_retry(reference, directory, prefix):
    """Retry only an immutable identity; never substitute a tag or registry mirror."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9./:_-]*@sha256:[0-9a-f]{64}", reference):
        raise ValueError("Registry pulls require an immutable image reference")
    for attempt in range(1, 5):
        log = directory / f"{prefix}-{attempt}.txt"
        try:
            docker("pull", "--platform", "linux/amd64", reference, log=log)
            return
        except (RuntimeError, subprocess.TimeoutExpired):
            text = log.read_text(encoding="utf-8") if log.exists() else "timeout"
            if re.search(r"unauthorized|authentication required|denied:|manifest unknown|no matching manifest", text, re.I):
                raise
            if attempt == 4:
                raise
            time.sleep(5 * 2 ** (attempt - 1))


def validate_base(image, reference):
    if image.get("Os") != "linux" or image.get("Architecture") != "amd64":
        raise ValueError("Base is not linux/amd64")
    # Docker canonicalizes unqualified Docker Hub references to docker.io in some versions.
    digests = {item.removeprefix("docker.io/") for item in image.get("RepoDigests", [])}
    if reference.removeprefix("docker.io/") not in digests:
        raise ValueError("Pulled base has no matching immutable RepoDigest")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image.get("Id", "")):
        raise ValueError("Missing base image ID")


def pull_base(backend, environ, directory):
    values = checked_inputs(backend, environ, directory)
    pull_with_retry(values["base_image"], directory, "base-pull")
    image = inspect(values["base_image"])
    validate_base(image, values["base_image"])
    dump(directory / "base-image-inspect.json", image)
    dump(directory / "base-resolved.json", {"status": "PASS_IMMUTABLE_BASE_PULL", "backend": backend,
         "base_image": values["base_image"], "image_id": image["Id"], "platform": "linux/amd64",
         "source": "Reviewed Docker Registry amd64 manifest; no per-run tag re-resolution"})


def env_map(config):
    values = {}
    for entry in config.get("Env", []):
        key, separator, value = entry.partition("=")
        if not separator or key in values:
            raise ValueError("Duplicate or malformed image environment")
        values[key] = value
    return values


def validate_image(image, values):
    if image.get("Os") != "linux" or image.get("Architecture") != "amd64":
        raise ValueError("Candidate is not linux/amd64")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image.get("Id", "")):
        raise ValueError("Invalid candidate image ID")
    config = image["Config"]
    if str(config.get("User", "")).split(":", 1)[0] not in {"1000", "sr-native"}:
        raise ValueError("Candidate must default to nonroot UID1000")
    if env_map(config).get("DYNAMO_COMMIT_SHA") != values["source_commit"]:
        raise ValueError("Wrong candidate source marker")
    labels = config.get("Labels") or {}
    if labels.get("org.opencontainers.image.revision") != values["source_commit"]:
        raise ValueError("Wrong image revision label")
    if labels.get("org.opencontainers.image.base.name") != values["base_image"]:
        raise ValueError("Wrong immutable base label")


INVENTORY_PROBE = r'''
import importlib.metadata as m, json, os, subprocess
packages=[]
for d in m.distributions():
 packages.append({"name":d.metadata.get("Name"),"version":d.version,
  "metadata":d.read_text("METADATA"),"direct_url":d.read_text("direct_url.json"),
  "license":d.metadata.get("License-Expression") or d.metadata.get("License") or "UNKNOWN"})
packages.sort(key=lambda x:(x["name"] or "", x["version"], x["metadata"] or "", x["direct_url"] or ""))
dpkg=subprocess.run(["dpkg-query","-W","-f=${binary:Package}\t${Version}\n"],check=True,capture_output=True,text=True)
print(json.dumps({"euid":os.geteuid(),"packages":packages,"dpkg":sorted(dpkg.stdout.splitlines()),
 "source_commit":os.environ.get("DYNAMO_COMMIT_SHA")}))
'''


def validate_inventory(value, backend, commit):
    if value.get("euid") != 1000 or value.get("source_commit") != commit:
        raise ValueError("Wrong inventory source/UID")
    if not value.get("packages") or not value.get("dpkg"):
        raise ValueError("Missing package inventory")
    packages = {}
    for item in value["packages"]:
        name = re.sub(r"[-_.]+", "-", item.get("name") or "").lower()
        if not name or name in packages or not item.get("version"):
            raise ValueError("Duplicate, unnamed or unversioned installed distribution")
        packages[name] = item["version"]
    if packages.get("torch") != TORCH[backend]:
        raise ValueError("Wrong native Torch stack")
    if any(name in packages for name in ("vllm", "vllm-omni", "ai-dingo", "ai-dingo-runtime")):
        raise ValueError("Native reference image must not mix the inherited Omni/Gateway stack")


def sbom(inventory):
    components = []
    for item in inventory["packages"]:
        name = re.sub(r"[-_.]+", "-", item["name"]).lower()
        components.append({"type": "library", "name": name, "version": item["version"],
            "purl": "pkg:pypi/" + name + "@" + item["version"],
            "properties": [{"name": "declared-license-unverified", "value": item["license"]}]})
    for item in inventory["dpkg"]:
        name, separator, version = item.partition("\t")
        if not separator:
            raise ValueError("Malformed dpkg inventory")
        components.append({"type": "library", "name": name, "version": version,
                           "properties": [{"name": "source", "value": "dpkg-query"}]})
    return {"bomFormat": "CycloneDX", "specVersion": "1.5", "version": 1, "components": components,
            "metadata": {"properties": [{"name": "scope", "value": "Package inventory; declared license evidence only. Not a license-policy approval or GPU qualification."}]}}


def validate_report(report, values):
    expected = {"status": "PASS_CPU_ONLY", "euid": 1000, "backend": values["backend"],
                "source_commit": values["source_commit"], "gpu_validation_status": "NOT_RUN"}
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError("Final CPU runtime gate failed: " + key)
    if report.get("errors"):
        raise ValueError("Final CPU runtime gate has errors")


def evidence_hashes(directory):
    return {p.relative_to(directory).as_posix(): sha256(p) for p in sorted(directory.rglob("*"))
            if p.is_file() and p.name not in {"candidate-verified.json", "image.json", "deployment-image.txt"}
            and not p.name.startswith("registry-")}


def run_cpu_validation(image_id, values, directory):
    cid = docker("create", "--network", "none", "--user", "1000:0", "--cap-drop", "ALL",
                 "--security-opt", "no-new-privileges", "--env", "HOME=/tmp", "--env", "TMPDIR=/tmp",
                 "--env", "PYTHONDONTWRITEBYTECODE=1", "--entrypoint", "python3", image_id, "-B",
                 f"/opt/sr-native/{values['backend']}/audit.py", "--output-dir", "/tmp/sr-native-final").strip()
    try:
        failed = None
        try:
            docker("start", "--attach", cid, log=directory / "final-container.log")
        except RuntimeError as exc:
            failed = exc
        state = json.loads(docker("container", "inspect", cid))[0]["State"]
        final = directory / "final"
        final.mkdir(parents=True, exist_ok=False)
        try:
            docker("cp", cid + ":/tmp/sr-native-final/.", str(final))
        except RuntimeError:
            if not failed and state.get("ExitCode") == 0:
                raise
        dump(directory / "final-container-state.json", {"exit_code": state.get("ExitCode"), "oom_killed": state.get("OOMKilled")})
        if failed or state.get("ExitCode") != 0 or state.get("OOMKilled"):
            raise RuntimeError("Offline final-image CPU validation failed; see final-container.log") from failed
        validate_report(read(final / "runtime-validation.json"), values)
    finally:
        docker("rm", cid)


def verify_candidate(backend, environ, directory):
    values = checked_inputs(backend, environ, directory)
    base = read(directory / "base-resolved.json")
    if base.get("status") != "PASS_IMMUTABLE_BASE_PULL" or base.get("base_image") != values["base_image"]:
        raise ValueError("No verified immutable base")
    image = inspect(values["candidate_image"])
    validate_image(image, values)
    dump(directory / "candidate-image-inspect.json", image)
    run_cpu_validation(image["Id"], values, directory)
    common = ["run", "--rm", "--network", "none", "--user", "1000:0", "--read-only",
              "--tmpfs", "/tmp:rw,nosuid,nodev,size=2g", "--env", "HOME=/tmp",
              "--env", "PYTHONDONTWRITEBYTECODE=1", "--entrypoint", "python3", image["Id"], "-B"]
    # pip check needs no resolver or shared uv cache. No known metadata exceptions exist in these clean stacks.
    docker(*common, "-m", "pip", "check", log=directory / "pip-check.txt")
    inventory = json.loads(docker(*common, "-c", INVENTORY_PROBE))
    validate_inventory(inventory, backend, values["source_commit"])
    dump(directory / "installed-inventory.json", inventory)
    dump(directory / "packages.cdx.json", sbom(inventory))
    dump(directory / "candidate-verified.json", {"status": "PASS_CANDIDATE_CPU_ONLY",
         "image_id": image["Id"], "backend": backend, "source_commit": values["source_commit"],
         "candidate_image": values["candidate_image"], "base_image": values["base_image"],
         "gpu_validation_status": "NOT_RUN", "evidence_sha256": evidence_hashes(directory)})


def publication_digest(output):
    values = set(re.findall(r"\bdigest:\s*(sha256:[0-9a-f]{64})\b", output))
    if len(values) != 1:
        raise ValueError("Expected exactly one Docker push digest")
    return values.pop()


def publish(backend, environ, directory, output=None):
    values = checked_inputs(backend, environ, directory)
    proof = read(directory / "candidate-verified.json")
    expected = {"status": "PASS_CANDIDATE_CPU_ONLY", "backend": backend,
                "source_commit": values["source_commit"], "candidate_image": values["candidate_image"],
                "base_image": values["base_image"], "gpu_validation_status": "NOT_RUN"}
    if any(proof.get(k) != v for k, v in expected.items()):
        raise ValueError("Candidate has not passed the exact CPU gate")
    if proof.get("evidence_sha256") != evidence_hashes(directory):
        raise ValueError("CPU evidence changed after verification")
    validate_report(read(directory / "final/runtime-validation.json"), values)
    validate_inventory(read(directory / "installed-inventory.json"), backend, values["source_commit"])
    image = inspect(values["candidate_image"])
    validate_image(image, values)
    if image["Id"] != proof.get("image_id"):
        raise ValueError("Candidate tag changed after CPU verification")
    docker("tag", image["Id"], values["final_image"])
    docker("push", values["final_image"], log=directory / "registry-push.txt")
    digest = publication_digest((directory / "registry-push.txt").read_text(encoding="utf-8"))
    reference = REPOSITORY + "@" + digest
    pull_with_retry(reference, directory, "registry-pull")
    published = inspect(reference)
    validate_image(published, values)
    if published["Id"] != image["Id"]:
        raise ValueError("Registry pull differs from the tested image")
    record = {"schema": "sr-native-publication/1", "status": "PASS_PUBLISHED_AND_PULLED_CPU_VALIDATED",
              "backend": backend, "branch": BRANCH, "source_commit": values["source_commit"],
              "image": values["final_image"], "deployment_image": reference, "digest": digest,
              "image_id": image["Id"], "base_image": values["base_image"],
              "run_id": values["run_id"], "run_attempt": values["run_attempt"],
              "actions_url": "https://github.com/githubgxll/dynamo/actions/runs/" + values["run_id"],
              "gpu_validation_status": "NOT_RUN", "evidence_sha256": evidence_hashes(directory),
              "not_run": ["CUDA native kernels", "distributed SeedVR2 SP4", "complete 17-frame inference",
                          "model weight mount identity", "media and human quality acceptance", "Gateway integration"]}
    dump(directory / "image.json", record)
    (directory / "deployment-image.txt").write_text(reference + "\n", encoding="utf-8", newline="\n")
    write_outputs(output, {"digest": digest, "deployment_image": reference})
    print(json.dumps({k: record[k] for k in ("status", "backend", "deployment_image", "gpu_validation_status")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "pull-base", "verify-candidate", "publish"))
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.command == "prepare":
        prepare(args.backend, os.environ, directory, args.github_output)
    elif args.command == "pull-base":
        pull_base(args.backend, os.environ, directory)
    elif args.command == "verify-candidate":
        verify_candidate(args.backend, os.environ, directory)
    else:
        publish(args.backend, os.environ, directory, args.github_output)


if __name__ == "__main__":
    main()
