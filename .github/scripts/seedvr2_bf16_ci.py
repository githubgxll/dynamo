#!/usr/bin/env python3
"""Validate one loaded SeedVR2 BF16 imageID before one publication operation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("seedvr2_bf16_ops", ROOT / "delivery/seedvr2-bf16-production/ops.py")
ops = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ops)
BRANCH, REPOSITORY, PARENT, PARENT_ID = ops.BRANCH, ops.HARBOR, ops.PARENT, ops.PARENT_ID
RUNTIME = "/opt/seedvr2-bf16/runtime.py"
CONFIG_FIELDS = ("User", "Entrypoint", "Cmd", "WorkingDir", "ExposedPorts", "Volumes", "Healthcheck",
                 "StopSignal", "Shell", "OnBuild")
read, dump = ops.read, ops.dump


def command(arguments, log=None, timeout=1800):
    try:
        result = subprocess.run(arguments, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                timeout=timeout)
    except subprocess.TimeoutExpired as error:
        if log is not None:
            Path(log).parent.mkdir(parents=True, exist_ok=True)
            output = "\n".join(value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
                               for value in (error.stdout, error.stderr))
            Path(log).write_text(output + "\nCommand timed out; outcome remains unconfirmed.\n", encoding="utf-8")
        raise
    if log is not None:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        Path(log).write_text(result.stdout + result.stderr, encoding="utf-8")
    return result


def execute(arguments, log=None):
    result = command(arguments, log)
    if result.returncode:
        raise RuntimeError("Command failed: " + arguments[0] + "\n" + result.stdout[-8000:] + result.stderr[-8000:])
    return result.stdout


def docker(*arguments, log=None):
    return execute(["docker", *arguments], log=log)


def inspect(image):
    entries = json.loads(docker("image", "inspect", image))
    if len(entries) != 1 or not re.fullmatch(r"sha256:[0-9a-f]{64}", entries[0].get("Id", "")):
        raise ValueError("Expected one exact Docker image identity")
    return entries[0]


def run_identity(environ=None):
    env = os.environ if environ is None else environ
    if (env.get("GITHUB_REF_NAME") != BRANCH or env.get("GITHUB_EVENT_NAME") != "push"
            or not re.fullmatch(r"[0-9a-f]{40}", env.get("GITHUB_SHA", ""))
            or not re.fullmatch(r"[1-9][0-9]*", env.get("GITHUB_RUN_ID", ""))
            or not re.fullmatch(r"[1-9][0-9]*", env.get("GITHUB_RUN_ATTEMPT", ""))):
        raise ValueError("A push on the designated branch with actual positive run/attempt is required")
    return {"commit": env["GITHUB_SHA"], "run_id": env["GITHUB_RUN_ID"],
            "run_attempt": int(env["GITHUB_RUN_ATTEMPT"])}


def require_current(record):
    if any(record.get(key) != value for key, value in run_identity().items()):
        raise ValueError("Evidence belongs to a different source/run/attempt")


def image_tag(commit):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("A full source commit is required")
    return REPOSITORY + ":seedvr2-dit-bf16-" + commit[:12]


def classify_manifest_lookup(result, image=None):
    if result.returncode == 0:
        return "EXISTS"
    if result.returncode != 1:
        return "UNKNOWN"
    raw = result.stdout + "\n" + result.stderr
    detail = raw.casefold()
    harbor_missing = False
    if isinstance(image, str) and image.startswith(REPOSITORY + ":"):
        tag = image[len(REPOSITORY) + 1:]
        if re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", tag):
            artifact = REPOSITORY.split("/", 1)[1] + ":" + tag
            harbor_missing = raw.strip() == "unknown: artifact " + artifact + " not found"
    missing = "manifest unknown" in detail or "no such manifest" in detail or harbor_missing
    uncertain = any(marker in detail for marker in (
        "unauthorized", "authentication", "authorization", "forbidden", "denied", "failed to fetch", "x509",
        "certificate", "tls", "ssl", "eof", "timeout", "timed out", "connection", "network", "no such host",
        "dial tcp", "proxyconnect", "lookup", "bad gateway", "service unavailable", "unexpected status",
        "toomanyrequests", "too many requests", "rate limit", "internal server error"))
    statuses = re.findall(r"\b(?:http(?:/\d(?:\.\d)?)?|status(?:\s+code)?)\s*[:=]?\s*([45][0-9]{2})\b", detail)
    return "ABSENT_CONFIRMED" if missing and not uncertain and all(code == "404" for code in statuses) else "UNKNOWN"


def lookup(image, directory, filename="tag-inspect.txt"):
    try:
        result = command(["docker", "manifest", "inspect", image], directory / filename, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as error:
        dump(directory / (filename + ".error.json"), {"type": type(error).__name__, "message": str(error)})
        return "UNKNOWN"
    return classify_manifest_lookup(result, image)


def pull_digest(output):
    matches = set(re.findall(r"(?m)^Digest:\s*(sha256:[0-9a-f]{64})\s*$", output))
    if len(matches) != 1:
        raise ValueError("Pull did not report one exact immutable digest")
    return matches.pop()


def prepare(directory, github_output=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    identity = run_identity()
    value, _, seal = ops.check_seal(committed=True)
    if ops.git("rev-parse", "HEAD") != identity["commit"]:
        raise ValueError("The actual checkout differs from the triggering commit")
    if not ops.valid_request_message(ops.git("show", "-s", "--format=%B", "HEAD")):
        raise ValueError("The source commit must contain only the unique BF16 build marker")
    image = image_tag(identity["commit"])
    state = lookup(image, directory)
    dump(directory / "initial-publication-observation.json", {**identity, "image": image, "publication_state": state})
    if state == "UNKNOWN":
        raise RuntimeError("Harbor tag state remains UNKNOWN; do not build or push")
    docker("pull", "--platform", "linux/amd64", PARENT, log=directory / "parent-pull.txt")
    parent = inspect(PARENT)
    if parent["Id"] != PARENT_ID or parent.get("Os") != "linux" or parent.get("Architecture") != "amd64":
        raise ValueError("The immutable production parent does not match its verified imageID/platform")
    dump(directory / "parent-inspect.json", parent)
    existing_digest = None
    if state == "EXISTS":
        existing_digest = pull_digest(docker("pull", "--platform", "linux/amd64", image,
                                              log=directory / "existing-tag-pull.txt"))
        reference = REPOSITORY + "@" + existing_digest
        docker("pull", "--platform", "linux/amd64", reference, log=directory / "existing-digest-pull.txt")
        compare_config(parent, inspect(reference), identity["commit"], value)
    record = {"schema": "seedvr2-bf16-preparation/1", **identity, "image": image, "parent_image": PARENT,
              "parent_image_id": PARENT_ID, "recipe_sha256": seal["recipe_sha256"], "review_tree": seal["review_tree"],
              "tag_inspection": state, "reused": state == "EXISTS", "existing_digest": existing_digest}
    claim(directory / "preparation.json", record)
    if github_output is not None:
        with Path(github_output).open("a", encoding="utf-8", newline="\n") as stream:
            outputs = {"image": image, "base_image": PARENT, "build_required": "false" if record["reused"] else "true"}
            for key, value in outputs.items():
                if "\n" in value or "\r" in value:
                    raise ValueError("Unsafe workflow output")
                stream.write(key + "=" + value + "\n")
    return record


def compare_config(parent, child, commit, value):
    if parent.get("Id") != PARENT_ID or child.get("Os") != "linux" or child.get("Architecture") != "amd64":
        raise ValueError("Candidate/parent platform differs")
    for name in CONFIG_FIELDS:
        if child["Config"].get(name) != parent["Config"].get(name):
            raise ValueError("The production parent's configuration changed: " + name)
    before = dict(item.split("=", 1) for item in parent["Config"].get("Env", []))
    after = dict(item.split("=", 1) for item in child["Config"].get("Env", []))
    expected = dict(before)
    expected.update({key: item.replace("{source_commit}", commit)
                     for key, item in value["expected_environment_additions"].items()})
    if after != expected:
        raise ValueError("Inherited environment or reviewed child identity changed")
    labels = child["Config"].get("Labels") or {}
    if labels.get("org.opencontainers.image.revision") != commit or labels.get("org.opencontainers.image.base.name") != PARENT:
        raise ValueError("Candidate labels differ from frozen source/parent")
    layers = parent.get("RootFS", {}).get("Layers", [])
    actual = child.get("RootFS", {}).get("Layers", [])
    if not layers or actual[:len(layers)] != layers or len(actual) <= len(layers):
        raise ValueError("Candidate must extend every pinned parent layer")


def validate_reports(directory, commit):
    directory = Path(directory)
    runtime, proof = read(directory / "runtime-validation.json"), read(directory / "source-verification.json")
    for key, expected in {"schema": "seedvr2-bf16-runtime/1", "status": "PASS_CPU_ONLY", "source_commit": commit,
                          "parent_image": PARENT, "parent_image_id": PARENT_ID, "euid": 1000,
                          "inventory_preserved": True, "errors": [], "gpu_status": "NOT_RUN", "cli_status": "PASS"}.items():
        if runtime.get(key) != expected:
            raise ValueError("Actual final runtime validation failed: " + key)
    if runtime.get("inventory_preserved") is not True:
        raise ValueError("The actual parent package inventory must be preserved")
    if runtime.get("dependency_status") not in ("PASS", "PASS_WITH_UPSTREAM_NCCL_OVERRIDE"):
        raise ValueError("The unchanged strict dependency auditor must succeed")
    tests = runtime.get("tests")
    if (not isinstance(tests, dict) or type(tests.get("tests")) is not int or tests["tests"] <= 0
            or any(type(tests.get(key)) is not int or tests[key] != 0 for key in ("failures", "errors", "skipped"))):
        raise ValueError("Actual CPU regressions must run nonempty with zero failures/errors/skips")
    precision = runtime.get("precision_cpu_checks", {})
    checks = precision.get("checks")
    required = ops.lock()["required_precision_cpu_checks"]
    if (precision.get("status") != "PASS_REAL_CPU_TORCH_HELPERS_ONLY" or precision.get("gpu_validation") != "NOT_RUN"
            or not isinstance(checks, list) or any(not isinstance(name, str) or not name for name in checks)
            or len(set(checks)) != len(checks) or not set(required).issubset(checks)):
        raise ValueError("Actual CPU conversion/RoPE/autocast/velocity bridge/unsafe checks are incomplete")
    expected = read(ROOT / "delivery/seedvr2-bf16-production/expected-source-manifest.json")
    files = expected.get("source_sha256")
    if (not isinstance(files, dict) or not files or expected.get("source_manifest_sha256") != ops.hash_value(files)
            or any(not isinstance(name, str) or not name.startswith("vllm_omni/") or "\\" in name
                   or any(part in ("", ".", "..") for part in name.split("/"))
                   or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                   for name, digest in files.items())):
        raise ValueError("Frozen complete expected Omni source map is malformed")
    for key, item in {"schema": "seedvr2-bf16-source-proof/1", "status": "PASS", "source_commit": commit,
                      "parent_image": PARENT, "parent_image_id": PARENT_ID, "source_sha256": files,
                      "source_manifest_sha256": ops.hash_value(files)}.items():
        if proof.get(key) != item:
            raise ValueError("Installed source proof differs from the frozen full map: " + key)
    if runtime.get("source_manifest_sha256") != ops.hash_value(files):
        raise ValueError("Runtime and complete installed source proof differ")
    return {"runtime": runtime, "source_proof": proof}


def validate(directory):
    directory = Path(directory)
    prepared = read(directory / "preparation.json")
    require_current(prepared)
    reference = REPOSITORY + "@" + prepared["existing_digest"] if prepared["reused"] else prepared["image"]
    image = inspect(reference)
    compare_config(read(directory / "parent-inspect.json"), image, prepared["commit"], ops.lock())
    output = directory / "final"
    output.mkdir(exist_ok=False)
    output.chmod(0o777)  # Only this transaction's newly created empty evidence directory.
    docker("run", "--rm", "--network", "none", "--user", "1000:1000", "--entrypoint", "python3",
           "--mount", "type=bind,src=" + str(output.resolve()) + ",dst=/tmp/seedvr2-bf16-evidence",
           image["Id"], "-B", RUNTIME, "validate", "--output-dir", "/tmp/seedvr2-bf16-evidence",
           log=directory / "actual-image-validation.txt")
    report = validate_reports(output, prepared["commit"])
    if inspect(reference)["Id"] != image["Id"]:
        raise ValueError("Candidate reference changed during imageID validation")
    record = {"schema": "seedvr2-bf16-candidate/1", "status": "PASS_CPU_ONLY", **run_identity(),
              "image": prepared["image"], "image_id": image["Id"], "parent_image": PARENT,
              "parent_image_id": PARENT_ID, "recipe_sha256": prepared["recipe_sha256"],
              "runtime_evidence_sha256": ops.hash_value(report), "gpu_validation_status": "NOT_RUN"}
    claim(directory / "candidate.json", record)
    return record


def claim(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def publish(directory):
    directory = Path(directory)
    prepared, candidate = read(directory / "preparation.json"), read(directory / "candidate.json")
    require_current(prepared)
    require_current(candidate)
    reference = REPOSITORY + "@" + prepared["existing_digest"] if prepared["reused"] else prepared["image"]
    if (candidate.get("status") != "PASS_CPU_ONLY" or candidate.get("image") != prepared["image"]
            or inspect(reference)["Id"] != candidate["image_id"]):
        raise ValueError("Only the exact currently CPU-validated imageID can be published")
    report = validate_reports(directory / "final", prepared["commit"])
    if (candidate.get("recipe_sha256") != prepared["recipe_sha256"]
            or candidate.get("runtime_evidence_sha256") != ops.hash_value(report)):
        raise ValueError("Candidate no longer matches the frozen recipe and runtime evidence")
    claim(directory / "publication-operation.json", {"operation": "REUSE" if prepared["reused"] else "PUSH",
                                                    **run_identity(), "candidate": candidate})
    if prepared["reused"]:
        digest = prepared["existing_digest"]
    else:
        if lookup(prepared["image"], directory, "pre-push-tag-inspect.txt") != "ABSENT_CONFIRMED":
            raise RuntimeError("Final tag is no longer confirmed absent; stop before push")
        output = docker("push", prepared["image"], log=directory / "push.txt")
        matches = set(re.findall(r"digest:\s*(sha256:[0-9a-f]{64})", output))
        if len(matches) != 1:
            raise ValueError("Push response is ambiguous; confirm actual Harbor state before any retry")
        digest = matches.pop()
    reference = REPOSITORY + "@" + digest
    record = {"schema": "seedvr2-bf16-publication/1", "status": "PUBLISHED_PULL_PENDING", **run_identity(),
              "image": prepared["image"], "published_image": reference, "digest": digest,
              "image_id": candidate["image_id"], "parent_image": PARENT, "parent_image_id": PARENT_ID,
              "recipe_sha256": prepared["recipe_sha256"], "runtime_evidence_sha256": candidate["runtime_evidence_sha256"],
              "push_count": 0 if prepared["reused"] else 1, "reused_existing": prepared["reused"],
              "gpu_validation_status": "NOT_RUN"}
    dump(directory / "publication.json", record)
    docker("pull", "--platform", "linux/amd64", reference, log=directory / "published-digest-pull.txt")
    pulled = inspect(reference)
    if pulled["Id"] != candidate["image_id"] or pulled.get("Os") != "linux" or pulled.get("Architecture") != "amd64":
        raise ValueError("Published digest pull does not match the tested imageID/platform")
    dump(directory / "published-digest-inspect.json", pulled)
    (directory / "deployment-image.txt").write_text(reference + "\n", encoding="utf-8")
    record.update(status="PASS_PUBLISHED_AND_PULLED_CPU_VALIDATED", evidence_sha256=evidence_hashes(directory))
    dump(directory / "publication.json", record)
    return record


def evidence_hashes(directory):
    return {path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob("*")) if path.is_file() and path != directory / "publication.json"}


def outcome(directory):
    directory = Path(directory)
    current = run_identity()
    directory.mkdir(parents=True, exist_ok=True)
    image = image_tag(current["commit"])
    status = lookup(image, directory, "final-publication-lookup.txt")
    record = {"schema": "seedvr2-bf16-publication-observation/1", **current,
              "image": image, "publication_state": status,
              "observed_at": datetime.now(timezone.utc).isoformat()}
    dump(directory / "publication-observation.json", record)
    publication = directory / "publication.json"
    if publication.is_file():
        result = read(publication)
        require_current(result)
        if result.get("status") == "PASS_PUBLISHED_AND_PULLED_CPU_VALIDATED":
            result["evidence_sha256"] = evidence_hashes(directory)
            dump(publication, result)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "validate", "publish", "outcome"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    actions = {"prepare": lambda: prepare(args.directory, args.github_output),
               "validate": lambda: validate(args.directory), "publish": lambda: publish(args.directory),
               "outcome": lambda: outcome(args.directory)}
    print(json.dumps(actions[args.action](), indent=2))


if __name__ == "__main__":
    main()
