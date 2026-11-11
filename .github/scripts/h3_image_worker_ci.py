#!/usr/bin/env python3
"""Validate and publish the single source-bound native H3-Image worker."""
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
SPEC = importlib.util.spec_from_file_location("h3_image_worker_ops", ROOT / "delivery/h3-image-worker/ops.py")
ops = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ops)
BRANCH, REPOSITORY, PARENT, PARENT_ID = ops.BRANCH, ops.HARBOR, ops.PARENT, ops.PARENT_ID
RUNTIME = "/opt/h3-image-worker/runtime.py"
MINIMUM_SOURCE_FILES = 2138
CONFIG_FIELDS = ("User", "Entrypoint", "Cmd", "WorkingDir", "ExposedPorts", "Volumes", "Healthcheck",
                 "StopSignal", "Shell", "OnBuild")
read, dump, hash_value = ops.read, ops.dump, ops.hash_value


def command(arguments, log=None):
    result = subprocess.run(arguments, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
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
    if len(entries) != 1:
        raise ValueError("Expected one image identity")
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
    return REPOSITORY + ":h3-image-worker-" + commit[:12]


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
    return classify_manifest_lookup(command(["docker", "manifest", "inspect", image], directory / filename), image)


def pull_digest(output):
    matches = set(re.findall(r"(?m)^Digest:\s*(sha256:[0-9a-f]{64})\s*$", output))
    if len(matches) != 1:
        raise ValueError("Pull did not report one exact immutable digest")
    return matches.pop()


def prepare(directory, github_output=None):
    current = run_identity()
    _, _, seal = ops.check_seal(committed=True)
    message = ops.git("log", "-1", "--format=%B")
    if not ops.valid_request_message(message) or ops.git("rev-parse", "HEAD") != current["commit"]:
        raise ValueError("Checkout/unique commit marker differs from the frozen worker source")
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "preparation.json").exists():
        raise ValueError("This run/attempt has already prepared a transaction")
    image = image_tag(current["commit"])
    state = lookup(image, directory)
    dump(directory / "tag-inspection.json", {"image": image, "status": state, **current})
    if state == "UNKNOWN":
        raise RuntimeError("Tag lookup is UNKNOWN; resolve actual registry state before continuing")
    docker("pull", "--platform", "linux/amd64", PARENT, log=directory / "parent-pull.txt")
    parent = inspect(PARENT)
    if parent.get("Id") != PARENT_ID or parent.get("Os") != "linux" or parent.get("Architecture") != "amd64":
        raise ValueError("Pinned official parent identity differs")
    dump(directory / "parent-inspect.json", parent)
    existing_digest = None
    if state == "EXISTS":
        existing_digest = pull_digest(docker("pull", "--platform", "linux/amd64", image,
                                            log=directory / "existing-pull.txt"))
    record = {"schema": "h3-image-worker-ci-preparation/1", "status": "PASS", **current,
              "image": image, "parent_image": PARENT, "parent_image_id": PARENT_ID,
              "recipe_sha256": seal["recipe_sha256"], "review_tree": seal["review_tree"],
              "tag_inspection": state, "reused": state == "EXISTS", "existing_digest": existing_digest}
    dump(directory / "preparation.json", record)
    if github_output:
        with Path(github_output).open("a", encoding="utf-8", newline="\n") as stream:
            outputs = {"image": image, "base_image": PARENT, "build_required": "false" if record["reused"] else "true"}
            for key, text in outputs.items():
                if "\n" in text or "\r" in text:
                    raise ValueError("Unsafe workflow output")
                stream.write(key + "=" + text + "\n")
    return record


def compare_config(parent, child, commit, value):
    if parent.get("Id") != PARENT_ID or child.get("Os") != "linux" or child.get("Architecture") != "amd64":
        raise ValueError("Candidate/parent platform differs")
    configured = value.get("expected_image_config", {})
    for name in CONFIG_FIELDS:
        if child["Config"].get(name) != configured.get(name, parent["Config"].get(name)):
            raise ValueError("Unreviewed child configuration: " + name)
    if child["Config"].get("User") not in ("1000", "1000:1000", "1000:0", "h3-image", "h3-image:1000"):
        raise ValueError("Final worker must use non-root UID1000")
    before = dict(item.split("=", 1) for item in parent["Config"].get("Env", []))
    after = dict(item.split("=", 1) for item in child["Config"].get("Env", []))
    expected = dict(before)
    expected.update({"H3_IMAGE_WORKER_SOURCE_COMMIT": commit, "H3_IMAGE_WORKER_PARENT_IMAGE": PARENT})
    expected.update(value.get("expected_environment_additions", {}))
    if after != expected:
        raise ValueError("Inherited environment or child identity changed")
    labels = child["Config"].get("Labels") or {}
    if labels.get("org.opencontainers.image.revision") != commit or labels.get("org.opencontainers.image.base.name") != PARENT:
        raise ValueError("Candidate labels differ from frozen source/parent")
    layers = parent.get("RootFS", {}).get("Layers", [])
    actual = child.get("RootFS", {}).get("Layers", [])
    if not layers or actual[:len(layers)] != layers or len(actual) <= len(layers):
        raise ValueError("Candidate must extend every pinned parent layer")


def validate_reports(directory, commit):
    runtime = read(directory / "runtime-validation.json")
    proof = read(directory / "source-verification.json")
    expected = {"status": "PASS_CPU_ONLY", "source_commit": commit, "parent_image": PARENT,
                "parent_image_id": PARENT_ID, "euid": 1000, "inventory_preserved": True, "errors": []}
    for key, value in expected.items():
        if runtime.get(key) != value:
            raise ValueError("Actual final runtime validation failed: " + key)
    if runtime.get("inventory_preserved") is not True:
        raise ValueError("The installed parent inventory must be preserved")
    checks = runtime.get("cpu_checks", {})
    if (checks.get("status") != "PASS_CPU_RUNTIME_ONLY" or type(checks.get("tests_run")) is not int
            or checks["tests_run"] < 10 or any(checks.get(name) != 0 for name in ("failures", "errors", "skipped"))):
        raise ValueError("Actual installed-module CPU regression evidence is incomplete")
    profiles = read(ROOT / "container/h3-image-worker/profiles.json")["profiles"]
    expected_profiles = [{"profile": item["id"], "status": "PASS", "gpus": item["gpus"],
                          "dit_tp": item["dit_tp"], "te_tp": item["te_tp"]} for item in profiles]
    if len(expected_profiles) != 4 or runtime.get("cli_profiles") != expected_profiles:
        raise ValueError("All four actual native CLI profiles must match the frozen configuration")
    files = proof.get("source_sha256")
    if (not isinstance(files, dict) or len(files) < MINIMUM_SOURCE_FILES
            or any(not isinstance(name, str) or not name or not isinstance(digest, str)
                   or not re.fullmatch(r"[0-9a-f]{64}", digest) for name, digest in files.items())):
        raise ValueError("Actual installed source map is incomplete or malformed")
    expected = {"status": "PASS", "source_commit": commit, "parent_image": PARENT,
                "source_manifest_sha256": hash_value(files)}
    if any(proof.get(key) != value for key, value in expected.items()):
        raise ValueError("Actual installed source proof identity differs")
    if runtime.get("source_manifest_sha256") != proof["source_manifest_sha256"]:
        raise ValueError("Runtime and installed source proof differ")
    return runtime, proof


def validate(directory):
    prepared = read(directory / "preparation.json")
    require_current(prepared)
    reference = REPOSITORY + "@" + prepared["existing_digest"] if prepared["reused"] else prepared["image"]
    image = inspect(reference)
    compare_config(read(directory / "parent-inspect.json"), image, prepared["commit"], ops.lock())
    output = directory / "final"
    output.mkdir(exist_ok=False)
    output.chmod(0o777)  # This transaction's new empty evidence directory.
    docker("run", "--rm", "--network", "none", "--user", "1000:1000", "--entrypoint", "python3",
           "--mount", "type=bind,src=" + str(output.resolve()) + ",dst=/tmp/h3-image-worker-evidence",
           image["Id"], "-B", RUNTIME, "validate", "--output-dir", "/tmp/h3-image-worker-evidence",
           log=directory / "actual-image-validation.txt")
    runtime, _ = validate_reports(output, prepared["commit"])
    if inspect(reference)["Id"] != image["Id"]:
        raise ValueError("Candidate tag changed during imageID validation")
    record = {"schema": "h3-image-worker-candidate/1", "status": "PASS_CPU_ONLY", **run_identity(),
              "image": prepared["image"], "image_id": image["Id"], "parent_image": PARENT,
              "parent_image_id": PARENT_ID, "source_manifest_sha256": runtime["source_manifest_sha256"],
              "recipe_sha256": prepared["recipe_sha256"], "gpu_validation_status": "NOT_RUN"}
    dump(directory / "candidate.json", record)
    return record


def claim(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def publish(directory):
    prepared, candidate = read(directory / "preparation.json"), read(directory / "candidate.json")
    require_current(prepared)
    require_current(candidate)
    reference = REPOSITORY + "@" + prepared["existing_digest"] if prepared["reused"] else prepared["image"]
    if candidate.get("status") != "PASS_CPU_ONLY" or inspect(reference)["Id"] != candidate["image_id"]:
        raise ValueError("Only the exact currently CPU-validated imageID can be published")
    runtime, _ = validate_reports(directory / "final", prepared["commit"])
    if (candidate.get("recipe_sha256") != prepared["recipe_sha256"]
            or candidate.get("source_manifest_sha256") != runtime["source_manifest_sha256"]):
        raise ValueError("Candidate no longer matches the frozen recipe and installed-source proof")
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
    record = {"schema": "h3-image-worker-publication/1", "status": "PUBLISHED_PULL_PENDING", **run_identity(),
              "image": prepared["image"], "published_image": reference, "digest": digest,
              "image_id": candidate["image_id"], "parent_image": PARENT, "parent_image_id": PARENT_ID,
              "recipe_sha256": prepared["recipe_sha256"], "source_manifest_sha256": candidate["source_manifest_sha256"],
              "push_count": 0 if prepared["reused"] else 1, "reused_existing": prepared["reused"],
              "gpu_validation_status": "NOT_RUN"}
    dump(directory / "publication.json", record)
    docker("pull", "--platform", "linux/amd64", reference, log=directory / "published-digest-pull.txt")
    pulled = inspect(reference)
    if pulled["Id"] != candidate["image_id"] or pulled.get("Os") != "linux" or pulled.get("Architecture") != "amd64":
        raise ValueError("Published digest pull does not match the tested imageID/platform")
    dump(directory / "published-digest-inspect.json", pulled)
    (directory / "deployment-image.txt").write_text(reference + "\n", encoding="utf-8")
    artifacts = {path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted(directory.rglob("*"))
                 if path.is_file() and path != directory / "publication.json"}
    record.update(status="PASS_PUBLISHED_AND_PULLED_CPU_VALIDATED", evidence_sha256=artifacts)
    dump(directory / "publication.json", record)
    return record


def outcome(directory):
    current = run_identity()
    directory.mkdir(parents=True, exist_ok=True)
    prepared = read(directory / "preparation.json") if (directory / "preparation.json").exists() else {
        **current, "image": image_tag(current["commit"])}
    require_current(prepared)
    status = lookup(prepared["image"], directory, "final-publication-lookup.txt")
    record = {"schema": "h3-image-worker-publication-observation/1", **run_identity(),
              "image": prepared["image"], "publication_state": status,
              "observed_at": datetime.now(timezone.utc).isoformat()}
    dump(directory / "publication-observation.json", record)
    publication = directory / "publication.json"
    if publication.is_file():
        result = read(publication)
        require_current(result)
        if result.get("status") == "PASS_PUBLISHED_AND_PULLED_CPU_VALIDATED":
            result["evidence_sha256"] = {
                path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(directory.rglob("*")) if path.is_file() and path != publication}
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
