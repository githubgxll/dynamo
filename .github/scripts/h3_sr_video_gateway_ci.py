#!/usr/bin/env python3
"""Single final c88-derived image transaction; never publishes a builder."""
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
SPEC = importlib.util.spec_from_file_location("production_ops", ROOT / "delivery/h3-sr-video-gateway/ops.py")
ops = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ops)
BRANCH, REPOSITORY, PARENT, PARENT_ID = ops.BRANCH, ops.HARBOR, ops.PARENT, ops.PARENT_ID
RUNTIME = "/opt/h3-sr/runtime.py"
CONFIG_FIELDS = ("User", "Entrypoint", "Cmd", "WorkingDir", "ExposedPorts", "Volumes", "Healthcheck", "StopSignal", "Shell", "OnBuild")
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
    return {"commit": env["GITHUB_SHA"], "run_id": env["GITHUB_RUN_ID"], "run_attempt": int(env["GITHUB_RUN_ATTEMPT"])}


def require_current(record):
    actual = run_identity()
    if any(record.get(k) != v for k, v in actual.items()):
        raise ValueError("Evidence belongs to a different source/run/attempt")


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
    uncertain = any(marker in detail for marker in ("unauthorized", "authentication", "authorization", "forbidden", "denied",
        "failed to fetch", "x509", "certificate", "tls", "ssl", "eof", "timeout", "timed out", "connection", "network",
        "no such host", "dial tcp", "proxyconnect", "lookup", "bad gateway", "service unavailable", "unexpected status",
        "toomanyrequests", "too many requests", "rate limit", "internal server error"))
    statuses = re.findall(r"\b(?:http(?:/\d(?:\.\d)?)?|status(?:\s+code)?)\s*[:=]?\s*([45][0-9]{2})\b", detail)
    return "ABSENT_CONFIRMED" if missing and not uncertain and all(code == "404" for code in statuses) else "UNKNOWN"


def lookup(image, directory, filename="tag-inspect.txt"):
    result = command(["docker", "manifest", "inspect", image], directory / filename)
    return classify_manifest_lookup(result, image)


def pull_digest(output):
    matches = re.findall(r"(?m)^Digest:\s*(sha256:[0-9a-f]{64})\s*$", output)
    if len(set(matches)) != 1:
        raise ValueError("Pull did not report one exact immutable digest")
    return matches[0]


def prepare(directory, github_output=None):
    current = run_identity()
    value, _, seal = ops.check_seal(committed=True)
    ops.test_dependencies()
    message = ops.git("log", "-1", "--format=%B")
    if not ops.valid_request_message(message) or ops.git("rev-parse", "HEAD") != current["commit"]:
        raise ValueError("Checkout/unique commit marker differs from the requested production source")
    changes = ops.git("diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD").splitlines()
    if not any(name not in ops.AUXILIARY for name in changes):
        raise ValueError("Empty/approval-only request is not a production change")
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "preparation.json").exists():
        raise ValueError("This run/attempt has already prepared a transaction")
    image = REPOSITORY + ":h3-sr-video-gateway-" + current["commit"][:12]
    state = lookup(image, directory)
    dump(directory / "tag-inspection.json", {"image": image, "status": state, **current})
    if state == "UNKNOWN":
        raise RuntimeError("Tag lookup is UNKNOWN; do not treat connectivity/authentication as an absent artifact")
    docker("pull", "--platform", "linux/amd64", PARENT, log=directory / "parent-pull.txt")
    parent = inspect(PARENT)
    if parent.get("Id") != PARENT_ID or parent.get("Os") != "linux" or parent.get("Architecture") != "amd64":
        raise ValueError("Pinned successful parent image identity differs")
    dump(directory / "parent-inspect.json", parent)
    existing_digest = None
    if state == "EXISTS":
        # A previous push can have succeeded before a tail SSL/artifact failure.
        # Revalidate and reuse this artifact; never build or push it again.
        existing_digest = pull_digest(docker("pull", "--platform", "linux/amd64", image, log=directory / "existing-pull.txt"))
    record = {"schema": "h3-sr-video-gateway-ci-preparation/1", "status": "PASS", **current,
              "image": image, "parent_image": PARENT, "recipe_sha256": seal["recipe_sha256"],
              "review_tree": seal["review_tree"], "tag_inspection": state,
              "reused": state == "EXISTS", "existing_digest": existing_digest}
    dump(directory / "preparation.json", record)
    if github_output:
        with Path(github_output).open("a", encoding="utf-8", newline="\n") as stream:
            for k, v in {"image": image, "base_image": PARENT, "build_required": "false" if record["reused"] else "true"}.items():
                if "\n" in v or "\r" in v:
                    raise ValueError("Unsafe workflow output")
                stream.write(k + "=" + v + "\n")
    return record


def compare_config(parent, child, commit, value):
    if parent.get("Id") != PARENT_ID or child.get("Os") != "linux" or child.get("Architecture") != "amd64":
        raise ValueError("Candidate/parent platform differs")
    configured = value.get("expected_image_config", {})
    for name in CONFIG_FIELDS:
        if child["Config"].get(name) != configured.get(name, parent["Config"].get(name)):
            raise ValueError("Unreviewed child service configuration: " + name)
    if child["Config"].get("User") not in ("dynamo", "dynamo:0", "1000", "1000:0"):
        raise ValueError("Final image must remain non-root UID1000")
    before = dict(item.split("=", 1) for item in parent["Config"].get("Env", []))
    after = dict(item.split("=", 1) for item in child["Config"].get("Env", []))
    additions = {k: v for k, v in after.items() if k not in before}
    expected_additions = {"H3_SR_PRODUCTION_SOURCE_COMMIT": commit, "H3_SR_PRODUCTION_PARENT_IMAGE": PARENT}
    expected_additions.update(value.get("expected_environment_additions", {}))
    if (any(after.get(k) != v for k, v in before.items()) or additions != expected_additions):
        raise ValueError("Inherited environment or child source identity changed")
    labels = child["Config"].get("Labels") or {}
    if labels.get("org.opencontainers.image.revision") != commit or labels.get("org.opencontainers.image.base.name") != PARENT:
        raise ValueError("Candidate labels differ from frozen source/parent")
    layers, actual = parent.get("RootFS", {}).get("Layers", []), child.get("RootFS", {}).get("Layers", [])
    if not layers or actual[:len(layers)] != layers or len(actual) <= len(layers):
        raise ValueError("Candidate must extend every verified parent layer")


def validate_reports(directory, commit):
    runtime, proof = read(directory / "runtime-validation.json"), read(directory / "source-verification.json")
    for key, expected in {"status": "PASS_CPU_ONLY", "source_commit": commit, "parent_image": PARENT,
                          "euid": 1000, "errors": []}.items():
        if runtime.get(key) != expected:
            raise ValueError("Actual final runtime validation failed: " + key)
    tests = runtime.get("tests")
    if isinstance(tests, dict):
        if any(tests.get(k, 0) != 0 for k in ("failures", "errors", "skipped")):
            raise ValueError("Final runtime regressions are incomplete")
        tests = tests.get("tests", tests.get("tests_run"))
    if type(tests) is not int or tests <= 0:
        raise ValueError("Actual final image must execute nonempty CPU checks")
    expected = read(ROOT / "delivery/h3-sr-production/expected-source-manifest.json")
    files = expected.get("source_sha256")
    if not isinstance(files, dict) or not files or expected.get("source_manifest_sha256") != hash_value(files):
        raise ValueError("Frozen complete expected source map is malformed")
    if (proof.get("status") != "PASS" or proof.get("source_commit") != commit
            or proof.get("source_sha256") != files or proof.get("source_manifest_sha256") != hash_value(files)
            or runtime.get("source_manifest_sha256") != hash_value(files)):
        raise ValueError("Actual installed source does not match the frozen complete child map")
    return runtime, proof


def validate(directory):
    prepared = read(directory / "preparation.json")
    require_current(prepared)
    value = ops.lock()
    candidate_reference = REPOSITORY + "@" + prepared["existing_digest"] if prepared["reused"] else prepared["image"]
    image = inspect(candidate_reference)
    compare_config(read(directory / "parent-inspect.json"), image, prepared["commit"], value)
    output = directory / "final"
    output.mkdir(exist_ok=False)
    output.chmod(0o777)  # This invocation's empty evidence directory, not a shared cache.
    docker("run", "--rm", "--network", "none", "--user", "1000:0", "--entrypoint", "python3",
           "--mount", "type=bind,src=" + str(output.resolve()) + ",dst=/tmp/h3-sr-video-gateway-evidence",
           image["Id"], "-B", RUNTIME, "validate", "--output-dir", "/tmp/h3-sr-video-gateway-evidence",
           log=directory / "actual-image-validation.txt")
    runtime, _ = validate_reports(output, prepared["commit"])
    gateway = validate_gateway(image["Id"], prepared["commit"], output)
    if inspect(candidate_reference)["Id"] != image["Id"]:
        raise ValueError("Candidate tag changed while validating the actual imageID")
    record = {"schema": "h3-sr-video-gateway-candidate/1", "status": "PASS_CPU_ONLY", **run_identity(),
              "image": prepared["image"], "image_id": image["Id"], "parent_image": PARENT,
              "source_manifest_sha256": runtime["source_manifest_sha256"], "recipe_sha256": prepared["recipe_sha256"],
              "gateway_tests": gateway["counts"],
              "gpu_validation_status": "NOT_RUN"}
    dump(directory / "candidate.json", record)
    return record


def validate_gateway(image_id, commit, output):
    wheels, dependencies = ops.test_dependencies()
    test_root = ROOT / "tests/video_gateway"
    sources = {"tests/video_gateway/" + path.relative_to(test_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(test_root.rglob("*.py"))}
    sources["tests/__init__.py"] = hashlib.sha256((ROOT / "tests/__init__.py").read_bytes()).hexdigest()
    wheel_path = ":".join("/tmp/gateway-test-deps/" + path.name for path in wheels)
    testpack = "/tmp/gateway-testpack"
    docker("run", "--rm", "--network", "none", "--user", "1000:0", "--workdir", "/tmp",
           "--env", "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1", "--env", "PYTHONNOUSERSITE=1",
           "--env", "PYTHONPATH=" + testpack + ":" + wheel_path, "--entrypoint", "python3",
           "--mount", "type=bind,src=" + str(test_root.resolve()) + ",dst=" + testpack + "/tests/video_gateway,readonly",
           "--mount", "type=bind,src=" + str((ROOT / "tests/__init__.py").resolve()) + ",dst=" + testpack + "/tests/__init__.py,readonly",
           "--mount", "type=bind,src=" + str((ops.HERE / "test-deps").resolve()) + ",dst=/tmp/gateway-test-deps,readonly",
           "--mount", "type=bind,src=" + str(output.resolve()) + ",dst=/tmp/h3-sr-video-gateway-evidence",
           image_id, "-P", "-B", "-m", "pytest", "-p", "pytest_asyncio.plugin", "-c", "/dev/null",
           "--rootdir=" + testpack, "--import-mode=importlib",
           "--confcutdir=" + testpack + "/tests/video_gateway", "-o", "asyncio_mode=auto", "-q",
           "--junitxml=/tmp/h3-sr-video-gateway-evidence/gateway-tests.xml",
           *(testpack + "/tests/video_gateway/" + name for name in ops.GATEWAY_TEST_FILES),
           log=output / "gateway-tests.log")
    counts = ops.gateway_test_counts(output / "gateway-tests.xml")
    report = {"schema": "h3sr-video-gateway-installed-tests/1", "status": "PASS",
              "source_commit": commit, "image_id": image_id, "counts": counts,
              "junit_sha256": hashlib.sha256((output / "gateway-tests.xml").read_bytes()).hexdigest(),
              "test_source_sha256": sources, "test_dependency_sha256": dependencies,
              "selected_test_files": list(ops.GATEWAY_TEST_FILES),
              "scope": "Installed Dingo seven-file adapter/configuration/ContextIR/production/output/artifact-store CPU suite, including target-Linux Ceph quota regressions; test-only package and plugin wheel mounted"}
    dump(output / "gateway-validation.json", report)
    return report


def publish(directory):
    prepared, candidate = read(directory / "preparation.json"), read(directory / "candidate.json")
    require_current(prepared)
    require_current(candidate)
    candidate_reference = REPOSITORY + "@" + prepared["existing_digest"] if prepared["reused"] else prepared["image"]
    if candidate.get("status") != "PASS_CPU_ONLY" or inspect(candidate_reference)["Id"] != candidate["image_id"]:
        raise ValueError("Only the exact currently CPU-validated imageID can be published")
    validate_reports(directory / "final", prepared["commit"])
    gateway = ops.check_gateway_test_evidence(directory / "final", prepared["commit"], candidate["image_id"])
    if candidate.get("gateway_tests") != gateway["counts"]:
        raise ValueError("Candidate Gateway regression evidence differs")
    operation = directory / "publication-operation.json"
    ops.claim(operation, {"operation": "REUSE" if prepared["reused"] else "PUSH", **run_identity(), "candidate": candidate})
    if prepared["reused"]:
        digest = prepared["existing_digest"]
    else:
        if lookup(prepared["image"], directory, "pre-push-tag-inspect.txt") != "ABSENT_CONFIRMED":
            raise RuntimeError("Final tag is no longer confirmed absent; stop before push")
        output = docker("push", prepared["image"], log=directory / "push.txt")
        matches = set(re.findall(r"digest:\s*(sha256:[0-9a-f]{64})", output))
        if len(matches) != 1:
            raise ValueError("Push response is ambiguous; inspect actual Harbor state before any retry")
        digest = matches.pop()
    reference = REPOSITORY + "@" + digest
    record = {"schema": "h3-sr-video-gateway-publication/1", "status": "PUBLISHED_PULL_PENDING", **run_identity(),
              "image": prepared["image"], "published_image": reference, "digest": digest,
              "image_id": candidate["image_id"], "parent_image": PARENT, "recipe_sha256": prepared["recipe_sha256"],
              "source_manifest_sha256": candidate["source_manifest_sha256"],
              "push_count": 0 if prepared["reused"] else 1, "reused_existing": prepared["reused"],
              "gpu_validation_status": "NOT_RUN"}
    dump(directory / "publication.json", record)  # Preserve actual publication before a possible tail network failure.
    docker("pull", "--platform", "linux/amd64", reference, log=directory / "published-digest-pull.txt")
    if inspect(reference)["Id"] != candidate["image_id"]:
        raise ValueError("Published digest pull does not match the CPU-tested imageID")
    artifacts = {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(directory.rglob("*")) if p.is_file() and p != directory / "publication.json"}
    record.update(status="PASS_PUBLISHED_AND_PULLED_CPU_VALIDATED", evidence_sha256=artifacts)
    dump(directory / "publication.json", record)
    (directory / "deployment-image.txt").write_text(reference + "\n", encoding="utf-8")
    return record


def outcome(directory):
    current = run_identity()
    directory.mkdir(parents=True, exist_ok=True)
    prepared = read(directory / "preparation.json") if (directory / "preparation.json").exists() else {
        **current, "image": REPOSITORY + ":h3-sr-video-gateway-" + current["commit"][:12]}
    require_current(prepared)
    status = lookup(prepared["image"], directory, "final-publication-lookup.txt")
    record = {"schema": "h3-sr-video-gateway-publication-observation/1", **run_identity(),
              "image": prepared["image"], "publication_state": status,
              "observed_at": datetime.now(timezone.utc).isoformat()}
    dump(directory / "publication-observation.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "validate", "publish", "outcome"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    result = (prepare(args.directory, args.github_output) if args.action == "prepare" else
              validate(args.directory) if args.action == "validate" else
              publish(args.directory) if args.action == "publish" else outcome(args.directory))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
