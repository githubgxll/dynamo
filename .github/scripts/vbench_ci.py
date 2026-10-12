#!/usr/bin/env python3
"""Small, credential-free helpers for VBench's prepare/publish workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

BASE_TAG = "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04"
BASE_REPOSITORY = "nvidia/cuda"
PLATFORM = "linux/amd64"
BASE_DEFINITION = Path(__file__).resolve().parents[2] / "container/vbench/base-image.json"
BASE_IDENTITY_FIELDS = ("schema_version", "source_tag", "platform", "digest", "image")
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
SHA = re.compile(r"^[0-9a-f]{40}$")
IMAGE_TAG = re.compile(r"^[a-z0-9][a-z0-9.:-]*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+:[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")
REQUIRED_LOCKS = ("base.lock.json", "requirements.lock", "assets.lock.json", "input-fingerprint.json", "preparation-license-status.json")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def amd64_digest(manifest: dict) -> str:
    """Select the actual Linux/amd64 manifest, excluding attestations/other CPUs."""
    matching = [
        item for item in manifest.get("manifests", [])
        if item.get("platform", {}).get("os") == "linux"
        and item.get("platform", {}).get("architecture") == "amd64"
        and item.get("platform", {}).get("variant", "") in {"", "v1"}
    ]
    if len(matching) != 1 or not DIGEST.fullmatch(matching[0].get("digest", "")):
        raise ValueError("Expected exactly one Linux/amd64 image digest in the CUDA image index")
    return matching[0]["digest"]


def validate_base_identity(lock: dict) -> str:
    if not isinstance(lock, dict):
        raise ValueError("Base identity must be a JSON object")
    digest = lock.get("digest", "")
    if (type(lock.get("schema_version")) is not int or lock.get("schema_version") != 1
            or lock.get("source_tag") != BASE_TAG or lock.get("platform") != PLATFORM
            or not isinstance(digest, str) or not DIGEST.fullmatch(digest)
            or lock.get("image") != f"{BASE_REPOSITORY}@{digest}"):
        raise ValueError("Base identity must pin the expected CUDA 12.1 Linux/amd64 image by digest")
    return lock["image"]


def load_base_definition() -> dict:
    """The reviewed pin is an input, not a tag resolved during every build."""
    definition = read_json(BASE_DEFINITION)
    validate_base_identity(definition)
    return definition


def validate_base_lock(lock: dict) -> str:
    image = validate_base_identity(lock)
    definition = load_base_definition()
    if any(lock.get(key) != definition.get(key) for key in BASE_IDENTITY_FIELDS):
        raise ValueError("base.lock.json differs from the reviewed base-image.json; prepare again")
    return image


def write_base_output(image: str, path: Path) -> None:
    # Only a validated, newline-free digest reference enters GITHUB_OUTPUT.
    with path.open("a", encoding="utf-8") as output:
        output.write(f"base_image={image}\n")


def prepare_base(lock_path: Path, output_path: Path, evidence: Path) -> int:
    """Materialize the reviewed pin offline; actual daemon pull follows separately."""
    evidence.mkdir(parents=True, exist_ok=True)
    summary = {"schema_version": 1, "status": "RUNNING", "mode": "reviewed_pin",
               "started_utc": datetime.now(timezone.utc).isoformat(),
               "network_used": False, "runtime_pull_verified": False}
    try:
        definition = load_base_definition()
        lock = {key: definition[key] for key in BASE_IDENTITY_FIELDS}
        image = validate_base_lock(lock)
        lock["resolved_utc"] = datetime.now(timezone.utc).isoformat()
        lock["resolution_method"] = "version_controlled_pin"
        summary.update(image=image, platform=PLATFORM,
                       definition_sha256=hashlib.sha256(BASE_DEFINITION.read_bytes()).hexdigest())
        write_json(evidence / "base-definition.json", definition)
        write_json(lock_path, lock)
        write_base_output(image, output_path)
        summary.update(status="PASS_PIN_VALIDATION_ONLY")
        print(f"Using reviewed base pin: {image} ({PLATFORM}); no tag lookup", flush=True)
    except (ValueError, OSError, TypeError) as error:
        summary.update(status="FAILED", category="invalid_base_definition", error=sanitized(str(error)))
        print("Base preparation failed: " + summary["error"], flush=True)
    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(evidence / "base-prepare-summary.json", summary)
    return 0 if summary["status"] == "PASS_PIN_VALIDATION_ONLY" else 1


def verify_locks(directory: Path, output_path: Path) -> None:
    for name in REQUIRED_LOCKS:
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Publish requires a nonempty reviewed lock: {path}")
        if name.endswith(".json"):
            read_json(path)
    image = validate_base_lock(read_json(directory / "base.lock.json"))
    license_status = read_json(directory / "preparation-license-status.json")
    if license_status.get("policy_pass") is not True or license_status.get("violations"):
        raise ValueError("Preparation license review has not passed; publish is blocked")
    write_base_output(image, output_path)
    print("Lock files present; Docker build will enforce fingerprints, hashes and license decisions.")


def sanitized(value) -> str:
    """Keep diagnostics without URL credentials, signed queries or auth values."""
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value or "")

    def clean_url(match):
        try:
            url = urlsplit(match.group())
            host = url.hostname or "redacted-host"
            if ":" in host:
                host = "[" + host + "]"
            if url.port is not None:
                host += ":" + str(url.port)
            return urlunsplit((url.scheme, host, url.path, "REDACTED" if url.query else "", ""))
        except ValueError:
            return "[REDACTED_URL]"

    text = re.sub(r'[A-Za-z][A-Za-z0-9+.-]*://[^\s<>"\']+', clean_url, text)
    text = re.sub(r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+", r"\1 [REDACTED]", text)
    text = re.sub(
        r'''(?ix)(["']?(?:authorization|proxy-authorization|x-registry-auth|auth|access_token|refresh_token|identitytoken|token|password|passwd|client_secret)["']?\s*[:=]\s*)(?:"[^"\r\n]*"|'[^'\r\n]*'|[^\s,;]+)''',
        r"\1[REDACTED]", text,
    )
    text = re.sub(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b", "[REDACTED_JWT]", text)
    # A failed layer can produce very large progress streams; retain the tail
    # and mark the truncation instead of exporting unbounded diagnostics.
    return text if len(text) <= 262144 else "[earlier output truncated]\n" + text[-262144:]


def failure_category(output: str, *, timed_out=False, missing_command=False) -> str:
    if missing_command:
        return "docker_unavailable"
    low = output.lower()
    if re.search(r"unauthorized|authentication required|access denied|\bdenied\b|\bforbidden\b|\b(?:401|403)\b", low):
        return "authentication_or_permission"
    if re.search(r"manifest unknown|no such manifest|manifest.*not found", low):
        return "manifest_unknown"
    if re.search(r"x509|certificate|unknown authority", low):
        return "tls_certificate"
    if timed_out or re.search(r"\beof\b|timeout|timed out|deadline exceeded|connection reset|\b429\b|too many requests|toomanyrequests|internal server error|bad gateway|service unavailable|gateway timeout|(?:status(?: code)?|http(?:/[0-9.]+)?)[^\n]{0,70}\b5[0-9]{2}\b", low):
        return "transient_transport_or_rate_limit"
    return "other_failure"


def diagnostic_command(command: list[str], timeout: int) -> tuple[dict, bytes, bytes]:
    """Run without a shell; raw bytes remain in memory only until sanitized."""
    try:
        result = subprocess.run(command, check=False, capture_output=True, timeout=timeout)
        stdout, stderr = result.stdout or b"", result.stderr or b""
        status = {"return_code": result.returncode, "timed_out": False,
                  "category": "success" if result.returncode == 0 else failure_category(sanitized(stdout) + "\n" + sanitized(stderr))}
    except subprocess.TimeoutExpired as error:
        stdout, stderr = error.stdout or b"", error.stderr or b""
        status = {"return_code": None, "timed_out": True,
                  "category": failure_category(sanitized(stdout) + "\n" + sanitized(stderr), timed_out=True)}
    except OSError as error:
        stdout, stderr = b"", sanitized(str(error)).encode()
        status = {"return_code": None, "timed_out": False, "category": "docker_unavailable"}
    status["timeout_seconds"] = timeout
    return status, stdout, stderr


def save_diagnostics(directory: Path, name: str, stdout, stderr) -> None:
    for channel, output in (("stdout", stdout), ("stderr", stderr)):
        (directory / f"{name}.{channel}.txt").write_text(sanitized(output), encoding="utf-8")


def report_failed_command(name: str, status: dict, stdout, stderr) -> None:
    if status["return_code"] == 0:
        return
    print(f"=== {name}: {status['category']} (exit={status['return_code']}) ===", flush=True)
    for channel, raw in (("stdout", stdout), ("stderr", stderr)):
        cleaned = sanitized(raw)
        if cleaned:
            print(channel + ":\n" + cleaned[-8192:], flush=True)
    print(f"=== end {name}; full redacted output is in the artifact ===", flush=True)


def pull_base(lock_path: Path, evidence: Path, probe_client: bool = False) -> int:
    """Pull the fixed digest with the build daemon; client inspection is opt-in."""
    evidence.mkdir(parents=True, exist_ok=True)
    summary_path = evidence / "base-pull-summary.json"
    summary = {"schema_version": 1, "status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(),
               "client_probe": None, "daemon_attempts": [], "identity": None}
    write_json(summary_path, summary)
    try:
        lock = read_json(lock_path)
        image = validate_base_lock(lock)
    except (ValueError, OSError, TypeError) as error:
        summary.update(status="FAILED", category="invalid_base_lock", error=sanitized(str(error)),
                       finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(summary_path, summary)
        print("Base pull failed: invalid_base_lock; see " + str(summary_path))
        return 1
    summary.update(image=image, platform=PLATFORM, expected_digest=lock["digest"])
    client = {"category": "not_requested", "reason": "Pinned prepare/publish do not depend on Buildx client registry access"}
    if probe_client:
        print("Checking locked base with the Buildx client (timeout: 60 seconds)", flush=True)
        client, stdout, stderr = diagnostic_command(["docker", "buildx", "imagetools", "inspect", "--raw", image], 60)
        save_diagnostics(evidence, "client-inspect", stdout, stderr)
        report_failed_command("client-inspect", client, stdout, stderr)
        raw = stdout.encode() if isinstance(stdout, str) else stdout
        client["raw_sha256"] = "sha256:" + hashlib.sha256(raw).hexdigest()
        client["raw_sha256_matches_expected"] = client["raw_sha256"] == lock["digest"]
        try:
            body = json.loads(raw)
            client["valid_manifest_json"] = isinstance(body, dict) and body.get("schemaVersion") == 2
        except (ValueError, UnicodeError):
            client["valid_manifest_json"] = False
        client["raw_hash_scope"] = "Diagnostic only: CLI formatting may alter bytes; daemon digest identity decides success"
    summary["client_probe"] = client
    write_json(summary_path, summary)
    for number in range(1, 4):
        print(f"Pulling locked base through Docker daemon: attempt {number}/3 (timeout: 600 seconds)", flush=True)
        attempt, stdout, stderr = diagnostic_command(["docker", "pull", "--platform", PLATFORM, image], 600)
        if attempt["timed_out"]:
            print(f"Docker daemon pull attempt {number} reached its 600-second timeout", flush=True)
        save_diagnostics(evidence, f"daemon-pull-{number}", stdout, stderr)
        report_failed_command(f"daemon-pull-{number}", attempt, stdout, stderr)
        attempt["attempt"] = number
        summary["daemon_attempts"].append(attempt)
        write_json(summary_path, summary)
        if attempt["return_code"] == 0:
            break
        if attempt["category"] != "transient_transport_or_rate_limit" or number == 3:
            summary.update(status="FAILED", category=attempt["category"])
            break
        print(f"Base daemon pull attempt {number}: {attempt['category']}; retrying in 10 seconds", flush=True)
        time.sleep(10)
    if summary["daemon_attempts"][-1]["return_code"] == 0:
        print("Verifying local locked-image identity (timeout: 60 seconds)", flush=True)
        # Ask Docker only for public identity fields, never Config.Env or auth.
        template = '{"Os":{{json .Os}},"Architecture":{{json .Architecture}},"RepoDigests":{{json .RepoDigests}},"Id":{{json .Id}}}'
        identity_status, stdout, stderr = diagnostic_command(["docker", "image", "inspect", "--format", template, image], 60)
        # stdout is validated/projected below rather than saved as a raw object.
        (evidence / "daemon-image-inspect.stderr.txt").write_text(sanitized(stderr), encoding="utf-8")
        report_failed_command("daemon-image-inspect", identity_status, b"", stderr)
        identity = {}
        try:
            parsed = json.loads(stdout)
            if not isinstance(parsed, dict):
                raise ValueError("Identity response is not an object")
            repos = parsed.get("RepoDigests") or []
            matched = isinstance(repos, list) and any(isinstance(repo, str) and repo in {image, "docker.io/" + image} for repo in repos)
            valid = identity_status["return_code"] == 0 and parsed.get("Os") == "linux" and parsed.get("Architecture") == "amd64" and matched
            identity = {key: sanitized(parsed[key]) if isinstance(parsed.get(key), str) else None
                        for key in ("Os", "Architecture", "Id")}
            identity["RepoDigests"] = [sanitized(repo) for repo in repos if isinstance(repo, str)] if isinstance(repos, list) else []
        except (ValueError, TypeError, UnicodeError):
            valid = False
        write_json(evidence / "daemon-image-identity.json", identity)
        summary["identity"] = {**identity_status, "matches_locked_image": valid}
        summary.update(status="PASS" if valid else "FAILED", category="verified_locked_image" if valid else "daemon_image_identity_mismatch")
    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(summary_path, summary)
    print(f"Base pull {summary['status']}: {summary['category']}; client={client['category']}; see {summary_path}", flush=True)
    return 0 if summary["status"] == "PASS" else 1


def record_image(args: argparse.Namespace) -> None:
    if not IMAGE_TAG.fullmatch(args.image) or not DIGEST.fullmatch(args.digest):
        raise ValueError("Invalid published image tag or digest")
    if not SHA.fullmatch(args.source_commit):
        raise ValueError("Source commit must be a full lowercase Git SHA")
    if not args.run_id.isdecimal() or not args.run_attempt.isdecimal():
        raise ValueError("Run ID and attempt must be numeric")
    dockerfile = Path(args.dockerfile)
    manifest = {
        "source_commit": args.source_commit, "run_id": args.run_id,
        "run_attempt": args.run_attempt, "image": args.image,
        "digest": args.digest,
        "deployment_image": args.image.rsplit(":", 1)[0] + "@" + args.digest,
        "builder_image": args.builder_image,
        "dockerfile": dockerfile.as_posix(),
        "dockerfile_sha256": hashlib.sha256(dockerfile.read_bytes()).hexdigest(),
        "validation_scope": "Image published; K8s pull, GPU and model/media E2E not yet validated",
    }
    write_json(args.output, manifest)
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as summary:
            summary.write(f"### Published image\n\n`{args.image}`\n\n")
            summary.write(f"Digest: `{args.digest}`\n\nSource commit: `{args.source_commit}`\n\n")
            summary.write(f"Deploy by digest: `{manifest['deployment_image']}`\n\n")
            summary.write(manifest["validation_scope"] + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    resolve = sub.add_parser("prepare-base", help="Validate and reuse the reviewed base pin without a registry lookup")
    resolve.add_argument("--lock", type=Path, required=True)
    resolve.add_argument("--github-output", type=Path, required=True)
    resolve.add_argument("--evidence", type=Path, required=True)
    verify = sub.add_parser("verify-locks")
    verify.add_argument("--directory", type=Path, required=True)
    verify.add_argument("--github-output", type=Path, required=True)
    pull = sub.add_parser("pull-base")
    pull.add_argument("--lock", type=Path, required=True)
    pull.add_argument("--evidence", type=Path, required=True)
    pull.add_argument("--probe-client", action="store_true", help="Optional client/daemon comparison; never required for building a fixed pin")
    record = sub.add_parser("record-image")
    for name in ("image", "digest", "source-commit", "run-id", "run-attempt", "dockerfile"):
        record.add_argument("--" + name, required=True)
    record.add_argument("--builder-image", default="")
    record.add_argument("--output", type=Path, required=True)
    record.add_argument("--summary", type=Path)
    args = parser.parse_args()
    if args.command == "prepare-base":
        raise SystemExit(prepare_base(args.lock, args.github_output, args.evidence))
    elif args.command == "verify-locks":
        verify_locks(args.directory, args.github_output)
    elif args.command == "pull-base":
        raise SystemExit(pull_base(args.lock, args.evidence, args.probe_client))
    else:
        record_image(args)


if __name__ == "__main__":
    main()
