"""Single-image Actions job for Regenerate IC FA3 VAE r1."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


BRANCH = "Dingorouter-H3-vllm-omni-jirx"
REGISTRY = "registry.hd-04.alayanew.com:8443"
REPOSITORY = REGISTRY + "/openclaw/vllm-omni"
PARENT = REPOSITORY + "@sha256:9387d41fca398d0061cc36be37a2ef4527e56cbbb679727e7494b93add9f9658"
CONTEXT = Path("container/ic-fa3-vae-r1")
VERIFIER = "/opt/ic-r1/verify_image.py"
DIGEST = r"sha256:[0-9a-f]{64}"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def capture(*args):
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")


def read_command(*args):
    done = capture(*args)
    if done.returncode:
        raise RuntimeError(f"{args[0]} {args[1]} failed ({done.returncode}): {(done.stderr or done.stdout)[-4000:]}")
    return done.stdout


def save(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def lookup(tag):
    done = capture("docker", "manifest", "inspect", tag)
    if done.returncode == 0:
        return True
    detail = (done.stdout + done.stderr).strip()
    short = tag.split("/", 1)[1]
    missing = {f"no such manifest: {tag}", f"unknown: artifact {short} not found",
               "manifest unknown", "manifest unknown: manifest unknown"}
    require(done.returncode == 1 and detail in missing,
            f"Harbor manifest lookup unresolved for {tag}: {detail[-2000:]}")
    return False


def resolve_digest(tag):
    output = read_command("docker", "buildx", "imagetools", "inspect", tag)
    values = re.findall("^Digest:\\s*(" + DIGEST + ")\\s*$", output, re.MULTILINE)
    require(len(values) == 1, "Missing or ambiguous registry manifest digest")
    return REPOSITORY + "@" + values[0]


def inspect_image(tag):
    return json.loads(read_command("docker", "image", "inspect", tag))[0]


def validate_image(tag, commit, manifest_hash, evidence):
    image = inspect_image(tag)
    image_id = image["Id"]
    labels = image["Config"].get("Labels") or {}
    require(re.fullmatch(DIGEST, image_id) and image["Os"] == "linux" and image["Architecture"] == "amd64",
            "Unexpected image identity or platform")
    require(labels.get("org.opencontainers.image.revision") == commit
            and labels.get("org.opencontainers.image.base.name") == PARENT
            and labels.get("org.opencontainers.image.ic-release-manifest") == manifest_hash,
            "Image does not belong to this frozen source, manifest and parent")
    check = capture("docker", "run", "--rm", "--network", "none", "--tmpfs", "/tmp:rw,exec",
                    "--env", "HOME=/tmp", "--env", "XDG_CACHE_HOME=/tmp/cache", "--entrypoint", "/usr/bin/python3",
                    image_id, "-B", VERIFIER)
    (evidence / "verify.stdout").write_text(check.stdout, encoding="utf-8")
    (evidence / "verify.stderr").write_text(check.stderr, encoding="utf-8")
    if check.returncode:
        raise RuntimeError("Offline CPU image validation failed: " + (check.stderr or check.stdout)[-3000:])
    lines = check.stdout.strip().splitlines()
    require(lines, "Offline image verifier returned no result")
    report = json.loads(lines[-1])
    require(report.get("status") == "PASS_IMAGE_CPU_AND_OFFLINE_PAYLOAD_ONLY"
            and report.get("release_manifest_sha256") == manifest_hash, "Image verifier rejected or frozen manifest differs")
    return image_id


def archive_publication(evidence, record):
    save(evidence / "publication.json", record)
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write(f"IC worker **{record['state']}**\n\n`{record['pull_reference']}`\n\n")
    print(json.dumps(record), flush=True)


def publish(evidence, commit):
    manifest_hash = hashlib.sha256((CONTEXT / "runtime/release-manifest.json").read_bytes()).hexdigest()
    tag = REPOSITORY + ":ic-fa3-vae-r1-" + commit[:12]
    record = {"tag": tag, "revision": commit, "parent_image": PARENT, "manifest_sha256": manifest_hash,
              "job_run": os.environ["GITHUB_RUN_ID"], "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
              "gpu_inference": "not_run", "pushes_this_job": 0}
    save(evidence / "request.json", record)
    if lookup(tag):
        reference = resolve_digest(tag)
        subprocess.run(["docker", "pull", reference], check=True)
        image_id = validate_image(reference, commit, manifest_hash, evidence)
        archive_publication(evidence, {**record, "state": "REUSED_VALIDATED", "image_id": image_id,
                                      "pull_reference": reference})
        return

    subprocess.run(["docker", "buildx", "build", "--platform", "linux/amd64", "--load",
                    "--provenance=false", "--sbom=false", "--progress=plain", "--tag", tag,
                    "--label", "org.opencontainers.image.base.name=" + PARENT,
                    "--label", "org.opencontainers.image.revision=" + commit,
                    "--label", "org.opencontainers.image.ic-release-manifest=" + manifest_hash,
                    "--file", str(CONTEXT / "Dockerfile"), str(CONTEXT)], check=True)
    image_id = validate_image(tag, commit, manifest_hash, evidence)
    require(not lookup(tag), "Harbor tag appeared while building; assess the existing digest before publication")
    require(inspect_image(tag)["Id"] == image_id, "Local tag no longer refers to the validated image")
    save(evidence / "push-intent.json", {**record, "image_id": image_id})
    pushed = capture("docker", "push", tag)
    (evidence / "push.log").write_text(pushed.stdout + pushed.stderr, encoding="utf-8")
    require(pushed.returncode == 0, "Push response failed; check original tag and receipt before retrying")
    digests = set(re.findall(r"\bdigest:\s*(" + DIGEST + ")", pushed.stdout + pushed.stderr))
    require(len(digests) == 1, "Could not identify a single registry digest in Docker push response")
    reference = REPOSITORY + "@" + digests.pop()
    record.update(image_id=image_id, pull_reference=reference, pushes_this_job=1, state="PUSH_ACKNOWLEDGED")
    archive_publication(evidence, record)
    require(resolve_digest(tag) == reference, "Registry tag digest changed after push")
    subprocess.run(["docker", "pull", reference], check=True)
    require(inspect_image(reference)["Id"] == image_id, "Pulled digest differs from CPU-verified local image")
    archive_publication(evidence, {**record, "state": "PUBLISHED_PULL_VERIFIED"})


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("evidence", type=Path)
    evidence = args.parse_args().evidence
    commit = os.environ.get("GITHUB_SHA", "")
    require(re.fullmatch("[0-9a-f]{40}", commit) and os.environ.get("GITHUB_REF_NAME") == BRANCH
            and os.environ.get("GITHUB_EVENT_NAME") == "push", "Wrong source branch or Actions event")
    require(read_command("git", "rev-parse", "HEAD").strip() == commit, "Checkout differs from the trigger")
    evidence.mkdir(parents=True, exist_ok=False)
    publish(evidence, commit)


if __name__ == "__main__":
    main()
