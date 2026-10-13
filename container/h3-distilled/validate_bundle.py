"""Offline CPU-only gate on the completed derivative image, before publication."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from runtime_snapshot import snapshot

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        raise ValueError("A full packaging source commit is required")
    lock = read(HERE / "image.lock.json")
    info = HERE / "build-info"
    info.mkdir(exist_ok=True)
    before = read(info / "base-environment.json")
    after = snapshot()
    (info / "runtime-environment.json").write_text(json.dumps(after, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if before != after:
        raise ValueError("Inherited Python packages or protected environment changed")
    if os.environ.get("DYNAMO_COMMIT_SHA") not in (None, lock["recipe_base_commit"]):
        raise ValueError("Installed framework provenance changed")
    source_report = info / "runtime-source-check.json"
    with source_report.open("w", encoding="utf-8") as output:
        subprocess.run([sys.executable, "-B", str(HERE / "image_reuse_check.py")], stdout=output, check=True)
    report = read(source_report)
    print("Runtime source check:", report["status"], flush=True)
    if report["status"] != "PASS_INSTALLED_CODE_ONLY":
        raise ValueError("Runtime source check failed")
    subprocess.run([sys.executable, "-B", str(HERE / "verify_assets.py"),
                    "--lock", str(HERE / "assets.lock.json"), "--root", lock["asset_root"],
                    "--output", str(info / "assets-check.json")], check=True)
    assets = read(info / "assets-check.json")
    if assets["status"] != "PASS_ASSETS_ONLY":
        raise ValueError("Asset verification failed")
    # Reading under the inherited non-root user checks permission on all payloads.
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        raise ValueError("Final verification must use inherited non-root dynamo user")
    result = {"status": "PASS_DISTILLED_CPU_ONLY", "source_commit": args.source_commit,
              "framework_source_commit": lock["recipe_base_commit"], "base_image": lock["base_image"],
              "asset_root": lock["asset_root"], "environment_unchanged": True,
              "asset_lock_sha256": hashlib.sha256((HERE / "assets.lock.json").read_bytes()).hexdigest(),
              "image_lock_sha256": hashlib.sha256((HERE / "image.lock.json").read_bytes()).hexdigest(),
              "validation_network": "Docker RUN --network=none",
              "limits": ["No GPU kernel or HSDP validation", "No adapter activation or media generation",
                         "Mounted Base revision remains to be verified for FastH3"]}
    (info / "bundle-check.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
