#!/usr/bin/env python3
"""Rerun inherited capability gates into a fresh, unprivileged evidence directory."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if os.geteuid() != 1000:
        raise RuntimeError("Retained capability checks must run as UID1000")
    out.mkdir(parents=True, exist_ok=True)
    root = Path("/opt/h3-experimental")
    checks = [
        ("assets", [root / "assets/verify_assets.py", "--lock", root / "assets/assets.lock.json",
                    "--root", "/opt/h3-adapters", "--output", out / "assets-report.json"]),
        ("ref8", [root / "ref8/check_contract.py", "--installed", "--report", out / "ref8-contract.json"]),
        ("profiles", [root / "profiles/check_profiles.py", "--installed", "--output", out / "profiles-check.json"]),
        ("environment", [root / "environment/audit_environment.py", "--output-dir", out]),
        ("media", [root / "environment/media_smoke.py", "--output-dir", out]),
        ("service", [root / "environment/service_smoke.py", "--output-dir", out]),
    ]
    env = dict(os.environ)
    # Input baseline stays where the parent installer wrote it. This is not an output path.
    env["H3_BUILD_INFO"] = "/opt/dynamo/build-info"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for name, command in checks:
        result = subprocess.run([sys.executable, "-B", *map(str, command)], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                encoding="utf-8", errors="replace", timeout=1800)
        (out / (name + ".log")).write_text(result.stdout, encoding="utf-8")
        print(result.stdout, flush=True)
        if result.returncode:
            raise RuntimeError(f"Inherited {name} gate failed: exit {result.returncode}")
    (out / "checks.json").write_text(json.dumps({
        "status": "PASS_RETAINED_CAPABILITIES_CPU_ONLY", "euid": os.geteuid(),
        "checks": [name for name, _ in checks], "source_commit": os.environ["H3_EXPERIMENTAL_SOURCE_COMMIT"],
        "scope": "Fresh adapter hashes, Ref8/profile/source, dependencies, CPU media, service import gates; no GPU inference or visual acceptance."
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
