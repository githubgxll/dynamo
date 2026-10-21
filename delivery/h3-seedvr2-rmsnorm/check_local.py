#!/usr/bin/env python3
"""Run inherited and derivative local checks without pretending to run the image."""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def validate_contract():
    source = json.loads((HERE / "source-lock.json").read_text(encoding="utf-8"))
    patch = json.loads((ROOT / "container/h3-seedvr2-rmsnorm/patch.lock.json").read_text(encoding="utf-8"))
    for outer, inner in {
        "parent_image": "parent_image", "parent_source_commit": "parent_source_commit",
        "upstream_commit": "upstream_commit", "omni_source_commit": "omni_base_commit",
        "source_before_sha256": "before_sha256", "source_after_sha256": "after_sha256",
    }.items():
        if source[outer] != patch[inner]:
            raise ValueError("Delivery/runtime locks disagree: " + outer)
    if source["upstream_test_sha256"] != patch["vendored_sha256"]["upstream/test_seedvr2_norm.py"]:
        raise ValueError("Delivery/runtime test identity differs")
    if any(patch["required_versions"].get(name) != version for name, version in source["versions"].items()):
        raise ValueError("Delivery/runtime protected versions differ")


def main():
    validate_contract()
    out = ROOT / ".h3-experimental-evidence/srnorm/local" / datetime.now(timezone.utc).strftime("checks-%Y%m%dT%H%M%S.%fZ")
    out.mkdir(parents=True)
    temp = ROOT / ".h3-experimental-evidence/srnorm/local/tmp"
    temp.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TEMP=str(temp), TMP=str(temp), TMPDIR=str(temp))
    jobs = [
        ("inherited-implementation", [sys.executable, "-B", "-X", "utf8", str(ROOT / "delivery/h3-experimental/check_local.py")]),
        ("patch-regressions", [sys.executable, "-B", "-X", "utf8", "-m", "unittest", "discover", "-s", str(ROOT / "container/h3-seedvr2-rmsnorm/tests"), "-v"]),
        ("submission-regressions", [sys.executable, "-B", "-X", "utf8", "-m", "unittest", "discover", "-s", str(HERE / "tests"), "-v"]),
        ("ci-publication-regressions", [sys.executable, "-B", "-X", "utf8", str(ROOT / ".github/scripts/test_h3_seedvr2_rmsnorm_ci.py")]),
    ]
    counts = {}
    for name, argv in jobs:
        result = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, encoding="utf-8", errors="replace")
        output = result.stdout + result.stderr
        (out / (name + ".txt")).write_text(output, encoding="utf-8")
        if result.returncode:
            print(output)
            raise RuntimeError(f"{name} failed; evidence: {out}")
        counts[name] = sum(int(n) for n in re.findall(r"Ran (\d+) tests?", output))
        if name == "inherited-implementation":
            counts[name] = int(re.findall(r'"regression_tests":\s*(\d+)', output)[-1])
        print("PASS " + name)
    files = list((ROOT / "container/h3-seedvr2-rmsnorm").rglob("*.py")) + list(HERE.rglob("*.py"))
    files += [ROOT / ".github/scripts/h3_seedvr2_rmsnorm_ci.py", ROOT / ".github/scripts/test_h3_seedvr2_rmsnorm_ci.py"]
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    report = {"status": "PASS_LOCAL_RECIPE_AND_REGRESSIONS_ONLY", "regression_counts": counts,
              "total_regression_tests": sum(counts.values()),
              "python_syntax_files": len(files), "local_torch_available": importlib.util.find_spec("torch") is not None,
              "evidence_directory": str(out), "not_run": ["Linux image actual module import",
              "Six upstream numerical cases in the parent Torch runtime", "Harbor publication/pull", "CUDA", "same-source video A/B"]}
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
