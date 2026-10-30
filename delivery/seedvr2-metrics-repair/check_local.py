#!/usr/bin/env python3
"""Check the bounded SeedVR2 candidate; no Docker, network, cluster or GPU activity."""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def check_contract():
    lock = json.loads((HERE / "source-lock.json").read_text(encoding="utf-8"))
    if lock["parent_source_commit"] != lock["recipe_base_commit"]:
        raise ValueError("Parent/source baseline mismatch")
    spec = importlib.util.spec_from_file_location("_local_sr_source_patches", ROOT / "container/seedvr2-metrics-repair/source_patches.py")
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    reviewed = patch.lock()
    manifest = json.loads((HERE / "parent-source-manifest.json").read_text(encoding="utf-8"))["source_sha256"]
    for relative in patch.CHANGES:
        original_path = ROOT / "container/seedvr2-metrics-repair/vendor/original" / relative
        if relative.endswith("sr_telemetry.py"):
            original_path = ROOT / "container/seedvr2-optimization/telemetry.py"
        elif relative.endswith("long_video_support.py"):
            original_path = ROOT / "container/seedvr2-optimization/long_video_support.py"
        before = original_path.read_bytes()
        if patch.digest(before) != manifest[relative]:
            raise ValueError("Pinned parent bytes differ: " + relative)
        after = patch.transform(relative, before)
        if reviewed["changes"][relative] != {"before_sha256": patch.digest(before), "after_sha256": patch.digest(after)}:
            raise ValueError("Repair transform differs from reviewed lock: " + relative)
    dockerfile = (ROOT / "container/seedvr2-metrics-repair/Dockerfile").read_text(encoding="utf-8")
    if "# syntax=" in dockerfile or re.search(r"\b(?:pip|uv)\s+(?:pip\s+)?install\b", dockerfile):
        raise ValueError("No frontend fetch or package install in derivative")
    import yaml
    workflow = yaml.load((ROOT / ".github/workflows/dingo-router-ci.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    job = workflow["jobs"]["seedvr2-metrics-repair"]
    if "[seedvr2-metrics-repair]" not in job["if"]:
        raise ValueError("New recipe marker missing")
    return [step["run"] for step in job["steps"] if "run" in step]


def main():
    shell_blocks = check_contract()
    out = ROOT / ".h3-experimental-evidence/sr-metrics-repair/local" / datetime.now(timezone.utc).strftime("checks-%Y%m%dT%H%M%S.%fZ")
    out.mkdir(parents=True)
    temp = out.parent / "tmp"
    temp.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TEMP=str(temp), TMP=str(temp), TMPDIR=str(temp))
    jobs = []
    for name, path in (
        ("inherited-environment", ROOT / "container/h3-experimental/environment"),
        ("inherited-rmsnorm", ROOT / "container/h3-seedvr2-rmsnorm/tests"),
        ("inherited-lite-runtime", ROOT / "container/omni-gateway-lite/tests"),
        ("inherited-c14-diagnostics", ROOT / "container/seedvr2-c14/tests"),
        ("hsdp-offload", ROOT / "container/h3-hsdp-offload/tests"),
        ("sr-metrics-repair", ROOT / "container/seedvr2-metrics-repair/tests"),
        ("sr-submission", HERE / "tests"),
    ):
        jobs.append((name, [sys.executable, "-B", "-X", "utf8", "-m", "unittest", "discover", "-s", str(path), "-p", "test_*.py", "-v"]))
    jobs.append(("sr-ci-publication", [sys.executable, "-B", "-X", "utf8", str(ROOT / ".github/scripts/test_seedvr2_metrics_repair_ci.py")]))
    counts, skips = {}, {}
    for name, argv in jobs:
        result = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, encoding="utf-8", errors="replace")
        output = result.stdout + result.stderr
        (out / (name + ".txt")).write_text(output, encoding="utf-8", newline="\n")
        if result.returncode:
            print(output[-20000:])
            raise RuntimeError(name + " failed; evidence: " + str(out))
        counts[name] = sum(map(int, re.findall(r"Ran (\d+) tests?", output)))
        skips[name] = sum(map(int, re.findall(r"OK \(skipped=(\d+)\)", output)))
        if counts[name] == 0:
            raise RuntimeError(name + " did not execute regression tests")
        print("PASS " + name)
    files = sorted((ROOT / "container/seedvr2-metrics-repair").rglob("*.py")) + sorted(HERE.rglob("*.py"))
    files += [ROOT / ".github/scripts/seedvr2_metrics_repair_ci.py", ROOT / ".github/scripts/test_seedvr2_metrics_repair_ci.py"]
    for path in files:
        target_python = (3, 12) if path.is_relative_to(ROOT / "container") else (3, 10)
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=target_python)
    # Parse the actual Linux workflow blocks without executing them.
    bash = Path(r"D:\workingtools\Git\bin\bash.exe") if os.name == "nt" else Path("/bin/bash")
    for index, block in enumerate(shell_blocks):
        path = out / ("workflow-" + str(index) + ".sh")
        path.write_text(block, encoding="utf-8", newline="\n")
        result = subprocess.run([str(bash), "-n", str(path)], cwd=ROOT, env=env, capture_output=True, encoding="utf-8", errors="replace")
        if result.returncode:
            raise RuntimeError("Workflow shell syntax check failed: " + result.stderr)
    report = {"status": "PASS_LOCAL_RECIPE_AND_REGRESSIONS_ONLY", "regression_counts": counts,
              "total_regression_tests": sum(counts.values()), "skipped_runtime_tests": sum(skips.values()),
              "executed_regression_tests": sum(counts.values()) - sum(skips.values()), "python_syntax_files": len(files),
              "workflow_shell_blocks": len(shell_blocks), "local_torch_available": importlib.util.find_spec("torch") is not None,
              "evidence_directory": str(out), "not_run": ["Linux Docker image", "Registry publish/pull",
              "Actual installed Omni imports/numerics", "Actual CUDA copy events on worker side stream", "Eight-rank real SeedVR2 metrics/2 canary", "off/basic perturbation", "Fresh image baseline and 17/33/65 ablation"]}
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
