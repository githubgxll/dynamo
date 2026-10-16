#!/usr/bin/env python3
"""Local regressions and configuration validation; not a container/GPU test."""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "container/h3-experimental"
RUN = ROOT / ".h3-experimental-evidence/local" / datetime.now(timezone.utc).strftime("checks-%Y%m%dT%H%M%S.%fZ")
def find_bash():
    # Windows PATH commonly resolves the WSL launcher, even without any WSL
    # distribution. Use the installed Git Bash for syntax checks on this host.
    if os.name == "nt":
        git_path = shutil.which("git")
        candidates = [Path(r"D:\workingtools\Git\bin\bash.exe")]
        if git_path:
            candidates.insert(0, Path(git_path).parent.parent / "bin/bash.exe")
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        raise RuntimeError("Git Bash is required for local shell syntax checks; WSL is not assumed")
    found = shutil.which("bash")
    if not found:
        raise RuntimeError("bash is required for shell syntax checks")
    return found


BASH = find_bash()


def command(name, argv):
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    (RUN / (name + ".txt")).write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        print(result.stdout + result.stderr)
        raise RuntimeError(f"{name} failed; see {RUN}")
    print("PASS " + name)
    matches = re.findall(r"Ran (\d+) tests?", result.stdout + result.stderr)
    return sum(map(int, matches))


def main():
    RUN.mkdir(parents=True)
    count = 0
    for name, path in [
        ("asset-download-and-integrity-regressions", PROFILE / "assets/tests"),
        ("ref8-contract-regressions", PROFILE / "ref8/tests"),
        ("six-profile-regressions", PROFILE / "profiles/tests"),
        ("environment-auditor-synthetic-regressions", PROFILE / "environment"),
        ("local-submission-evidence-regressions", Path(__file__).resolve().parent / "tests"),
    ]:
        if not path.is_dir():
            raise FileNotFoundError(path)
        count += command(name, [sys.executable, "-B", "-X", "utf8", "-m", "unittest", "discover", "-s", str(path), "-p", "test_*.py", "-v"])
    count += command("ci-and-gateway-renderer-regressions", [sys.executable, "-B", "-X", "utf8", str(ROOT / ".github/scripts/test_h3_experimental_ci.py")])
    command("ref8-local-fixed-source-contract", [sys.executable, "-B", "-X", "utf8", str(PROFILE / "ref8/check_contract.py"), "--local", "--report", str(RUN / "ref8-local.json")])
    command("six-profile-local-contract", [sys.executable, "-B", "-X", "utf8", str(PROFILE / "profiles/check_profiles.py"), "--local", "--output", str(RUN / "profiles-local.json")])
    scripts = list(PROFILE.rglob("*.py")) + list(Path(__file__).resolve().parent.rglob("*.py"))
    for path in scripts:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    workflow = yaml.load((ROOT / ".github/workflows/dingo-router-ci.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    shell_blocks = [s["run"] for job in workflow["jobs"].values() for s in job["steps"] if "run" in s]
    with tempfile.TemporaryDirectory(prefix="h3-shell-syntax-") as temp:
        for index, block in enumerate(shell_blocks):
            path = Path(temp) / f"step-{index}.sh"
            path.write_text(re.sub(r"\$\{\{.*?\}\}", "SYNTHETIC_CI_VALUE", block), encoding="utf-8", newline="\n")
            command(f"workflow-shell-{index}", [BASH, "-n", path.as_posix()])
    for index, path in enumerate(sorted(PROFILE.rglob("*.sh")) + [ROOT / "container/use-sccache.sh"]):
        command(f"script-shell-{index}", [BASH, "-n", path.as_posix()])
    report = {"status": "PASS_LOCAL_IMPLEMENTATION_CHECKS_ONLY", "regression_tests": count,
              "python_syntax_files": len(scripts), "workflow_shell_blocks": len(shell_blocks),
              "evidence_directory": str(RUN),
              "boundaries": "Dependency-auditor regression fixtures are simulated metadata; Ref8 local checks use fixed source/AST. Actual Linux packages, Docker build, GPU and media acceptance are not run locally."}
    (RUN / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
