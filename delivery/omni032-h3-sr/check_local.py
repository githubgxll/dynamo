#!/usr/bin/env python3
"""Local recipe regressions/syntax only; never builds an image or writes online."""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def export_check(path):
    result = subprocess.run([sys.executable, "-B", "-X", "utf8", str(HERE / "ops.py"), "verify-export"],
                            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    path.write_text(result.stdout + ("\n" + result.stderr if result.stderr else ""), encoding="utf-8", newline="\n")
    if result.returncode:
        raise RuntimeError("Git export bytes failed before approval/build; see " + str(path))
    proof = json.loads(result.stdout)
    if proof.get("status") != "PASS_LOCAL_EXPORT_BYTES_ONLY":
        raise ValueError("Missing actual Git export byte proof")
    return proof


def main():
    import yaml
    output = ROOT / ".h3-experimental-evidence/omni032-h3-sr/local" / ("checks-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    output.mkdir(parents=True)
    before_export = export_check(output / "export-bytes-before.json")
    suites = {"framework-patches": ["-m", "unittest", "discover", "-s", "container/omni032-h3-sr/tests", "-p", "test*.py"],
              "submission-and-publication": ["-m", "unittest", "discover", "-s", ".github/scripts", "-p", "test_omni032_h3_sr_ci.py"]}
    counts = {}
    for name, arguments in suites.items():
        result = subprocess.run([sys.executable, "-B", "-X", "utf8", *arguments], cwd=ROOT,
                                capture_output=True, text=True, encoding="utf-8", errors="replace")
        text = result.stdout + result.stderr
        (output / (name + ".txt")).write_text(text, encoding="utf-8")
        match = re.search(r"Ran ([0-9]+) tests?", text)
        if result.returncode or not match or int(match.group(1)) < 1 or re.search(r"skipped=[1-9]", text):
            raise RuntimeError(name + " failed; see " + str(output / (name + ".txt")))
        counts[name] = int(match.group(1))
        print("PASS " + name)
    python_files = list((ROOT / "container/omni032-h3-sr").rglob("*.py")) + list(HERE.glob("*.py"))
    python_files += [ROOT / ".github/scripts/omni032_h3_sr_ci.py", ROOT / ".github/scripts/test_omni032_h3_sr_ci.py"]
    for path in python_files:
        target_python = (3, 12) if path.is_relative_to(ROOT / "container") else (3, 10)
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path), feature_version=target_python)
    workflow = yaml.load((ROOT / ".github/workflows/omni032-h3-sr.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    bash_candidates = [Path(r"D:\workingtools\Git\usr\bin\bash.exe"), Path(r"D:\workingtools\Git\bin\bash.exe")]
    found_bash = shutil.which("bash")
    if found_bash: bash_candidates.append(Path(found_bash))
    bash_candidates = list(dict.fromkeys(str(path) for path in bash_candidates if path.is_file()))
    if not bash_candidates: raise RuntimeError("Bash syntax verification is required")
    blocks = 0
    shell_scripts = []
    for job in workflow["jobs"].values():
        for step in job["steps"]:
            if "run" not in step: continue
            script = output / f"workflow-{blocks}.sh"
            script.write_text(step["run"], encoding="utf-8", newline="\n")
            shell_scripts.append(script)
            blocks += 1
    packaged_shell = sorted((ROOT / "container/omni032-h3-sr").glob("*.sh"))
    if not packaged_shell:
        raise RuntimeError("The actual dependency installer shell is required")
    for script in [*shell_scripts, *packaged_shell]:
        startup_errors = []
        for bash in bash_candidates:
            result = subprocess.run([bash, "-n", script.as_posix()], cwd=ROOT, capture_output=True,
                                    text=True, encoding="utf-8", errors="replace")
            if result.returncode == 0: break
            startup_errors.append({"bash": bash, "exit_code": result.returncode, "stderr": result.stderr})
            # Bash's normal syntax-error exit is authoritative. Only a
            # Windows/MSYS startup failure permits trying another binary.
            if result.returncode in range(1, 256):
                raise RuntimeError("Shell syntax failed in " + str(script) + ": " + result.stderr)
        else:
            (output / "bash-startup-errors.json").write_text(json.dumps(startup_errors, indent=2), encoding="utf-8")
            raise RuntimeError("Bash could not initialize. Windows managed isolation may deny MSYS named objects; run these local syntax checks in an allowed environment. No shell syntax PASS was claimed. Evidence: " + str(output))
    after_export = export_check(output / "export-bytes-after.json")
    if before_export["recipe_sha256"] != after_export["recipe_sha256"]:
        raise ValueError("Actual source inputs changed while the local checks were running")
    report = {"status": "PASS_LOCAL_IMPLEMENTATION_ONLY_NOT_FROZEN", "regression_counts": counts,
              "tests_run": sum(counts.values()), "skipped": 0, "python_syntax_files": len(python_files),
              "workflow_shell_blocks": blocks, "packaged_shell_files": len(packaged_shell), "evidence_directory": str(output),
              "recipe_sha256": after_export["recipe_sha256"], "git_export_checked_inputs": after_export["checked_input_paths"],
              "not_run": ["Linux image/package installation", "Docker build", "Harbor push/pull", "CUDA", "Ref2VA media"]}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
