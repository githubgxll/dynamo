#!/usr/bin/env python3
"""Resolve/install the full six-metric Python closure; no VBench setup.py GPU probe.

The sole metadata adjustment replaces facexlib's GUI OpenCV distribution with
its headless equivalent. All importable code and bundled licenses are unchanged.
The local version and wheel manifest expose this adjustment to later audits.
"""
from __future__ import annotations

import argparse
import base64
import csv
import difflib
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import zipfile
from pathlib import Path

from package_sources import load_sources

ORIGINAL_SHA256 = "245d58861537b820c616e8b3ef618ccfad2a24724a2d74be2b0542643c01a878"
ORIGINAL_FILENAME = "facexlib-0.3.0-py3-none-any.whl"
PATCHED_FILENAME = "facexlib-0.3.0+vbench1-py3-none-any.whl"
OLD_INFO = "facexlib-0.3.0.dist-info"
NEW_INFO = "facexlib-0.3.0+vbench1.dist-info"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def patch_facexlib(original: Path, output_dir: Path, evidence: Path) -> Path:
    """Rebuild deterministically, limiting changes to metadata identity/RECORD."""
    if sha256(original) != ORIGINAL_SHA256:
        raise ValueError("facexlib upstream wheel SHA256 does not match reviewed 0.3.0")
    with zipfile.ZipFile(original) as source:
        names = source.namelist()
        if len(names) != len(set(names)) or any(n.startswith("/") or ".." in Path(n).parts for n in names):
            raise ValueError("unsafe or duplicate wheel members")
        files = {n: source.read(n) for n in names if not n.endswith("/")}
    before = files[f"{OLD_INFO}/METADATA"].decode("utf-8")
    if before.count("Version: 0.3.0\n") != 1 or before.count("Requires-Dist: opencv-python\n") != 1:
        raise ValueError("facexlib metadata differs from reviewed input")
    after = before.replace("Version: 0.3.0\n", "Version: 0.3.0+vbench1\n").replace(
        "Requires-Dist: opencv-python\n", "Requires-Dist: opencv-python-headless\n"
    )
    files[f"{OLD_INFO}/METADATA"] = after.encode("utf-8")
    del files[f"{OLD_INFO}/RECORD"]
    files = {n.replace(OLD_INFO + "/", NEW_INFO + "/", 1): data for n, data in files.items()}
    rows = []
    for name, data in sorted(files.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        rows.append([name, "sha256=" + digest, str(len(data))])
    rows.append([NEW_INFO + "/RECORD", "", ""])
    record = io.StringIO(newline="")
    csv.writer(record, lineterminator="\n").writerows(rows)
    files[NEW_INFO + "/RECORD"] = record.getvalue().encode()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / PATCHED_FILENAME
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            wheel.writestr(info, data)
    evidence.mkdir(parents=True, exist_ok=True)
    diff = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                     fromfile=OLD_INFO + "/METADATA", tofile=NEW_INFO + "/METADATA"))
    (evidence / "facexlib-METADATA.patch").write_text(diff, encoding="utf-8")
    (evidence / "facexlib-patched-RECORD.csv").write_text(record.getvalue(), encoding="utf-8")
    (evidence / "facexlib-patch.json").write_text(json.dumps({
        "original_filename": ORIGINAL_FILENAME, "original_sha256": ORIGINAL_SHA256,
        "patched_filename": PATCHED_FILENAME, "patched_sha256": sha256(output),
        "version": "0.3.0+vbench1", "code_changed": False,
        "change": "Replace Requires-Dist opencv-python with opencv-python-headless; local version; dist-info rename; RECORD regeneration",
        "licenses_preserved": True,
        "payload_sha256": {n: hashlib.sha256(data).hexdigest() for n, data in sorted(files.items())
                           if not n.startswith(NEW_INFO + "/")},
    }, indent=2) + "\n", encoding="utf-8")
    return output


def acquire_original(wheels: Path, sources: dict | None = None) -> Path:
    sources = sources or load_sources()
    wheels.mkdir(parents=True, exist_ok=True)
    original = wheels / ORIGINAL_FILENAME
    if original.exists():
        if sha256(original) != ORIGINAL_SHA256:
            raise ValueError("cached facexlib original has wrong SHA256")
        return original
    # Fetch this one reviewed artifact before its metadata adjustment. --no-deps
    # applies only to acquisition; the later uv install resolves the full graph.
    # Failed/partial downloads must never poison the next attempt's cache.
    with tempfile.TemporaryDirectory(prefix="facexlib-download-", dir=wheels) as temporary:
        root = Path(temporary)
        requirement = root / "original-wheel.txt"
        requirement.write_text(f"facexlib==0.3.0 --hash=sha256:{ORIGINAL_SHA256}\n", encoding="utf-8")
        destination = root / "download"
        destination.mkdir()
        run([sys.executable, "-m", "pip", "--isolated", "--disable-pip-version-check", "download",
             "--index-url", sources["python_index"], "--only-binary=:all:", "--no-deps", "--require-hashes",
             "--retries", "1", "--timeout", "15", "--dest", str(destination), "-r", str(requirement)])
        downloaded = destination / ORIGINAL_FILENAME
        if {item.name for item in destination.iterdir()} != {ORIGINAL_FILENAME} or not downloaded.is_file():
            raise ValueError("facexlib download did not produce exactly the reviewed wheel")
        if sha256(downloaded) != ORIGINAL_SHA256:
            raise ValueError("downloaded facexlib wheel has wrong SHA256")
        os.replace(downloaded, original)
    return original


def run(command: list[str], output: Path | None = None) -> None:
    if output:
        with output.open("w", encoding="utf-8") as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
    else:
        subprocess.run(command, check=True)


def check_dependencies(evidence: Path) -> None:
    """Expose both installed-environment checks before enforcing their result."""
    checks = [
        ("pip-check.txt", [sys.executable, "-m", "pip", "check"]),
        ("uv-pip-check.txt", ["uv", "pip", "check", "--python", sys.executable]),
    ]
    records = []
    for filename, command in checks:
        output = evidence / filename
        exit_code, error_type = 0, None
        try:
            run(command, output)
        except subprocess.CalledProcessError as error:
            exit_code = error.returncode
        except OSError as error:
            exit_code, error_type = None, type(error).__name__
        if output.is_file():
            diagnostic = output.read_text(encoding="utf-8", errors="replace")
            # Checker messages normally contain package names and constraints.
            # Omit any embedded URL rather than emitting credentials/CDN tokens.
            def redact_url(match):
                try:
                    return "<URL host=" + str(urllib.parse.urlsplit(match.group()).hostname) + ">"
                except ValueError:
                    return "<URL omitted>"
            diagnostic = re.sub(r"https?://\S+", redact_url, diagnostic)
            output.write_text(diagnostic, encoding="utf-8")
        else:
            diagnostic = "Checker did not produce its diagnostic file.\n"
            exit_code, error_type = None, error_type or "MissingDiagnostic"
        print(f"=== {filename} (exit={exit_code}) ===", flush=True)
        print(diagnostic, end="" if diagnostic.endswith("\n") else "\n", flush=True)
        if error_type:
            print("Checker execution error: " + error_type, flush=True)
        print(f"=== end {filename} ===", flush=True)
        records.append({"file": filename, "exit_code": exit_code, "error_type": error_type})
    passed = all(record["exit_code"] == 0 for record in records)
    summary = {"status": "PASS" if passed else "FAIL", "checks": records}
    (evidence / "dependency-checks.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Dependency checks: " + json.dumps(summary), flush=True)
    if not passed:
        raise RuntimeError("Installed dependency checks failed; review the pip-check.txt and uv-pip-check.txt sections above")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["prepare", "publish"], required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 10) or sys.platform != "linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise SystemExit("Resolve/install only inside the Linux amd64 Python 3.10 image")
    args.evidence.mkdir(parents=True, exist_ok=True)
    sources = load_sources()
    print(f"Dependency Python index: {sources['python_index']}; Torch index: {sources['torch_index']}", flush=True)
    wheels = Path("/opt/vbench-build/wheels")
    patch_facexlib(acquire_original(wheels, sources), wheels, args.evidence)
    if args.phase == "prepare":
        args.lock.parent.mkdir(parents=True, exist_ok=True)
        # Re-preparation after a reviewed policy change resolves anew inside the
        # disposable build filesystem. No source checkout lock is rewritten.
        generated = args.evidence / "requirements.lock"
        run(["uv", "pip", "compile", str(args.inputs), "--python", sys.executable,
             "--generate-hashes", "--upgrade", "--no-build-isolation", "--default-index", sources["python_index"],
             "--index", sources["torch_index"], "--index-strategy", "unsafe-best-match",
             "--output-file", str(generated)])
        if generated.resolve() != args.lock.resolve():
            shutil.copy2(generated, args.lock)
    elif not args.lock.is_file():
        raise SystemExit("Publish requires the reviewed dependency lock from preparation")
    elif args.lock.resolve() != (args.evidence / "requirements.lock").resolve():
        shutil.copy2(args.lock, args.evidence / "requirements.lock")
    # Some reviewed dependencies publish only sdists (e.g. openai-clip).
    # Use the Dockerfile's pinned setuptools/wheel instead of resolving a
    # second, unrecorded build-isolation dependency environment.
    run(["uv", "pip", "install", "--python", sys.executable, "--require-hashes", "--no-build-isolation",
         "--default-index", sources["python_index"], "--index", sources["torch_index"],
         "--index-strategy", "unsafe-best-match", "-r", str(args.lock)])
    check_dependencies(args.evidence)
    run([sys.executable, "-m", "pip", "inspect", "--local"], args.evidence / "pip-inspect.json")
    run([sys.executable, "-m", "pip", "freeze", "--all"], args.evidence / "installed-freeze.txt")
    (args.evidence / "dependency-build.json").write_text(json.dumps({
        "phase": args.phase, "python": sys.version, "platform": platform.platform(),
        "package_sources": sources,
        "inputs_sha256": sha256(args.inputs), "lock_sha256": sha256(args.lock),
        "full_declared_dependency_closure": True, "pip_check": "passed", "uv_pip_check": "passed",
        "build_isolation": False, "build_tools": "Pinned pip/setuptools/wheel/uv from Dockerfile and requirements.in",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
