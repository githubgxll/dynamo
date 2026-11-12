#!/usr/bin/env python3
"""Check and freeze the SeedVR2 BF16 recipe; this tool never submits a build."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EVIDENCE = ROOT / ".h3-experimental-evidence/seedvr2-bf16/local"
BRANCH = "Dingorouter-H3-vllm-omni-jirx"
REPOSITORY = "https://github.com/githubgxll/dynamo.git"
HARBOR = "registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm"
PARENT = HARBOR + "@sha256:26d3767f090fa2fecb2d5a94cf5b326657dd2fef891a13ded41476056f99fe18"
PARENT_ID = "sha256:7b0f8c28d87fe914b1c1061c732746de7171862e6c6da3c6223164b1d9d5fd87"
BASE = "a24e3fada566810d3389f00277e8e3b9478f7a25"
MARKER = "[seedvr2-bf16]"
WORKFLOW = "seedvr2-bf16.yml"
EXTERNAL_REVIEW = Path(r"D:\AI\Issue_solution\SeedVR2_P001_BF16_20261112\release\reviews\BUILD_REVIEW.json")
AUXILIARY = {"delivery/seedvr2-bf16-production/input-seal.json",
             "delivery/seedvr2-bf16-production/freeze-approval.json"}
PREFIXES = ("container/seedvr2-bf16-production/", "delivery/seedvr2-bf16-production/")
FILES = {".github/scripts/seedvr2_bf16_ci.py", ".github/scripts/test_seedvr2_bf16_ci.py",
         ".github/workflows/seedvr2-bf16.yml", ".github/workflows/dingo-router-ci.yml",
         ".github/workflows/h3-image-worker.yml", ".github/workflows/h3-sr-production.yml",
         ".github/workflows/h3-sr-video-gateway.yml", ".github/workflows/omni032-h3-sr.yml",
         ".dockerignore", ".gitattributes"}
GIT = (r"D:\workingtools\Git\cmd\git.exe" if os.name == "nt" else shutil.which("git")) or "git"


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object: " + str(path))
    return value


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def hash_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def git(*args, binary=False, env=None):
    command = [GIT, "-C", str(ROOT), "-c", "safe.directory=" + ROOT.as_posix(),
               "-c", "filter.lfs.process=", "-c", "filter.lfs.required=false"]
    result = subprocess.run([*command, *args], capture_output=True, env=env, timeout=120)
    if result.returncode:
        raise RuntimeError("Git " + args[0] + " failed:\n" + (result.stdout + result.stderr).decode("utf-8", errors="replace"))
    return result.stdout if binary else result.stdout.decode("utf-8").strip()


def validate_lock(value):
    expected = {"schema": "seedvr2-bf16-release-lock/1", "repository": REPOSITORY,
                "remote_branch": BRANCH, "harbor_repository": HARBOR, "parent_image": PARENT,
                "parent_image_id": PARENT_ID, "platform": "linux/amd64", "recipe_base_commit": BASE,
                "tag_prefix": "seedvr2-dit-bf16", "marker": MARKER, "model_weights_in_image": False}
    for key, item in expected.items():
        if value.get(key) != item:
            raise ValueError("Release lock differs: " + key)
    environment = {"SEEDVR2_BF16_SOURCE_COMMIT": "{source_commit}", "SEEDVR2_BF16_PARENT_IMAGE": PARENT,
                   "SEEDVR2_DIT_BF16": "1", "H3_SR_PRODUCTION_SOURCE_COMMIT": "{source_commit}",
                   "H3_SR_PRODUCTION_PARENT_IMAGE": PARENT}
    if value.get("expected_environment_additions") != environment:
        raise ValueError("Only the five reviewed BF16/production identity environment values may change")
    checks = value.get("required_precision_cpu_checks")
    if (not isinstance(checks, list) or not checks or any(not isinstance(name, str) or not name for name in checks)
            or len(set(checks)) != len(checks)):
        raise ValueError("The actual precision CPU helper contract must be frozen explicitly")


def lock():
    value = read(HERE / "release-lock.json")
    validate_lock(value)
    return value


def check_scope(names):
    bad = [name for name in names if "\\" in name or any(part in ("", ".", "..") for part in name.split("/"))
           or (name not in FILES and not name.startswith(PREFIXES))]
    if bad:
        raise ValueError("Unreviewed files outside the BF16 recipe: " + ", ".join(sorted(bad)))


def temporary_file(suffix):
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix="seedvr2-bf16-", suffix=suffix, dir=EVIDENCE)
    os.close(descriptor)
    return Path(name)


def review_tree(full_tree):
    index = temporary_file(".index")
    index.unlink()
    env = dict(os.environ, GIT_INDEX_FILE=str(index))
    try:
        git("read-tree", full_tree, env=env)
        for name in sorted(AUXILIARY):
            git("update-index", "--force-remove", "--", name, env=env)
        return git("write-tree", env=env)
    finally:
        for path in (index, Path(str(index) + ".lock")):
            if path.exists():
                path.unlink()


def recipe_identity(value, full_tree):
    reviewed = review_tree(full_tree)
    names = list(filter(None, git("diff", "--name-only", "--no-renames", "-z", value["recipe_base_commit"],
                                 reviewed, binary=True).decode().split("\0")))
    check_scope(names)
    if not names:
        raise ValueError("An empty or approval-only recipe cannot request a BF16 image")
    files = {}
    for name in sorted(names):
        entry = git("ls-tree", "-z", reviewed, "--", name, binary=True)
        if not entry:
            files[name] = None
            continue
        metadata, actual = entry.rstrip(b"\0").split(b"\t", 1)
        if metadata.split()[0] not in (b"100644", b"100755") or actual.decode() != name:
            raise ValueError("Recipe symlinks, submodules and non-file inputs are rejected")
        files[name] = hashlib.sha256(git("show", reviewed + ":" + name, binary=True)).hexdigest()
    payload = {"base_commit": value["recipe_base_commit"], "parent_image": value["parent_image"],
               "parent_image_id": value["parent_image_id"], "files": files}
    return {"schema": "seedvr2-bf16-input-seal/1", **payload,
            "recipe_sha256": hash_value(payload), "review_tree": reviewed}


def prospective(value):
    origin = git("remote", "get-url", "origin").rstrip("/").removesuffix(".git")
    if origin != REPOSITORY.removesuffix(".git"):
        raise ValueError("Unexpected origin repository")
    git("merge-base", "--is-ancestor", value["recipe_base_commit"], "HEAD")
    modified = git("diff", "HEAD", "--name-only", "--no-renames", "-z", binary=True).decode().split("\0")
    untracked = git("ls-files", "--others", "--exclude-standard", "-z", binary=True).decode().split("\0")
    names = sorted(set(filter(None, modified + untracked)))
    check_scope(names)
    index, pathspec = temporary_file(".index"), temporary_file(".names")
    index.unlink()
    env = dict(os.environ, GIT_INDEX_FILE=str(index))
    try:
        git("read-tree", "HEAD", env=env)
        if names:
            pathspec.write_bytes(b"\0".join(name.encode() for name in names) + b"\0")
            # Enumerated, scope-checked paths only. --force preserves historical tracked/ignored paths.
            git("--literal-pathspecs", "add", "--force", "--all", "--pathspec-from-file=" + str(pathspec),
                "--pathspec-file-nul", env=env)
        full = git("write-tree", env=env)
        recipe = recipe_identity(value, full)
        for name, expected in recipe["files"].items():
            if expected is not None and hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise ValueError("Working bytes differ from Git export: " + name)
        return full, recipe
    finally:
        for path in (index, Path(str(index) + ".lock"), pathspec):
            if path.exists():
                path.unlink()


def check_review(review, value, recipe):
    for key, item in {"status": "PASS", "recipe_sha256": recipe["recipe_sha256"],
                      "review_tree": recipe["review_tree"], "parent_image": value["parent_image"],
                      "parent_image_id": value["parent_image_id"]}.items():
        if review.get(key) != item:
            raise ValueError("Independent root review is not bound to the exact recipe: " + key)


def check_approval(approval, value, recipe):
    for key, item in {"schema": "seedvr2-bf16-freeze/1", "status": "READY_FOR_BUILD",
                      "recipe_sha256": recipe["recipe_sha256"], "review_tree": recipe["review_tree"],
                      "parent_image": value["parent_image"], "parent_image_id": value["parent_image_id"]}.items():
        if approval.get(key) != item:
            raise ValueError("Frozen approval differs: " + key)
    if approval.get("reviews", {}).get("root") != "PASS":
        raise ValueError("Independent root approval is required")
    review = approval.get("root_review", {})
    check_review(review, value, recipe)
    if approval.get("root_review_sha256") != hash_value(review):
        raise ValueError("Embedded coordinating root review hash differs")


def check_seal(committed=False, approved=True):
    value = lock()
    full = git("rev-parse", "HEAD^{tree}") if committed else prospective(value)[0]
    recipe = recipe_identity(value, full)
    if read(HERE / "input-seal.json") != recipe:
        raise ValueError("The frozen recipe changed; repeat affected checks and reseal")
    if approved:
        check_approval(read(HERE / "freeze-approval.json"), value, recipe)
    return value, full, recipe


def valid_request_message(message):
    known = re.findall(r"\[(?:h3-image-worker|omni032-h3-sr|omni-gateway-lite|h3-experimental|seedvr2-[a-z0-9-]+|h3-hsdp-offload|sr-native|h3-sr-production|h3-sr-video-gateway)\]", message)
    return (known == [MARKER] and not re.search(r"\[(?:skip ci|ci skip|no ci|skip actions|actions skip)\]", message, re.I)
            and not re.search(r"(?mi)^skip-checks:\s*true\s*$", message))


def check(skip_shell=False):
    value = lock()
    paths = set((ROOT / "container/seedvr2-bf16-production").rglob("*.py")) | set(HERE.rglob("*.py"))
    paths.update(ROOT / ".github/scripts" / name for name in ("seedvr2_bf16_ci.py", "test_seedvr2_bf16_ci.py"))
    for path in sorted(paths):
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path), feature_version=(3, 10))
    output = EVIDENCE / ("checks-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    output.mkdir(parents=True)
    suites = [(ROOT / ".github/scripts", "test_seedvr2_bf16_ci.py"), (HERE / "tests", "test_*.py"),
              (ROOT / "container/seedvr2-bf16-production/tests", "test_*.py")]
    counts = {}
    for directory, pattern in suites:
        if not directory.is_dir() or not list(directory.glob(pattern)):
            continue
        result = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", str(directory), "-p", pattern],
                                cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        log = result.stdout + result.stderr
        suite_name = directory.relative_to(ROOT).as_posix()
        (output / (suite_name.replace("/", "-") + ".txt")).write_text(log, encoding="utf-8")
        count = re.search(r"Ran ([0-9]+) tests?", log)
        if result.returncode or count is None or int(count[1]) <= 0 or re.search(r"\bskipped=", log):
            raise RuntimeError("Local regression suite failed: " + str(directory) + "\n" + log[-12000:])
        counts[suite_name] = int(count[1])
    workflow = (ROOT / ".github/workflows" / WORKFLOW).read_text(encoding="utf-8")
    blocks = re.findall(r"(?m)^        run: \|\n((?:          [^\n]*\n|\n)+)", workflow)
    if not blocks or len(blocks) != len(re.findall(r"(?m)^        run:", workflow)):
        raise ValueError("Every workflow shell block must be checked")
    if not skip_shell:
        bash = Path(r"D:\workingtools\Git\usr\bin\bash.exe") if os.name == "nt" else Path(shutil.which("bash") or "bash")
        for number, block in enumerate(blocks):
            path = output / ("workflow-" + str(number) + ".sh")
            path.write_text("\n".join(line[10:] for line in block.splitlines()) + "\n", encoding="utf-8", newline="\n")
            result = subprocess.run([str(bash), "--noprofile", "--norc", "-n", path.as_posix()], capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError("Shell syntax check failed: " + result.stdout + result.stderr)
    full, recipe = prospective(value)
    result = {"status": "PASS_LOCAL_IMPLEMENTATION_CHECKS_ONLY" if not skip_shell else "PASS_LOCAL_TESTS_SHELL_NOT_RUN",
              "regression_counts": counts, "regression_tests": sum(counts.values()), "python_syntax_files": len(paths),
              "workflow_shell_blocks": len(blocks), "shell_checked": not skip_shell, "evidence_directory": str(output),
              "prospective_tree": full, "recipe_sha256": recipe["recipe_sha256"], "review_tree": recipe["review_tree"],
              "boundaries": "Local regression fixtures and syntax only; Linux runtime, Harbor and GPU are separate."}
    dump(output / "summary.json", result)
    dump(EVIDENCE / "latest-checks.json", result)
    return result


def seal():
    value = lock()
    full, recipe = prospective(value)
    dump(HERE / "input-seal.json", recipe)
    result = {"status": "FROZEN_AWAITING_INDEPENDENT_REVIEW", "prospective_tree": prospective(value)[0], **recipe}
    dump(EVIDENCE / "freeze-candidate.json", result)
    return result


def import_review(path):
    if Path(path).resolve() != EXTERNAL_REVIEW.resolve():
        raise ValueError("Only the designated coordinating root review can approve this image")
    value, _, recipe = check_seal(approved=False)
    review = read(path)
    check_review(review, value, recipe)
    approval = {"schema": "seedvr2-bf16-freeze/1", "status": "READY_FOR_BUILD",
                "recipe_sha256": recipe["recipe_sha256"], "review_tree": recipe["review_tree"],
                "parent_image": value["parent_image"], "parent_image_id": value["parent_image_id"],
                "reviews": {"root": "PASS"}, "root_review": review, "root_review_sha256": hash_value(review),
                "root_review_file_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    dump(HERE / "freeze-approval.json", approval)
    return approval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    check_command = commands.add_parser("check")
    check_command.add_argument("--skip-shell", action="store_true")
    commands.add_parser("seal")
    review = commands.add_parser("import-review")
    review.add_argument("path", type=Path, nargs="?", default=EXTERNAL_REVIEW)
    args = parser.parse_args()
    actions = {"check": lambda: check(args.skip_shell), "seal": seal,
               "import-review": lambda: import_review(args.path)}
    print(json.dumps(actions[args.action](), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
