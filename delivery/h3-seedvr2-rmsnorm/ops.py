#!/usr/bin/env python3
"""Prepare, seal and submit a bounded derivative on the existing H3 branch."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SEAL = "delivery/h3-seedvr2-rmsnorm/input-seal.json"
EVIDENCE = ROOT / ".h3-experimental-evidence/srnorm/local"
ALLOWED_PREFIXES = ("container/h3-seedvr2-rmsnorm/", "delivery/h3-seedvr2-rmsnorm/")
ALLOWED_FILES = {".github/workflows/dingo-router-ci.yml", ".github/scripts/h3_seedvr2_rmsnorm_ci.py",
                 ".github/scripts/test_h3_seedvr2_rmsnorm_ci.py", ".github/scripts/test_h3_experimental_ci.py",
                 ".gitattributes"}


def load_legacy():
    # Reuse the tested command-scoped Git proxy and index backup/staging logic.
    spec = importlib.util.spec_from_file_location("h3_parent_ops", ROOT / "delivery/h3-experimental/ops.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = EVIDENCE
    return module


LEGACY = load_legacy()
git = LEGACY.git


def load_lock():
    return json.loads((HERE / "source-lock.json").read_text(encoding="utf-8"))


def permitted(name):
    return name in ALLOWED_FILES or name.startswith(ALLOWED_PREFIXES)


def check_scope(names):
    unexpected = sorted(name for name in names if not permitted(name))
    if unexpected:
        raise ValueError("Unrelated changes must be preserved and reviewed separately: " + ", ".join(unexpected))


def identity(lock):
    if git("branch", "--show-current") != lock["remote_branch"]:
        raise ValueError("Use existing branch " + lock["remote_branch"])
    if git("remote", "get-url", "origin").rstrip("/").removesuffix(".git") != lock["repository"].removesuffix(".git"):
        raise ValueError("Unexpected origin repository")
    git("merge-base", "--is-ancestor", lock["parent_source_commit"], "HEAD")


def delta_for_tree(tree, lock):
    fields = git("diff", "--raw", "--no-renames", "--no-abbrev", "-z", lock["parent_source_commit"], tree,
                 binary=True).decode("utf-8").split("\0")
    delta = {}
    for i in range(0, len(fields) - 1, 2):
        metadata, name = fields[i:i + 2]
        pieces = metadata.split()
        if len(pieces) != 5 or pieces[4] not in {"A", "M", "D", "T"}:
            raise ValueError("Unrecognized Git delta: " + metadata)
        if name != SEAL:
            delta[name] = {"mode": pieces[1], "blob": pieces[3], "status": pieces[4]}
    check_scope(delta)
    return delta


def prospective(lock):
    changed = git("diff", "HEAD", "--name-only", "--no-renames", "-z", binary=True).decode("utf-8").split("\0")
    untracked = git("ls-files", "--others", "--exclude-standard", "-z", binary=True).decode("utf-8").split("\0")
    names = sorted(set(filter(None, changed + untracked)))
    check_scope(names)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="index-", dir=EVIDENCE) as tmp:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(tmp) / "index"))
        git("read-tree", "HEAD", env=env)
        # Only explicit changed paths are added. Inherited ignored files and
        # executable modes remain exactly as stored in HEAD.
        if names:
            pathspec = Path(tmp) / "paths.bin"
            pathspec.write_bytes(b"\0".join(name.encode("utf-8") for name in names) + b"\0")
            git("--literal-pathspecs", "add", "--all", "--pathspec-file-nul",
                "--pathspec-from-file=" + str(pathspec), env=env)
        tree = git("write-tree", env=env)
    return tree, delta_for_tree(tree, lock)


def expected_seal(lock, delta):
    return {"schema": "h3-seedvr2-rmsnorm-input-seal/1", "parent_source_commit": lock["parent_source_commit"],
            "parent_image": lock["parent_image"], "delta": delta,
            "scope": "Exact Git modes/blobs relative to successful parent source; only derivative inputs, CI routing and their tests. Excludes this seal itself."}


def seal():
    lock = load_lock()
    identity(lock)
    _, delta = prospective(lock)
    (ROOT / SEAL).write_text(json.dumps(expected_seal(lock, delta), indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Sealed {len(delta)} reviewed derivative paths; no index/branch mutation")


def check_seal(committed=False):
    lock = load_lock()
    # Actions checkout uses detached HEAD. Branch identity is enforced in CI.
    if not committed:
        identity(lock)
    if committed:
        tree = git("rev-parse", "HEAD^{tree}")
        delta = delta_for_tree(tree, lock)
    else:
        tree, delta = prospective(lock)
    actual = json.loads((ROOT / SEAL).read_text(encoding="utf-8"))
    if actual != expected_seal(lock, delta):
        raise ValueError("Inputs differ from their reviewed seal; inspect the diff then explicitly reseal")
    for name, entry in delta.items():
        if entry["status"] != "D" and (ROOT / name).read_bytes() != git("show", tree + ":" + name, binary=True):
            raise ValueError("Local bytes differ from the publishable Git blob: " + name)
    if (ROOT / SEAL).read_bytes() != git("show", tree + ":" + SEAL, binary=True):
        raise ValueError("Seal bytes differ from publishable Git blob")
    git("diff", "--check", lock["parent_source_commit"], tree, "--", *delta)
    return lock, tree


def check():
    lock, tree = check_seal()
    subprocess.run([sys.executable, "-B", "-X", "utf8", str(HERE / "check_local.py")], cwd=ROOT, check=True)
    print(json.dumps({"status": "PASS_LOCAL_RECIPE_ONLY", "parent_image": lock["parent_image"],
                      "branch": lock["remote_branch"], "prospective_tree": tree,
                      "not_run": ["Linux Docker build", "registry publish/pull", "CUDA", "same-input media A/B"]}, indent=2))
    return lock, tree


def submit(confirmed):
    if not confirmed:
        raise ValueError("Use 02-submit-build.cmd to explicitly commit/push and trigger publishing")
    lock, tree = check()
    branch = lock["remote_branch"]
    remote = LEGACY.remote_head(branch)
    if remote is None:
        raise ValueError("The designated existing remote branch is absent; no new branch is created")
    git("fetch", "--no-tags", "origin", f"refs/heads/{branch}:refs/remotes/origin/{branch}", network=True)
    if git("rev-parse", f"refs/remotes/origin/{branch}") != remote:
        raise ValueError("Remote moved during inspection; inspect before retrying")
    git("merge-base", "--is-ancestor", remote, "HEAD")
    if prospective(lock)[0] != tree:
        raise ValueError("Files changed during validation")
    if git("rev-parse", "HEAD^{tree}") != tree:
        git("var", "GIT_AUTHOR_IDENT")
        LEGACY.stage_validated_tree(tree)
        git("commit", "-s", "-m", "fix(seedvr2): backport RMSNorm FP32 accumulation to H3 image")
    if git("rev-parse", "HEAD^{tree}") != tree or git("status", "--porcelain"):
        raise ValueError("Checkout changed while committing; stop before push")
    if LEGACY.remote_head(branch) == git("rev-parse", "HEAD"):
        print("This exact commit is already uploaded. Inspect its existing Actions run.")
        return
    git("push", "origin", f"HEAD:refs/heads/{branch}", network=True)
    print("Submitted to existing branch; patch-only CI: https://github.com/githubgxll/dynamo/actions")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("seal")
    commands.add_parser("check")
    verify = commands.add_parser("check-seal")
    verify.add_argument("--committed", action="store_true")
    publish = commands.add_parser("submit")
    publish.add_argument("--confirm-build-and-publish", action="store_true")
    args = parser.parse_args()
    if args.command == "seal":
        seal()
    elif args.command == "check":
        check()
    elif args.command == "check-seal":
        check_seal(args.committed)
        print("PASS sealed derivative inputs")
    else:
        submit(args.confirm_build_and_publish)


if __name__ == "__main__":
    main()
