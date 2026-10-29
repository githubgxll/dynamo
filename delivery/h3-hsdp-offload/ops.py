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
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SEAL = "delivery/h3-hsdp-offload/input-seal.json"
EVIDENCE = ROOT / ".h3-experimental-evidence/hsdp/local"
ALLOWED_PREFIXES = ("container/h3-hsdp-offload/", "delivery/h3-hsdp-offload/")
ALLOWED_FILES = {".github/workflows/dingo-router-ci.yml", ".github/scripts/h3_hsdp_offload_ci.py",
                 ".github/scripts/test_h3_hsdp_offload_ci.py"}
SHARED_PREFIXES = ("container/h3-seedvr2-rmsnorm/", "container/h3-experimental/environment/",
                   "container/omni-gateway-lite/", "delivery/h3-experimental/", "delivery/omni-gateway-lite/",
                   "container/seedvr2-c14/", "delivery/seedvr2-c14/",
                   ".github/scripts/seedvr2_c14_ci.py", ".github/scripts/omni_gateway_lite_ci.py")


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


def triggers_hsdp(message):
    """A marker cannot countermand GitHub's commit-message skip instructions."""
    skipped = re.search(r"\[(?:skip ci|ci skip|no ci|skip actions|actions skip)\]", message, re.I)
    skipped = skipped or re.search(r"(?mi)^skip-checks:\s*true\s*$", message)
    return "[h3-hsdp-offload]" in message and not skipped


def check_scope(names):
    unexpected = sorted(name for name in names if not permitted(name))
    if unexpected:
        raise ValueError("Unrelated changes must be preserved and reviewed separately: " + ", ".join(unexpected))


def identity(lock):
    if git("branch", "--show-current") != lock["remote_branch"]:
        raise ValueError("Use existing branch " + lock["remote_branch"])
    if git("remote", "get-url", "origin").rstrip("/").removesuffix(".git") != lock["repository"].removesuffix(".git"):
        raise ValueError("Unexpected origin repository")
    validate_anchors(lock)


def working_base(lock):
    # Older locks used one baseline for both recipe export and change scope.
    return lock.get("working_tree_base_commit", lock["recipe_base_commit"])


def validate_anchors(lock):
    recipe, working = lock["recipe_base_commit"], working_base(lock)
    for commit in dict.fromkeys((recipe, working)):
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("Recipe and working-tree anchors must be full Git commit SHA values")
        if git("rev-parse", "--verify", commit + "^{commit}") != commit:
            raise ValueError("Recipe and working-tree anchors must identify commits directly")
    git("merge-base", "--is-ancestor", recipe, working)
    git("merge-base", "--is-ancestor", working, "HEAD")


def raw_delta(base, tree):
    fields = git("diff", "--raw", "--no-renames", "--no-abbrev", "-z", base, tree,
                 binary=True).decode("utf-8").split("\0")
    delta = {}
    for i in range(0, len(fields) - 1, 2):
        metadata, name = fields[i:i + 2]
        pieces = metadata.split()
        if len(pieces) != 5 or pieces[4] not in {"A", "M", "D", "T"}:
            raise ValueError("Unrecognized Git delta: " + metadata)
        if name != SEAL:
            delta[name] = {"mode": pieces[1], "blob": pieces[3], "status": pieces[4]}
    return delta


def delta_for_tree(tree, lock):
    # This also runs for detached CI. Historical unrelated commits are not a
    # permission to change their files in this candidate, including reverting
    # one to the recipe baseline so that it vanishes from the cumulative diff.
    validate_anchors(lock)
    check_scope(raw_delta(working_base(lock), tree))
    cumulative = raw_delta(lock["recipe_base_commit"], tree)
    shared_changes = sorted(name for name in cumulative if any(
        name.startswith(prefix) if prefix.endswith("/") else name == prefix
        for prefix in SHARED_PREFIXES))
    if shared_changes:
        raise ValueError("Shared inputs differ from the fixed recipe baseline: " + ", ".join(shared_changes))
    # Keep all reviewed HSDP history for the complete exported recipe, while
    # excluding unrelated history that is unchanged since the working anchor.
    return {name: entry for name, entry in cumulative.items() if permitted(name)}


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
    return {"schema": "h3-hsdp-offload-input-seal/1", "recipe_base_commit": lock["recipe_base_commit"],
            "working_tree_base_commit": working_base(lock),
            "parent_image": lock["parent_image"], "delta": delta, "shared_inputs": shared_inputs(lock),
            "scope": "Complete allowed recipe delta from recipe_base_commit; all candidate changes from working_tree_base_commit must stay in the original scope. Unrelated history is excluded unchanged; shared inputs remain fixed at recipe_base_commit. Excludes seal itself."}


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
    git("diff", "--check", lock["recipe_base_commit"], tree, "--", *delta)
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
        raise ValueError("Use 02-submit-build.cmd to explicitly commit/push the reviewed recipe")
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
        git("commit", "-s", "-m", "build(omni): submit H3 HSDP offload candidate [h3-hsdp-offload]")
    if git("rev-parse", "HEAD^{tree}") != tree or git("status", "--porcelain"):
        raise ValueError("Checkout changed while committing; stop before push")
    if LEGACY.remote_head(branch) == git("rev-parse", "HEAD"):
        print("This exact recipe is already uploaded; no duplicate push was made.")
        print("Check Actions for commit " + git("rev-parse", "HEAD") + ". Use 03-run-build.cmd only to request another build.")
        return
    if not triggers_hsdp(git("log", "-1", "--format=%B")):
        # A user may have committed the validated tree manually. The push event
        # routes by its final commit message, so append a bounded empty request.
        git("var", "GIT_AUTHOR_IDENT")
        git("commit", "--allow-empty", "-s", "-m", "ci(omni): build reviewed HSDP offload recipe [h3-hsdp-offload]")
        if git("rev-parse", "HEAD^{tree}") != tree or git("status", "--porcelain"):
            raise ValueError("Checkout changed while preparing the HSDP offload build request")
    git("push", "origin", f"HEAD:refs/heads/{branch}", network=True)
    print("Uploaded commit " + git("rev-parse", "HEAD") + "; its push triggers ONLY the HSDP offload candidate image build.")
    print("Progress: https://github.com/githubgxll/dynamo/actions/workflows/dingo-router-ci.yml")


def shared_inputs(lock):
    # Every reused source is locked to the existing recipe baseline; no local edit
    # to shared code is accepted by this alternative's bounded submission scope.
    paths = []
    for prefix in SHARED_PREFIXES:
        paths.extend(git("ls-tree", "-r", "--name-only", lock["recipe_base_commit"], "--", prefix).splitlines())
    return {name: git("rev-parse", lock["recipe_base_commit"] + ":" + name) for name in sorted(set(paths))}


def export_recipe(directory):
    check_seal(committed=True)
    sealed = json.loads((ROOT / SEAL).read_text(encoding="utf-8"))
    names = {SEAL, *sealed["shared_inputs"], *(name for name, item in sealed["delta"].items() if item["status"] != "D")}
    for name in sorted(names):
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(git("show", "HEAD:" + name, binary=True))


def dispatch():
    # The workflow does not exist on main, so GitHub's webpage dispatch button
    # is unavailable. An explicit empty commit uses the existing working push
    # trigger without requiring gh, a separate token, or changes to main.
    lock, tree = check()
    if git("status", "--porcelain") or LEGACY.remote_head(lock["remote_branch"]) != git("rev-parse", "HEAD"):
        raise ValueError("Submit this exact clean recipe with 02-submit-build.cmd before requesting a repeat build")
    git("var", "GIT_AUTHOR_IDENT")
    git("commit", "--allow-empty", "-s", "-m", "ci(omni): rerun H3 HSDP offload candidate [h3-hsdp-offload]")
    if git("rev-parse", "HEAD^{tree}") != tree or git("status", "--porcelain"):
        raise ValueError("Checkout changed during the repeat-build request; refusing to push")
    git("push", "origin", f"HEAD:refs/heads/{lock['remote_branch']}", network=True)
    print("Repeat build requested by empty commit " + git("rev-parse", "HEAD") + "; recipe files are unchanged.")
    print("Progress: https://github.com/githubgxll/dynamo/actions/workflows/dingo-router-ci.yml")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("seal")
    commands.add_parser("check")
    commands.add_parser("dispatch")
    export = commands.add_parser("export-recipe")
    export.add_argument("--directory", type=Path, required=True)
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
    elif args.command == "export-recipe":
        export_recipe(args.directory)
    elif args.command == "dispatch":
        dispatch()
    else:
        submit(args.confirm_build_and_publish)


if __name__ == "__main__":
    main()
