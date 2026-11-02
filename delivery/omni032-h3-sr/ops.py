#!/usr/bin/env python3
"""Bounded local review/freeze and one-shot submission; no build retry command."""
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
EVIDENCE = ROOT / ".h3-experimental-evidence/omni032-h3-sr/local"
SEAL = "delivery/omni032-h3-sr/input-seal.json"
APPROVAL = "delivery/omni032-h3-sr/freeze-approval.json"
AUXILIARY = {SEAL, APPROVAL}
CONTRACT = Path(r"D:\AI\H3_experiments_H100_HBM3_20261012\delivery\H3_SR_NextStage_20261102\image-upgrade-contract.json")
MARKER = "[omni032-h3-sr]"
ALLOWED_PREFIXES = ("container/omni032-h3-sr/", "delivery/omni032-h3-sr/")
ALLOWED_FILES = {".github/scripts/omni032_h3_sr_ci.py", ".github/scripts/test_omni032_h3_sr_ci.py",
                 ".github/workflows/omni032-h3-sr.yml", ".github/workflows/dingo-router-ci.yml"}
SHARED_PREFIXES = ("container/h3-experimental/environment/", "container/omni-gateway-lite/",
                   "container/seedvr2-c14/", "container/h3-hsdp-offload/",
                   "container/seedvr2-optimization/", "container/seedvr2-metrics-repair/", "container/compliance/")


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object: " + str(path))
    return value


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def hash_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


spec = importlib.util.spec_from_file_location("_omni032_shared_git_ops", ROOT / "delivery/h3-experimental/ops.py")
LEGACY = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LEGACY)
LEGACY.EVIDENCE = EVIDENCE
git = LEGACY.git


def lock():
    return read(HERE / "source-lock.json")


def permitted(name):
    return name in ALLOWED_FILES or name.startswith(ALLOWED_PREFIXES)


def check_scope(names):
    unexpected = sorted(name for name in names if not permitted(name))
    if unexpected:
        raise ValueError("Unreviewed changes outside this upgrade: " + ", ".join(unexpected))


def identity(value):
    if git("branch", "--show-current") != value["remote_branch"]:
        raise ValueError("Use the designated existing branch")
    if git("remote", "get-url", "origin").rstrip("/").removesuffix(".git") != value["repository"].removesuffix(".git"):
        raise ValueError("Unexpected origin repository")
    base = value["working_tree_base_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", base) or git("rev-parse", "--verify", base + "^{commit}") != base:
        raise ValueError("Working baseline must be a full existing commit")
    git("merge-base", "--is-ancestor", base, "HEAD")
    if value["recipe_base_commit"] != base:
        raise ValueError("This upgrade requires one common working/recipe baseline")


def raw_delta(base, tree):
    fields = git("diff", "--raw", "--no-renames", "--no-abbrev", "-z", base, tree, binary=True).decode("utf-8").split("\0")
    result = {}
    for index in range(0, len(fields) - 1, 2):
        info, name = fields[index:index + 2]
        parts = info.split()
        if len(parts) != 5 or parts[4] not in {"A", "M", "D", "T"}:
            raise ValueError("Unexpected Git delta")
        if parts[1] not in {"100644", "100755", "000000"} or (parts[1] == "000000") != (parts[4] == "D"):
            raise ValueError("Symlinks/submodules and non-file recipe objects are forbidden")
        result[name] = {"mode": parts[1], "blob": parts[3], "status": parts[4]}
    check_scope(result)
    return result


def shared_inputs(value):
    entries = git("ls-tree", "-r", "-z", value["recipe_base_commit"], "--", *SHARED_PREFIXES, binary=True)
    result = {}
    for entry in filter(None, entries.decode("utf-8").split("\0")):
        metadata, name = entry.split("\t", 1)
        mode, kind, blob = metadata.split()
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise ValueError("Unexpected shared source object")
        result[name] = blob
    return dict(sorted(result.items()))


def valid_request_message(message):
    markers = ("[seedvr2-metrics-repair]", "[seedvr2-optimization]", "[h3-hsdp-offload]", "[sr-native]", "[seedvr2-c14]")
    return (message.count(MARKER) == 1 and not any(marker in message for marker in markers)
            and not re.search(r"\[(?:skip ci|ci skip|no ci|skip actions|actions skip)\]", message, re.I)
            and not re.search(r"(?mi)^skip-checks:\s*true\s*$", message))


def recipe_identity(value, full_tree):
    delta = {name: item for name, item in raw_delta(value["recipe_base_commit"], full_tree).items() if name not in AUXILIARY}
    if not delta:
        raise ValueError("An empty recipe cannot request an image build")
    # The recipe tree excludes the seal and approval snapshot to avoid a
    # self-referential Git tree/hash. All actual recipe inputs remain covered.
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="review-tree-", dir=EVIDENCE) as temporary:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temporary) / "index"))
        git("read-tree", full_tree, env=env)
        for name in AUXILIARY:
            git("update-index", "--force-remove", "--", name, env=env)
        review_tree = git("write-tree", env=env)
    payload = {"schema": "omni032-h3-sr-recipe/1", "recipe_base_commit": value["recipe_base_commit"],
               "parent_image": value["parent_image"], "delta": delta, "shared_inputs": shared_inputs(value)}
    return {**payload, "recipe_sha256": hash_value(payload), "tree": review_tree}


def prospective(value):
    changed = git("diff", "HEAD", "--name-only", "--no-renames", "-z", binary=True).decode("utf-8").split("\0")
    untracked = git("ls-files", "--others", "--exclude-standard", "-z", binary=True).decode("utf-8").split("\0")
    names = sorted(set(filter(None, changed + untracked)))
    check_scope(names)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="index-", dir=EVIDENCE) as temporary:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temporary) / "index"))
        git("read-tree", "HEAD", env=env)
        if names:
            paths = Path(temporary) / "paths.bin"
            paths.write_bytes(b"\0".join(name.encode("utf-8") for name in names) + b"\0")
            git("--literal-pathspecs", "add", "--all", "--pathspec-file-nul", "--pathspec-from-file=" + str(paths), env=env)
        tree = git("write-tree", env=env)
    return tree, recipe_identity(value, tree)


def _blob_contents(object_ids):
    """Read exact Git blob bytes in one process; filters are never applied."""
    object_ids = sorted(set(object_ids))
    if not object_ids or any(not re.fullmatch(r"[0-9a-f]{40}", item) for item in object_ids):
        raise ValueError("Export verification requires exact Git blob identities")
    command = [LEGACY.GIT, "-c", f"safe.directory={ROOT.as_posix()}", "cat-file", "--batch"]
    result = subprocess.run(command, cwd=ROOT, input=("\n".join(object_ids) + "\n").encode("ascii"),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        raise RuntimeError("Cannot read frozen Git blob bytes: " + result.stderr.decode("utf-8", errors="replace"))
    cursor, values = 0, {}
    for expected in object_ids:
        end = result.stdout.find(b"\n", cursor)
        header = result.stdout[cursor:end].decode("ascii").split()
        if end < cursor or len(header) != 3 or header[:2] != [expected, "blob"] or not header[2].isdigit():
            raise ValueError("Malformed or non-blob Git export object")
        size = int(header[2])
        cursor = end + 1
        body = result.stdout[cursor:cursor + size]
        cursor += size
        if len(body) != size or result.stdout[cursor:cursor + 1] != b"\n":
            raise ValueError("Truncated frozen Git blob bytes")
        cursor += 1
        values[expected] = body
    if cursor != len(result.stdout):
        raise ValueError("Unexpected trailing Git export bytes")
    return values


def check_export(value=None, tree=None, recipe=None):
    """Check local raw bytes against prospective blobs without approvals.

    This checks source inputs, not a built image. It never normalizes files,
    stages the real index, requests a build or writes an approval snapshot.
    """
    if value is None:
        value = lock()
    if tree is None or recipe is None:
        identity(value)
        tree, recipe = prospective(value)
    delta = raw_delta(value["recipe_base_commit"], tree)
    inputs = {name: item["blob"] for name, item in delta.items() if item["status"] != "D"}
    shared = recipe.get("shared_inputs", {})
    for name, blob in shared.items():
        if name in inputs and inputs[name] != blob:
            raise ValueError("Candidate attempts to replace a locked shared export input: " + name)
        inputs[name] = blob
    bodies = _blob_contents(inputs.values())
    changed = []
    for name, blob in sorted(inputs.items()):
        path = ROOT / name
        if path.is_symlink() or not path.is_file() or path.read_bytes() != bodies[blob]:
            changed.append(name)
    if changed:
        raise ValueError("Working bytes differ from frozen Git export input (including CRLF/LF/filter differences); "
                         "no automatic conversion was made: " + ", ".join(changed[:20])
                         + (f"; {len(changed)} mismatched inputs total" if len(changed) > 20 else ""))
    return {"status": "PASS_LOCAL_EXPORT_BYTES_ONLY", "tree": tree,
            "recipe_sha256": recipe["recipe_sha256"], "candidate_paths": len(delta),
            "shared_paths": len(shared), "checked_input_paths": len(inputs),
            "scope": "Actual working bytes equal exact prospective/locked Git blobs; no approval/build/installation claim"}


def review_contract(contract, value, recipe):
    if contract.get("schema") != "h3-sr-omni032-upgrade/1":
        raise ValueError("Unknown shared review contract")
    for key, expected in (("repository", value["repository"]), ("branch", value["remote_branch"]),
                          ("functional_parent_image", value["parent_image"]), ("harbor_repository", value["harbor_repository"])):
        if contract.get(key) != expected:
            raise ValueError("Review contract differs: " + key)
    if contract.get("sole_online_submitter_thread") != "01a0eb06-bc59-7ee0-b2d8-b2467e740cd1":
        raise ValueError("Wrong designated online submitter")
    if type(contract.get("build_request_count")) is not int or contract["build_request_count"] != 0:
        raise ValueError("A build was already requested; no automatic retry")
    reviews = contract.get("reviews", {})
    if set(reviews) != {"image", "k8s", "independent_audit"} or any(
            not isinstance(item, dict) or item.get("status") != "PASS" for item in reviews.values()):
        raise ValueError("Image, K8S and independent reviews must all pass")
    if contract.get("freeze") != {"recipe_sha256": recipe["recipe_sha256"], "tree": recipe["tree"]}:
        raise ValueError("Shared reviews do not approve this exact frozen recipe")
    return contract


def seal():
    value = lock()
    identity(value)
    tree, recipe = prospective(value)
    check_export(value, tree, recipe)
    dump(ROOT / SEAL, recipe)
    print(json.dumps({"status": "SEALED_LOCAL_NOT_APPROVED", "recipe_sha256": recipe["recipe_sha256"],
                      "tree": recipe["tree"], "prospective_tree": tree}, indent=2))


def freeze(contract_path=CONTRACT):
    value = lock()
    identity(value)
    _, recipe = prospective(value)
    if read(ROOT / SEAL) != recipe:
        raise ValueError("Recipe changed; review and reseal before freezing")
    approved = review_contract(read(contract_path), value, recipe)
    dump(ROOT / APPROVAL, approved)
    print("Frozen the exact independently reviewed recipe; no online action")


def check_seal(committed=False, external=False):
    value = lock()
    if committed:
        tree = git("rev-parse", "HEAD^{tree}")
        recipe = recipe_identity(value, tree)
    else:
        identity(value)
        tree, recipe = prospective(value)
    if read(ROOT / SEAL) != recipe:
        raise ValueError("Recipe no longer matches its reviewed seal")
    check_export(value, tree, recipe)
    review_contract(read(ROOT / APPROVAL), value, recipe)
    if external and read(CONTRACT) != read(ROOT / APPROVAL):
        raise ValueError("Shared coordination contract differs from the frozen snapshot")
    return value, tree, recipe


def check():
    value, tree, recipe = check_seal(external=True)
    subprocess.run([sys.executable, "-B", "-X", "utf8", str(HERE / "check_local.py")], cwd=ROOT, check=True)
    print(json.dumps({"status": "PASS_LOCAL_RECIPE_ONLY", "parent_image": value["parent_image"],
                      "prospective_tree": tree, "recipe_sha256": recipe["recipe_sha256"],
                      "not_run": ["Linux image build", "Harbor publication", "CUDA", "Ref2VA media"]}, indent=2))
    return value, tree, recipe


def claim_request(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def submit(confirmed):
    journal = EVIDENCE / "one-build-journal.json"
    if not confirmed or journal.exists():
        raise ValueError("Explicit one-shot submission required; existing journal forbids automatic retry")
    value, tree, recipe = check()
    remote = LEGACY.remote_head(value["remote_branch"])
    if remote != value["working_tree_base_commit"]:
        raise ValueError("Remote changed from the reviewed baseline; stop before push")
    if prospective(value)[0] != tree:
        raise ValueError("Recipe changed during review")
    if git("rev-parse", "HEAD^{tree}") != tree:
        LEGACY.stage_validated_tree(tree)
        git("commit", "-s", "-m", "build(omni): upgrade reviewed H3 SR to 0.32.0 " + MARKER)
    if not valid_request_message(git("log", "-1", "--format=%B")):
        raise ValueError("Existing commit lacks the unique marker; no empty request commit is created")
    commit = git("rev-parse", "HEAD")
    if git("rev-parse", "HEAD^{tree}") != tree or git("status", "--porcelain"):
        raise ValueError("Checkout changed while committing")
    check_seal(committed=True, external=True)
    record = {"schema": "omni032-h3-sr-one-build/1", "status": "REQUESTED", "commit": commit,
              "tree": tree, "recipe_sha256": recipe["recipe_sha256"], "request_count": 1,
              "created_at": datetime.now(timezone.utc).isoformat()}
    claim_request(journal, record)
    try:
        git("push", "origin", f"HEAD:refs/heads/{value['remote_branch']}", network=True)
    except BaseException:
        dump(journal, {**record, "status": "UNKNOWN", "reason": "Push failed or response interrupted; preserve original request and investigate, never automatically push again"})
        raise
    dump(journal, {**record, "push_completed": True})
    print("One build request submitted for " + commit + "; preserve its Actions run. No rerun/dispatch/empty-commit command exists.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("seal")
    commands.add_parser("freeze")
    commands.add_parser("check")
    commands.add_parser("verify-export")
    verify = commands.add_parser("check-seal")
    verify.add_argument("--committed", action="store_true")
    publish = commands.add_parser("submit")
    publish.add_argument("--confirm-build-and-publish", action="store_true")
    args = parser.parse_args()
    if args.command == "seal": seal()
    elif args.command == "freeze": freeze()
    elif args.command == "check": check()
    elif args.command == "verify-export": print(json.dumps(check_export(), indent=2))
    elif args.command == "check-seal": check_seal(args.committed)
    else: submit(args.confirm_build_and_publish)


if __name__ == "__main__":
    main()
