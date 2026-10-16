#!/usr/bin/env python3
"""Local preparation, ordinary branch submission, and immutable CI evidence import."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SEAL_PATH = "delivery/h3-experimental/input-seal.json"
EVIDENCE = ROOT / ".h3-experimental-evidence" / "local"
GIT = shutil.which("git") or r"D:\workingtools\Git\cmd\git.exe"
PROXY = "http://127.0.0.1:17890"
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
EXPERIMENT_BRANCH = "Dingorouter-H3-vllm-omni-jirx"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def run_id():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def git(*args, env=None, network=False, binary=False):
    command = [GIT, "-c", f"safe.directory={ROOT.as_posix()}"]
    if network:
        command += ["-c", f"http.proxy={PROXY}"]
    result = subprocess.run(command + list(args), cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        raise RuntimeError(f"Git {args[0]} failed ({result.returncode}):\n" + (result.stdout + result.stderr).decode("utf-8", errors="replace"))
    return result.stdout if binary else result.stdout.decode("utf-8", errors="strict").strip()


def git_environment(index):
    env = dict(os.environ)
    env["GIT_INDEX_FILE"] = str(index)
    return env


def prospective_tree(lock):
    """Use a temporary index; never stage unrelated work in the user's index."""
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="index-", dir=EVIDENCE) as temp:
        env = git_environment(Path(temp) / "index")
        git("read-tree", lock["main_commit"], env=env)
        git("add", "--all", "--", ".", env=env)
        tree = git("write-tree", env=env)
        raw = git("diff", "--cached", "--raw", "--no-renames", "--no-abbrev", "-z", lock["main_commit"], env=env, binary=True)
        fields = raw.decode("utf-8").split("\0")
        delta = {}
        for i in range(0, len(fields) - 1, 2):
            metadata, name = fields[i], fields[i + 1]
            if name == SEAL_PATH:
                continue
            parts = metadata.split()
            if len(parts) != 5 or parts[4] not in {"A", "M", "D", "T"}:
                raise ValueError("Unexpected tree delta; use --no-renames: " + metadata)
            delta[name] = {"mode": parts[1], "blob": parts[3], "status": parts[4]}
        return tree, delta


def identity(lock):
    if lock["remote_branch"] != EXPERIMENT_BRANCH:
        raise ValueError("Use the designated MiniMax-H3 experimental branch")
    if Path(git("rev-parse", "--show-toplevel")).resolve() != ROOT:
        raise ValueError("Unexpected repository root")
    origin = git("remote", "get-url", "origin").rstrip("/")
    if origin not in {lock["repository"], lock["repository"].removesuffix(".git")}:
        raise ValueError("Origin is not the reviewed githubgxll/dynamo repository")
    if git("branch", "--show-current") != lock["remote_branch"]:
        raise ValueError("Use the designated experimental branch in this prepared checkout")
    git("cat-file", "-e", lock["main_commit"] + "^{commit}")
    git("merge-base", "--is-ancestor", lock["remote_baseline"], "HEAD")


def seal():
    lock = read_json(HERE / "source-lock.json")
    identity(lock)
    _, delta = prospective_tree(lock)
    dump(ROOT / SEAL_PATH, {"schema": "h3-main-overlay-seal/1", "main_commit": lock["main_commit"],
                            "remote_baseline": lock["remote_baseline"], "delta": delta,
                            "scope": "Exact Git blobs and deletions relative to the reviewed main snapshot; seal excludes itself"})
    print(f"Sealed {len(delta)} H3 overlay paths against main {lock['main_commit']}")


def check():
    lock = read_json(HERE / "source-lock.json")
    identity(lock)
    sealed = read_json(ROOT / SEAL_PATH)
    tree, delta = prospective_tree(lock)
    if sealed["main_commit"] != lock["main_commit"] or sealed["remote_baseline"] != lock["remote_baseline"] or sealed["delta"] != delta:
        expected = sealed["delta"]
        changed = sorted(p for p in set(expected) | set(delta) if expected.get(p) != delta.get(p))
        raise ValueError("Prepared inputs changed; review and reseal before submission: " + ", ".join(changed[:30]))
    # Raw source/asset/patch hashes are also used inside Linux images. Prove the
    # local tested bytes survive Git attributes, rather than testing CRLF while
    # publishing normalized LF bytes under a different identity.
    byte_paths = [p for p, item in delta.items() if item["status"] != "D"] + [SEAL_PATH]
    mismatches = [p for p in byte_paths if (ROOT / p).read_bytes() != git("show", tree + ":" + p, binary=True)]
    if mismatches:
        raise ValueError("Local bytes differ from prospective Git checkout; normalize unsealed text or preserve locked bytes in attributes: " + ", ".join(mismatches))
    result = subprocess.run([sys.executable, "-B", "-X", "utf8", str(HERE / "check_local.py")], cwd=ROOT, check=False)
    if result.returncode:
        raise RuntimeError("Local implementation checks failed")
    # Only the new overlay must meet whitespace requirements; inherited main is verbatim.
    paths = [p for p, value in delta.items() if value["status"] != "D"]
    git("diff", "--check", lock["main_commit"], tree, "--", *paths)
    report = {"status": "PASS_LOCAL_RECIPE_ONLY", "main_commit": lock["main_commit"],
              "head": git("rev-parse", "HEAD"), "prospective_tree": tree, "sealed_overlay_paths": len(delta),
              "not_run": ["Linux Docker build", "Harbor publish/pull", "H100", "Base mount identity", "generated media E2E"]}
    dump(EVIDENCE / run_id() / "preparation.json", report)
    print(json.dumps(report, indent=2))
    return lock, tree


def remote_head(branch):
    """An empty successful lookup means absent; network/auth failures still fail."""
    ref = "refs/heads/" + branch
    output = git("ls-remote", "--refs", "origin", ref, network=True)
    if not output:
        return None
    rows = [line.split() for line in output.splitlines()]
    if len(rows) != 1 or len(rows[0]) != 2 or rows[0][1] != ref or not HEX40.fullmatch(rows[0][0]):
        raise ValueError("Unexpected remote branch identity response")
    return rows[0][0]


def prepare_remote(lock):
    branch = lock["remote_branch"]
    if branch != EXPERIMENT_BRANCH:
        raise ValueError("Only the designated MiniMax-H3 branch may be submitted")
    git("fetch", "--no-tags", "origin", "+refs/heads/main:refs/remotes/origin/main", network=True)
    if git("rev-parse", "refs/remotes/origin/main") != lock["main_commit"]:
        raise ValueError("Upstream main has changed since review. Review its new dependencies before rebuilding.")
    remote = remote_head(branch)
    if remote is None:
        if lock.get("allow_remote_branch_creation") is not True:
            raise ValueError("Target branch is absent and first creation is not enabled")
        print(f"First submission will create the user-designated branch {branch}.")
        return None
    git("fetch", "--no-tags", "origin", f"refs/heads/{branch}:refs/remotes/origin/{branch}", network=True)
    if git("rev-parse", f"refs/remotes/origin/{branch}") != remote:
        raise ValueError("Remote branch moved during inspection; inspect and retry")
    # Existing remote work must be included, including on later experiment upgrades.
    # This command fails on divergence; the eventual normal push also checks races.
    git("merge-base", "--is-ancestor", remote, "HEAD")
    return remote


def tree_entries(ref):
    entries = {}
    for record in git("ls-tree", "-r", "-z", ref, binary=True).split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, kind, oid = metadata.decode("ascii").split()
        entries[raw_path.decode("utf-8")] = (mode, oid)
    return entries


def index_entries():
    entries = {}
    for record in git("ls-files", "--stage", "-z", binary=True).split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, oid, stage = metadata.decode("ascii").split()
        if stage != "0":
            raise ValueError("Resolve index merge conflicts before submission")
        entries[raw_path.decode("utf-8")] = (mode, oid)
    return entries


def stage_validated_tree(expected_tree):
    """Install only the checked tree in the index; never change working files.

    A main-tracked path can also match .gitignore. Re-adding those paths from
    the old branch index both rejects ignored paths and loses POSIX file modes
    on Windows. The validated tree already has their exact blobs and modes.
    """
    head = tree_entries("HEAD")
    expected = tree_entries(expected_tree)
    staged = index_entries()
    hidden = []
    for name in set(head) | set(staged):
        current = staged.get(name)
        if current == head.get(name):
            continue
        # Tolerate a partial earlier staging attempt and Windows mode changes
        # only when its bytes equal HEAD or the validated destination. Never
        # discard staged-only edits that disappeared from the working tree.
        current_oid = current[1] if current else None
        allowed = {entry[1] if entry else None for entry in (head.get(name), expected.get(name))}
        if current_oid not in allowed:
            hidden.append(name)
    if hidden:
        raise ValueError("Staged-only content differs from HEAD and the reviewed tree; preserve/reconcile it before submission: " + ", ".join(sorted(hidden)[:30]))
    index_path = Path(git("rev-parse", "--git-path", "index"))
    if not index_path.is_absolute():
        index_path = ROOT / index_path
    original = index_path.read_bytes()
    backup = EVIDENCE / "index-backups" / run_id()
    backup.mkdir(parents=True)
    (backup / "index").write_bytes(original)
    dump(backup / "record.json", {"head": git("rev-parse", "HEAD"), "expected_tree": expected_tree,
                                  "index_sha256": hashlib.sha256(original).hexdigest(),
                                  "scope": "Index backup only; working files are never reset"})
    # No -u: read-tree writes the index only. It does not reset/delete local files,
    # resolve ignored directories again, or alter the branch/commit history.
    git("read-tree", expected_tree)
    if git("write-tree") != expected_tree:
        raise ValueError(f"Index does not match the checked tree; stop and inspect. Backup: {backup}")
    print(f"Staged the exact checked tree; previous index saved at {backup}")
    return backup


def submit(confirmed):
    if not confirmed:
        raise ValueError("Use the explicit 02-submit-build.cmd entry point")
    lock, expected_tree = check()
    branch = lock["remote_branch"]
    remote = prepare_remote(lock)
    head = git("rev-parse", "HEAD")
    if remote == head and git("rev-parse", "HEAD^{tree}") == expected_tree:
        print("This exact prepared tree is already submitted. Inspect the existing Actions run.")
        return
    changed = git("diff", "--name-only", "--no-renames", "-z", "HEAD", expected_tree, binary=True)
    if changed:
        # Validate DCO identity before mutating the index. Values are not logged.
        git("var", "GIT_AUTHOR_IDENT")
        refreshed_tree, _ = prospective_tree(lock)
        if refreshed_tree != expected_tree:
            raise ValueError("Working files changed after validation; rerun checks before submission")
        stage_validated_tree(expected_tree)
        git("commit", "-s", "-m", "feat(h3): migrate experimental distilled image to main and Omni 0.31")
    if git("rev-parse", "HEAD^{tree}") != expected_tree or git("status", "--porcelain"):
        raise ValueError("Files changed during submission; inspect and rerun checks")
    git("push", "--set-upstream", "origin", f"HEAD:refs/heads/{branch}", network=True)
    print(f"Submitted to {branch}. Watch https://github.com/githubgxll/dynamo/actions")


def safe_relative(name):
    if "\\" in name or ":" in name:
        raise ValueError("Unsafe evidence path")
    path = PurePosixPath(name)
    if path.is_absolute() or any(p in {"..", "."} for p in path.parts) or not path.parts:
        raise ValueError("Unsafe evidence path")
    return path


def verify_publication(directory, lock):
    record = read_json(directory / "image.json")
    if record.get("schema") != "h3-experimental-publication/1" or record.get("main_commit") != lock["main_commit"]:
        raise ValueError("Wrong publication/main source")
    if not HEX40.fullmatch(record.get("source_commit", "")) or not DIGEST.fullmatch(record.get("digest", "")):
        raise ValueError("Invalid source/image identity")
    if record.get("deployment_image") != lock["harbor_repository"] + "@" + record["digest"]:
        raise ValueError("Wrong Harbor destination")
    expected_tag = lock["harbor_repository"] + ":omni031rc1-distilled-ref8shift12-" + record["source_commit"][:12]
    if record.get("image") != expected_tag:
        raise ValueError("Publication is not the submitted final distilled image tag")
    parent = record.get("framework_image", "")
    if not parent.startswith(lock["harbor_repository"] + "@") or not DIGEST.fullmatch(parent.split("@")[-1]):
        raise ValueError("Invalid framework parent")
    if parent == record["deployment_image"]:
        raise ValueError("Framework parent cannot be the final adapter image")
    if not record.get("evidence_sha256"):
        raise ValueError("Missing publication evidence hashes")
    required = {
        "experimental/assets-report.json": "PASS_ASSETS_ONLY",
        "experimental/profiles-check.json": "PASS_INSTALLED_PROFILES_CPU_ONLY",
        "experimental/environment-check.json": "PASS_EXPERIMENTAL_ENVIRONMENT_CPU_ONLY",
        "experimental/media-smoke.json": "PASS_REAL_CPU_MEDIA_SMOKE",
        "framework/ref8-patch.json": "PASS_PATCH_APPLIED_CPU_ONLY",
        "framework/ref8-contract.json": "PASS_INSTALLED_CONTRACT_CPU_ONLY",
        "experimental/dependency-license-inventory.json": "METADATA_INVENTORY_ONLY_NOT_LEGAL_CLEARANCE",
    }
    required_files = set(required) | {"experimental/image-inputs.json", "inputs/source-lock.json",
                                    "inputs/assets.lock.json", "inputs/profiles.json", "inputs/Dockerfile",
                                    "legal/osrb-deps.csv", "legal/osrb.cdx.json", "experimental/dpkg-packages.txt"}
    if not required_files.issubset(record["evidence_sha256"]):
        raise ValueError("Missing required final evidence")
    for name, digest in record["evidence_sha256"].items():
        file = directory.joinpath(*safe_relative(name).parts)
        if file.is_symlink() or not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            raise ValueError("Evidence hash mismatch: " + name)
    for name, status in required.items():
        data = read_json(directory / name)
        if data.get("status") != status or data.get("errors"):
            raise ValueError("Final evidence gate failed: " + name)
    inputs = read_json(directory / "experimental/image-inputs.json")
    if inputs.get("source_commit") != record["source_commit"] or inputs.get("base_image") != parent or inputs.get("versions") != {"vllm": "0.31.0", "vllm-omni": "0.31.0rc1"}:
        raise ValueError("Final image inputs disagree with publication record")
    if read_json(directory / "inputs/source-lock.json") != lock:
        raise ValueError("Published source lock differs from this prepared checkout")
    if inputs.get("assets_lock_sha256") != hashlib.sha256((directory / "inputs/assets.lock.json").read_bytes()).hexdigest():
        raise ValueError("Published asset lock does not match installed evidence")
    if (directory / "deployment-image.txt").read_text(encoding="utf-8").strip() != record["deployment_image"]:
        raise ValueError("Digest text disagrees with publication record")
    return record


def verify_submitted_inputs(directory, record):
    """Bind archive inputs to actual committed bytes, not just their own hashes."""
    files = {"inputs/source-lock.json": "delivery/h3-experimental/source-lock.json",
             "inputs/assets.lock.json": "container/h3-experimental/assets/assets.lock.json",
             "inputs/profiles.json": "container/h3-experimental/profiles/profiles.json",
             "inputs/Dockerfile": "container/h3-experimental/Dockerfile"}
    for archived, committed in files.items():
        expected = git("show", record["source_commit"] + ":" + committed, binary=True)
        if (directory / archived).read_bytes() != expected:
            raise ValueError("Artifact inputs differ from submitted Git bytes: " + archived)


def import_evidence(archive):
    archive = Path(archive).resolve(strict=True)
    lock = read_json(HERE / "source-lock.json")
    destination = EVIDENCE / ("import-" + run_id())
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            relative = safe_relative(member.filename.rstrip("/"))
            if ((member.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError("Symlinks are forbidden in evidence ZIP")
            target = destination.joinpath(*relative.parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise ValueError("Duplicate evidence ZIP member")
                with bundle.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    candidates = list(destination.rglob("image.json"))
    if len(candidates) != 1:
        raise ValueError("Expected exactly one completed final image publication record")
    record = verify_publication(candidates[0].parent, lock)
    # Bind downloaded evidence to a local submitted commit, never a framework/artifact ZIP digest.
    if record["source_commit"] != git("rev-parse", "HEAD"):
        raise ValueError("Artifact source is not this checkout HEAD; select this submission's artifact")
    verify_submitted_inputs(candidates[0].parent, record)
    print(json.dumps({"status": "PASS_IMPORTED_PUBLICATION_RECORD_ONLY", "directory": str(destination),
                      "image": record["deployment_image"], "not_verified": ["independent registry pull", "GPU", "media E2E"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["seal", "check", "submit", "import-evidence"])
    parser.add_argument("archive", nargs="?")
    parser.add_argument("--confirm-build-and-publish", action="store_true")
    args = parser.parse_args()
    if args.command == "seal":
        seal()
    elif args.command == "check":
        check()
    elif args.command == "submit":
        submit(args.confirm_build_and_publish)
    elif args.archive:
        import_evidence(args.archive)
    else:
        parser.error("import-evidence requires the Actions artifact ZIP path")


if __name__ == "__main__":
    main()
