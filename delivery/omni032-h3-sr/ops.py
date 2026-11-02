#!/usr/bin/env python3
"""Submit reviewed work sequentially after the prior outcome is known."""
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
import urllib.error
import urllib.request

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
                 ".github/scripts/test_omni032_retry_ci.py", ".github/scripts/test_omni032_submission_ci.py",
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
    validate_anchors(value)


def validate_anchors(value):
    recipe, working = value["recipe_base_commit"], value["working_tree_base_commit"]
    for commit in dict.fromkeys((recipe, working)):
        if not re.fullmatch(r"[0-9a-f]{40}", commit) or git("rev-parse", "--verify", commit + "^{commit}") != commit:
            raise ValueError("Recipe and working baselines must be full existing commits")
    git("merge-base", "--is-ancestor", recipe, working)
    git("merge-base", "--is-ancestor", working, "HEAD")


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
    validate_anchors(value)
    raw_delta(value["working_tree_base_commit"], full_tree)
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
    count = contract.get("build_request_count")
    if type(count) is not int or count < 0:
        raise ValueError("Submission history count must be a nonnegative integer")
    if count:
        validate_previous_result(contract.get("previous_build_result"), value,
                                 expected_commit=contract.get("submitted_commit"))
    reviews = contract.get("reviews", {})
    if set(reviews) != {"image", "k8s", "independent_audit"} or any(
            not isinstance(item, dict) or item.get("status") != "PASS" for item in reviews.values()):
        raise ValueError("Image, K8S and independent reviews must all pass")
    if contract.get("freeze") != {"recipe_sha256": recipe["recipe_sha256"], "tree": recipe["tree"]}:
        raise ValueError("Shared reviews do not approve this exact frozen recipe")
    return contract


def validate_previous_result(result, value, *, expected_commit):
    if not isinstance(result, dict):
        raise ValueError("The previous build result is unknown; inspect it before submitting again")
    if (result.get("repository") != value["repository"] or result.get("branch") != value["remote_branch"]
            or result.get("workflow_path") != ".github/workflows/omni032-h3-sr.yml"
            or result.get("commit") != expected_commit or not re.fullmatch(r"[0-9a-f]{40}", str(expected_commit))
            or type(result.get("run_id")) is not int or result["run_id"] <= 0
            or type(result.get("run_attempt")) is not int or result["run_attempt"] <= 0):
        raise ValueError("Prior result does not identify the previous repository/commit/run/attempt")
    if result.get("status") != "completed" or result.get("conclusion") not in {"failure", "skipped", "cancelled", "timed_out", "startup_failure"}:
        raise ValueError("Prior request is active, successful or unknown; do not submit another build")
    if result.get("publication_status") != "not_started":
        raise ValueError("Prior publication exists or is unknown; verify/reuse the original image")
    return result


def github_json(endpoint):
    """Read only this repository using existing auth; never print/store tokens."""
    if not endpoint.startswith("/") or ".." in endpoint or endpoint.startswith("//"):
        raise ValueError("Unsafe GitHub read endpoint")
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never")
        result = subprocess.run([LEGACY.GIT, "-C", str(ROOT), "-c", "safe.directory=" + ROOT.as_posix(),
            "-c", "credential.interactive=never", "credential", "fill"],
            input=b"url=https://github.com/githubgxll/dynamo.git\n\n", capture_output=True, env=env, timeout=30)
        if result.returncode:
            raise RuntimeError("Existing repository authentication helper failed; prior run status remains unknown")
        token = dict(line.split("=", 1) for line in result.stdout.decode().splitlines() if "=" in line).get("password")
    if not token:
        raise RuntimeError("No existing repository API authentication; prior run status remains unknown")
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None
    handlers = [NoRedirect()]
    if os.name == "nt":
        handlers.append(urllib.request.ProxyHandler({"https": LEGACY.PROXY}))
    opener = urllib.request.build_opener(*handlers)
    request = urllib.request.Request("https://api.github.com/repos/githubgxll/dynamo" + endpoint,
        headers={"Authorization": "Bearer " + token, "User-Agent": "Omni032-prior-result-read-only",
                 "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"})
    try:
        with opener.open(request, timeout=30) as response:
            return json.load(response)
    except (OSError, urllib.error.HTTPError, ValueError) as error:
        raise RuntimeError("Cannot confirm original run outcome: " + type(error).__name__
                           + "; no additional request was sent") from None


def observe_run(run_id, getter=None):
    getter = getter or github_json
    if type(run_id) is not int or run_id <= 0:
        raise ValueError("A positive original run ID is required")
    run = getter(f"/actions/runs/{run_id}")
    attempt = run.get("run_attempt")
    if run.get("id") != run_id or type(attempt) is not int or attempt <= 0:
        raise ValueError("Original run identity is missing or malformed")
    history = []
    for number in range(1, attempt + 1):
        jobs = getter(f"/actions/runs/{run_id}/attempts/{number}/jobs?per_page=100")
        entries = jobs.get("jobs")
        if not isinstance(entries, list) or jobs.get("total_count") != len(entries):
            raise ValueError("Incomplete original-attempt job inventory")
        if any(j.get("run_id") != run_id or j.get("run_attempt") != number
               or j.get("head_sha") != run.get("head_sha") for j in entries):
            raise ValueError("Original jobs belong to a different run, attempt or commit")
        publication = "unknown"
        if entries and all(j.get("status") == "completed" for j in entries):
            publish = [step for j in entries for step in (j.get("steps") or [])
                       if step.get("name") == "Publish the validated image once and verify its pull digest"]
            setup_only = all(j.get("conclusion") == "skipped" or
                             (len(j.get("steps") or []) == 1 and j["steps"][0].get("name") == "Set up job"
                              and j["steps"][0].get("conclusion") != "success") for j in entries)
            if publish and all(step.get("conclusion") == "skipped" for step in publish):
                publication = "not_started"
            elif not publish and setup_only:
                publication = "not_started"
            elif any(step.get("conclusion") == "success" for step in publish):
                publication = "published"
        history.append({"run_attempt": number, "publication_status": publication, "jobs": entries})
    latest = getter(f"/actions/runs/{run_id}")
    if any(latest.get(key) != run.get(key) for key in
           ("id", "run_attempt", "status", "conclusion", "head_sha", "head_branch", "path", "updated_at")):
        raise ValueError("Original run changed during observation; inspect its latest result first")
    states = {item["publication_status"] for item in history}
    publication = "published" if "published" in states else "not_started" if states == {"not_started"} else "unknown"
    return {"schema": "omni032-prior-result/1", "repository": "https://github.com/githubgxll/dynamo.git",
            "branch": run.get("head_branch"), "workflow_path": run.get("path"), "commit": run.get("head_sha"),
            "run_id": run_id, "run_attempt": attempt, "status": run.get("status"),
            "conclusion": run.get("conclusion"), "publication_status": publication,
            "observed_at": datetime.now(timezone.utc).isoformat(), "jobs": history[-1]["jobs"],
            "attempt_history": history,
            "run_url": run.get("html_url")}


def observe_previous(run_id):
    journal = read(EVIDENCE / "one-build-journal.json")
    result = observe_run(run_id)
    validate_previous_result(result, lock(), expected_commit=journal["commit"])
    bind_journal_result(journal, result, allow_discovery=True)
    output = EVIDENCE / "observations" / f"{run_id}-{result['run_attempt']}.json"
    dump(output, result)
    dump(EVIDENCE / "one-build-journal.json", {**journal, "status": "CONFIRMED_TERMINAL_UNPUBLISHED",
         "run_id": run_id, "run_attempt": result["run_attempt"], "confirmed_result": result})
    print(json.dumps({"status": "CONFIRMED_TERMINAL_UNPUBLISHED", "evidence": str(output),
                      "run_id": run_id, "run_attempt": result["run_attempt"]}, indent=2))


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


def bind_journal_result(journal, result, *, allow_discovery=False):
    known_run = journal.get("run_id")
    confirmed = journal.get("confirmed_result") or {}
    nested_run = confirmed.get("run_id")
    if known_run is not None and nested_run is not None and known_run != nested_run:
        raise ValueError("Original journal and confirmed-result run IDs conflict")
    if known_run is None:
        known_run = nested_run
    if known_run is None:
        if not allow_discovery:
            raise ValueError("Original run is not yet bound to the journal; observe and confirm it first")
        # A lost push response must not be resolved by selecting a convenient
        # skipped run while another matching request is active or unknown.
        matches = github_json("/actions/workflows/omni032-h3-sr.yml/runs?event=push&head_sha="
                              + result["commit"] + "&per_page=100")
        runs = matches.get("workflow_runs")
        if (not isinstance(runs, list) or matches.get("total_count") != 1
                or len(runs) != 1 or runs[0].get("id") != result["run_id"]):
            raise ValueError("Original push has ambiguous matching runs; resolve the original request before submitting")
    elif type(known_run) is not int or known_run != result["run_id"]:
        raise ValueError("Observed run does not match the original journal run ID")
    attempts = (journal.get("run_attempt", 0), confirmed.get("run_attempt", 0))
    if any(type(number) is not int or number < 0 for number in attempts) or result["run_attempt"] < max(attempts):
        raise ValueError("Observed attempt cannot move the original journal backwards")


def prior_request(value, contract, journal, getter=None):
    """Require the latest terminal, definitely unpublished result, not a count cap."""
    count = contract["build_request_count"]
    if not count:
        if journal is not None:
            raise ValueError("An existing request is missing from shared history; inspect it first")
        return None
    if (journal is None or journal.get("request_count") != count
            or journal.get("commit") != contract.get("submitted_commit")):
        raise ValueError("Shared history and local submission journal disagree")
    reviewed = validate_previous_result(contract.get("previous_build_result"), value,
                                        expected_commit=journal["commit"])
    bind_journal_result(journal, reviewed)
    latest = observe_run(reviewed["run_id"], getter)
    validate_previous_result(latest, value, expected_commit=journal["commit"])
    if latest["run_attempt"] != reviewed["run_attempt"]:
        raise ValueError("A newer attempt exists; update the observed result before another submission")
    return latest


def submit(confirmed):
    journal = EVIDENCE / "one-build-journal.json"
    if not confirmed:
        raise ValueError("Explicit build and publication submission is required")
    value, tree, recipe = check()
    contract = read(CONTRACT)
    previous = read(journal) if journal.exists() else None
    prior = prior_request(value, contract, previous)
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
    if previous is not None and commit == previous["commit"]:
        raise ValueError("This exact commit was already submitted; inspect that run instead of pushing it again")
    if git("rev-parse", "HEAD^{tree}") != tree or git("status", "--porcelain"):
        raise ValueError("Checkout changed while committing")
    check_seal(committed=True, external=True)
    # Query again immediately before claiming/pushing; a manual rerun could have
    # started while local checks or commit creation were running.
    if prior is not None:
        prior = prior_request(value, contract, previous)
    sequence = contract["build_request_count"] + 1
    record = {"schema": "omni032-h3-sr-sequential-build/2", "status": "REQUESTED", "commit": commit,
              "tree": tree, "recipe_sha256": recipe["recipe_sha256"], "request_count": sequence,
              "count_scope": "Controlled submit invocations, not GitHub run_attempt or Action download attempts",
              "previous_result": prior,
              "created_at": datetime.now(timezone.utc).isoformat()}
    if previous is not None:
        archive = EVIDENCE / "requests" / f"request-{sequence - 1:04d}.json"
        if not archive.exists():
            claim_request(archive, previous)
    # One exclusive file per local sequence prevents competing submitters from
    # pushing different commits for the same next request. Preserve it on error.
    claim = EVIDENCE / "requests" / f"request-{sequence:04d}.json"
    claim_request(claim, record)
    dump(journal, record)
    try:
        git("push", "origin", f"{commit}:refs/heads/{value['remote_branch']}", network=True)
    except BaseException:
        failed = {**record, "status": "UNKNOWN", "reason": "Push failed or response interrupted; inspect the original request before another submission"}
        dump(journal, failed)
        dump(claim, failed)
        raise
    dump(journal, {**record, "push_completed": True})
    dump(claim, {**record, "push_completed": True})
    print("Build request " + str(sequence) + " submitted for " + commit + "; follow its result before any additional request.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("seal")
    commands.add_parser("freeze")
    commands.add_parser("check")
    commands.add_parser("verify-export")
    observe = commands.add_parser("observe-previous")
    observe.add_argument("--run-id", required=True, type=int)
    verify = commands.add_parser("check-seal")
    verify.add_argument("--committed", action="store_true")
    publish = commands.add_parser("submit")
    publish.add_argument("--confirm-build-and-publish", action="store_true")
    args = parser.parse_args()
    if args.command == "seal": seal()
    elif args.command == "freeze": freeze()
    elif args.command == "check": check()
    elif args.command == "verify-export": print(json.dumps(check_export(), indent=2))
    elif args.command == "observe-previous": observe_previous(args.run_id)
    elif args.command == "check-seal": check_seal(args.committed)
    else: submit(args.confirm_build_and_publish)


if __name__ == "__main__":
    main()
