#!/usr/bin/env python3
"""Seal, review and submit one source-bound production image sequentially."""
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
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EVIDENCE = ROOT / ".h3-experimental-evidence/h3-sr-production/local"
BRANCH = "Dingorouter-H3-vllm-omni-jirx"
REPOSITORY = "https://github.com/githubgxll/dynamo.git"
HARBOR = "registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm"
PARENT = HARBOR + "@sha256:c88bef388ed25359c78c48ae841736049100d7447c572678992699ae6291e25d"
PARENT_ID = "sha256:cd87bc74b19402c536fc775ea2ebbc15f31aad0013756e8d8c448afe1c895a6b"
BASE = "b42388a1a65067ed92dfe64f9364c4b044a8d666"
MARKER = "[h3-sr-production]"
WORKFLOW = "h3-sr-production.yml"
PUBLISH_STEP = "Publish or reuse the validated exact image and verify its digest pull"
AUXILIARY = {"delivery/h3-sr-production/input-seal.json", "delivery/h3-sr-production/freeze-approval.json"}
PREFIXES = ("container/h3-sr-production/", "delivery/h3-sr-production/", "dingo/vllm/omni/", "dingo/video_gateway/")
FILES = {".github/scripts/h3_sr_production_ci.py", ".github/scripts/test_h3_sr_production_ci.py",
         ".github/workflows/h3-sr-production.yml", ".github/workflows/omni032-h3-sr.yml",
         ".github/workflows/dingo-router-ci.yml", ".dockerignore", ".gitattributes",
         "dingo/common/video_timing_schemas.py", "dingo/common/video_result_file.py"}
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


def git(*args, binary=False, env=None, network=False):
    command = [GIT, "-C", str(ROOT), "-c", "safe.directory=" + ROOT.as_posix(),
               "-c", "filter.lfs.process=", "-c", "filter.lfs.required=false"]
    if network and os.name == "nt":
        command += ["-c", "http.proxy=http://127.0.0.1:17890"]
    result = subprocess.run([*command, *args], capture_output=True, env=env, timeout=120)
    if result.returncode:
        raise RuntimeError("Git " + args[0] + " failed:\n" + (result.stdout + result.stderr).decode("utf-8", errors="replace"))
    return result.stdout if binary else result.stdout.decode("utf-8").strip()


def validate_lock(value):
    for key, expected in {"repository": REPOSITORY, "remote_branch": BRANCH, "harbor_repository": HARBOR,
                          "parent_image": PARENT, "parent_image_id": PARENT_ID, "platform": "linux/amd64",
                          "recipe_base_commit": BASE, "tag_prefix": "h3-sr-production"}.items():
        if value.get(key) != expected:
            raise ValueError("Production source lock differs: " + key)
    if not re.fullmatch(r"[0-9a-f]{40}", value.get("working_tree_base_commit", "")):
        raise ValueError("A complete working-tree anchor is required")
    if value.get("versions", {}).get("vllm-omni") != "0.32.0" or value.get("versions", {}).get("torch") != "2.13.0+cu130":
        raise ValueError("This task inherits the verified native runtime; no dependency upgrade")


def lock():
    value = read(HERE / "source-lock.json")
    validate_lock(value)
    return value


def permitted(name):
    return name in FILES or name.startswith(PREFIXES)


def check_scope(names):
    bad = sorted(name for name in names if not permitted(name))
    if bad:
        raise ValueError("Unreviewed files outside this production change: " + ", ".join(bad))


def valid_request_message(message):
    markers = re.findall(r"\[(?:omni032-h3-sr|omni-gateway-lite|h3-experimental|seedvr2-[a-z0-9-]+|h3-hsdp-offload|sr-native|h3-sr-production)\]", message)
    return (markers == [MARKER] and not re.search(r"\[(?:skip ci|ci skip|no ci|skip actions|actions skip)\]", message, re.I)
            and not re.search(r"(?mi)^skip-checks:\s*true\s*$", message))


def remote_head():
    lines = git("ls-remote", "origin", "refs/heads/" + BRANCH, network=True).splitlines()
    if len(lines) != 1 or not re.fullmatch(r"[0-9a-f]{40}\trefs/heads/" + re.escape(BRANCH), lines[0]):
        raise ValueError("The exact designated remote branch must already exist")
    return lines[0].split("\t")[0]


def identity(value):
    origin = git("remote", "get-url", "origin").rstrip("/").removesuffix(".git")
    if origin != REPOSITORY.removesuffix(".git"):
        raise ValueError("Unexpected origin repository")
    # Detached worktrees are supported; only the explicit remote ref is writable.
    git("merge-base", "--is-ancestor", value["recipe_base_commit"], "HEAD")


def temporary_file(suffix):
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="production-", suffix=suffix, dir=EVIDENCE)
    os.close(fd)
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
    names = list(filter(None, git("diff", "--name-only", "--no-renames", "-z", value["recipe_base_commit"], reviewed,
                                 binary=True).decode("utf-8").split("\0")))
    check_scope(names)
    if not names:
        raise ValueError("Empty or approval-only recipes cannot request another image")
    files = {}
    for name in sorted(names):
        entry = git("ls-tree", "-z", reviewed, "--", name, binary=True)
        if not entry:
            files[name] = None
            continue
        metadata, actual = entry.rstrip(b"\0").split(b"\t", 1)
        if metadata.split()[0] not in (b"100644", b"100755") or actual.decode() != name:
            raise ValueError("Symlinks/submodules/non-file recipe inputs are not allowed")
        files[name] = hashlib.sha256(git("show", reviewed + ":" + name, binary=True)).hexdigest()
    payload = {"base_commit": value["recipe_base_commit"], "parent_image": value["parent_image"], "files": files}
    return {"schema": "h3-sr-production-input-seal/1", **payload,
            "recipe_sha256": hash_value(payload), "review_tree": reviewed}


def prospective(value):
    identity(value)
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
            git("--literal-pathspecs", "add", "--all", "--pathspec-from-file=" + str(pathspec), "--pathspec-file-nul", env=env)
        full = git("write-tree", env=env)
        recipe = recipe_identity(value, full)
        for name, expected in recipe["files"].items():
            if expected is not None and hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise ValueError("Working bytes differ from Git export; normalize the checked input first: " + name)
        return full, recipe
    finally:
        for path in (index, Path(str(index) + ".lock"), pathspec):
            if path.exists():
                path.unlink()


def check_approval(approval, value, recipe):
    for key, expected in {"schema": "h3-sr-production-freeze/1", "status": "READY_FOR_BUILD",
                          "recipe_sha256": recipe["recipe_sha256"], "review_tree": recipe["review_tree"],
                          "parent_image": value["parent_image"]}.items():
        if approval.get(key) != expected:
            raise ValueError("Final internal review is not bound to the exact recipe: " + key)
    for name in ("root", "k8s"):
        item = approval.get("reviews", {}).get(name)
        if item != "PASS" and not (isinstance(item, dict) and item.get("status") == "PASS"):
            raise ValueError("Final review is incomplete: " + name)


def check_seal(committed=False, approved=True):
    value = lock()
    full = git("rev-parse", "HEAD^{tree}") if committed else prospective(value)[0]
    recipe = recipe_identity(value, full)
    if read(HERE / "input-seal.json") != recipe:
        raise ValueError("Frozen recipe no longer matches; complete affected checks and reseal")
    if approved:
        check_approval(read(HERE / "freeze-approval.json"), value, recipe)
    return value, full, recipe


def check_syntax():
    paths = set((ROOT / "container/h3-sr-production").rglob("*.py"))
    paths.update(HERE.glob("*.py"))
    paths.update(ROOT / ".github/scripts" / name for name in ("h3_sr_production_ci.py", "test_h3_sr_production_ci.py"))
    paths.update((ROOT / "dingo/vllm/omni").rglob("*.py"))
    paths.update((ROOT / "dingo/video_gateway").rglob("*.py"))
    for path in sorted(paths):
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path),
                  feature_version=(3, 10) if path == HERE / "ops.py" or ".github" in path.parts else (3, 12))
    workflow = (ROOT / ".github/workflows/h3-sr-production.yml").read_text(encoding="utf-8")
    blocks = re.findall(r"(?m)^        run: \|\n((?:          [^\n]*\n|\n)+)", workflow)
    if not blocks or len(blocks) != len(re.findall(r"(?m)^        run:", workflow)):
        raise ValueError("Every production workflow shell block must be included in syntax checks")
    candidates = [Path(r"D:\workingtools\Git\usr\bin\bash.exe"), Path(r"D:\workingtools\Git\bin\bash.exe")]
    found = shutil.which("bash")
    if found and not (os.name == "nt" and "system32" in found.casefold()):
        candidates.append(Path(found))
    binaries = list(dict.fromkeys(str(path) for path in candidates if path.is_file()))
    if not binaries:
        raise RuntimeError("Bash is required to verify actual workflow shell syntax")
    for number, block in enumerate(blocks):
        body = "\n".join(line[10:] if line.startswith("          ") else line for line in block.splitlines()) + "\n"
        attempts = []
        for binary in binaries:
            result = subprocess.run([binary, "-n"], input=body, cwd=ROOT, capture_output=True,
                                    text=True, encoding="utf-8", errors="replace")
            attempts.append({"binary": binary, "exit_code": result.returncode, "stderr": result.stderr})
            if result.returncode == 0:
                break
            if result.returncode in range(1, 256):
                raise RuntimeError("Workflow shell syntax failed: " + result.stderr)
        else:
            dump(EVIDENCE / "bash-startup-errors.json", {"block": number, "attempts": attempts})
            raise RuntimeError("Bash failed to initialize; shell syntax was not verified. See bash-startup-errors.json")
    return {"python_syntax_files": len(paths), "workflow_shell_blocks": len(blocks)}


def run_local_checks():
    value = lock()
    _, before = prospective(value)
    groups = {"production-runtime": ["container/h3-sr-production/tests", "test*.py"],
              "production-ci": [".github/scripts", "test_h3_sr_production_ci.py"]}
    results = {}
    environment = dict(os.environ)
    if environment.get("H3_SR_CHECK_DEPS"):
        dependencies = Path(environment["H3_SR_CHECK_DEPS"]).resolve()
        if not dependencies.is_dir():
            raise ValueError("H3_SR_CHECK_DEPS must point to an existing independent local dependency directory")
        environment["PYTHONPATH"] = str(dependencies) + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    for name, (directory, pattern) in groups.items():
        result = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", directory, "-p", pattern],
                                cwd=ROOT, env=environment, capture_output=True, text=True, encoding="utf-8", errors="replace")
        dump(EVIDENCE / (name + ".json"), {"exit_code": result.returncode, "output": result.stdout + result.stderr})
        count = re.search(r"Ran ([0-9]+) tests?", result.stdout + result.stderr)
        if result.returncode or not count or int(count.group(1)) == 0 or re.search(r"skipped=[1-9]", result.stdout + result.stderr):
            raise RuntimeError("Local checks failed: " + name)
        results[name] = int(count.group(1))
    syntax = check_syntax()
    _, after = prospective(value)
    if after != before:
        raise ValueError("Recipe changed while local checks were running")
    report = {"status": "PASS_LOCAL_ONLY_GPU_NOT_RUN", "recipe_sha256": after["recipe_sha256"],
              "review_tree": after["review_tree"], "tests": results, **syntax,
              "not_run": ["Linux image", "CUDA", "Kubernetes", "Harbor"]}
    dump(EVIDENCE / "local-checks.json", report)
    return report


def github_json(endpoint):
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        result = subprocess.run([GIT, "-C", str(ROOT), "-c", "safe.directory=" + ROOT.as_posix(),
            "-c", "credential.interactive=never", "credential", "fill"],
            input=b"url=https://github.com/githubgxll/dynamo.git\n\n", capture_output=True,
            env=dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never"), timeout=30)
        if result.returncode:
            raise RuntimeError("Cannot authenticate the original-run lookup; result remains UNKNOWN")
        token = dict(line.split("=", 1) for line in result.stdout.decode().splitlines() if "=" in line).get("password")
    if not token:
        raise RuntimeError("Original-run lookup authentication unavailable; do not submit")
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None
    handlers = [NoRedirect()]
    if os.name == "nt":
        handlers.append(urllib.request.ProxyHandler({"https": "http://127.0.0.1:17890"}))
    opener = urllib.request.build_opener(*handlers)
    request = urllib.request.Request("https://api.github.com/repos/githubgxll/dynamo" + endpoint,
        headers={"Authorization": "Bearer " + token, "User-Agent": "h3-sr-production-result-observer",
                 "Accept": "application/vnd.github+json"})
    try:
        with opener.open(request, timeout=30) as response:
            return json.load(response)
    except Exception as error:
        raise RuntimeError("Original Actions outcome remains UNKNOWN: " + type(error).__name__) from None


def observe_run(run_id, getter=None):
    getter = getter or github_json
    run = getter(f"/actions/runs/{run_id}")
    if run.get("id") != run_id or run.get("head_branch") != BRANCH or run.get("path") != ".github/workflows/" + WORKFLOW:
        raise ValueError("Observed run is not the designated production workflow/branch")
    attempt = run.get("run_attempt")
    if type(attempt) is not int or attempt <= 0:
        raise ValueError("Original run attempt identity missing")
    history = []
    for number in range(1, attempt + 1):
        response = getter(f"/actions/runs/{run_id}/attempts/{number}/jobs?per_page=100")
        jobs = response.get("jobs")
        if not isinstance(jobs, list) or response.get("total_count") != len(jobs):
            raise ValueError("Incomplete original-attempt job inventory")
        if any(j.get("run_id") != run_id or j.get("run_attempt") != number or j.get("head_sha") != run.get("head_sha") for j in jobs):
            raise ValueError("Mixed original run/attempt/commit jobs")
        publication = "unknown"
        if jobs and all(j.get("status") == "completed" for j in jobs):
            steps = [s for j in jobs for s in j.get("steps", []) if s.get("name") == PUBLISH_STEP]
            setup_only = all(j.get("conclusion") == "skipped" or
                             (len(j.get("steps", [])) == 1 and j["steps"][0].get("name") == "Set up job"
                              and j["steps"][0].get("conclusion") != "success") for j in jobs)
            if steps and any(s.get("conclusion") == "success" for s in steps):
                publication = "published"
            elif (steps and all(s.get("conclusion") == "skipped" for s in steps)) or (not steps and setup_only):
                publication = "not_started"
        history.append({"attempt": number, "publication": publication})
    latest = getter(f"/actions/runs/{run_id}")
    if any(latest.get(k) != run.get(k) for k in ("id", "head_sha", "status", "conclusion", "run_attempt", "updated_at")):
        raise ValueError("Original result changed during observation")
    states = {item["publication"] for item in history}
    publication = "published" if "published" in states else "not_started" if states == {"not_started"} else "unknown"
    return {"run_id": run_id, "run_attempt": attempt, "commit": run["head_sha"], "status": run["status"],
            "conclusion": run.get("conclusion"), "publication_status": publication,
            "history": history, "run_url": run.get("html_url"), "observed_at": datetime.now(timezone.utc).isoformat()}


def previous_result(journal, getter=None):
    if journal is None:
        return None
    if journal.get("status") == "PUBLISHED":
        raise ValueError("A successful digest already exists; reuse it: " + journal.get("image", ""))
    getter = getter or github_json
    matches = getter("/actions/workflows/" + WORKFLOW + "/runs?event=push&head_sha=" + journal["commit"] + "&per_page=100")
    runs = matches.get("workflow_runs")
    if not isinstance(runs, list) or matches.get("total_count") != 1 or len(runs) != 1:
        raise ValueError("Prior request has no unique confirmed run; remains UNKNOWN")
    result = observe_run(runs[0]["id"], getter)
    if result["commit"] != journal["commit"]:
        raise ValueError("Original request commit differs")
    if result["status"] != "completed":
        raise ValueError("Original request remains queued/running; observe it without another request")
    if result["publication_status"] == "published" or result["conclusion"] not in ("failure", "cancelled", "timed_out", "action_required"):
        raise ValueError("Original publication is successful or UNKNOWN; inspect Harbor before any new request")
    observation = journal.get("publication_observation", {})
    expected = {"schema": "h3-sr-production-publication-observation/1", "commit": journal["commit"],
                "run_id": str(result["run_id"]), "run_attempt": result["run_attempt"],
                "image": HARBOR + ":h3-sr-production-" + journal["commit"][:12],
                "publication_state": "ABSENT_CONFIRMED"}
    if any(observation.get(k) != v for k, v in expected.items()):
        raise ValueError("Prior run failed, but actual Harbor absence is not confirmed for its latest attempt; import publication-observation.json")
    path = Path(journal.get("publication_observation_path", ""))
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != journal.get("publication_observation_sha256"):
        raise ValueError("Recorded original Harbor observation changed or is unavailable")
    return result


def record_outcome(receipt, journal):
    result = read(receipt)
    expected = {"schema": "h3-sr-production-publication-observation/1", "commit": journal["commit"],
                "image": HARBOR + ":h3-sr-production-" + journal["commit"][:12]}
    if (any(result.get(k) != v for k, v in expected.items())
            or not re.fullmatch(r"[1-9][0-9]*", str(result.get("run_id", "")))
            or type(result.get("run_attempt")) is not int or result["run_attempt"] <= 0
            or result.get("publication_state") not in ("ABSENT_CONFIRMED", "EXISTS", "UNKNOWN")
            or not isinstance(result.get("observed_at"), str)):
        raise ValueError("Harbor observation is not bound to this source request")
    datetime.fromisoformat(result["observed_at"].replace("Z", "+00:00"))
    return {**journal, "publication_observation": result, "publication_observation_path": str(receipt.resolve()),
            "publication_observation_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest()}


def record_publication(receipt, journal):
    result = read(receipt)
    expected = {"schema": "h3-sr-production-publication/1", "status": "PASS_PUBLISHED_AND_PULLED_CPU_VALIDATED",
                "commit": journal["commit"], "parent_image": PARENT, "recipe_sha256": journal["recipe_sha256"],
                "image": HARBOR + ":h3-sr-production-" + journal["commit"][:12]}
    if (any(result.get(k) != v for k, v in expected.items())
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", result.get("digest", ""))
            or result.get("published_image") != HARBOR + "@" + result["digest"]
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", result.get("image_id", ""))):
        raise ValueError("Publication receipt does not establish this request's immutable validated result")
    files = result.get("evidence_sha256", {})
    required = {"candidate.json", "preparation.json", "parent-inspect.json", "final/runtime-validation.json", "final/source-verification.json"}
    if not isinstance(files, dict) or not required.issubset(files):
        raise ValueError("Complete actual-image publication evidence is required")
    directory = receipt.resolve().parent
    for name, expected_hash in files.items():
        if "\\" in name or Path(name).is_absolute():
            raise ValueError("Unsafe publication evidence path")
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise ValueError("Unsafe publication evidence identity")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError("Publication evidence changed: " + name)
    candidate = read(directory / "candidate.json")
    if (candidate.get("status") != "PASS_CPU_ONLY" or candidate.get("image_id") != result["image_id"]
            or candidate.get("commit") != journal["commit"] or candidate.get("recipe_sha256") != journal["recipe_sha256"]
            or read(directory / "parent-inspect.json").get("Id") != PARENT_ID):
        raise ValueError("Actual CPU-tested imageID differs from the published receipt")
    runtime, proof = read(directory / "final/runtime-validation.json"), read(directory / "final/source-verification.json")
    tests = runtime.get("tests")
    if isinstance(tests, dict):
        if any(tests.get(k, 0) for k in ("errors", "failures", "skipped")):
            raise ValueError("Imported actual-image tests were incomplete")
        tests = tests.get("tests", tests.get("tests_run"))
    if (type(tests) is not int or tests <= 0 or runtime.get("status") != "PASS_CPU_ONLY" or runtime.get("euid") != 1000
            or runtime.get("errors") != [] or runtime.get("source_commit") != journal["commit"]
            or runtime.get("parent_image") != PARENT or proof.get("status") != "PASS"
            or proof.get("source_commit") != journal["commit"] or not proof.get("source_sha256")):
        raise ValueError("Imported final runtime/source acceptance is invalid")
    manifest = hash_value(proof["source_sha256"])
    if any(item.get("source_manifest_sha256") != manifest for item in (proof, runtime, result, candidate)):
        raise ValueError("Imported complete source map does not match the validated publication")
    return {**journal, "status": "PUBLISHED", "image": result["published_image"],
            "publication_receipt": str(receipt.resolve()), "publication_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest()}


def claim(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def submit(confirmed):
    if not confirmed:
        raise ValueError("Use the explicitly authorized --confirm-build-and-publish submission")
    value, full, recipe = check_seal()
    cached = read(EVIDENCE / "local-checks.json") if (EVIDENCE / "local-checks.json").exists() else {}
    if cached.get("status") != "PASS_LOCAL_ONLY_GPU_NOT_RUN" or cached.get("recipe_sha256") != recipe["recipe_sha256"]:
        run_local_checks()
    path = EVIDENCE / "request-journal.json"
    previous = read(path) if path.exists() else None
    result = previous_result(previous)
    expected_remote = previous["commit"] if previous else value["working_tree_base_commit"]
    if remote_head() != expected_remote:
        raise ValueError("Remote branch changed from the observed baseline; do not overwrite it")
    if prospective(value)[0] != full:
        raise ValueError("Checked source changed before staging")
    if git("rev-parse", "HEAD^{tree}") != full:
        index = (ROOT / git("rev-parse", "--git-path", "index")).resolve()
        backup = EVIDENCE / ("index-before-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
        if index.exists():
            shutil.copy2(index, backup)
        git("read-tree", full)
        git("commit", "-s", "-m", "build(h3): freeze common production H3 SR runtime " + MARKER)
    commit = git("rev-parse", "HEAD")
    if not valid_request_message(git("log", "-1", "--format=%B")) or (previous and previous["commit"] == commit):
        raise ValueError("Commit message invalid or this exact request was already submitted; no empty commit")
    if git("status", "--porcelain") or git("rev-parse", "HEAD^{tree}") != full:
        raise ValueError("Checkout changed during commit")
    check_seal(committed=True)
    previous_result(previous)  # Recheck immediately before the online request.
    if remote_head() != expected_remote:
        raise ValueError("Remote moved before push; stop without force")
    sequence = previous["sequence"] + 1 if previous else 1
    record = {"schema": "h3-sr-production-request/1", "status": "REQUESTED", "sequence": sequence,
              "commit": commit, "review_tree": recipe["review_tree"], "recipe_sha256": recipe["recipe_sha256"],
              "parent_image": PARENT, "previous_result": result, "created_at": datetime.now(timezone.utc).isoformat()}
    claim(EVIDENCE / "requests" / f"request-{sequence:04d}.json", record)
    if previous:
        dump(EVIDENCE / "history" / f"request-{sequence - 1:04d}.json", previous)
    dump(path, record)
    try:
        git("push", "origin", "HEAD:refs/heads/" + BRANCH, network=True)
    except BaseException:
        dump(path, {**record, "status": "UNKNOWN", "reason": "Push response failed; inspect original commit/run/Harbor before another request"})
        raise
    dump(path, {**record, "push_completed": True})
    print(json.dumps({"status": "REQUESTED", "commit": commit, "sequence": sequence,
                      "branch": BRANCH, "note": "Observe the original result; no repeated submission while active/unknown"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("seal", "check", "check-seal"):
        sub.add_parser(name)
    submit_parser = sub.add_parser("submit")
    submit_parser.add_argument("--confirm-build-and-publish", action="store_true")
    observe = sub.add_parser("observe")
    observe.add_argument("--run-id", type=int, required=True)
    recorded = sub.add_parser("record-publication")
    recorded.add_argument("--receipt", type=Path, required=True)
    outcome_parser = sub.add_parser("record-outcome")
    outcome_parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "seal":
        value = lock()
        _, recipe = prospective(value)
        dump(HERE / "input-seal.json", recipe)
        print(json.dumps({"status": "SEALED_LOCAL_WAITING_FINAL_REVIEW", **recipe}, indent=2))
    elif args.command == "check":
        print(json.dumps(run_local_checks(), indent=2))
    elif args.command == "check-seal":
        _, full, recipe = check_seal()
        print(json.dumps({"status": "READY_FOR_BUILD", "tree": full, **recipe}, indent=2))
    elif args.command == "observe":
        result = observe_run(args.run_id)
        journal = read(EVIDENCE / "request-journal.json")
        if result["commit"] != journal["commit"]:
            raise ValueError("Observed run is not the local request")
        dump(EVIDENCE / "observations" / f"{args.run_id}-{result['run_attempt']}.json", result)
        dump(EVIDENCE / "request-journal.json", {**journal, "last_observation": result})
        print(json.dumps(result, indent=2))
    elif args.command == "record-outcome":
        updated = record_outcome(args.receipt, read(EVIDENCE / "request-journal.json"))
        dump(EVIDENCE / "request-journal.json", updated)
        print(json.dumps(updated["publication_observation"], indent=2))
    elif args.command == "record-publication":
        updated = record_publication(args.receipt, read(EVIDENCE / "request-journal.json"))
        dump(EVIDENCE / "request-journal.json", updated)
        print(updated["image"])
    else:
        submit(args.confirm_build_and_publish)


if __name__ == "__main__":
    main()
