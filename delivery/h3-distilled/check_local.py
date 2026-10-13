"""Offline preparation checks, not a Docker/GPU/publication acceptance test."""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONTEXT = ROOT / "container/h3-distilled"
OUT = Path(os.environ.get("H3_DISTILLED_EVIDENCE", r"D:\AI\_workspace\H3_Distilled_Image_20261013"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files():
    selected = [ROOT / ".github/workflows/dingo-router-ci.yml", ROOT / ".github/scripts/h3_distilled_ci.py",
                ROOT / ".github/scripts/test_h3_distilled_ci.py"]
    for folder in (CONTEXT, HERE):
        selected.extend(p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return sorted(selected)


def check_prospective_index(root, paths, documents, evidence):
    """Check the actual proposed Git blobs, without changing the user's index."""
    root = Path(root).resolve()
    evidence = Path(evidence).resolve()
    index = evidence / "prospective.index"
    require(not index.exists(), "Prospective index must start absent")
    env = {**os.environ, "GIT_INDEX_FILE": str(index)}
    real_index = Path(subprocess.run(["git", "rev-parse", "--git-path", "index"], cwd=root,
                                    check=True, capture_output=True, text=True).stdout.strip())
    if not real_index.is_absolute():
        real_index = root / real_index
    before = sha(real_index) if real_index.exists() else None

    def command(name, *args):
        result = subprocess.run(["git", *args], cwd=root, env=env, capture_output=True)
        (evidence / (name + ".txt")).write_bytes(result.stdout + result.stderr)
        require(result.returncode == 0, name + " failed; see " + str(evidence))
        return result.stdout

    try:
        command("git-preview-read-tree", "read-tree", "HEAD")
        if paths:
            command("git-preview-add", "add", "--", *sorted(set(paths)))
        command("git-preview-whitespace", "diff", "--cached", "--check")
        for relative, expected_hash in documents.items():
            blob = command("git-preview-" + Path(relative).name, "cat-file", "blob", ":" + relative)
            require(hashlib.sha256(blob).hexdigest() == expected_hash,
                    "Git would alter a pinned upstream document: " + relative)
    finally:
        require((sha(real_index) if real_index.exists() else None) == before,
                "Real Git index changed during prospective check")
    return {"status": "PASS_PROSPECTIVE_GIT_ONLY", "paths_checked": len(set(paths)),
            "documents_checked": len(documents), "real_index_unchanged": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    evidence = OUT / "checks" / run_id
    evidence.mkdir(parents=True)
    os.environ["TMPDIR"] = str(evidence)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    fingerprints = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in files()}
    config = load(CONTEXT / "image.lock.json")
    log_index = 0

    def execute(name, command):
        nonlocal log_index
        log_index += 1
        result = subprocess.run([str(x) for x in command], cwd=ROOT, text=True,
                                encoding="utf-8", errors="replace", capture_output=True)
        (evidence / (name + ".txt")).write_text(result.stdout + result.stderr, encoding="utf-8")
        require(result.returncode == 0, name + " failed; see " + str(evidence))
        return result.stdout

    execute("baseline-ancestor", ["git", "merge-base", "--is-ancestor", config["recipe_base_commit"], "HEAD"])
    unchanged = ["container/context.yaml", "container/templates/vllm_runtime.Dockerfile",
                 "container/templates/wheel_builder.Dockerfile", "container/deps/vllm/install_vllm_omni.sh",
                 "container/deps/vllm/collect_runtime_manifest.py", "container/deps/vllm/protected_packages.txt",
                 "container/compliance/policy/licenses.toml", ".github/dingo-images.json", "pyproject.toml"]
    for relative in unchanged:
        old = subprocess.run(["git", "show", config["recipe_base_commit"] + ":" + relative], cwd=ROOT,
                             check=True, capture_output=True).stdout.decode("utf-8").replace("\r\n", "\n")
        require((ROOT / relative).read_text(encoding="utf-8").replace("\r\n", "\n") == old,
                "Successful framework recipe changed: " + relative)
    for p in files():
        if p.suffix == ".py":
            ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        require(p.stat().st_size < 2 * 1024 * 1024, "Large file in recipe: " + str(p))

    sys.path.insert(0, str(CONTEXT))
    from verify_assets import check_file, load_lock, safe_path
    assets = load_lock(CONTEXT / "assets.lock.json")
    require({a["id"] for a in assets["assets"]} == {"turbo-fl8", "turbo-ref8", "turbo-fl4", "fasth3-dense4"}, "Wrong adapter selection")
    for d in assets["documents"]:
        check_file(safe_path(CONTEXT, d["local_source"]), d)
    # Ordinary git diff misses untracked inputs. Preview the same add operation
    # that submit will perform, then check the indexed license bytes as well.
    from ops import changed_paths
    prospective = check_prospective_index(ROOT, changed_paths(),
        {(CONTEXT / d["local_source"]).relative_to(ROOT).as_posix(): d["sha256"]
         for d in assets["documents"]}, evidence)
    profiles = load(CONTEXT / "profiles.json")
    require(len(profiles["profiles"]) == 6, "Expected six reference contracts")
    for a in assets["assets"]:
        p = next(x for x in profiles["profiles"] if x["id"] == a["id"])
        require(p["adapter_path"] == config["asset_root"] + "/" + a["relative_path"], "Profile adapter path mismatch")
        require(p["sha256"] == a["sha256"] and p["steps"] == a["steps"], "Profile asset identity/step mismatch")

    dockerfile = (CONTEXT / "Dockerfile").read_text(encoding="utf-8")
    require("# syntax=" not in dockerfile, "Unexpected external frontend dependency")
    require(not re.search(r"\b(?:pip|apt-get|dnf|uv pip)\s+(?:install|upgrade)", dockerfile), "Do not reinstall framework")
    require("ARG BASE_IMAGE=" + config["base_image"] in dockerfile, "Dockerfile base is not the lock")
    require("FROM ${BASE_IMAGE} AS runtime" in dockerfile and "COPY --from=runtime /opt/h3-distilled/build-info/ /" in dockerfile, "Evidence must depend on runtime gate")
    require("USER dynamo\nRUN --network=none" in dockerfile.replace("\r\n", "\n"), "Offline non-root gate missing")
    require("ARG DYNAMO_COMMIT_SHA" not in dockerfile, "Do not replace installed framework provenance")

    workflow = yaml.load((ROOT / ".github/workflows/dingo-router-ci.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    jobs = workflow["jobs"]
    require("!startsWith(github.ref_name, 'DingoRouter-h3-distilled-')" in jobs["prepare"]["if"], "Normal build also triggers for distilled branch")
    require("inputs.frameworks != 'h3-distilled'" in jobs["prepare"]["if"], "Manual distilled selection must exclude normal build")
    job = jobs["h3-distilled"]
    require(job["runs-on"] == ["self-hosted", "linux", "x64", "dingo"], "Wrong Runner")
    builds = [s for s in job["steps"] if str(s.get("uses", "")).startswith("docker/build-push-action")]
    require([s["with"]["target"] for s in builds] == ["evidence", "runtime"], "Wrong prepublish ordering")
    require([s["with"]["push"] for s in builds] == ["false", "true"], "Evidence must not publish")
    for step in builds:
        require(step["with"]["context"] == "container/h3-distilled", "Wrong build context")
        require("DYNAMO_COMMIT_SHA=" not in step["with"]["build-args"], "Build arg changes framework provenance")
    bash = Path(r"D:\workingtools\Git\bin\bash.exe")
    require(bash.exists(), "Existing Git Bash required for shell syntax checks")
    blocks = 0
    for j in jobs.values():
        for step in j.get("steps", []):
            if "run" in step:
                blocks += 1
                text = re.sub(r"\$\{\{.*?\}\}", "synthetic_value", step["run"], flags=re.S)
                path = evidence / ("workflow-" + str(blocks) + ".sh")
                path.write_text(text, encoding="utf-8", newline="\n")
                execute("bash-" + str(blocks), [bash, "-n", path])
    execute("asset-regression", [sys.executable, "-B", "-m", "unittest", "discover", "-s", "container/h3-distilled/tests", "-v"])
    execute("ci-regression", [sys.executable, "-B", ".github/scripts/test_h3_distilled_ci.py"])
    execute("operator-regression", [sys.executable, "-B", HERE / "test_ops.py"])
    execute("git-preflight-regression", [sys.executable, "-B", HERE / "test_git_preflight.py"])
    wheel = Path(r"D:\AI\_workspace\H3_Omni030_Image_20260929\evidence\vllm_omni-0.30.0-py3-none-any.whl")
    execute("official-wheel-source", [sys.executable, "-B", CONTEXT / "image_reuse_check.py", "--wheel", wheel])
    execute("git-whitespace", ["git", "diff", "--check"])
    execute("git-staged-whitespace", ["git", "diff", "--cached", "--check"])
    status = execute("git-status", ["git", "status", "--porcelain"])
    if args.require_clean:
        require(not status.strip(), "Commit prepared changes before require-clean")
    require(fingerprints == {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in files()}, "Inputs changed during check")
    report = {"status": "PASS_LOCAL_PREPARATION_ONLY", "run_id": run_id, "evidence_directory": str(evidence),
              "checks": ["Successful 0.30 framework recipe unchanged", "Pinned adapter and document locks",
                         "Six profile reference paths", "Offline non-root Docker gate and no dependency installation",
                         "Dedicated group workflow routing and registry target", "Downloader/CI/operator regression tests",
                         "Official wheel and 53 source hashes", "Prospective staged inputs and exact upstream document bytes",
                         "Git whitespace and stable inputs"],
              "prospective_git": prospective,
              "workflow_shell_blocks_checked": blocks, "input_sha256": fingerprints,
              "not_run": ["Full production adapter download", "Docker/Linux build", "Harbor publish or pull",
                          "H100/HSDP or adapter activation", "Mounted Base revision verification", "Video/audio generation"],
              "git_head": execute("git-head", ["git", "rev-parse", "HEAD"]).strip(), "require_clean": args.require_clean}
    content = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    (evidence / "local-checks.json").write_text(content, encoding="utf-8")
    (OUT / "local-checks.json").write_text(content, encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
