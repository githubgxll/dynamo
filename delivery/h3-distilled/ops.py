"""Operator commands; only explicit submit writes Git and triggers publication."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONTEXT = ROOT / "container/h3-distilled"
OUT = Path(os.environ.get("H3_DISTILLED_EVIDENCE", r"D:\AI\_workspace\H3_Distilled_Image_20261013"))
ALLOWED_PREFIXES = ("container/h3-distilled/", "delivery/h3-distilled/")
ALLOWED_FILES = {".github/workflows/dingo-router-ci.yml", ".github/scripts/h3_distilled_ci.py",
                 ".github/scripts/test_h3_distilled_ci.py", ".gitignore"}


def run(*args, capture=False):
    return subprocess.run([str(x) for x in args], cwd=ROOT, check=True, capture_output=capture,
                          text=True, encoding="utf-8")


def git(*args):
    return run("git", *args, capture=True).stdout.strip()


def lock():
    return json.loads((CONTEXT / "image.lock.json").read_text(encoding="utf-8"))


def changed_paths():
    paths = set()
    for args in (("diff", "--cached", "--name-only", "-z"), ("diff", "--name-only", "-z"),
                 ("ls-files", "--others", "--exclude-standard", "-z")):
        paths.update(p for p in run("git", *args, capture=True).stdout.split("\0") if p)
    return sorted(paths)


def allowed(path):
    if path.endswith((".safetensors", ".part", ".whl", ".zip", ".env", ".pyc")):
        return False
    return path in ALLOWED_FILES or path.startswith(ALLOWED_PREFIXES)


def submit(confirmed):
    if not confirmed:
        raise ValueError("This operation builds AND publishes; use the documented CMD entrypoint")
    config = lock()
    if git("branch", "--show-current") != config["local_branch"]:
        raise ValueError("Use the isolated distilled branch")
    if git("remote", "get-url", "origin") != config["repository"]:
        raise ValueError("Unexpected Git origin")
    if not re.fullmatch(r"DingoRouter-h3-distilled-\d{8}", config["publish_branch"]):
        raise ValueError("Unexpected publication branch")
    run("git", "merge-base", "--is-ancestor", config["recipe_base_commit"], "HEAD")
    paths = changed_paths()
    if any(not allowed(p) for p in paths):
        raise ValueError("Unrelated or binary input present: " + repr([p for p in paths if not allowed(p)]))
    for path in paths:
        file = ROOT / path
        if file.is_file() and file.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("Large input must not enter Git: " + path)
    run(sys.executable, "-B", HERE / "check_local.py")
    if paths:
        run("git", "add", "--", *paths)
        run("git", "diff", "--cached", "--check")
        run("git", "commit", "-m", "Package H3 distilled adapters on verified Omni 0.30 image")
    run(sys.executable, "-B", HERE / "check_local.py", "--require-clean")
    run("git", "-c", "http.proxy=http://127.0.0.1:17890", "push", "origin",
        "HEAD:refs/heads/" + config["publish_branch"])


def fetch():
    run(sys.executable, "-B", CONTEXT / "fetch_assets.py", "--lock", CONTEXT / "assets.lock.json",
        "--cache-dir", OUT / "downloads/cache", "--output", OUT / "downloads/adapters",
        "--proxy", "http://127.0.0.1:17890")


def read_archive(path):
    with zipfile.ZipFile(path) as archive:
        items = archive.infolist()
        if len(items) > 150 or sum(i.file_size for i in items) > 25 * 1024 * 1024:
            raise ValueError("Unexpected artifact size; weights must not be in evidence ZIP")
        files = {}
        names = set()
        for info in items:
            p = PurePosixPath(info.filename)
            if not p.name or p.is_absolute() or ".." in p.parts or "\\" in info.filename or ":" in info.filename:
                raise ValueError("Unsafe ZIP path")
            if info.is_dir():
                continue
            if p.name.casefold() in names:
                raise ValueError("Ambiguous duplicate evidence filename")
            names.add(p.name.casefold())
            files[p.name] = archive.read(info)
        return files


def import_evidence(path):
    files = read_archive(path)
    record = json.loads(files["image.json"])
    config = lock()
    if record.get("schema") != "h3-distilled-publication/1" or record.get("platform") != "linux/amd64":
        raise ValueError("Wrong publication schema or platform")
    mandatory = {"image.lock.json", "assets.lock.json", "bundle-check.json", "assets-check.json",
                 "runtime-source-check.json", "base-environment.json", "runtime-environment.json",
                 "Dockerfile", "dingo-images.json"}
    fingerprints = record.get("evidence_sha256")
    if not isinstance(fingerprints, dict) or not mandatory.issubset(fingerprints):
        raise ValueError("Incomplete evidence fingerprints")
    if set(files) - set(fingerprints) - {"image.json", "deployment-image.txt"}:
        raise ValueError("Unmanifested extra evidence files")
    for name, expected_hash in fingerprints.items():
        if name not in files or not re.fullmatch(r"[0-9a-f]{64}", str(expected_hash)) or hashlib.sha256(files[name]).hexdigest() != expected_hash:
            raise ValueError("Evidence hash mismatch: " + name)
    for name, field in (("image.lock.json", "image_lock_sha256"), ("assets.lock.json", "assets_lock_sha256")):
        if record.get(field) != fingerprints[name]:
            raise ValueError("Publication lock checksum mismatch")
    commit = record["source_commit"]
    digest = record["digest"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError("Invalid source commit or image digest")
    expected = config["publication_repository"] + ":" + config["tag_prefix"] + "-" + commit[:12]
    if record["image"] != expected or record["deployment_image"] != config["publication_repository"] + "@" + digest:
        raise ValueError("Wrong publication target")
    if record["base_image"] != config["base_image"]:
        raise ValueError("Wrong inherited runtime")
    if record.get("framework_source_commit") != config["recipe_base_commit"]:
        raise ValueError("Wrong inherited framework source")
    if "deployment-image.txt" in files and files["deployment-image.txt"].decode("utf-8").strip() != record["deployment_image"]:
        raise ValueError("Conflicting deployment image reference")
    for name in ("image.lock.json", "assets.lock.json"):
        expected_blob = subprocess.run(["git", "show", commit + ":container/h3-distilled/" + name],
                                       cwd=ROOT, check=True, capture_output=True).stdout
        # Git and checkout may use different newline conventions; compare parsed lock objects.
        if json.loads(files[name]) != json.loads(expected_blob):
            raise ValueError("Evidence differs from committed lock: " + name)
    for name, status in (("assets-check.json", "PASS_ASSETS_ONLY"),
                         ("runtime-source-check.json", "PASS_INSTALLED_CODE_ONLY"),
                         ("bundle-check.json", "PASS_DISTILLED_CPU_ONLY")):
        if json.loads(files[name])["status"] != status:
            raise ValueError("Missing successful build gate: " + name)
    if json.loads(files["bundle-check.json"])["source_commit"] != commit:
        raise ValueError("Build evidence belongs to another source commit")
    bundle = json.loads(files["bundle-check.json"])
    if bundle.get("framework_source_commit") != config["recipe_base_commit"]:
        raise ValueError("Bundle belongs to another framework source")
    if bundle.get("base_image") != config["base_image"] or bundle.get("environment_unchanged") is not True:
        raise ValueError("Incomplete inherited-runtime verification")
    if bundle.get("asset_lock_sha256") != fingerprints["assets.lock.json"] or bundle.get("image_lock_sha256") != fingerprints["image.lock.json"]:
        raise ValueError("Bundle was validated against another lock")
    if json.loads(files["assets-check.json"]).get("lock_sha256") != fingerprints["assets.lock.json"]:
        raise ValueError("Assets were validated against another lock")
    if json.loads(files["base-environment.json"]) != json.loads(files["runtime-environment.json"]):
        raise ValueError("Installed runtime changed during packaging")
    target = OUT / "published" / commit
    if (target / "image.json").exists() and json.loads((target / "image.json").read_text(encoding="utf-8")) != record:
        raise ValueError("Different publication already recorded for this commit; preserve both runs separately")
    target.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        (target / name).write_bytes(payload)
    (target / "image.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    (target / "deployment-image.txt").write_text(record["deployment_image"] + "\n", encoding="utf-8")
    print("PASS_PUBLICATION_RECORD_ONLY:", target)
    print("This does not independently pull Harbor or validate H100/model/media execution.")


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("fetch")
    submit_parser = commands.add_parser("submit")
    submit_parser.add_argument("--confirm-build-and-publish", action="store_true")
    importer = commands.add_parser("import-evidence")
    importer.add_argument("zip")
    args = parser.parse_args()
    if args.command == "fetch":
        fetch()
    elif args.command == "submit":
        submit(args.confirm_build_and_publish)
    else:
        import_evidence(args.zip)


if __name__ == "__main__":
    main()
