#!/usr/bin/env python3
"""Validate a downloaded Actions artifact; its ZIP hash is not the image digest."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import zipfile

import ops


def safe_name(name):
    value = name.rstrip("/")
    if not value or "\\" in value or ":" in value or value.startswith("/") or any(x in {"", ".", ".."} for x in value.split("/")):
        raise ValueError("Unsafe artifact member name: " + name)
    for component in value.split("/"):
        if (component.endswith((".", " ")) or any(ord(c) < 32 for c in component)
                or re.fullmatch(r"(?i)(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", component)):
            raise ValueError("Ambiguous Windows artifact member name: " + name)
    return PurePosixPath(value)


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def validate_repair_reports(ci, root, record, commit):
    """Use the same source-content gate for import and registry publication."""
    ci.validate_candidate_reports(root, commit)
    if record.get("dit_entry_status") != "PASS" or record.get("patched_source_sha256") != ci.PATCHED_SOURCE_SHA256:
        raise ValueError("Publication does not identify the reviewed HSDP and DiT entry repair")


def import_archive(archive):
    ops.check_seal(committed=True)
    archive = archive.resolve(strict=True)
    destination = ops.EVIDENCE / datetime.now(timezone.utc).strftime("import-%Y%m%dT%H%M%S.%fZ")
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if len(members) > 20000 or sum(m.file_size for m in members) > 2 * 1024**3:
            raise ValueError("Unexpectedly large evidence archive")
        names = set()
        for member in members:
            relative = safe_name(member.filename)
            key = relative.as_posix().casefold()
            if key in names or ((member.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError("Duplicate path or symlink in evidence archive")
            names.add(key)
            target = destination.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError("Artifact path escapes its new import directory")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
    records = list(destination.rglob("image.json"))
    if len(records) != 1:
        raise ValueError("Expected exactly one successful publication record")
    record_path = records[0]
    root = record_path.parent
    record = json.loads(record_path.read_text(encoding="utf-8"))
    lock = ops.load_lock()
    if record.get("schema") != "h3-hsdp-offload-publication/1" or record.get("status") != "PASS_PUBLISHED_AND_PULLED_CPU_VALIDATED":
        raise ValueError("Artifact is not a successful HSDP CPU candidate publication")
    if record.get("gpu_validation_status") != "NOT_RUN":
        raise ValueError("Publication must preserve GPU_NOT_RUN candidate status")
    commit = record.get("source_commit", "")
    digest = record.get("digest", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError("Missing source commit or complete IMAGE digest")
    if commit != ops.git("rev-parse", "HEAD"):
        raise ValueError("Artifact does not belong to this checkout HEAD; select the intended submitted commit")
    expected_tag = lock["harbor_repository"] + ":omni031rc1-gateway-hsdp-offload-" + commit[:12]
    if record.get("image") != expected_tag or record.get("deployment_image") != lock["harbor_repository"] + "@" + digest:
        raise ValueError("Unexpected publication destination")
    for key in ("parent_image", "parent_source_commit", "gateway_commit"):
        if record.get(key) != lock[key]:
            raise ValueError("Publication identity differs: " + key)
    hashes = record.get("evidence_sha256", {})
    required = {"inputs.json", "candidate-verified.json", "package-preservation.json", "inheritance.json",
                "registry-push.txt", "registry-pull.txt"}
    required.add("parent-source.json")
    for phase in ("build", "final"):
        required.update(phase + "/" + name for name in (
            "runtime-validation.json", "preservation.json", "source-verification.json", "regression.json", "torch-contract.json"))
    if not required.issubset(hashes):
        raise ValueError("Required fresh publication evidence is missing")
    for name, expected in hashes.items():
        path = root.joinpath(*safe_name(name).parts)
        if not path.is_file() or sha(path) != expected:
            raise ValueError("Artifact evidence hash mismatch: " + name)
    spec = importlib.util.spec_from_file_location("srnorm_ci_import", ops.ROOT / ".github/scripts/h3_hsdp_offload_ci.py")
    ci = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ci)
    validate_repair_reports(ci, root, record, commit)
    inputs = ci.read(root / "inputs.json")
    for key, expected in {"source_commit": commit, "parent_image": lock["parent_image"],
                          "run_id": record["run_id"], "run_attempt": record["run_attempt"],
                          "final_image": expected_tag}.items():
        if inputs.get(key) != expected:
            raise ValueError("Publication input differs: " + key)
    verified = ci.read(root / "candidate-verified.json")
    for key, expected in {"status": "PASS_CANDIDATE_CPU_ONLY", "source_commit": commit,
                          "parent_image": lock["parent_image"], "image_id": record["image_id"],
                          "candidate_image": inputs["candidate_image"], "gpu_validation_status": "NOT_RUN",
                          "dit_entry_status": "PASS", "patched_source_sha256": ci.PATCHED_SOURCE_SHA256}.items():
        if verified.get(key) != expected:
            raise ValueError("Verified candidate differs: " + key)
    inheritance = ci.read(root / "inheritance.json")
    if (inheritance.get("status") != "PASS" or inheritance.get("candidate_image_id") != record["image_id"]
            or inheritance.get("parent_image_id") != lock["parent_image_id"]
            or ci.read(root / "package-preservation.json").get("status") != "PASS"):
        raise ValueError("Missing parent/config/package preservation proof")
    if ci.publication_digest((root / "registry-push.txt").read_text(encoding="utf-8")) != digest:
        raise ValueError("Push receipt differs from the published IMAGE digest")
    if (root / "deployment-image.txt").read_text(encoding="utf-8").strip() != record["deployment_image"]:
        raise ValueError("Deployment image text differs from publication record")
    seal = ci.read(ops.ROOT / ops.SEAL)
    for relative in set([name for name, entry in seal["delta"].items() if entry["status"] != "D"] + [ops.SEAL] + list(seal["shared_inputs"])):
        name = "recipe/" + relative
        archived = root / name
        if name not in hashes or archived.read_bytes() != ops.git("show", commit + ":" + relative, binary=True):
            raise ValueError("Artifact recipe does not match committed inputs: " + relative)
    source_snapshot = ci.read(root / "parent-source.json")
    if (source_snapshot.get("status") != "PASS" or source_snapshot.get("image_id") != lock["parent_image_id"]
            or source_snapshot.get("versions") != lock["versions"] or source_snapshot.get("parent_image") != lock["parent_image"]):
        raise ValueError("Missing actual parent source snapshot")
    for name, expected in lock["parent_source_sha256"].items():
        if source_snapshot.get("files", {}).get(name, {}).get("sha256") != expected:
            raise ValueError("Parent source snapshot does not match reviewed original: " + name)
    for name, entry in source_snapshot["files"].items():
        relative = "parent-source/" + safe_name(name).as_posix()
        if relative not in hashes or sha(root / relative) != entry["sha256"]:
            raise ValueError("Actual parent source evidence differs: " + name)
    result = {"gpu_validation_status":"NOT_RUN", "status": "PASS_IMPORTED_PUBLICATION_EVIDENCE_ONLY", "deployment_image": record["deployment_image"],
              "dit_entry_status":"PASS", "patched_source_sha256":ci.PATCHED_SOURCE_SHA256,
              "image_tag": record["image"], "source_commit": commit, "actions_url": record["actions_url"],
              "artifact_zip": str(archive), "artifact_zip_sha256": sha(archive), "evidence_directory": str(root),
              "not_run_locally": ["registry pull", "CUDA", "H100 generation", "visual A/B"]}
    (destination / "import-receipt.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    import_archive(parser.parse_args().archive)
