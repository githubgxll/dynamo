#!/usr/bin/env python3
"""Prepare and record the single approved H3 experiment pipeline.

No registry calls or credentials. The Actions workflow owns building/publishing;
these gates require the actual returned digest and the exported CPU evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import yaml

ROOT = Path(__file__).resolve().parents[2]
BRANCH = "Dingorouter-H3-vllm-omni-jirx"
REPOSITORY = "registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm"
SERVICE_BASE_SHA = "e8c9195672f761ec09344a641730844c9e6c9e99"
TAG_PREFIX = "omni031rc1-gateway-distilled-ref8shift12"
SHA = re.compile(r"[0-9a-f]{40}\Z")
DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
PROFILE = Path("container/h3-experimental")
SOURCE_LOCK = Path("delivery/h3-experimental/source-lock.json")


def read(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_outputs(path, values):
    if path:
        with path.open("a", encoding="utf-8", newline="\n") as out:
            for key, value in values.items():
                if "\n" in str(value) or "\r" in str(value):
                    raise ValueError("Multiline GitHub output is forbidden")
                out.write(f"{key}={value}\n")


def identity(environ):
    commit = environ.get("GITHUB_SHA", "")
    if not SHA.fullmatch(commit):
        raise ValueError("GITHUB_SHA must be a full lowercase commit")
    if environ.get("GITHUB_REF_NAME") != BRANCH:
        raise ValueError(f"This workflow is restricted to the dedicated {BRANCH} branch")
    event = environ.get("GITHUB_EVENT_NAME", "push")
    if event not in {"push", "workflow_dispatch"}:
        raise ValueError("Unsupported triggering event")
    return commit


def prepare(root, environ, output=None, evidence=None):
    commit = identity(environ)
    config = read(root / ".github/dingo-images.json")
    context = yaml.safe_load((root / "container/context.yaml").read_text(encoding="utf-8"))
    vllm = context["vllm"]
    runtime = vllm["cuda13.0"]
    if runtime.get("runtime_image_digest") != "sha256:3fb023cc987cc7c2bca7ca14c1e0aff5ae06bd9431bf07773731ce7e8a682460":
        raise ValueError("Official vLLM runtime digest differs from the reviewed 0.31 image")
    if runtime.get("runtime_image_tag") != "v0.31.0-ubuntu2404" or runtime.get("vllm_omni_ref") != "v0.31.0rc1":
        raise ValueError("Unexpected CUDA vLLM/Omni versions")
    if vllm.get("enable_kvbm") != "false" or vllm.get("enable_modelexpress") != "false":
        raise ValueError("H3 profile must disable optional KVBM and ModelExpress")
    spec = importlib.util.spec_from_file_location("h3_image_matrix", root / ".github/scripts/prepare_dingo_image_matrix.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entries = module.build_matrix(config, commit, "configured")
    if len(entries) != 1 or entries[0]["framework"] != "vllm" or entries[0]["docker_target"] != "runtime":
        raise ValueError("Only the gateway vLLM runtime is allowed")
    entry = entries[0]
    if entry["image"].rsplit(":", 1)[0] != REPOSITORY:
        raise ValueError("Unexpected Harbor destination")
    if entry["platform"] != "linux/amd64" or entry["cuda_version"] != "13.0":
        raise ValueError("Unsupported platform")
    # The source lock is retained verbatim; the service base is pinned independently here.
    lock = read(root / SOURCE_LOCK)
    if lock.get("service_base_commit") != SERVICE_BASE_SHA:
        raise ValueError("Source lock does not record the reviewed service base commit")
    assets = read(root / PROFILE / "assets/assets.lock.json")
    ids = {item["id"] for item in assets.get("assets", [])}
    if ids != {"turbo-fl8", "turbo-ref8", "turbo-fl4", "fasth3-dense4"}:
        raise ValueError(f"Expected exactly the four reviewed adapters, got {sorted(ids)}")
    ref8 = next(item for item in assets["assets"] if item["id"] == "turbo-ref8")
    if (ref8["steps"], ref8["video_shift"], ref8["audio_shift"]) != (8, 12, 3):
        raise ValueError("Ref8 must declare its patched 8-step/12/3 contract")
    values = {
        "registry": entry["registry"], "platform": entry["platform"],
        "cuda_version": entry["cuda_version"], "dockerfile": entry["dockerfile"],
        "builder_image": entry["builder_image"],
        "sccache_cache_version": entry["sccache_cache_version"] + "-gateway-" + SERVICE_BASE_SHA[:12],
        "framework_image": REPOSITORY + ":omni031rc1-gateway-framework-" + commit[:12],
        "final_image": REPOSITORY + ":" + TAG_PREFIX + "-" + commit[:12],
    }
    write_outputs(output, values)
    if evidence:
        dump(evidence / "inputs.json", {
            "schema": "h3-experimental-ci-inputs/1", "source_commit": commit,
            "service_base_commit": SERVICE_BASE_SHA, "branch": BRANCH, **values,
            "source_lock_sha256": sha256(root / SOURCE_LOCK),
            "assets_lock_sha256": sha256(root / PROFILE / "assets/assets.lock.json"),
            "validation_scope": "Configuration only; build and GPU not yet accepted",
        })
    return values


def record_framework(root, environ, digest, directory, output=None):
    values = prepare(root, environ)
    if not DIGEST.fullmatch(digest):
        raise ValueError("Missing/invalid actual framework digest")
    ref = REPOSITORY + "@" + digest
    record = {"schema": "h3-experimental-framework/1", "source_commit": identity(environ),
              "service_base_commit": SERVICE_BASE_SHA, "image": values["framework_image"], "digest": digest,
              "deployment_image": ref, "builder_image": values["builder_image"],
              "run_id": environ.get("GITHUB_RUN_ID"), "run_attempt": environ.get("GITHUB_RUN_ATTEMPT")}
    dump(directory / "framework-image.json", record)
    write_outputs(output, {"framework_ref": ref})
    return record


def validate_evidence(root, environ, directory, framework_ref):
    commit = identity(environ)
    prepare(root, environ)
    if not framework_ref.startswith(REPOSITORY + "@") or not DIGEST.fullmatch(framework_ref.split("@", 1)[-1]):
        raise ValueError("Final parent must be the actual immutable framework digest")
    reports = {
        "experimental/service-smoke.json": "PASS_SERVICE_IMPORTS_AND_HELP_ONLY",
        "experimental/assets-report.json": "PASS_ASSETS_ONLY",
        "experimental/profiles-check.json": "PASS_INSTALLED_PROFILES_CPU_ONLY",
        "framework/ref8-patch.json": "PASS_PATCH_APPLIED_CPU_ONLY",
        "framework/ref8-contract.json": "PASS_INSTALLED_CONTRACT_CPU_ONLY",
    }
    for name, status in reports.items():
        report = read(directory / name)
        if report.get("status") != status or report.get("errors"):
            raise ValueError(f"Failed or missing successful CPU report: {name}")
    inputs = read(directory / "experimental/image-inputs.json")
    if inputs.get("schema") != "h3-experimental-inputs/1" or inputs.get("source_commit") != commit:
        raise ValueError("Final evidence belongs to another source")
    if inputs.get("base_image") != framework_ref:
        raise ValueError("Final evidence did not inherit this run's framework digest")
    if inputs.get("versions") != {"vllm": "0.31.0", "vllm-omni": "0.31.0rc1"}:
        raise ValueError("Unexpected installed framework versions")
    if inputs.get("assets_lock_sha256") != sha256(root / PROFILE / "assets/assets.lock.json"):
        raise ValueError("Final evidence has another adapter lock")
    assets = read(directory / "experimental/assets-report.json")
    if assets.get("lock_sha256") != inputs["assets_lock_sha256"]:
        raise ValueError("Assets were checked against another lock")
    # Environment agent emits the final dependency audit in this directory.
    environment = read(directory / "experimental/environment-check.json")
    if environment.get("status") != "PASS_EXPERIMENTAL_ENVIRONMENT_CPU_ONLY":
        raise ValueError("Final full-Omni dependency/environment gate did not pass")
    media = read(directory / "experimental/media-smoke.json")
    if media.get("status") != "PASS_REAL_CPU_MEDIA_SMOKE":
        raise ValueError("Final real CPU media paths did not pass")
    licenses = read(directory / "experimental/dependency-license-inventory.json")
    if licenses.get("status") != "METADATA_INVENTORY_ONLY_NOT_LEGAL_CLEARANCE":
        raise ValueError("Missing explicitly scoped overlay dependency/license inventory")
    if not isinstance(licenses.get("packages"), list) or not licenses["packages"]:
        raise ValueError("Overlay package inventory is empty")
    if not isinstance(licenses.get("python_dependency_delta_from_inherited_vllm_plus_dingo"), list):
        raise ValueError("Missing delta from the gateway framework's dependency inventory")
    if not isinstance(licenses.get("media_bundled_files"), dict):
        raise ValueError("Missing media bundled-library inventory")
    if not (directory / "experimental/dpkg-packages.txt").is_file():
        raise ValueError("Missing final system package inventory")
    for name in ("legal/osrb-deps.csv", "legal/osrb.cdx.json"):
        if not (directory / name).is_file() or (directory / name).stat().st_size == 0:
            raise ValueError("Missing inherited gateway compliance report: " + name)
    for name, relative in (("assets.lock.json", "assets/assets.lock.json"),
                           ("profiles.json", "profiles/profiles.json"), ("Dockerfile", "Dockerfile")):
        target = directory / "inputs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        source = root / PROFILE / relative
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError("Conflicting evidence input: " + name)
        shutil.copyfile(source, target)
    shutil.copyfile(root / SOURCE_LOCK, directory / "inputs/source-lock.json")
    return inputs


def record_final(root, environ, digest, directory, framework_ref):
    values = prepare(root, environ)
    validate_evidence(root, environ, directory, framework_ref)
    if not DIGEST.fullmatch(digest):
        raise ValueError("Missing/invalid actual final image digest")
    files = {}
    for file in sorted(directory.rglob("*")):
        if file.is_symlink():
            raise ValueError("Symlink in exported evidence")
        if file.is_file() and file.name not in {"image.json", "deployment-image.txt"}:
            files[file.relative_to(directory).as_posix()] = sha256(file)
    record = {"schema": "h3-experimental-publication/1", "source_commit": identity(environ),
              "service_base_commit": SERVICE_BASE_SHA, "framework_image": framework_ref,
              "image": values["final_image"], "digest": digest,
              "deployment_image": REPOSITORY + "@" + digest,
              "run_id": environ.get("GITHUB_RUN_ID"), "run_attempt": environ.get("GITHUB_RUN_ATTEMPT"),
              "evidence_sha256": files,
              "validation_scope": "Image published and CPU gates passed; H100, Base mounts and media E2E still required",
              "license_scope": "Inherited gateway compliance plus overlay metadata/library inventory; not a final complete legal or native codec clearance"}
    dump(directory / "image.json", record)
    (directory / "deployment-image.txt").write_text(record["deployment_image"] + "\n", encoding="utf-8")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "record-framework", "validate-evidence", "record-final"])
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--directory", type=Path, default=Path(".h3-experimental-evidence"))
    parser.add_argument("--digest")
    parser.add_argument("--framework-ref")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(ROOT, os.environ, args.github_output, args.directory)
    elif args.command == "record-framework":
        result = record_framework(ROOT, os.environ, args.digest or "", args.directory, args.github_output)
    elif args.command == "validate-evidence":
        result = validate_evidence(ROOT, os.environ, args.directory, args.framework_ref or "")
    else:
        result = record_final(ROOT, os.environ, args.digest or "", args.directory, args.framework_ref or "")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
