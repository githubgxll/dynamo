#!/usr/bin/env python3
"""Publish an exact CPU-validated HSDP candidate; GPU acceptance remains separate."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = Path("delivery/h3-hsdp-offload/source-lock.json")
BRANCH = "Dingorouter-H3-vllm-omni-jirx"
REPOSITORY = "registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm"
PARENT = REPOSITORY + "@sha256:87365073cbdf95fa17ca352a24a93cb74fde144a47c96135e91011d1a03df72e"
PARENT_COMMIT = "0b11a41b083facf74d3387696a72893d0d3e40e1"
PARENT_IMAGE_ID = "sha256:c1298a2a8403e20b689bb73c380d0f9f74fec8cc698bf9461a0721361af87e99"
GATEWAY = "9ccee2f56d0122be94d7436ae2f5fec96d15ea9b"
BUILD_INFO = "/opt/h3-hsdp-offload/build-info"
BACKEND = "vllm_omni/diffusion/offloader/sequential_backend.py"
HELPER = "vllm_omni/diffusion/offloader/hsdp_migration.py"
PIPELINE = "vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py"
# The accepted candidate combines the previous migration fix with the DiT
# entry fix. An old HSDP-only PASS must not authorize this publication.
PATCHED_SOURCE_SHA256 = {
    BACKEND: "17c3ea33434e924b78eea98d28f3a75032c73e9e175af46a625b0980557e1e08",
    HELPER: "b1656d629dd27d1fc7295f320ba2bfd5aea9178b0a6a16a52089f6565f0da12a",
    PIPELINE: "9a2b75e9989a4fdb3dcd09094250f82a39d8cc9956c6773481e3c9cf969c0024",
}
spec = importlib.util.spec_from_file_location("_hsdp_parent_ci", ROOT / ".github/scripts/seedvr2_c14_ci.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
read, dump, sha256 = base.read, base.dump, base.sha256
execute, docker, inspect = base.execute, base.docker, base.inspect
identity, env_map, write_outputs = base.identity, base.env_map, base.write_outputs
copy_from_image, publication_digest = base.copy_from_image, base.publication_digest

INVENTORY_PROBE = r'''
import hashlib, importlib.metadata as m, json, os, pathlib, subprocess
packages = [{"name": d.metadata["Name"], "version": d.version,
 "metadata": d.read_text("METADATA"), "direct_url": d.read_text("direct_url.json")}
 for d in m.distributions()]
packages.sort(key=lambda x: (x["name"] or "", x["version"], x["metadata"] or "", x["direct_url"] or ""))
dpkg = subprocess.run(["dpkg-query", "-W", "-f=${binary:Package}\t${Version}\n"], check=True, capture_output=True, text=True)
print(json.dumps({"packages": packages, "dpkg": sorted(dpkg.stdout.splitlines()),
 "gateway_commit": os.environ.get("DYNAMO_GATEWAY_SOURCE_SHA"),
 "c14_source_commit": os.environ.get("H3_C14_SOURCE_COMMIT"),
 "source_commit": os.environ.get("H3_HSDP_SOURCE_COMMIT"),
 "adapters_present": pathlib.Path("/opt/h3-adapters").exists(), "euid": os.geteuid()}))
'''
SOURCE_PROBE = r'''
import hashlib, importlib.metadata as m, json, pathlib
sources = {}
for dist_name, paths in (("vllm-omni", ["vllm_omni/diffusion/offloader/sequential_backend.py", "vllm_omni/diffusion/distributed/hsdp.py", "vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py"]), ("torch", [])):
 root = pathlib.Path(m.distribution(dist_name).locate_file("")).resolve()
 if dist_name == "torch":
  paths = [p.relative_to(root).as_posix() for p in (root / "torch/distributed/fsdp").rglob("*.py")] + ["torch/version.py", "torch/nn/modules/module.py"]
 for relative in paths:
  path = root / relative
  if path.is_symlink() or not path.is_file(): raise RuntimeError("Missing or symlink installed source: " + relative)
  data = path.read_bytes()
  sources[relative] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "text": data.decode("utf-8")}
print(json.dumps({"versions": {n:m.version(n) for n in ("vllm", "vllm-omni", "torch", "cache-dit")}, "sources": sources}))
'''

def validate_lock(lock):
    expected = {"schema": "h3-hsdp-offload-source-lock/1", "parent_image": PARENT,
        "parent_source_commit": PARENT_COMMIT, "parent_image_id": PARENT_IMAGE_ID,
        "recipe_base_commit": PARENT_COMMIT, "gateway_commit": GATEWAY,
        "remote_branch": BRANCH, "repository": "https://github.com/githubgxll/dynamo.git",
        "harbor_repository": REPOSITORY, "build_recipe": "h3-hsdp-offload",
        "tag_prefix": "omni031rc1-gateway-hsdp-offload", "platform": "linux/amd64",
        "gpu_validation_status": "NOT_RUN",
        "versions": {"vllm":"0.31.0", "vllm-omni":"0.31.0rc1", "torch":"2.13.0+cu130", "cache-dit":"1.5.0"}}
    for key, value in expected.items():
        if lock.get(key) != value:
            raise ValueError("Unreviewed source lock: " + key)
    archived = read(ROOT / "delivery/h3-hsdp-offload/parent-evidence/image-release-evidence.json")
    if lock.get("parent_source_sha256") != archived.get("parent_source_sha256"):
        raise ValueError("Parent source hashes differ from archived successful image")

def probe(image, code):
    return json.loads(docker("run", "--rm", "--network", "none", "--read-only", "--user", "1000:0",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=2g", "--entrypoint", "python3", image, "-B", "-c", code))

def compare_images(parent, child, commit):
    if parent.get("Id") != PARENT_IMAGE_ID:
        raise ValueError("Unexpected parent image ID")
    for image in (parent, child):
        if image.get("Os") != "linux" or image.get("Architecture") != "amd64":
            raise ValueError("Expected linux/amd64")
    old, new = parent["Config"], child["Config"]
    for key in base.CONFIG_FIELDS:
        if old.get(key) != new.get(key):
            raise ValueError("Inherited config changed: " + key)
    before, after = env_map(old), env_map(new)
    if before.get("H3_C14_SOURCE_COMMIT") != PARENT_COMMIT or before.get("DYNAMO_GATEWAY_SOURCE_SHA") != GATEWAY:
        raise ValueError("Unexpected inherited source identity")
    expected = {**before, "H3_HSDP_SOURCE_COMMIT":commit, "H3_HSDP_PARENT_IMAGE":PARENT}
    if after != expected:
        raise ValueError("Unexpected environment mutation")
    layers, added = parent.get("RootFS", {}).get("Layers", []), child.get("RootFS", {}).get("Layers", [])
    if not layers or added[:len(layers)] != layers or len(added) <= len(layers):
        raise ValueError("Candidate must extend entire immutable parent")
    labels = new.get("Labels") or {}
    for key, value in (old.get("Labels") or {}).items():
        if key not in {"org.opencontainers.image.title","org.opencontainers.image.revision","org.opencontainers.image.base.name"} and labels.get(key) != value:
            raise ValueError("Inherited label changed: " + key)
    if labels.get("org.opencontainers.image.revision") != commit or labels.get("org.opencontainers.image.base.name") != PARENT:
        raise ValueError("Wrong image revision/base label")
    if "GPU_NOT_RUN" not in labels.get("org.opencontainers.image.title", ""):
        raise ValueError("Candidate image must declare GPU_NOT_RUN")
    return {"status":"PASS", "parent_image_id":parent["Id"], "candidate_image_id":child["Id"],
        "inherited_layers":len(layers), "added_layers":len(added)-len(layers)}

def compare_inventories(parent, child, commit):
    for key in ("packages", "dpkg"):
        if not parent.get(key) or parent[key] != child.get(key):
            raise ValueError("Inherited dependency metadata changed: " + key)
    for item in (parent, child):
        if item.get("gateway_commit") != GATEWAY or item.get("c14_source_commit") != PARENT_COMMIT or item.get("euid") != 1000 or item.get("adapters_present") is not False:
            raise ValueError("Inherited source, UID or adapter state changed")
    if child.get("source_commit") != commit:
        raise ValueError("Wrong final source marker")
    return {"status":"PASS", "python_distribution_count":len(parent["packages"]), "dpkg_package_count":len(parent["dpkg"])}

def validate_reports(directory, commit):
    report = read(directory / "runtime-validation.json")
    expected = {"schema":"h3-hsdp-offload-runtime/1", "status":"PASS_CPU_ONLY", "euid":1000,
        "parent_image":PARENT, "source_commit":commit, "gpu_validation_status":"NOT_RUN",
        "retained_packages_unchanged":True, "dit_entry_status":"PASS"}
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError("Runtime gate failed: " + key)
    for name in ("source-verification.json", "preservation.json", "regression.json", "torch-contract.json"):
        result = read(directory / name)
        if result.get("status") != "PASS" or result.get("errors"):
            raise ValueError("Missing/failed gate: " + name)
    source = read(directory / "source-verification.json")
    source_hash = source.get("source_manifest_sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", source_hash) or report.get("source_manifest_sha256") != source_hash:
        raise ValueError("Runtime receipt not bound to installed source manifest")
    installed = source.get("source_sha256")
    if not isinstance(installed, dict) or not installed or any(
            not isinstance(name, str) or not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
            for name, value in installed.items()):
        raise ValueError("Missing complete installed source manifest")
    actual_hash = hashlib.sha256(json.dumps(installed, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if actual_hash != source_hash:
        raise ValueError("Installed source manifest digest differs from its contents")
    if source.get("changed_sources") != [BACKEND, PIPELINE] or source.get("added_sources") != [HELPER]:
        raise ValueError("Source delta does not include the exact HSDP and DiT entry repair")
    for name, expected_hash in PATCHED_SOURCE_SHA256.items():
        if installed.get(name) != expected_hash:
            raise ValueError("Required installed repair source differs: " + name)
    regression = read(directory / "regression.json")
    if type(regression.get("tests_run")) is not int or regression["tests_run"] < 1 or any(regression.get(key) != 0 for key in ("failures","errors","skipped")):
        raise ValueError("Required CPU regression tests did not all execute")
    return report

def validate_candidate_reports(directory, commit):
    build = validate_reports(directory / "build", commit)
    final = validate_reports(directory / "final", commit)
    if build["source_manifest_sha256"] != final["source_manifest_sha256"]:
        raise ValueError("Final container sources differ from the built candidate")
    return final

def candidate_tag(commit, env):
    return "h3-hsdp-offload-ci:" + commit[:12] + "-" + env["GITHUB_RUN_ID"] + "-" + env["GITHUB_RUN_ATTEMPT"]

def final_tag(commit):
    return REPOSITORY + ":omni031rc1-gateway-hsdp-offload-" + commit[:12]

def prepare(environ, directory, output=None):
    commit = identity(environ)
    validate_lock(read(ROOT / LOCK_PATH))
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Use a fresh evidence directory")
    directory.mkdir(parents=True, exist_ok=True)
    execute([sys.executable,"-B",str(ROOT / "delivery/h3-hsdp-offload/ops.py"),"check-seal","--committed"], directory / "input-seal-check.txt")
    values = {"registry":REPOSITORY.split("/",1)[0], "parent_image":PARENT,
        "candidate_image":candidate_tag(commit,environ), "final_image":final_tag(commit),
        "dockerfile":"container/h3-hsdp-offload/Dockerfile"}
    dump(directory / "inputs.json", {"schema":"h3-hsdp-offload-ci-inputs/1", "source_commit":commit,
        "run_id":environ["GITHUB_RUN_ID"], "run_attempt":environ["GITHUB_RUN_ATTEMPT"],
        "source_lock_sha256":sha256(ROOT / LOCK_PATH), **values})
    execute([sys.executable,"-B",str(ROOT / "delivery/h3-hsdp-offload/ops.py"),"export-recipe","--directory",str(directory / "recipe")], directory / "recipe-export.txt")
    write_outputs(output, values)

def checked_inputs(environ, directory):
    commit = identity(environ)
    value = read(directory / "inputs.json")
    expected = {"source_commit":commit, "parent_image":PARENT,"run_id":environ["GITHUB_RUN_ID"],
        "run_attempt":environ["GITHUB_RUN_ATTEMPT"],"source_lock_sha256":sha256(ROOT / LOCK_PATH),
        "candidate_image":candidate_tag(commit,environ),"final_image":final_tag(commit)}
    if any(value.get(key) != item for key,item in expected.items()):
        raise ValueError("Evidence does not belong to this run/recipe")
    return value

def snapshot_parent(environ, directory):
    checked_inputs(environ, directory)
    image = inspect(PARENT)
    if image["Id"] != PARENT_IMAGE_ID:
        raise ValueError("Parent image ID differs from successful publication")
    source = probe(image["Id"], SOURCE_PROBE)
    lock = read(ROOT / LOCK_PATH)
    if source.get("versions") != lock["versions"]:
        raise ValueError("Actual parent package versions differ")
    files = source["sources"]
    for name, expected in lock["parent_source_sha256"].items():
        if files.get(name, {}).get("sha256") != expected:
            raise ValueError("Actual installed parent differs: " + name)
    for required in ("torch/distributed/fsdp/_fully_shard/_fsdp_param.py", "torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py"):
        if required not in files:
            raise ValueError("Missing actual FSDP implementation: " + required)
    manifest = {}
    for name, entry in files.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
            raise ValueError("Unsafe installed source path")
        data = entry["text"].encode("utf-8")
        if hashlib.sha256(data).hexdigest() != entry["sha256"] or len(data) != entry["bytes"]:
            raise ValueError("Source export bytes differ: " + name)
        target = directory / "parent-source" / path
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(data)
        manifest[name] = {"sha256":entry["sha256"],"bytes":len(data)}
    dump(directory / "parent-source.json", {"status":"PASS", "image_id":image["Id"], "parent_image":PARENT,
        "versions":source["versions"], "files":manifest,"scope":"Actual image sources exported before patch; no GPU claim"})

def run_final_validation(image, directory):
    cid = docker("create","--network","none","--user","1000:0","--env","TMPDIR=/tmp",
        "--entrypoint","python3",image,"-B","/opt/h3-hsdp-offload/runtime.py","validate",
        "--output-dir","/tmp/hsdp-final").strip()
    try:
        failed = None
        try:
            docker("start","--attach",cid,log=directory / "final-container.log")
        except RuntimeError as exc:
            failed = exc
        state = json.loads(docker("container","inspect",cid))[0]["State"]
        target = directory / "final"
        target.mkdir(parents=True,exist_ok=False)
        try:
            docker("cp",cid + ":/tmp/hsdp-final/.",str(target))
        except RuntimeError:
            if not failed and state.get("ExitCode") == 0: raise
        dump(directory / "final-container-state.json", {"exit_code":state.get("ExitCode"),"oom_killed":state.get("OOMKilled")})
        if failed or state.get("ExitCode") != 0 or state.get("OOMKilled"):
            raise RuntimeError("Final nonroot offline CPU candidate validation failed") from failed
    finally:
        docker("rm",cid)

def verify_candidate(environ, directory):
    inputs = checked_inputs(environ,directory)
    proof = read(directory / "parent-source.json")
    if proof.get("status") != "PASS" or proof.get("image_id") != PARENT_IMAGE_ID:
        raise ValueError("Missing actual parent source snapshot")
    commit = inputs["source_commit"]
    parent, candidate = inspect(PARENT), inspect(inputs["candidate_image"])
    dump(directory / "inheritance.json", compare_images(parent,candidate,commit))
    old, new = probe(parent["Id"], INVENTORY_PROBE), probe(candidate["Id"], INVENTORY_PROBE)
    dump(directory / "parent-inventory.json",old)
    dump(directory / "candidate-inventory.json",new)
    dump(directory / "package-preservation.json",compare_inventories(old,new,commit))
    copy_from_image(candidate["Id"],BUILD_INFO,directory / "build")
    validate_reports(directory / "build",commit)
    run_final_validation(candidate["Id"],directory)
    validate_candidate_reports(directory,commit)
    dump(directory / "candidate-verified.json", {"status":"PASS_CANDIDATE_CPU_ONLY","image_id":candidate["Id"],
        "source_commit":commit,"candidate_image":inputs["candidate_image"],"parent_image":PARENT,"gpu_validation_status":"NOT_RUN",
        "dit_entry_status":"PASS", "patched_source_sha256":PATCHED_SOURCE_SHA256})

def publish(environ, directory, output=None):
    inputs = checked_inputs(environ,directory)
    verified = read(directory / "candidate-verified.json")
    commit = inputs["source_commit"]
    expected = {"status":"PASS_CANDIDATE_CPU_ONLY","source_commit":commit,"candidate_image":inputs["candidate_image"],"parent_image":PARENT,"gpu_validation_status":"NOT_RUN",
        "dit_entry_status":"PASS", "patched_source_sha256":PATCHED_SOURCE_SHA256}
    if any(verified.get(k) != v for k,v in expected.items()):
        raise ValueError("Candidate has not passed CPU validation")
    image_id = verified["image_id"]
    if inspect(inputs["candidate_image"])["Id"] != image_id:
        raise ValueError("Candidate tag changed after verification")
    validate_candidate_reports(directory,commit)
    docker("tag",image_id,inputs["final_image"])
    docker("push",inputs["final_image"],log=directory / "registry-push.txt")
    digest = publication_digest((directory / "registry-push.txt").read_text(encoding="utf-8"))
    reference = REPOSITORY + "@" + digest
    docker("pull","--platform","linux/amd64",reference,log=directory / "registry-pull.txt")
    pulled = inspect(reference)
    if pulled["Id"] != image_id: raise ValueError("Registry image differs from tested image")
    compare_images(inspect(PARENT),pulled,commit)
    record = {"schema":"h3-hsdp-offload-publication/1","status":"PASS_PUBLISHED_AND_PULLED_CPU_VALIDATED",
        "dit_entry_status":"PASS", "patched_source_sha256":PATCHED_SOURCE_SHA256,
        "gpu_validation_status":"NOT_RUN", "source_commit":commit,"branch":BRANCH,"image":inputs["final_image"],
        "image_id":image_id,"deployment_image":reference,"digest":digest,"parent_image":PARENT,
        "parent_source_commit":PARENT_COMMIT,"gateway_commit":GATEWAY,
        "run_id":environ["GITHUB_RUN_ID"],"run_attempt":environ["GITHUB_RUN_ATTEMPT"],
        "actions_url":"https://github.com/githubgxll/dynamo/actions/runs/"+environ["GITHUB_RUN_ID"],
        "not_run":["Real CUDA HSDP migration","8xH100 C02 cold/warm/repeat","Ref7","GPU memory and CPU memory acceptance","generated media acceptance"]}
    record["evidence_sha256"] = {p.relative_to(directory).as_posix():sha256(p) for p in sorted(directory.rglob("*")) if p.is_file() and p.name not in {"image.json","deployment-image.txt"}}
    dump(directory / "image.json",record)
    (directory / "deployment-image.txt").write_text(reference+"\n",encoding="utf-8")
    write_outputs(output,{"digest":digest,"deployment_image":reference})
    print(json.dumps(record,indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare","snapshot-parent","verify-candidate","publish"))
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--github-output",type=Path)
    args=parser.parse_args(); directory=args.directory.resolve()
    if args.command == "prepare": prepare(os.environ,directory,args.github_output)
    elif args.command == "snapshot-parent": snapshot_parent(os.environ,directory)
    elif args.command == "verify-candidate": verify_candidate(os.environ,directory)
    else: publish(os.environ,directory,args.github_output)

if __name__ == "__main__": main()
