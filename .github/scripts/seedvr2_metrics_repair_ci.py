#!/usr/bin/env python3
"""Validate and publish the exact source-only SeedVR2 transport metrics repair image."""
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
BRANCH = "Dingorouter-H3-vllm-omni-jirx"
REPOSITORY = "registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm"
PARENT = REPOSITORY + "@sha256:9547a5e924383dcaba4cdb8cc4fc9c5b5b3b80c24a5290edbd64e4cf96992d69"
PARENT_IMAGE_ID = "sha256:62671b8a4097a41d57ebd6aac79c6513fa40f3b34fc393e5ff17a340a6730272"
PARENT_COMMIT = "bd3f34bd37d6b82c52a5af0c2a7aa2349285a0bc"
HSDP_COMMIT = "430c62be0e38179535b4a6a1aa38dc67e8367d00"
PARENT_SOURCE_MANIFEST = "387886a6f99ac514aa6977419100e28c9b5ad18061a0225ddf91c7de3bdf5ceb"
REQUIRED_CHANGES = {"vllm_omni/diffusion/models/seedvr2/sr_telemetry.py",
    "vllm_omni/diffusion/models/seedvr2/long_video_support.py", "vllm_omni/diffusion/ipc.py"}
GATEWAY = "9ccee2f56d0122be94d7436ae2f5fec96d15ea9b"
LOCK_PATH = Path("delivery/seedvr2-metrics-repair/source-lock.json")
PATCH_PATH = Path("container/seedvr2-metrics-repair/source-patches.lock.json")
BUILD_INFO = "/opt/seedvr2-metrics-repair/build-info"

spec = importlib.util.spec_from_file_location("_sr_optimization_shared_ci", ROOT / ".github/scripts/seedvr2_c14_ci.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
read, dump, sha256 = base.read, base.dump, base.sha256
execute, docker, inspect = base.execute, base.docker, base.inspect
identity, env_map, write_outputs = base.identity, base.env_map, base.write_outputs
copy_from_image, publication_digest = base.copy_from_image, base.publication_digest

INVENTORY_PROBE = r'''
import importlib.metadata as m, json, os, pathlib, subprocess
packages = [{"name":d.metadata["Name"], "version":d.version,
 "metadata":d.read_text("METADATA"), "direct_url":d.read_text("direct_url.json")}
 for d in m.distributions()]
packages.sort(key=lambda x:(x["name"] or "",x["version"],x["metadata"] or "",x["direct_url"] or ""))
dpkg=subprocess.run(["dpkg-query","-W","-f=${binary:Package}\t${Version}\n"],check=True,capture_output=True,text=True)
print(json.dumps({"packages":packages,"dpkg":sorted(dpkg.stdout.splitlines()),"euid":os.geteuid(),
 "gateway_commit":os.environ.get("DYNAMO_GATEWAY_SOURCE_SHA"),
 "hsdp_source_commit":os.environ.get("H3_HSDP_SOURCE_COMMIT"),
 "optimization_source_commit":os.environ.get("SEEDVR2_OPT_SOURCE_COMMIT"),
 "source_commit":os.environ.get("SEEDVR2_METRICS_REPAIR_SOURCE_COMMIT"),
 "adapters_present":pathlib.Path("/opt/h3-adapters").exists()}))
'''

PARENT_SOURCE_PROBE = r'''
import hashlib,importlib.metadata as m,importlib.util,json,pathlib,sys
sys.path.insert(0,"/opt/seedvr2-optimization")
patch_spec=importlib.util.spec_from_file_location("source_patches","/opt/seedvr2-optimization/source_patches.py")
patch_module=importlib.util.module_from_spec(patch_spec);sys.modules[patch_spec.name]=patch_module;patch_spec.loader.exec_module(patch_module)
spec=importlib.util.spec_from_file_location("_verified_parent_hsdp", "/opt/seedvr2-optimization/runtime.py")
runtime=importlib.util.module_from_spec(spec);sys.modules[spec.name]=runtime;spec.loader.exec_module(runtime)
proof=runtime.verify_installed()
root=pathlib.Path(m.distribution("vllm-omni").locate_file("")).resolve()
files={}
for name,value in proof["source_sha256"].items():
 p=root/name
 if p.is_symlink() or not p.is_file():raise ValueError("Unsafe installed parent source: "+name)
 data=p.read_bytes()
 if hashlib.sha256(data).hexdigest()!=value:raise ValueError("Parent changed during snapshot")
 files[name]={"sha256":value,"bytes":len(data)}
 if name.endswith(("/sr_telemetry.py","/long_video_support.py","/ipc.py")):files[name]["text"]=data.decode("utf-8")
print(json.dumps({"proof":proof,"versions":{n:m.version(n) for n in ("vllm","vllm-omni","torch","cache-dit")},"files":files}))
'''


def hash_map(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_lock(lock):
    expected = {"schema":"seedvr2-metrics-repair-source-lock/1", "parent_image":PARENT,
        "parent_source_commit":PARENT_COMMIT, "recipe_base_commit":PARENT_COMMIT,
        "parent_image_id":PARENT_IMAGE_ID, "gateway_commit":GATEWAY,
        "parent_source_manifest_sha256":PARENT_SOURCE_MANIFEST,
        "remote_branch":BRANCH, "repository":"https://github.com/githubgxll/dynamo.git",
        "harbor_repository":REPOSITORY, "build_recipe":"seedvr2-metrics-repair", "platform":"linux/amd64",
        "tag_prefix":"omni031rc1-gateway-sr-metrics-repair",
        "gpu_validation_status":"NOT_RUN", "metrics_schema":"sr-timing-memory/2",
        "versions":{"vllm":"0.31.0", "vllm-omni":"0.31.0rc1", "torch":"2.13.0+cu130", "cache-dit":"1.5.0"}}
    for key,value in expected.items():
        if lock.get(key) != value:
            raise ValueError("Unreviewed source lock: " + key)


def patch_delta():
    value = read(ROOT / PATCH_PATH)
    if value.get("schema") != "seedvr2-metrics-repair-source-patches/1":
        raise ValueError("Unknown source delta schema")
    changes, additions = value.get("changes"), value.get("additions")
    if not isinstance(changes,dict) or set(changes)!=REQUIRED_CHANGES or additions!={}:
        raise ValueError("Missing exact source delta")
    if set(changes) & set(additions):
        raise ValueError("Changed and added paths overlap")
    for name,entry in [*changes.items(), *additions.items()]:
        if not isinstance(name,str) or not name.startswith("vllm_omni/") or ".." in Path(name).parts or "\\" in name:
            raise ValueError("Unsafe source delta path")
        if name in changes and not isinstance(entry,dict):
            raise ValueError("Expected changed-source hash object")
        hashes = [entry.get("before_sha256"),entry.get("after_sha256")] if name in changes else [entry]
        if any(not isinstance(v,str) or not re.fullmatch(r"[a-f0-9]{64}",v) for v in hashes):
            raise ValueError("Invalid source delta hash")
    return changes,additions


def probe(image,code):
    return json.loads(docker("run","--rm","--network","none","--read-only","--user","1000:0",
        "--tmpfs","/tmp:rw,nosuid,nodev,size=2g","--entrypoint","python3",image,"-B","-c",code))


def compare_images(parent,child,commit):
    if parent.get("Id") != PARENT_IMAGE_ID:
        raise ValueError("Parent config differs from successful immutable publication")
    for image in (parent,child):
        if image.get("Os") != "linux" or image.get("Architecture") != "amd64":
            raise ValueError("Expected linux/amd64")
    old,new = parent["Config"],child["Config"]
    for key in base.CONFIG_FIELDS:
        if old.get(key) != new.get(key):
            raise ValueError("Inherited config changed: " + key)
    before,after=env_map(old),env_map(new)
    if (before.get("H3_HSDP_SOURCE_COMMIT") != HSDP_COMMIT
            or before.get("SEEDVR2_OPT_SOURCE_COMMIT") != PARENT_COMMIT
            or before.get("DYNAMO_GATEWAY_SOURCE_SHA") != GATEWAY):
        raise ValueError("Parent inherited source markers differ")
    if after != {**before,"SEEDVR2_METRICS_REPAIR_SOURCE_COMMIT":commit,"SEEDVR2_METRICS_REPAIR_PARENT_IMAGE":PARENT}:
        raise ValueError("Unexpected environment mutation")
    layers,new_layers=parent.get("RootFS",{}).get("Layers",[]),child.get("RootFS",{}).get("Layers",[])
    if not layers or new_layers[:len(layers)] != layers or len(new_layers)<=len(layers):
        raise ValueError("Candidate must extend the complete immutable parent")
    labels=new.get("Labels") or {}
    for key,value in (old.get("Labels") or {}).items():
        if key not in {"org.opencontainers.image.title","org.opencontainers.image.revision","org.opencontainers.image.base.name"} and labels.get(key)!=value:
            raise ValueError("Inherited label changed: " + key)
    if labels.get("org.opencontainers.image.revision")!=commit or labels.get("org.opencontainers.image.base.name")!=PARENT:
        raise ValueError("Wrong final revision/base label")
    if "GPU_NOT_RUN" not in labels.get("org.opencontainers.image.title",""):
        raise ValueError("CPU publication must state GPU_NOT_RUN")
    return {"status":"PASS","parent_image_id":parent["Id"],"candidate_image_id":child["Id"],
        "inherited_layers":len(layers),"added_layers":len(new_layers)-len(layers),
        "parent_size_bytes":parent.get("Size"),"candidate_size_bytes":child.get("Size")}


def compare_inventories(parent,child,commit):
    for key in ("packages","dpkg"):
        if not parent.get(key) or parent[key]!=child.get(key):
            raise ValueError("Inherited dependency metadata changed: " + key)
    for item in (parent,child):
        if (item.get("gateway_commit")!=GATEWAY or item.get("hsdp_source_commit")!=HSDP_COMMIT
                or item.get("optimization_source_commit")!=PARENT_COMMIT
                or item.get("euid")!=1000 or item.get("adapters_present") is not False):
            raise ValueError("Inherited source, UID or adapter state changed")
    if child.get("source_commit")!=commit:
        raise ValueError("Wrong candidate source commit")
    return {"status":"PASS","python_distribution_count":len(parent["packages"]),"dpkg_package_count":len(parent["dpkg"])}


def validate_reports(directory,commit,parent_source=None):
    report=read(directory/"runtime-validation.json")
    expected={"schema":"seedvr2-metrics-repair-runtime/1","status":"PASS_CPU_ONLY","euid":1000,
        "parent_image":PARENT,"source_commit":commit,"gpu_validation_status":"NOT_RUN","retained_packages_unchanged":True,
        "metrics_schema":"sr-timing-memory/2"}
    for key,value in expected.items():
        if report.get(key)!=value:
            raise ValueError("Runtime gate failed: " + key)
    for name in ("source-verification.json","preservation.json","regression.json","torch-contract.json"):
        proof=read(directory/name)
        if proof.get("status")!="PASS" or proof.get("errors"):
            raise ValueError("Missing or failed gate: " + name)
    source=read(directory/"source-verification.json")
    files=source.get("source_sha256")
    if not isinstance(files,dict) or not files or any(not isinstance(n,str) or not isinstance(v,str) or not re.fullmatch(r"[a-f0-9]{64}",v) for n,v in files.items()):
        raise ValueError("Missing complete installed source hashes")
    if source.get("source_manifest_sha256")!=hash_map(files) or report.get("source_manifest_sha256")!=hash_map(files):
        raise ValueError("Installed source manifest identity differs")
    changes,additions=patch_delta()
    if sorted(source.get("changed_sources",[]))!=sorted(changes) or sorted(source.get("added_sources",[]))!=sorted(additions):
        raise ValueError("Unexpected declared source delta")
    for name,entry in changes.items():
        if files.get(name)!=entry["after_sha256"]:
            raise ValueError("Installed patched source differs: " + name)
    for name,value in additions.items():
        if files.get(name)!=value:
            raise ValueError("Installed helper differs: " + name)
    if parent_source is not None:
        target=dict(parent_source)
        for name,entry in changes.items():
            if target.get(name)!=entry["before_sha256"]:
                raise ValueError("Actual parent differs from source delta: " + name)
            target[name]=entry["after_sha256"]
        for name,value in additions.items():
            if name in target:
                raise ValueError("New helper would overwrite parent source")
            target[name]=value
        if target!=files:
            raise ValueError("Source changes outside the exact reviewed delta")
    regression=read(directory/"regression.json")
    if type(regression.get("tests_run")) is not int or regression["tests_run"]<1 or any(regression.get(k)!=0 for k in ("failures","errors","skipped")):
        raise ValueError("CPU regression suite incomplete")
    return report


def validate_candidate_reports(directory,commit):
    parent=read(directory/"parent-source.json")
    if parent.get("status")!="PASS" or parent.get("image_id")!=PARENT_IMAGE_ID:
        raise ValueError("Missing verified actual parent source snapshot")
    before={n:v["sha256"] for n,v in parent["files"].items()}
    if hash_map(before)!=PARENT_SOURCE_MANIFEST:
        raise ValueError("Archived parent source identity differs from the published parent")
    built=validate_reports(directory/"build",commit,before)
    final=validate_reports(directory/"final",commit,before)
    if built["source_manifest_sha256"]!=final["source_manifest_sha256"]:
        raise ValueError("Final container sources differ from built candidate")
    return final


def candidate_tag(commit,environ):
    return "seedvr2-metrics-repair-ci:"+commit[:12]+"-"+environ["GITHUB_RUN_ID"]+"-"+environ["GITHUB_RUN_ATTEMPT"]


def final_tag(commit):
    return REPOSITORY+":omni031rc1-gateway-sr-metrics-repair-"+commit[:12]


def prepare(environ,directory,output=None):
    commit=identity(environ)
    validate_lock(read(ROOT/LOCK_PATH))
    patch_delta()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Use a fresh evidence directory")
    directory.mkdir(parents=True,exist_ok=True)
    execute([sys.executable,"-B",str(ROOT/"delivery/seedvr2-metrics-repair/ops.py"),"check-seal","--committed"],directory/"input-seal-check.txt")
    values={"registry":REPOSITORY.split("/",1)[0],"parent_image":PARENT,
        "candidate_image":candidate_tag(commit,environ),"final_image":final_tag(commit),"dockerfile":"container/seedvr2-metrics-repair/Dockerfile"}
    dump(directory/"inputs.json",{"schema":"seedvr2-metrics-repair-ci-inputs/1","source_commit":commit,
        "run_id":environ["GITHUB_RUN_ID"],"run_attempt":environ["GITHUB_RUN_ATTEMPT"],
        "source_lock_sha256":sha256(ROOT/LOCK_PATH),"patch_lock_sha256":sha256(ROOT/PATCH_PATH),**values})
    execute([sys.executable,"-B",str(ROOT/"delivery/seedvr2-metrics-repair/ops.py"),"export-recipe","--directory",str(directory/"recipe")],directory/"recipe-export.txt")
    write_outputs(output,values)


def checked_inputs(environ,directory):
    commit=identity(environ)
    value=read(directory/"inputs.json")
    expected={"source_commit":commit,"parent_image":PARENT,"run_id":environ["GITHUB_RUN_ID"],
        "run_attempt":environ["GITHUB_RUN_ATTEMPT"],"source_lock_sha256":sha256(ROOT/LOCK_PATH),
        "patch_lock_sha256":sha256(ROOT/PATCH_PATH),"candidate_image":candidate_tag(commit,environ),"final_image":final_tag(commit)}
    if any(value.get(k)!=v for k,v in expected.items()):
        raise ValueError("Evidence does not belong to this run/recipe")
    return value


def snapshot_parent(environ,directory):
    checked_inputs(environ,directory)
    image=inspect(PARENT)
    if image["Id"]!=PARENT_IMAGE_ID:
        raise ValueError("Parent image ID differs from successful publication")
    result=probe(image["Id"],PARENT_SOURCE_PROBE)
    if result.get("versions")!=read(ROOT/LOCK_PATH)["versions"] or result.get("proof",{}).get("status")!="PASS":
        raise ValueError("Actual parent versions/source verification failed")
    files=result["files"]
    manifest={}
    for name,entry in files.items():
        path=Path(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
            raise ValueError("Unsafe actual source path")
        manifest[name]={"sha256":entry["sha256"],"bytes":entry["bytes"]}
        if "text" in entry:
            data=entry["text"].encode("utf-8")
            if hashlib.sha256(data).hexdigest()!=entry["sha256"] or len(data)!=entry["bytes"]:
                raise ValueError("Actual source export differs")
            target=directory/"parent-source"/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    before={name:entry["sha256"] for name,entry in files.items()}
    if result["proof"].get("source_sha256")!=before:
        raise ValueError("Parent source manifest changed during probe")
    if hash_map(before)!=PARENT_SOURCE_MANIFEST or result["proof"].get("source_manifest_sha256")!=PARENT_SOURCE_MANIFEST:
        raise ValueError("Actual parent source manifest differs from the published accepted candidate")
    changes,additions=patch_delta()
    if any(before.get(n)!=entry["before_sha256"] for n,entry in changes.items()) or set(additions)&set(before):
        raise ValueError("Parent does not match this exact patch recipe")
    dump(directory/"parent-source.json",{"status":"PASS","parent_image":PARENT,"image_id":image["Id"],
        "versions":result["versions"],"files":manifest,"parent_verification":result["proof"]})


def run_final_validation(image,directory):
    cid=docker("create","--network","none","--user","1000:0","--env","TMPDIR=/tmp",
        "--entrypoint","python3",image,"-B","/opt/seedvr2-metrics-repair/runtime.py","validate",
        "--output-dir","/tmp/seedvr2-metrics-repair-final").strip()
    try:
        failed=None
        try:docker("start","--attach",cid,log=directory/"final-container.log")
        except RuntimeError as error:failed=error
        state=json.loads(docker("container","inspect",cid))[0]["State"]
        target=directory/"final";target.mkdir(parents=True,exist_ok=False)
        try:docker("cp",cid+":/tmp/seedvr2-metrics-repair-final/.",str(target))
        except RuntimeError:
            if not failed and state.get("ExitCode")==0:raise
        dump(directory/"final-container-state.json",{"exit_code":state.get("ExitCode"),"oom_killed":state.get("OOMKilled")})
        if failed or state.get("ExitCode")!=0 or state.get("OOMKilled"):
            raise RuntimeError("Final nonroot offline CPU validation failed") from failed
    finally:docker("rm",cid)


def verify_candidate(environ,directory):
    inputs=checked_inputs(environ,directory);commit=inputs["source_commit"]
    parent,candidate=inspect(PARENT),inspect(inputs["candidate_image"])
    dump(directory/"inheritance.json",compare_images(parent,candidate,commit))
    old,new=probe(parent["Id"],INVENTORY_PROBE),probe(candidate["Id"],INVENTORY_PROBE)
    dump(directory/"parent-inventory.json",old);dump(directory/"candidate-inventory.json",new)
    dump(directory/"package-preservation.json",compare_inventories(old,new,commit))
    copy_from_image(candidate["Id"],BUILD_INFO,directory/"build")
    validate_reports(directory/"build",commit)
    run_final_validation(candidate["Id"],directory)
    final=validate_candidate_reports(directory,commit)
    dump(directory/"candidate-verified.json",{"status":"PASS_CANDIDATE_CPU_ONLY","source_commit":commit,
        "candidate_image":inputs["candidate_image"],"parent_image":PARENT,"image_id":candidate["Id"],
        "source_manifest_sha256":final["source_manifest_sha256"],"patch_lock_sha256":sha256(ROOT/PATCH_PATH),"gpu_validation_status":"NOT_RUN"})


def publish(environ,directory,output=None):
    inputs=checked_inputs(environ,directory);commit=inputs["source_commit"]
    verified=read(directory/"candidate-verified.json")
    expected={"status":"PASS_CANDIDATE_CPU_ONLY","source_commit":commit,"candidate_image":inputs["candidate_image"],
        "parent_image":PARENT,"gpu_validation_status":"NOT_RUN","patch_lock_sha256":sha256(ROOT/PATCH_PATH)}
    if any(verified.get(k)!=v for k,v in expected.items()):
        raise ValueError("Candidate has not passed this recipe's CPU gates")
    image_id=verified["image_id"]
    if inspect(inputs["candidate_image"])["Id"]!=image_id:
        raise ValueError("Candidate tag changed after validation")
    final=validate_candidate_reports(directory,commit)
    if verified.get("source_manifest_sha256")!=final["source_manifest_sha256"]:
        raise ValueError("Verified source identity changed")
    docker("tag",image_id,inputs["final_image"])
    docker("push",inputs["final_image"],log=directory/"registry-push.txt")
    digest=publication_digest((directory/"registry-push.txt").read_text(encoding="utf-8"))
    reference=REPOSITORY+"@"+digest
    docker("pull","--platform","linux/amd64",reference,log=directory/"registry-pull.txt")
    pulled=inspect(reference)
    if pulled["Id"]!=image_id:raise ValueError("Registry pull differs from tested candidate")
    compare_images(inspect(PARENT),pulled,commit)
    record={"schema":"seedvr2-metrics-repair-publication/1","status":"PASS_PUBLISHED_AND_PULLED_CPU_VALIDATED",
        "source_commit":commit,"branch":BRANCH,"image":inputs["final_image"],"image_id":image_id,
        "deployment_image":reference,"digest":digest,"parent_image":PARENT,"parent_source_commit":PARENT_COMMIT,
        "source_manifest_sha256":final["source_manifest_sha256"],"gateway_commit":GATEWAY,"gpu_validation_status":"NOT_RUN",
        "run_id":environ["GITHUB_RUN_ID"],"run_attempt":environ["GITHUB_RUN_ATTEMPT"],
        "actions_url":"https://github.com/githubgxll/dynamo/actions/runs/"+environ["GITHUB_RUN_ID"],
        "not_run":["CUDA events and 8-rank identity","off/basic perturbation","17/33/65 capacity and inference","same-source legacy/single media","human quality acceptance"]}
    record["evidence_sha256"]={p.relative_to(directory).as_posix():sha256(p) for p in sorted(directory.rglob("*")) if p.is_file() and p.name not in {"image.json","deployment-image.txt"}}
    dump(directory/"image.json",record)
    (directory/"deployment-image.txt").write_text(reference+"\n",encoding="utf-8")
    write_outputs(output,{"digest":digest,"deployment_image":reference})
    print(json.dumps(record,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","snapshot-parent","verify-candidate","publish"))
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--github-output",type=Path)
    args=parser.parse_args();directory=args.directory.resolve()
    if args.command=="prepare":prepare(os.environ,directory,args.github_output)
    elif args.command=="snapshot-parent":snapshot_parent(os.environ,directory)
    elif args.command=="verify-candidate":verify_candidate(os.environ,directory)
    else:publish(os.environ,directory,args.github_output)


if __name__=="__main__":main()
