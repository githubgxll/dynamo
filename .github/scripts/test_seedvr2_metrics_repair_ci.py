"""Publication rejection tests; no Docker daemon, registry or CUDA side effects."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

spec=importlib.util.spec_from_file_location("sr_optimization_ci",Path(__file__).with_name("seedvr2_metrics_repair_ci.py"))
ci=importlib.util.module_from_spec(spec);spec.loader.exec_module(ci)
COMMIT="a"*40
ENV={"GITHUB_SHA":COMMIT,"GITHUB_REF_NAME":ci.BRANCH,"GITHUB_EVENT_NAME":"push","GITHUB_RUN_ID":"123","GITHUB_RUN_ATTEMPT":"1"}
CHANGED="vllm_omni/diffusion/models/seedvr2/sr_telemetry.py"
IPC="vllm_omni/diffusion/ipc.py"
SUPPORT="vllm_omni/diffusion/models/seedvr2/long_video_support.py"
DELTA=({name:{"before_sha256":"1"*64,"after_sha256":"2"*64} for name in (CHANGED,IPC,SUPPORT)},{})
PARENT_SOURCES={**{name:"1"*64 for name in DELTA[0]},"vllm_omni/retained.py":"4"*64}


def images():
    parent={"Id":ci.PARENT_IMAGE_ID,"Os":"linux","Architecture":"amd64","RootFS":{"Layers":["inherited"]},
        "Config":{"User":"dynamo","Entrypoint":["python"],"WorkingDir":"/workspace",
        "Env":["PATH=/usr/bin","H3_HSDP_SOURCE_COMMIT="+ci.HSDP_COMMIT,"SEEDVR2_OPT_SOURCE_COMMIT="+ci.PARENT_COMMIT,"DYNAMO_GATEWAY_SOURCE_SHA="+ci.GATEWAY],"Labels":{"retain":"yes"}}}
    child=copy.deepcopy(parent);child["Id"]="sha256:"+"b"*64;child["RootFS"]["Layers"].append("new")
    child["Config"]["Env"] += ["SEEDVR2_METRICS_REPAIR_SOURCE_COMMIT="+COMMIT,"SEEDVR2_METRICS_REPAIR_PARENT_IMAGE="+ci.PARENT]
    child["Config"]["Labels"].update({"org.opencontainers.image.title":"SeedVR2 CPU candidate GPU_NOT_RUN",
        "org.opencontainers.image.revision":COMMIT,"org.opencontainers.image.base.name":ci.PARENT})
    return parent,child


def inventory():
    old={"packages":[{"name":"torch","version":"2.13.0+cu130","metadata":"exact Requires-Dist","direct_url":None}],
        "dpkg":["libc\texact"],"gateway_commit":ci.GATEWAY,"hsdp_source_commit":ci.HSDP_COMMIT,
        "optimization_source_commit":ci.PARENT_COMMIT,"euid":1000,"adapters_present":False}
    return old,{**old,"source_commit":COMMIT}


def reports(root):
    sources={**PARENT_SOURCES,**{name:item["after_sha256"] for name,item in DELTA[0].items()}}
    ci.dump(root/"runtime-validation.json",{"schema":"seedvr2-metrics-repair-runtime/1","status":"PASS_CPU_ONLY","euid":1000,
        "parent_image":ci.PARENT,"source_commit":COMMIT,"gpu_validation_status":"NOT_RUN",
        "retained_packages_unchanged":True,"source_manifest_sha256":ci.hash_map(sources),"metrics_schema":"sr-timing-memory/2"})
    ci.dump(root/"source-verification.json",{"status":"PASS","source_sha256":sources,"source_manifest_sha256":ci.hash_map(sources),
        "changed_sources":sorted(DELTA[0]),"added_sources":[]})
    for name in ("preservation.json","torch-contract.json"):
        ci.dump(root/name,{"status":"PASS"})
    ci.dump(root/"regression.json",{"status":"PASS","tests_run":20,"failures":0,"errors":0,"skipped":0})


def rewrite_source(root,sources):
    source=ci.read(root/"source-verification.json");source.update(source_sha256=sources,source_manifest_sha256=ci.hash_map(sources))
    ci.dump(root/"source-verification.json",source)
    runtime=ci.read(root/"runtime-validation.json");runtime["source_manifest_sha256"]=ci.hash_map(sources)
    ci.dump(root/"runtime-validation.json",runtime)


class Publication(unittest.TestCase):
    def test_exact_config_and_inventory(self):
        ci.compare_images(*images(),COMMIT);ci.compare_inventories(*inventory(),COMMIT)

    def test_wrong_parent_or_rebase_rejected(self):
        parent,child=images();parent["Id"]="sha256:"+"c"*64
        with self.assertRaises(ValueError):ci.compare_images(parent,child,COMMIT)
        parent,child=images();child["RootFS"]["Layers"]=["other"]
        with self.assertRaises(ValueError):ci.compare_images(parent,child,COMMIT)

    def test_platform_and_entrypoint_preserved(self):
        for field,value in (("User","root"),("Entrypoint",["bash"]),("WorkingDir","/tmp"),("Cmd",["new"])):
            parent,child=images();child["Config"][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):ci.compare_images(parent,child,COMMIT)
        parent,child=images();child["Architecture"]="arm64"
        with self.assertRaises(ValueError):ci.compare_images(parent,child,COMMIT)

    def test_no_injected_runtime_controls_or_proxy(self):
        for setting in ("SEEDVR2_METRICS_DIR=/tmp/active","HTTP_PROXY=http://host","H3_FASTPATH=0"):
            parent,child=images();child["Config"]["Env"].append(setting)
            with self.subTest(setting=setting),self.assertRaises(ValueError):ci.compare_images(parent,child,COMMIT)

    def test_inherited_identity_and_labels_preserved(self):
        for setting in ("H3_HSDP_SOURCE_COMMIT=wrong","SEEDVR2_OPT_SOURCE_COMMIT=wrong","DYNAMO_GATEWAY_SOURCE_SHA=wrong"):
            parent,child=images();parent["Config"]["Env"].append(setting)
            with self.assertRaises(ValueError):ci.compare_images(parent,child,COMMIT)
        parent,child=images();child["Config"]["Labels"]["retain"]="mutated"
        with self.assertRaises(ValueError):ci.compare_images(parent,child,COMMIT)

    def test_cpu_only_publication_is_explicit(self):
        parent,child=images();child["Config"]["Labels"]["org.opencontainers.image.title"]="GPU passed"
        with self.assertRaises(ValueError):ci.compare_images(parent,child,COMMIT)

    def test_metadata_and_direct_url_changes_rejected(self):
        for key,value in (("metadata","changed dependency"),("direct_url","another-wheel"),("version","different")):
            old,new=inventory();new=copy.deepcopy(new);new["packages"][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):ci.compare_inventories(old,new,COMMIT)

    def test_inventory_uid_and_model_scope_preserved(self):
        for key,value in (("dpkg",[]),("adapters_present",True),("euid",0),("gateway_commit","wrong"),("hsdp_source_commit","wrong"),("source_commit","wrong")):
            old,new=inventory();new[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):ci.compare_inventories(old,new,COMMIT)

    def test_source_lock_exact_parent_and_runtime(self):
        lock={"schema":"seedvr2-metrics-repair-source-lock/1","parent_image":ci.PARENT,"parent_source_commit":ci.PARENT_COMMIT,
            "recipe_base_commit":ci.PARENT_COMMIT,"remote_branch":ci.BRANCH,"repository":"https://github.com/githubgxll/dynamo.git",
            "parent_image_id":ci.PARENT_IMAGE_ID,"gateway_commit":ci.GATEWAY,"gpu_validation_status":"NOT_RUN","metrics_schema":"sr-timing-memory/2",
            "parent_source_manifest_sha256":ci.PARENT_SOURCE_MANIFEST,
            "harbor_repository":ci.REPOSITORY,"build_recipe":"seedvr2-metrics-repair","platform":"linux/amd64","tag_prefix":"omni031rc1-gateway-sr-metrics-repair",
            "versions":{"vllm":"0.31.0","vllm-omni":"0.31.0rc1","torch":"2.13.0+cu130","cache-dit":"1.5.0"}}
        ci.validate_lock(lock)
        for key,value in (("parent_image",":latest"),("recipe_base_commit","wrong"),("versions",{}),("platform","linux/arm64"),("parent_image_id","wrong")):
            with self.subTest(key=key),self.assertRaises(ValueError):ci.validate_lock({**lock,key:value})

    def test_patch_lock_validates_paths_and_hashes(self):
        good={"schema":"seedvr2-metrics-repair-source-patches/1","changes":DELTA[0],"additions":DELTA[1]}
        with patch.object(ci,"read",return_value=good):self.assertEqual(ci.patch_delta(),DELTA)
        for value in ({**good,"changes":{}},{**good,"changes":{"../other.py":next(iter(DELTA[0].values()))}},
                      {**good,"additions":{"vllm_omni/new.py":"1"*64}},{**good,"additions":{CHANGED:"1"*64}}):
            with patch.object(ci,"read",return_value=value),self.assertRaises(ValueError):ci.patch_delta()

    def test_complete_reports_match_exact_parent_plus_delta(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(ci,"patch_delta",return_value=DELTA):
            root=Path(temp);reports(root);ci.validate_reports(root,COMMIT,PARENT_SOURCES)

    def test_unknown_or_skipped_tests_are_not_passed(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(ci,"patch_delta",return_value=DELTA):
            root=Path(temp)
            for key,value in (("tests_run",0),("tests_run",True),("failures",1),("errors",1),("skipped",1)):
                reports(root);proof=ci.read(root/"regression.json");proof[key]=value;ci.dump(root/"regression.json",proof)
                with self.subTest(key=key),self.assertRaises(ValueError):ci.validate_reports(root,COMMIT)

    def test_failed_dependency_or_native_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(ci,"patch_delta",return_value=DELTA):
            root=Path(temp)
            for name in ("source-verification.json","preservation.json","torch-contract.json","regression.json"):
                reports(root);ci.dump(root/name,{"status":"FAIL"})
                with self.subTest(name=name),self.assertRaises(ValueError):ci.validate_reports(root,COMMIT)

    def test_gpu_claim_and_stale_report_rejected(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(ci,"patch_delta",return_value=DELTA):
            root=Path(temp)
            for key,value in (("gpu_validation_status","PASS"),("source_commit",ci.PARENT_COMMIT),("euid",0),("retained_packages_unchanged",False),("metrics_schema","sr-timing-memory/1")):
                reports(root);report=ci.read(root/"runtime-validation.json");report[key]=value;ci.dump(root/"runtime-validation.json",report)
                with self.subTest(key=key),self.assertRaises(ValueError):ci.validate_reports(root,COMMIT)

    def test_tampered_full_source_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(ci,"patch_delta",return_value=DELTA):
            root=Path(temp);reports(root);proof=ci.read(root/"source-verification.json")
            proof["source_sha256"]["vllm_omni/retained.py"]="f"*64;ci.dump(root/"source-verification.json",proof)
            with self.assertRaisesRegex(ValueError,"manifest identity"):ci.validate_reports(root,COMMIT)

    def test_valid_new_hash_cannot_hide_unreviewed_mutation(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(ci,"patch_delta",return_value=DELTA):
            root=Path(temp);reports(root);sources=ci.read(root/"source-verification.json")["source_sha256"]
            sources["vllm_omni/retained.py"]="f"*64;rewrite_source(root,sources)
            with self.assertRaisesRegex(ValueError,"outside the exact"):ci.validate_reports(root,COMMIT,PARENT_SOURCES)

    def test_missing_or_wrong_new_patch_rejected(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(ci,"patch_delta",return_value=DELTA):
            root=Path(temp)
            for name in (CHANGED,IPC,SUPPORT):
                reports(root);sources=ci.read(root/"source-verification.json")["source_sha256"]
                sources[name]="f"*64;rewrite_source(root,sources)
                with self.subTest(name=name),self.assertRaises(ValueError):ci.validate_reports(root,COMMIT)

    def test_parent_before_hash_and_unrelated_source_rejected(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(ci,"patch_delta",return_value=DELTA):
            root=Path(temp);reports(root)
            for before in ({**PARENT_SOURCES,CHANGED:"f"*64},{**PARENT_SOURCES,"vllm_omni/unreviewed.py":"3"*64}):
                with self.assertRaises(ValueError):ci.validate_reports(root,COMMIT,before)

    def test_build_and_final_must_match(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(ci,"patch_delta",return_value=DELTA),patch.object(ci,"PARENT_SOURCE_MANIFEST",ci.hash_map(PARENT_SOURCES)):
            root=Path(temp);reports(root/"build");reports(root/"final")
            ci.dump(root/"parent-source.json",{"status":"PASS","image_id":ci.PARENT_IMAGE_ID,"files":{n:{"sha256":v} for n,v in PARENT_SOURCES.items()}})
            ci.validate_candidate_reports(root,COMMIT)
            sources=ci.read(root/"final/source-verification.json")["source_sha256"];sources["vllm_omni/retained.py"]="f"*64
            rewrite_source(root/"final",sources)
            with self.assertRaises(ValueError):ci.validate_candidate_reports(root,COMMIT)

    def test_no_unverified_publish_calls_docker(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(ci,"checked_inputs",return_value={"source_commit":COMMIT,"candidate_image":"candidate"}),patch.object(ci,"sha256",return_value="a"*64),patch.object(ci,"docker") as docker:
            root=Path(temp);ci.dump(root/"candidate-verified.json",{"status":"FAIL"})
            with self.assertRaises(ValueError):ci.publish(ENV,root)
            docker.assert_not_called()

    def test_digest_from_registry_not_archive(self):
        digest="sha256:"+"c"*64
        self.assertEqual(ci.publication_digest("tag: digest: "+digest+" size: 100\n"),digest)
        for value in ("","zip sha256: "+"c"*64,"tag: digest: "+digest+" size: 1\nother: digest: sha256:"+"d"*64+" size: 1"):
            with self.assertRaises(ValueError):ci.publication_digest(value)

    def test_candidate_name_distinguishes_attempt_and_run(self):
        self.assertNotEqual(ci.candidate_tag(COMMIT,ENV),ci.candidate_tag(COMMIT,{**ENV,"GITHUB_RUN_ATTEMPT":"2"}))
        self.assertNotEqual(ci.candidate_tag(COMMIT,ENV),ci.candidate_tag(COMMIT,{**ENV,"GITHUB_RUN_ID":"124"}))
        for key,value in (("GITHUB_REF_NAME","main"),("GITHUB_SHA","short"),("GITHUB_RUN_ID","0")):
            with self.subTest(key=key),self.assertRaises(ValueError):ci.identity({**ENV,key:value})

    def test_single_route_and_gate_before_publication(self):
        workflow=yaml.load((ci.ROOT/".github/workflows/dingo-router-ci.yml").read_text(),Loader=yaml.BaseLoader)
        jobs=workflow["jobs"];new=jobs["seedvr2-metrics-repair"]
        self.assertIn("[seedvr2-metrics-repair]",new["if"])
        self.assertNotIn("!contains(github.event.head_commit.message, '[seedvr2-metrics-repair]')",new["if"])
        for name in ("omni-gateway-lite","seedvr2-c14","h3-hsdp-offload","sr-native","seedvr2-optimization"):
            self.assertIn("!contains(github.event.head_commit.message, '[seedvr2-metrics-repair]')",jobs[name]["if"])
        self.assertIn("seedvr2-metrics-repair",workflow["on"]["workflow_dispatch"]["inputs"]["build_recipe"]["options"])
        steps=new["steps"]
        snapshot=next(i for i,v in enumerate(steps) if "snapshot-parent" in v.get("run",""))
        build=next(i for i,v in enumerate(steps) if v.get("uses","").startswith("docker/build-push-action"))
        verify=next(i for i,v in enumerate(steps) if "verify-candidate" in v.get("run",""))
        publish=next(i for i,v in enumerate(steps) if "_ci.py publish" in v.get("run",""))
        self.assertLess(snapshot,build);self.assertLess(build,verify);self.assertLess(verify,publish)
        self.assertEqual(steps[build]["with"]["push"],"false")
        self.assertEqual(steps[build]["with"]["load"],"true")
        self.assertIn("SEEDVR2_METRICS_REPAIR_SOURCE_COMMIT",steps[build]["with"]["build-args"])
        self.assertIn("GPU_NOT_RUN",steps[publish]["run"])

    def test_workflow_does_not_prune_shared_docker_or_bypass_admission(self):
        workflow=yaml.load((ci.ROOT/".github/workflows/dingo-router-ci.yml").read_text(),Loader=yaml.BaseLoader)
        job=workflow["jobs"]["seedvr2-metrics-repair"]
        self.assertEqual(job["runs-on"],["self-hosted","linux","x64","dingo"])
        scripts="\n".join(v.get("run","") for v in job["steps"])
        for unsafe in ("docker system prune","docker builder prune","manager.py","100GiB"):
            self.assertNotIn(unsafe,scripts)

    def test_parent_probe_binds_its_own_source_patches_module(self):
        compile(ci.PARENT_SOURCE_PROBE,"parent-source-probe","exec")
        self.assertIn('"/opt/seedvr2-optimization/source_patches.py"',ci.PARENT_SOURCE_PROBE)
        self.assertIn('sys.modules[patch_spec.name]=patch_module',ci.PARENT_SOURCE_PROBE)
        self.assertLess(ci.PARENT_SOURCE_PROBE.index('sys.modules[patch_spec.name]=patch_module'),
                        ci.PARENT_SOURCE_PROBE.index('runtime.verify_installed()'))

    def test_parent_snapshot_requires_published_full_source_map(self):
        result={"proof":{"status":"PASS","source_sha256":PARENT_SOURCES,"source_manifest_sha256":ci.hash_map(PARENT_SOURCES)},
            "versions":{"vllm":"0.31.0"},"files":{n:{"sha256":v,"bytes":10} for n,v in PARENT_SOURCES.items()}}
        with tempfile.TemporaryDirectory() as temp,patch.object(ci,"checked_inputs"),\
                patch.object(ci,"inspect",return_value={"Id":ci.PARENT_IMAGE_ID}),\
                patch.object(ci,"probe",return_value=result),patch.object(ci,"read",return_value={"versions":result["versions"]}),\
                patch.object(ci,"patch_delta",return_value=DELTA),patch.object(ci,"PARENT_SOURCE_MANIFEST",ci.hash_map(PARENT_SOURCES)):
            root=Path(temp);ci.snapshot_parent(ENV,root)
            self.assertTrue((root/"parent-source.json").is_file())
            with patch.object(ci,"PARENT_SOURCE_MANIFEST","f"*64),self.assertRaisesRegex(ValueError,"source manifest differs"):
                ci.snapshot_parent(ENV,root)


if __name__=="__main__":unittest.main()
