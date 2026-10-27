"""Exercise publication rejection paths without registry or GPU side effects."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import yaml

spec=importlib.util.spec_from_file_location("hsdp_ci",Path(__file__).with_name("h3_hsdp_offload_ci.py"))
ci=importlib.util.module_from_spec(spec); spec.loader.exec_module(ci)
COMMIT="a"*40
ENV={"GITHUB_SHA":COMMIT,"GITHUB_REF_NAME":ci.BRANCH,"GITHUB_EVENT_NAME":"push","GITHUB_RUN_ID":"123","GITHUB_RUN_ATTEMPT":"1"}

def images():
    parent={"Id":ci.PARENT_IMAGE_ID,"Os":"linux","Architecture":"amd64","RootFS":{"Layers":["old"]},
        "Config":{"User":"dynamo","Entrypoint":["python"],"WorkingDir":"/workspace","Env":["PATH=/usr/bin","H3_C14_SOURCE_COMMIT="+ci.PARENT_COMMIT,"DYNAMO_GATEWAY_SOURCE_SHA="+ci.GATEWAY],"Labels":{"keep":"same"}}}
    child=copy.deepcopy(parent); child["Id"]="sha256:"+"b"*64; child["RootFS"]["Layers"].append("new")
    child["Config"]["Env"] += ["H3_HSDP_SOURCE_COMMIT="+COMMIT,"H3_HSDP_PARENT_IMAGE="+ci.PARENT]
    child["Config"]["Labels"].update({"org.opencontainers.image.title":"HSDP candidate GPU_NOT_RUN","org.opencontainers.image.revision":COMMIT,"org.opencontainers.image.base.name":ci.PARENT})
    return parent,child

def inventory():
    old={"packages":[{"name":"torch","version":"2.13.0+cu130","metadata":"exact","direct_url":None}],"dpkg":["libc\tversion"],"gateway_commit":ci.GATEWAY,"c14_source_commit":ci.PARENT_COMMIT,"source_commit":None,"euid":1000,"adapters_present":False}
    return old,{**old,"source_commit":COMMIT}

def reports(root):
    ci.dump(root/"runtime-validation.json",{"schema":"h3-hsdp-offload-runtime/1","status":"PASS_CPU_ONLY","euid":1000,"parent_image":ci.PARENT,"source_commit":COMMIT,"gpu_validation_status":"NOT_RUN","source_manifest_sha256":"e"*64,"retained_packages_unchanged":True})
    ci.dump(root/"source-verification.json",{"status":"PASS","source_manifest_sha256":"e"*64})
    ci.dump(root/"torch-contract.json",{"status":"PASS"})
    ci.dump(root/"preservation.json",{"status":"PASS"})
    ci.dump(root/"regression.json",{"status":"PASS","tests_run":3,"failures":0,"errors":0,"skipped":0})

class Gates(unittest.TestCase):
    def test_exact_candidate_config_and_inventory_pass(self):
        ci.compare_images(*images(),COMMIT)
        ci.compare_inventories(*inventory(),COMMIT)
    def test_parent_wrong_id_fails(self):
        parent,child=images(); parent["Id"]="wrong"
        with self.assertRaises(ValueError): ci.compare_images(parent,child,COMMIT)
    def test_config_or_platform_changes_fail(self):
        for key,value in (("User","root"),("Entrypoint",["bash"]),("WorkingDir","/tmp")):
            parent,child=images(); child["Config"][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): ci.compare_images(parent,child,COMMIT)
        parent,child=images(); child["Architecture"]="arm64"
        with self.assertRaises(ValueError): ci.compare_images(parent,child,COMMIT)
    def test_environment_changes_fail(self):
        for value in ("HTTP_PROXY=http://unexpected","H3_C14_SOURCE_COMMIT=wrong","H3_OFFLOAD_ENABLED=1"):
            parent,child=images(); child["Config"]["Env"].append(value)
            with self.subTest(value=value),self.assertRaises(ValueError): ci.compare_images(parent,child,COMMIT)
    def test_layer_rebase_fails(self):
        parent,child=images(); child["RootFS"]["Layers"]=["replacement"]
        with self.assertRaises(ValueError): ci.compare_images(parent,child,COMMIT)
    def test_gpu_unqualified_label_required(self):
        parent,child=images(); child["Config"]["Labels"]["org.opencontainers.image.title"]="success"
        with self.assertRaises(ValueError): ci.compare_images(parent,child,COMMIT)
    def test_same_version_metadata_mutation_fails(self):
        old,new=inventory(); new=copy.deepcopy(new); new["packages"][0]["metadata"]="different requires-dist"
        with self.assertRaises(ValueError): ci.compare_inventories(old,new,COMMIT)
    def test_direct_url_mutation_fails(self):
        old,new=inventory(); new=copy.deepcopy(new); new["packages"][0]["direct_url"]="different wheel"
        with self.assertRaises(ValueError): ci.compare_inventories(old,new,COMMIT)
    def test_package_dpki_and_mode_changes_fail(self):
        for key,value in (("packages",[]),("dpkg",["changed"]),("adapters_present",True),("euid",0),("gateway_commit","wrong"),("c14_source_commit","wrong"),("source_commit","wrong")):
            old,new=inventory(); new[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): ci.compare_inventories(old,new,COMMIT)
    def test_source_lock_and_reviewed_parent(self):
        lock=ci.read(ci.ROOT/ci.LOCK_PATH); ci.validate_lock(lock)
        for key,value in (("parent_image",":latest"),("recipe_base_commit","wrong"),("gpu_validation_status","PASS"),("versions",{}),("parent_source_sha256",{})):
            with self.subTest(key=key),self.assertRaises(ValueError): ci.validate_lock({**lock,key:value})
    def test_complete_reports_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); reports(root); ci.validate_reports(root,COMMIT)
    def test_report_cannot_claim_gpu_acceptance(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); reports(root); value=ci.read(root/"runtime-validation.json"); value["gpu_validation_status"]="PASS"; ci.dump(root/"runtime-validation.json",value)
            with self.assertRaises(ValueError): ci.validate_reports(root,COMMIT)
    def test_stale_commit_and_root_reports_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for key,value in (("source_commit",ci.PARENT_COMMIT),("parent_image","old"),("euid",0),("status","PASS")):
                reports(root); report=ci.read(root/"runtime-validation.json"); report[key]=value; ci.dump(root/"runtime-validation.json",report)
                with self.subTest(key=key),self.assertRaises(ValueError): ci.validate_reports(root,COMMIT)
    def test_failed_or_skipped_cpu_tests_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for key,value in (("tests_run",0),("tests_run",True),("failures",1),("errors",1),("skipped",1),("status","SKIP")):
                reports(root); report=ci.read(root/"regression.json"); report[key]=value; ci.dump(root/"regression.json",report)
                with self.subTest(key=key),self.assertRaises(ValueError): ci.validate_reports(root,COMMIT)
    def test_failed_preservation_or_source_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name in ("preservation.json","source-verification.json","torch-contract.json"):
                reports(root); ci.dump(root/name,{"status":"FAIL"})
                with self.subTest(name=name),self.assertRaises(ValueError): ci.validate_reports(root,COMMIT)
    def test_runtime_must_match_actual_source_proof(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); reports(root)
            ci.dump(root/"source-verification.json",{"status":"PASS","source_manifest_sha256":"f"*64})
            with self.assertRaises(ValueError): ci.validate_reports(root,COMMIT)
    def test_publication_digest_is_not_archive_hash(self):
        digest="sha256:"+"c"*64
        self.assertEqual(ci.publication_digest("tag: digest: "+digest+" size: 123\n"),digest)
        for value in ("","zip sha256: abc","tag: digest: "+digest+" size: 1\nother: digest: sha256:"+"d"*64+" size: 1"):
            with self.assertRaises(ValueError): ci.publication_digest(value)
    def test_run_identity_and_unique_candidate(self):
        self.assertEqual(ci.identity(ENV),COMMIT)
        self.assertNotEqual(ci.candidate_tag(COMMIT,ENV),ci.candidate_tag(COMMIT,{**ENV,"GITHUB_RUN_ATTEMPT":"2"}))
        for key,value in (("GITHUB_SHA","short"),("GITHUB_REF_NAME","main"),("GITHUB_RUN_ID","0")):
            with self.subTest(key=key),self.assertRaises(ValueError): ci.identity({**ENV,key:value})
    def test_unverified_publish_does_not_call_docker(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(ci,"checked_inputs",return_value={"source_commit":COMMIT,"candidate_image":"candidate"}),patch.object(ci,"docker") as docker:
            root=Path(temp); ci.dump(root/"candidate-verified.json",{"status":"FAIL"})
            with self.assertRaises(ValueError): ci.publish(ENV,root)
            docker.assert_not_called()
    def test_workflow_single_new_route_and_publication_order(self):
        workflow=yaml.load((ci.ROOT/".github/workflows/dingo-router-ci.yml").read_text(),Loader=yaml.BaseLoader)
        jobs=workflow["jobs"]; new=jobs["h3-hsdp-offload"]
        self.assertIn("[h3-hsdp-offload]",new["if"])
        for job in ("omni-gateway-lite","seedvr2-c14"):
            self.assertIn("!contains(github.event.head_commit.message, '[h3-hsdp-offload]')",jobs[job]["if"])
        steps=new["steps"]
        snapshot=next(i for i,s in enumerate(steps) if "snapshot-parent" in s.get("run",""))
        build=next(i for i,s in enumerate(steps) if s.get("uses","").startswith("docker/build-push-action"))
        verify=next(i for i,s in enumerate(steps) if "verify-candidate" in s.get("run",""))
        publish=next(i for i,s in enumerate(steps) if "_ci.py publish" in s.get("run",""))
        self.assertLess(snapshot,build); self.assertLess(build,verify); self.assertLess(verify,publish)
        self.assertEqual(steps[build]["with"]["push"],"false")
        self.assertEqual(steps[build]["with"]["load"],"true")
        self.assertIn("GPU_NOT_RUN",steps[publish]["run"])

if __name__=="__main__": unittest.main()
