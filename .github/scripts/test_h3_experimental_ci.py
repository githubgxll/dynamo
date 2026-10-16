"""CPU-only CI identity, evidence, routing and main-template regression tests."""
from __future__ import annotations
import argparse
import copy
import fnmatch
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location("ci", HERE / "h3_experimental_ci.py")
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = {"GITHUB_SHA": "a"*40, "GITHUB_REF_NAME": ci.BRANCH,
                    "GITHUB_EVENT_NAME": "push", "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1"}
        self.values = {"framework_image": ci.REPOSITORY+":framework-"+"a"*12,
                       "final_image": ci.REPOSITORY+":"+ci.TAG_PREFIX+"-"+"a"*12,
                       "builder_image": ci.REPOSITORY+":builder-test"}
        self.base = ci.REPOSITORY+"@sha256:"+"1"*64
        self.out = self.root / "reports"
        write(self.root / ci.PROFILE / "assets/assets.lock.json", {"test": "assetlock"})
        write(self.root / ci.PROFILE / "profiles/profiles.json", {"test": "profiles"})
        (self.root / ci.PROFILE / "Dockerfile").write_text("ARG BASE_IMAGE\nFROM ${BASE_IMAGE}\n")
        write(self.root / ci.SOURCE_LOCK, {"service_base_commit": ci.SERVICE_BASE_SHA})
        for path, status in {
            "experimental/service-smoke.json": "PASS_SERVICE_IMPORTS_AND_HELP_ONLY",
            "experimental/assets-report.json": "PASS_ASSETS_ONLY",
            "experimental/profiles-check.json": "PASS_INSTALLED_PROFILES_CPU_ONLY",
            "framework/ref8-patch.json": "PASS_PATCH_APPLIED_CPU_ONLY",
            "framework/ref8-contract.json": "PASS_INSTALLED_CONTRACT_CPU_ONLY",
            "experimental/environment-check.json": "PASS_EXPERIMENTAL_ENVIRONMENT_CPU_ONLY",
            "experimental/media-smoke.json": "PASS_REAL_CPU_MEDIA_SMOKE",
        }.items():
            write(self.out/path, {"status": status, "lock_sha256": ci.sha256(self.root/ci.PROFILE/"assets/assets.lock.json")})
        write(self.out/"experimental/image-inputs.json", {
            "schema": "h3-experimental-inputs/1", "source_commit": "a"*40,
            "base_image": self.base, "versions": {"vllm": "0.31.0", "vllm-omni": "0.31.0rc1"},
            "assets_lock_sha256": ci.sha256(self.root/ci.PROFILE/"assets/assets.lock.json"),
        })
        for name in ("legal/osrb-deps.csv", "legal/osrb.cdx.json"):
            write(self.out/name, {"fixture": "not a real compliance result"})
        write(self.out/"experimental/dependency-license-inventory.json", {
            "status":"METADATA_INVENTORY_ONLY_NOT_LEGAL_CLEARANCE", "packages":[{"name":"fixture"}],
            "python_dependency_delta_from_inherited_vllm_plus_dingo":[], "media_bundled_files":{},
        })
        (self.out/"experimental/dpkg-packages.txt").write_text("fixture\n")
        self.preparer = patch.object(ci, "prepare", return_value=self.values)
        self.preparer.start()

    def tearDown(self):
        self.preparer.stop()
        self.temp.cleanup()

    def test_identity_rejects_other_branch(self):
        for branch in (
            "main", "DingoRouter-h3-omni031-20261012", "DingoRouter-h3-distilled-new",
            "DingoRouter-h3-distilled-20261013", "DingoRouter-H3-vllm-omni-jirx",
            "dingorouter-h3-vllm-omni-jirx",
        ):
            with self.subTest(branch=branch), self.assertRaises(ValueError):
                ci.identity({**self.env, "GITHUB_REF_NAME": branch})

    def test_identity_accepts_exact_new_branch_for_push_and_dispatch(self):
        self.assertEqual(ci.BRANCH, "Dingorouter-H3-vllm-omni-jirx")
        for event in ("push", "workflow_dispatch"):
            with self.subTest(event=event):
                self.assertEqual(ci.identity({**self.env, "GITHUB_EVENT_NAME": event}), "a"*40)

    def test_identity_rejects_wrong_sha_and_event(self):
        with self.assertRaises(ValueError): ci.identity({**self.env, "GITHUB_SHA": "abc"})
        with self.assertRaises(ValueError): ci.identity({**self.env, "GITHUB_EVENT_NAME": "pull_request"})

    def test_framework_actual_digest_is_used(self):
        data = ci.record_framework(self.root, self.env, "sha256:"+"1"*64, self.out)
        self.assertEqual(data["deployment_image"], self.base)
        with self.assertRaises(ValueError): ci.record_framework(self.root, self.env, "", self.out)

    def test_final_record_preserves_harbor_port_and_parent_digest(self):
        record = ci.record_final(self.root, self.env, "sha256:"+"2"*64, self.out, self.base)
        self.assertEqual(record["framework_image"], self.base)
        self.assertEqual(record["deployment_image"], ci.REPOSITORY+"@sha256:"+"2"*64)
        self.assertIn("inputs/assets.lock.json", record["evidence_sha256"])
        self.assertNotIn("image.json", record["evidence_sha256"])

    def test_ref8_failure_stops_publication_record(self):
        write(self.out/"framework/ref8-contract.json", {"status": "FAIL"})
        with self.assertRaises(ValueError): ci.record_final(self.root, self.env, "sha256:"+"2"*64, self.out, self.base)
        self.assertFalse((self.out/"image.json").exists())

    def test_wrong_parent_fails(self):
        with self.assertRaises(ValueError): ci.validate_evidence(self.root, self.env, self.out, ci.REPOSITORY+"@sha256:"+"9"*64)

    def test_wrong_asset_lock_fails(self):
        data = ci.read(self.out/"experimental/image-inputs.json")
        data["assets_lock_sha256"] = "0"*64
        write(self.out/"experimental/image-inputs.json", data)
        with self.assertRaises(ValueError): ci.validate_evidence(self.root, self.env, self.out, self.base)

    def test_missing_environment_gate_fails(self):
        (self.out/"experimental/environment-check.json").unlink()
        with self.assertRaises(FileNotFoundError): ci.validate_evidence(self.root, self.env, self.out, self.base)

    def test_missing_media_or_license_inventory_stops_record(self):
        write(self.out/"experimental/media-smoke.json", {"status":"FAIL"})
        with self.assertRaises(ValueError): ci.validate_evidence(self.root,self.env,self.out,self.base)
        write(self.out/"experimental/media-smoke.json", {"status":"PASS_REAL_CPU_MEDIA_SMOKE"})
        (self.out/"experimental/dependency-license-inventory.json").unlink()
        with self.assertRaises(FileNotFoundError): ci.validate_evidence(self.root,self.env,self.out,self.base)


class GatewayTemplateTests(unittest.TestCase):
    def test_only_h3_workflow_matches_experimental_push(self):
        matched = []
        for path in sorted((ROOT/".github/workflows").glob("*.y*ml")):
            workflow = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
            events = workflow.get("on", {})
            if isinstance(events, dict):
                if "push" not in events:
                    continue
                config = events["push"] or {}
            elif events == "push" or (isinstance(events, list) and "push" in events):
                config = {}
            else:
                continue
            branches = config.get("branches")
            ignores = config.get("branches-ignore", [])
            if branches is None and ("tags" in config or "tags-ignore" in config):
                continue  # Explicit tag-only workflows do not run on branch pushes.
            accepted = branches is None or any(fnmatch.fnmatchcase(ci.BRANCH, p) for p in branches)
            rejected = any(fnmatch.fnmatchcase(ci.BRANCH, p) for p in ignores)
            if accepted and not rejected:
                matched.append(path.name)
        self.assertEqual(matched, ["dingo-router-ci.yml"])

    def test_workflow_routes_only_dedicated_branch_and_keeps_two_phase_digest(self):
        workflow = yaml.load((ROOT/".github/workflows/dingo-router-ci.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        self.assertEqual(workflow["on"]["push"]["branches"], [ci.BRANCH])
        self.assertEqual(set(workflow["jobs"]), {"h3-experimental"})
        job = workflow["jobs"]["h3-experimental"]
        self.assertEqual(job["if"], f"github.ref_name == '{ci.BRANCH}'")
        self.assertNotIn("DingoRouter-h3-distilled-20261013", workflow["on"]["push"]["branches"])
        self.assertEqual(job["runs-on"], ["self-hosted", "linux", "x64", "dingo"])
        final_builds = [s for s in job["steps"] if s.get("with",{}).get("file") == "container/h3-experimental/Dockerfile"]
        self.assertEqual([s["with"]["target"] for s in final_builds], ["evidence", "runtime"])
        for build in final_builds:
            self.assertIn("BASE_IMAGE=${{ steps.framework_record.outputs.framework_ref }}", build["with"]["build-args"])
        self.assertNotIn("http.proxy", (ROOT/".github/workflows/dingo-router-ci.yml").read_text(encoding="utf-8"))

    def test_gateway_renderer_both_native_paths_and_dependency_order(self):
        spec = importlib.util.spec_from_file_location("render", ROOT/"container/render.py")
        render = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(render)
        context = yaml.safe_load((ROOT/"container/context.yaml").read_text(encoding="utf-8"))
        args = argparse.Namespace(framework="vllm",device="cuda",target="runtime",platform="amd64",cuda_version="13.0",make_efa=False,builder_image="")
        render.validate_args(args)
        env = render._make_jinja_env(ROOT/"container")
        def rendered():
            raw = env.get_template("Dockerfile.template").render(context=context, **render._render_context(args, context))
            return raw
        first = rendered()
        self.assertIn("ARG RUNTIME_IMAGE_TAG=v0.31.0-ubuntu2404@sha256:3fb023cc987cc7c2bca7ca14c1e0aff5ae06bd9431bf07773731ce7e8a682460", first)
        self.assertIn("ARG ENABLE_KVBM=false", first)
        self.assertIn("FROM wheel_builder_base AS reusable_builder_base", first)
        self.assertIn('git clone "https://github.com/ai-dynamo/nixl.git"', first)
        reusable = first.split("FROM wheel_builder_base AS reusable_builder_base",1)[1].split("FROM reusable_builder_base AS wheel_builder",1)[0]
        self.assertNotIn("COPY lib/", reusable)
        self.assertIn("--mount=type=secret,id=actions-runtime-token,env=ACTIONS_RUNTIME_TOKEN", reusable)
        self.assertIn("repo.huaweicloud.com", first)
        self.assertIn("manylinux_2_28_x86_64:2026.09.05-1@sha256:", first)
        self.assertIn("--mount=type=secret,id=pip-index-url", first)
        self.assertLess(first.index("install_environment.sh"), first.index(" AS licenses"))
        args.builder_image = ci.REPOSITORY+":builder-test"
        reused = rendered()
        self.assertIn("FROM "+args.builder_image+" AS wheel_builder_base", reused)
        self.assertNotIn('git clone "https://github.com/ai-dynamo/nixl.git"', reused)
        self.assertNotIn("dnf install", reused)
        self.assertIn("COPY dingo/", reused)
        self.assertIn("--mount=type=secret,id=pip-index-url", reused)
        self.assertIn("validate_media_probe.py", reused)
        self.assertIn("validate_torch_compile_smoke.py", reused)
        args.builder_image = "bad ref\nRUN echo bad"
        with self.assertRaises(ValueError): render.validate_args(args)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        files = [".github/scripts/prepare_dingo_image_matrix.py", ".github/dingo-images.json",
                 ".dockerignore", "container/Dockerfile.template", "container/context.yaml",
                 "container/render.py", "container/templates/args.Dockerfile",
                 "container/templates/dynamo_base.Dockerfile", "container/templates/wheel_builder.Dockerfile",
                 "container/use-sccache.sh"]
        for name in files:
            destination = self.root/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, destination)
        write(self.root/ci.SOURCE_LOCK, {"service_base_commit": ci.SERVICE_BASE_SHA})
        write(self.root/ci.PROFILE/"assets/assets.lock.json", {"assets": [
            {"id":"turbo-fl8"}, {"id":"turbo-fl4"}, {"id":"fasth3-dense4"},
            {"id":"turbo-ref8", "steps":8, "video_shift":12, "audio_shift":3},
        ]})
        self.env = {"GITHUB_SHA":"a"*40, "GITHUB_REF_NAME":ci.BRANCH, "GITHUB_EVENT_NAME":"push"}

    def tearDown(self):
        self.temp.cleanup()

    def test_prepare_reuses_native_fingerprint_for_source_only_changes(self):
        first = ci.prepare(self.root,self.env)
        (self.root/"app.py").write_text("# source-only change\n")
        second = ci.prepare(self.root,{**self.env,"GITHUB_SHA":"b"*40})
        self.assertEqual(first["builder_image"],second["builder_image"])
        self.assertNotEqual(first["final_image"],second["final_image"])
        path = self.root/"container/use-sccache.sh"
        path.write_text(path.read_text(encoding="utf-8")+"\n# native input change\n",encoding="utf-8")
        self.assertNotEqual(first["builder_image"],ci.prepare(self.root,self.env)["builder_image"])

    def test_prepare_rejects_wrong_main_or_missing_adapters(self):
        write(self.root/ci.SOURCE_LOCK, {"service_base_commit":"0"*40})
        with self.assertRaises(ValueError): ci.prepare(self.root,self.env)
        write(self.root/ci.SOURCE_LOCK, {"service_base_commit":ci.SERVICE_BASE_SHA})
        write(self.root/ci.PROFILE/"assets/assets.lock.json", {"assets":[]})
        with self.assertRaises(ValueError): ci.prepare(self.root,self.env)

    def test_prepare_rejects_wrong_registry(self):
        config = ci.read(self.root/".github/dingo-images.json")
        config["registry"] = "registry.hd-05.alayanew.com:8443"
        write(self.root/".github/dingo-images.json",config)
        with self.assertRaises(ValueError): ci.prepare(self.root,self.env)


if __name__ == "__main__":
    unittest.main()
