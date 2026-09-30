"""Regression checks for VBench's isolated, non-publishing preparation path."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


matrix_helper = module("prepare_dingo_image_matrix")
ci = module("vbench_ci")
SHA = "a" * 40
DIGEST = "sha256:" + "b" * 64


class VBenchWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.config = matrix_helper.load_config(ROOT / ".github/dingo-images.json")
        # The operator legitimately changes the checkout to publish after
        # reviewing preparation. Exercise both phases using explicit fixtures;
        # never make the test suite reject that valid configuration change.
        self.config["vbench"]["phase"] = "prepare"

    def test_checked_out_configuration_has_a_valid_vbench_phase(self):
        actual = matrix_helper.load_config(ROOT / ".github/dingo-images.json")
        entries = matrix_helper.build_matrix(actual, SHA, "configured")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["framework"], "vbench")
        self.assertIn(entries[0]["vbench_phase"], {"prepare", "publish"})

    def test_configured_is_only_static_non_publishing_vbench(self):
        entries = matrix_helper.build_matrix(self.config, SHA, "configured")
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry["framework"], "vbench")
        self.assertEqual(entry["dockerfile"], "container/vbench/Dockerfile")
        self.assertEqual(entry["docker_target"], "preparation_artifact")
        self.assertEqual(entry["publish"], "false")
        self.assertEqual(entry["builder_image"], "")
        self.assertEqual(entry["sccache_cache_version"], "")
        self.assertEqual(entry["cuda_version"], "12.1")
        self.assertEqual(entry["image"], "registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-testing/vench:v0.1.5-fd18b3d-cu121-r1-aaaaaaaaaaaa")

    def test_publish_requires_explicit_configuration(self):
        self.config["vbench"]["phase"] = "publish"
        entry = matrix_helper.build_matrix(self.config, SHA, "configured")[0]
        self.assertEqual((entry["docker_target"], entry["publish"]), ("runtime", "true"))
        for invalid in ("", "runtime", "Prepare", "publish\n"):
            self.config["vbench"]["phase"] = invalid
            with self.assertRaises(ValueError):
                matrix_helper.build_matrix(self.config, SHA, "configured")

    def test_vbench_target_cannot_override_preparation(self):
        self.config["images"][-1]["docker_target"] = "runtime"
        with self.assertRaises(ValueError):
            matrix_helper.build_matrix(self.config, SHA, "configured")

    def test_repository_accepts_nested_path_but_rejects_uppercase(self):
        self.config["images"][-1]["repository"] = "ai-dingo-testing/Vench"
        with self.assertRaises(ValueError):
            matrix_helper.build_matrix(self.config, SHA, "configured")

    def test_existing_frameworks_keep_renderer_builder_and_cuda13(self):
        for framework in ("dynamo", "vllm", "sglang"):
            entries = matrix_helper.build_matrix(self.config, SHA, framework)
            self.assertTrue(entries)
            for entry in entries:
                self.assertEqual(entry["publish"], "true")
                self.assertEqual(entry["cuda_version"], "13.0")
                self.assertIn("rendered.Dockerfile", entry["dockerfile"])
                if entry["render_target"] != "router":
                    self.assertTrue(entry["builder_image"])

    def test_workflow_keeps_prepare_out_of_registry_and_native_steps(self):
        workflow = yaml.safe_load((ROOT / ".github/workflows/dingo-router-ci.yml").read_text())
        steps = workflow["jobs"]["build"]["steps"]
        by_name = {step["name"]: step for step in steps}
        self.assertEqual(by_name["Log in to the private registry"]["if"], "matrix.publish == 'true'")
        for name in ("Configure GitHub Actions cache for sccache", "Render ${{ matrix.framework }} ${{ matrix.render_target }} Dockerfile", "Build and push ${{ matrix.image }}"):
            self.assertEqual(by_name[name]["if"], "matrix.framework != 'vbench'")
        prep = by_name["Prepare VBench dependency and asset evidence (no push)"]
        self.assertIs(prep["with"]["push"], False)
        self.assertNotIn("tags", prep["with"])
        self.assertEqual(prep["with"]["target"], "preparation_artifact")
        publish = by_name["Build and publish reviewed VBench image"]
        self.assertIn("matrix.vbench_phase == 'publish'", publish["if"])
        self.assertIn("steps.vbench_publish_base.outputs.base_image", publish["with"]["build-args"])
        uploads = [step for step in steps if step.get("uses", "").startswith("actions/upload-artifact")]
        self.assertTrue(all("always()" in step["if"] for step in uploads))


class VBenchEvidenceTests(unittest.TestCase):
    def base(self):
        return {"schema_version": 1, "source_tag": ci.BASE_TAG, "platform": "linux/amd64", "digest": DIGEST, "image": f"nvidia/cuda@{DIGEST}"}

    def test_selects_amd64_image_not_arm_or_attestation(self):
        entry = {"digest": DIGEST, "platform": {"os": "linux", "architecture": "amd64"}}
        index = {"manifests": [entry, {"digest": "other", "platform": {"os": "linux", "architecture": "arm64"}}, {"digest": "attestation", "platform": {"os": "unknown", "architecture": "unknown"}}]}
        self.assertEqual(ci.amd64_digest(index), DIGEST)
        index["manifests"].append(copy.deepcopy(entry))
        with self.assertRaises(ValueError):
            ci.amd64_digest(index)

    def test_base_lock_rejects_mutable_tag_wrong_architecture_and_output_injection(self):
        self.assertEqual(ci.validate_base_lock(self.base()), f"nvidia/cuda@{DIGEST}")
        for key, value in (("image", ci.BASE_TAG), ("platform", "linux/arm64"), ("digest", DIGEST + "\npublish=true")):
            lock = self.base()
            lock[key] = value
            with self.assertRaises(ValueError):
                ci.validate_base_lock(lock)

    def test_publish_requires_all_locks_and_license_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ci.write_json(root / "base.lock.json", self.base())
            with self.assertRaises(ValueError):
                ci.verify_locks(root, root / "output")
            (root / "requirements.lock").write_text("demo==1 --hash=sha256:" + "c" * 64)
            ci.write_json(root / "assets.lock.json", {})
            ci.write_json(root / "input-fingerprint.json", {})
            ci.write_json(root / "preparation-license-status.json", {"policy_pass": False})
            with self.assertRaisesRegex(ValueError, "license review has not passed"):
                ci.verify_locks(root, root / "output")
            ci.write_json(root / "preparation-license-status.json", {"policy_pass": True})
            ci.verify_locks(root, root / "output")
            self.assertEqual((root / "output").read_text(), f"base_image=nvidia/cuda@{DIGEST}\n")

    def test_image_evidence_preserves_registry_port_and_has_no_environment_dump(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dockerfile = root / "Dockerfile"
            dockerfile.write_text("FROM scratch\n")
            args = argparse.Namespace(image="registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-testing/vench:v1-abc", digest=DIGEST, source_commit=SHA, run_id="123", run_attempt="1", dockerfile=str(dockerfile), builder_image="", output=root / "image.json", summary=None)
            ci.record_image(args)
            evidence = json.loads(args.output.read_text())
            self.assertEqual(evidence["deployment_image"], f"registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-testing/vench@{DIGEST}")
            self.assertEqual(set(evidence), {"source_commit", "run_id", "run_attempt", "image", "digest", "deployment_image", "builder_image", "dockerfile", "dockerfile_sha256", "validation_scope"})


if __name__ == "__main__":
    unittest.main()
