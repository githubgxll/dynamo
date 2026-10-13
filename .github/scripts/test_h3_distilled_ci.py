#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline tests for target restrictions, publication evidence and job routing."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("h3_distilled_ci", Path(__file__).with_name("h3_distilled_ci.py"))
ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ci)


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.profile = self.root / ci.PROFILE
        self.profile.mkdir(parents=True)
        (self.root / ".github").mkdir()
        self.config = {
            "registry": "registry.hd-04.alayanew.com:8443", "namespace": "openclaw",
            "platform": "linux/amd64", "commit_sha_length": 12,
            "images": [{"framework": "vllm", "repository": "ai-dingo-vllm", "enabled": True}],
        }
        self.lock = {
            "schema": "h3-distilled-image/1", "base_image": ci.EXPECTED_BASE,
            "publication_repository": ci.EXPECTED_REPOSITORY,
            "tag_prefix": ci.EXPECTED_TAG_PREFIX, "platform": "linux/amd64",
            "recipe_base_commit": ci.EXPECTED_FRAMEWORK_COMMIT,
        }
        self.write(self.root / ".github/dingo-images.json", self.config)
        self.write(self.profile / "image.lock.json", self.lock)
        self.write(self.profile / "assets.lock.json", {"assets": [{"sha256": "a" * 64}]})
        (self.profile / "Dockerfile").write_text("ARG BASE_IMAGE\nFROM ${BASE_IMAGE}\n", encoding="utf-8")
        self.env = {"GITHUB_SHA": "1234567890abcdef1234567890abcdef12345678", "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1"}
        self.directory = self.root / "evidence"
        self.directory.mkdir()
        for name, status in ci.REPORTS.items():
            self.write(self.directory / name, {
                "status": status, "source_commit": self.env["GITHUB_SHA"],
                "base_image": ci.EXPECTED_BASE, "framework_source_commit": ci.EXPECTED_FRAMEWORK_COMMIT,
                "environment_unchanged": True,
                "asset_lock_sha256": ci.sha256_file(self.profile / "assets.lock.json"),
                "image_lock_sha256": ci.sha256_file(self.profile / "image.lock.json"),
                "lock_sha256": ci.sha256_file(self.profile / "assets.lock.json"),
            })
        ci.copy_inputs(self.root, self.directory)

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    def test_prepare_and_record_preserve_registry_port(self):
        output = self.root / "github-output"
        values = ci.prepare(self.root, self.env, output)
        self.assertEqual(values["image"], ci.EXPECTED_REPOSITORY + ":v0.30.0-ubuntu2404-distilled-1234567890ab")
        self.assertIn("registry=registry.hd-04.alayanew.com:8443\n", output.read_text())
        digest = "sha256:" + "a" * 64
        result = ci.record(self.root, self.env, values["image"], digest, self.directory)
        self.assertEqual(result["deployment_image"], ci.EXPECTED_REPOSITORY + "@" + digest)
        self.assertEqual(result["base_image"], ci.EXPECTED_BASE)
        self.assertEqual(set(ci.COPIED_INPUTS).union(ci.REPORTS), set(result["evidence_sha256"]))
        self.assertEqual(result["assets_lock_sha256"], ci.sha256_file(self.profile / "assets.lock.json"))
        self.assertEqual(ci.record(self.root, self.env, values["image"], digest, self.directory), result)

    def test_prepare_rejects_other_targets_and_versions(self):
        for field, wrong in [("registry", "registry.hd-05.alayanew.com:8443"), ("namespace", "other"), ("platform", "linux/arm64"), ("commit_sha_length", 7)]:
            with self.subTest(field=field):
                self.write(self.root / ".github/dingo-images.json", {**self.config, field: wrong})
                with self.assertRaises(ValueError):
                    ci.prepare_values(self.root, self.env)
        self.write(self.root / ".github/dingo-images.json", self.config)
        for field, wrong in [("schema", "2"), ("publication_repository", "registry.test/image"), ("base_image", "vllm/vllm-openai:latest"), ("tag_prefix", "v0.30.0"), ("platform", "linux/arm64"), ("recipe_base_commit", "a" * 40)]:
            with self.subTest(field=field):
                self.write(self.profile / "image.lock.json", {**self.lock, field: wrong})
                with self.assertRaises(ValueError):
                    ci.prepare_values(self.root, self.env)

    def test_rejects_wrong_repository_and_duplicate_framework(self):
        for images in [[{"framework": "vllm", "repository": "other"}], self.config["images"] * 2, []]:
            with self.subTest(images=images):
                self.write(self.root / ".github/dingo-images.json", {**self.config, "images": images})
                with self.assertRaises(ValueError):
                    ci.prepare_values(self.root, self.env)

    def test_rejects_invalid_sha_and_digest(self):
        for sha in ["", "1234567", "A" * 40, "g" * 40, "a" * 40 + "\n"]:
            with self.subTest(sha=sha), self.assertRaises(ValueError):
                ci.prepare_values(self.root, {**self.env, "GITHUB_SHA": sha})
        image = ci.prepare_values(self.root, self.env)["image"]
        for digest in ["", "sha256:" + "a" * 63, "sha256:" + "A" * 64, "sha512:" + "a" * 64, "sha256:" + "a" * 64 + "\n"]:
            with self.subTest(digest=digest), self.assertRaises(ValueError):
                ci.record(self.root, self.env, image, digest, self.directory)

    def test_rejects_wrong_published_tag_and_run_identity(self):
        image = ci.prepare_values(self.root, self.env)["image"]
        digest = "sha256:" + "a" * 64
        with self.assertRaises(ValueError):
            ci.record(self.root, self.env, image + "-wrong", digest, self.directory)
        for key in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                ci.record(self.root, {**self.env, key: ""}, image, digest, self.directory)

    def test_rejects_missing_failed_and_stale_reports(self):
        report = self.directory / "assets-check.json"
        report.unlink()
        with self.assertRaisesRegex(ValueError, "Missing regular evidence"):
            ci.validate_evidence(self.root, self.env, self.directory)
        self.write(report, {"status": "FAIL"})
        with self.assertRaisesRegex(ValueError, "Unsuccessful"):
            ci.validate_evidence(self.root, self.env, self.directory)
        self.write(report, {"status": ci.REPORTS[report.name]})
        self.write(self.directory / "bundle-check.json", {"status": "PASS_DISTILLED_CPU_ONLY", "source_commit": "a" * 40})
        with self.assertRaisesRegex(ValueError, "source_commit"):
            ci.validate_evidence(self.root, self.env, self.directory)

    def test_rejects_changed_inputs_without_overwriting_them(self):
        target = self.directory / "assets.lock.json"
        self.write(target, {"tampered": True})
        with self.assertRaisesRegex(ValueError, "differs from checkout"):
            ci.copy_inputs(self.root, self.directory)
        self.assertTrue(ci.read_object(target)["tampered"])
        with self.assertRaisesRegex(ValueError, "differs from checkout"):
            ci.validate_evidence(self.root, self.env, self.directory)

    def test_rejects_crossed_base_and_lock_reports(self):
        path = self.directory / "bundle-check.json"
        original = ci.read_object(path)
        for name, wrong in [("base_image", "other"), ("framework_source_commit", "a" * 40), ("environment_unchanged", False), ("asset_lock_sha256", "a" * 64), ("image_lock_sha256", "b" * 64)]:
            with self.subTest(name=name):
                self.write(path, {**original, name: wrong})
                with self.assertRaisesRegex(ValueError, "Bundle evidence"):
                    ci.validate_evidence(self.root, self.env, self.directory)
        self.write(path, original)
        path = self.directory / "assets-check.json"
        self.write(path, {**ci.read_object(path), "lock_sha256": "a" * 64})
        with self.assertRaisesRegex(ValueError, "different asset lock"):
            ci.validate_evidence(self.root, self.env, self.directory)

    def test_refuses_overwrite_of_different_publication(self):
        image = ci.prepare_values(self.root, self.env)["image"]
        ci.record(self.root, self.env, image, "sha256:" + "a" * 64, self.directory)
        with self.assertRaisesRegex(ValueError, "different publication"):
            ci.record(self.root, self.env, image, "sha256:" + "b" * 64, self.directory)


class WorkflowRoutingTests(unittest.TestCase):
    def test_distilled_has_separate_route_and_checked_export_before_push(self):
        import yaml

        path = Path(__file__).resolve().parents[1] / "workflows/dingo-router-ci.yml"
        workflow = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        self.assertIn("h3-distilled", workflow["on"]["workflow_dispatch"]["inputs"]["frameworks"]["options"])
        jobs = workflow["jobs"]
        for name in ("prepare", "build"):
            self.assertIn("!startsWith(github.ref_name, 'DingoRouter-h3-distilled-')", jobs[name]["if"])
            self.assertIn("inputs.frameworks != 'h3-distilled'", jobs[name]["if"])
        job = jobs["h3-distilled"]
        self.assertEqual(job["if"], "${{ startsWith(github.ref_name, 'DingoRouter-h3-distilled-') || inputs.frameworks == 'h3-distilled' }}")
        self.assertEqual(job["runs-on"], ["self-hosted", "linux", "x64", "dingo"])
        steps = job["steps"]
        builds = [(i, step) for i, step in enumerate(steps) if step.get("uses", "").startswith("docker/build-push-action@")]
        self.assertEqual(len(builds), 2)
        evidence_index, evidence = builds[0]
        runtime_index, runtime = builds[1]
        self.assertEqual(evidence["with"]["target"], "evidence")
        self.assertEqual(evidence["with"]["push"], "false")
        self.assertEqual(evidence["with"]["outputs"], "type=local,dest=.h3-distilled-evidence")
        self.assertEqual(runtime["with"]["target"], "runtime")
        self.assertEqual(runtime["with"]["push"], "true")
        self.assertEqual(evidence["with"]["build-args"], runtime["with"]["build-args"])
        self.assertNotIn("DYNAMO_COMMIT_SHA=", runtime["with"]["build-args"])
        self.assertIn("H3_DISTILLED_SOURCE_COMMIT=${{ github.sha }}", runtime["with"]["build-args"])
        self.assertTrue(any("h3_distilled_ci.py validate-evidence" in s.get("run", "") for s in steps[evidence_index + 1:runtime_index]))
        self.assertFalse(any("container/render.py" in s.get("run", "") for s in steps))
        self.assertFalse(any(s.get("with", {}).get("target") == "reusable_builder_base" for s in steps))


if __name__ == "__main__":
    unittest.main()
