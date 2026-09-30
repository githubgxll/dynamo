"""Regression checks for VBench's isolated, non-publishing preparation path."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
        self.assertIs(prep["with"]["pull"], False)
        self.assertIs(publish["with"]["pull"], False)
        self.assertIs(by_name["Build and push ${{ matrix.image }}"]["with"]["pull"], True)
        base_pull = by_name["Check and pull locked VBench base with Docker daemon"]
        self.assertEqual(base_pull["if"], "matrix.framework == 'vbench'")
        self.assertIn("pull-base", base_pull["run"])
        self.assertIn(".vbench-base/pull-evidence", base_pull["run"])
        self.assertLess(steps.index(base_pull), steps.index(prep))
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


class VBenchBasePullTests(unittest.TestCase):
    def result(self, code=0, stdout=b"", stderr=b""):
        return subprocess.CompletedProcess([], code, stdout, stderr)

    def identity(self, **changes):
        doc = {"Os": "linux", "Architecture": "amd64", "RepoDigests": ["docker.io/nvidia/cuda@" + DIGEST], "Id": "sha256:" + "c" * 64}
        doc.update(changes)
        return self.result(stdout=json.dumps(doc).encode())

    def run_pull(self, responses, expected):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / "base.lock.json"
            ci.write_json(lock, {"schema_version": 1, "source_tag": ci.BASE_TAG, "platform": "linux/amd64", "digest": DIGEST, "image": "nvidia/cuda@" + DIGEST})
            with patch.object(ci.subprocess, "run", side_effect=responses) as run, patch.object(ci.time, "sleep") as sleep:
                self.assertEqual(ci.pull_base(lock, root / "evidence"), expected)
            summary = json.loads((root / "evidence/base-pull-summary.json").read_text())
            files = {p.name: p.read_text() for p in (root / "evidence").glob("*") if p.is_file()}
            return summary, run.call_args_list, sleep.call_args_list, files

    def test_transient_eof_recovers_using_same_digest_without_mutable_tag(self):
        summary, calls, sleeps, files = self.run_pull([
            self.result(stdout=b'{"schemaVersion":2}\n'), self.result(1, stderr=b"unexpected EOF"),
            self.result(), self.identity()], 0)
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(len(summary["daemon_attempts"]), 2)
        self.assertEqual([x.args for x in sleeps], [(10,)])
        self.assertEqual(calls[0].kwargs["timeout"], 60)
        self.assertEqual(calls[1].kwargs["timeout"], 600)
        for call in calls:
            self.assertEqual(call.args[0][-1], "nvidia/cuda@" + DIGEST)
            self.assertNotIn(ci.BASE_TAG, call.args[0])
        self.assertFalse(summary["client_probe"]["raw_sha256_matches_expected"])
        self.assertIn("daemon-pull-2.stderr.txt", files)

    def test_auth_manifest_and_certificate_errors_stop_without_retry(self):
        for message, category in [(b"unauthorized: authentication required", "authentication_or_permission"),
                                  (b"manifest unknown", "manifest_unknown"),
                                  (b"x509: certificate signed by unknown authority", "tls_certificate")]:
            with self.subTest(category=category):
                summary, calls, sleeps, files = self.run_pull([self.result(), self.result(1, stderr=message)], 1)
                self.assertEqual(summary["category"], category)
                self.assertEqual(len(calls), 2)
                self.assertFalse(sleeps)
                self.assertIn("base-pull-summary.json", files)

    def test_bounded_timeouts_stop_after_three_attempts(self):
        timeout = subprocess.TimeoutExpired(["docker", "pull"], 600, output=b"partial progress", stderr=b"waiting")
        summary, calls, sleeps, files = self.run_pull([self.result(), timeout, timeout, timeout], 1)
        self.assertEqual(len(summary["daemon_attempts"]), 3)
        self.assertTrue(all(item["timed_out"] for item in summary["daemon_attempts"]))
        self.assertEqual(len(sleeps), 2)
        self.assertEqual(len(calls), 4)
        self.assertEqual(files["daemon-pull-3.stdout.txt"], "partial progress")

    def test_identity_must_match_architecture_and_exact_digest(self):
        for changes in ({"Architecture": "arm64"}, {"RepoDigests": ["nvidia/cuda@sha256:" + "a" * 64]}, {"Os": "windows"}):
            with self.subTest(changes=changes):
                summary, calls, sleeps, files = self.run_pull([self.result(), self.result(), self.identity(**changes)], 1)
                self.assertEqual(summary["category"], "daemon_image_identity_mismatch")
                self.assertFalse(summary["identity"]["matches_locked_image"])
                self.assertFalse(sleeps)

    def test_client_failure_or_invalid_json_still_allows_verified_daemon_success(self):
        for client in (self.result(1, stderr=b"client request EOF"), self.result(stdout=b"invalid body"), subprocess.TimeoutExpired(["docker"], 60)):
            with self.subTest(client=client):
                summary, calls, sleeps, files = self.run_pull([client, self.result(), self.identity()], 0)
                self.assertEqual(summary["status"], "PASS")
                self.assertEqual(len(calls), 3)

    def test_missing_docker_stops_without_retry(self):
        summary, calls, sleeps, files = self.run_pull([FileNotFoundError("docker missing"), FileNotFoundError("docker missing")], 1)
        self.assertEqual(summary["category"], "docker_unavailable")
        self.assertEqual(len(calls), 2)
        self.assertFalse(sleeps)

    def test_sensitive_diagnostics_are_redacted_and_image_config_is_not_saved(self):
        secret_error = b'EOF https://alice:secretpassword@registry.test/v2?token=signedsecret&other=moresecret Authorization: Bearer authsecret access_token="plainsecret"'
        summary, calls, sleeps, files = self.run_pull([
            self.result(1, stderr=secret_error), self.result(),
            self.identity(Config={"Env": ["PRIVATE=envsecret"]})], 0)
        saved = "\n".join(files.values())
        for secret in ("alice", "secretpassword", "signedsecret", "moresecret", "authsecret", "plainsecret", "envsecret", '"Config"'):
            self.assertNotIn(secret, saved)
        self.assertIn("registry.test/v2?REDACTED", saved)
        self.assertEqual(set(json.loads(files["daemon-image-identity.json"])), {"Os", "Architecture", "RepoDigests", "Id"})

    def test_only_transient_categories_retry(self):
        for message in ("429 Too Many Requests", "toomanyrequests: You have reached your pull rate limit", "unexpected status code: 503", "connection reset by peer", "net/http: TLS handshake timeout"):
            self.assertEqual(ci.failure_category(message), "transient_transport_or_rate_limit")
        self.assertEqual(ci.failure_category("an unknown error"), "other_failure")

    def test_invalid_lock_never_calls_docker_and_still_writes_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ci.write_json(root / "base.lock.json", {"image": ci.BASE_TAG})
            with patch.object(ci.subprocess, "run") as run:
                self.assertEqual(ci.pull_base(root / "base.lock.json", root / "evidence"), 1)
                run.assert_not_called()
            summary = json.loads((root / "evidence/base-pull-summary.json").read_text())
            self.assertEqual(summary["category"], "invalid_base_lock")
            self.assertIn("finished_utc", summary)


if __name__ == "__main__":
    unittest.main()
