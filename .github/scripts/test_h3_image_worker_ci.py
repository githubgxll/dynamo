"""Regression fixtures for publication identity, unknown state and retry handling."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("h3_image_worker_ci_under_test", ROOT / ".github/scripts/h3_image_worker_ci.py")
ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ci)
COMMIT = "a" * 40
DIGEST = "sha256:" + "b" * 64
CHILD_ID = "sha256:" + "c" * 64
CURRENT = {"commit": COMMIT, "run_id": "431", "run_attempt": 3}
ENV = {"GITHUB_REF_NAME": ci.BRANCH, "GITHUB_EVENT_NAME": "push", "GITHUB_SHA": COMMIT,
       "GITHUB_RUN_ID": "431", "GITHUB_RUN_ATTEMPT": "3"}
IMAGE = ci.image_tag(COMMIT)


def configs():
    parent = {"Id": ci.PARENT_ID, "Os": "linux", "Architecture": "amd64", "RootFS": {"Layers": ["parent"]},
              "Config": {"User": "", "Env": ["PATH=/usr/bin", "LD_PRELOAD=/opt/frozen-cudnn/libcudnn.so.9"],
                         "Entrypoint": None, "Cmd": None, "WorkingDir": "/app"}}
    child = copy.deepcopy(parent)
    child["Id"] = CHILD_ID
    child["RootFS"]["Layers"].append("worker")
    child["Config"]["User"] = "1000:1000"
    child["Config"]["Env"] += ["H3_IMAGE_WORKER_SOURCE_COMMIT=" + COMMIT,
                               "H3_IMAGE_WORKER_PARENT_IMAGE=" + ci.PARENT]
    child["Config"]["Labels"] = {"org.opencontainers.image.revision": COMMIT,
                                "org.opencontainers.image.base.name": ci.PARENT}
    return parent, child


def reports(directory):
    files = {"vllm_omni/fixture/module_" + str(number) + ".py": "d" * 64
             for number in range(ci.MINIMUM_SOURCE_FILES - 1)}
    files["vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3_image.py"] = "d" * 64
    source_hash = ci.hash_value(files)
    profiles = ci.read(ROOT / "container/h3-image-worker/profiles.json")["profiles"]
    runtime = {"status": "PASS_CPU_ONLY", "source_commit": COMMIT, "parent_image": ci.PARENT,
               "parent_image_id": ci.PARENT_ID, "euid": 1000, "inventory_preserved": True,
               "errors": [], "source_manifest_sha256": source_hash,
               "cpu_checks": {"status": "PASS_CPU_RUNTIME_ONLY", "tests_run": 12,
                              "failures": 0, "errors": 0, "skipped": 0},
               "cli_profiles": [{"profile": item["id"], "status": "PASS", "gpus": item["gpus"],
                                 "dit_tp": item["dit_tp"], "te_tp": item["te_tp"]} for item in profiles]}
    proof = {"status": "PASS", "source_commit": COMMIT, "parent_image": ci.PARENT,
             "source_sha256": files, "source_manifest_sha256": source_hash}
    ci.dump(directory / "runtime-validation.json", runtime)
    ci.dump(directory / "source-verification.json", proof)
    return runtime, proof


def transaction(directory, reused=False):
    parent, child = configs()
    prepared = {"schema": "h3-image-worker-ci-preparation/1", **CURRENT, "image": IMAGE,
                "reused": reused, "existing_digest": DIGEST if reused else None, "recipe_sha256": "e" * 64}
    runtime, proof = reports(directory / "final")
    candidate = {"schema": "h3-image-worker-candidate/1", **CURRENT, "status": "PASS_CPU_ONLY",
                 "image": IMAGE, "image_id": CHILD_ID, "recipe_sha256": prepared["recipe_sha256"],
                 "source_manifest_sha256": runtime["source_manifest_sha256"]}
    ci.dump(directory / "preparation.json", prepared)
    ci.dump(directory / "candidate.json", candidate)
    return prepared, candidate, parent, child


class ManifestClassification(unittest.TestCase):
    def classify(self, text, returncode=1):
        return ci.classify_manifest_lookup(SimpleNamespace(returncode=returncode, stdout="", stderr=text), IMAGE)

    def test_success_reports_exists(self):
        self.assertEqual(self.classify("", 0), "EXISTS")

    def test_precise_missing_manifest_reports_absence(self):
        for text in ("manifest unknown", "no such manifest", "manifest unknown status 404",
                     "unknown: artifact " + IMAGE.split("/", 1)[1] + " not found"):
            with self.subTest(text=text): self.assertEqual(self.classify(text), "ABSENT_CONFIRMED")

    def test_tls_network_auth_rate_limit_never_report_absence(self):
        for text in ("manifest unknown TLS EOF", "manifest unknown unauthorized", "manifest unknown forbidden",
                     "manifest unknown status code 500", "manifest unknown status: 429", "manifest unknown rate limit",
                     "no such manifest: proxyconnect failed", "manifest unknown timed out"):
            with self.subTest(text=text): self.assertEqual(self.classify(text), "UNKNOWN")

    def test_unrelated_404_and_wrong_harbor_repository_are_unknown(self):
        for text in ("404 not found", "unknown: artifact other/repo:tag not found", "repository not found"):
            with self.subTest(text=text): self.assertEqual(self.classify(text), "UNKNOWN")

    def test_tool_failure_is_unknown(self):
        self.assertEqual(self.classify("manifest unknown", 2), "UNKNOWN")

    def test_pull_digest_requires_one_full_value(self):
        self.assertEqual(ci.pull_digest("Digest: " + DIGEST + "\n"), DIGEST)
        for text in ("", "Digest: sha256:short", "Digest: " + DIGEST + "\nDigest: sha256:" + "f" * 64):
            with self.subTest(text=text), self.assertRaises(ValueError): ci.pull_digest(text)


class IdentityAndConfig(unittest.TestCase):
    def test_real_attempt_three_is_valid(self):
        self.assertEqual(ci.run_identity(ENV), CURRENT)

    def test_wrong_branch_event_and_nonpositive_identity_fail(self):
        for key, value in (("GITHUB_REF_NAME", "main"), ("GITHUB_EVENT_NAME", "workflow_dispatch"),
                           ("GITHUB_RUN_ATTEMPT", "0"), ("GITHUB_RUN_ID", ""), ("GITHUB_SHA", "short")):
            with self.subTest(key=key), self.assertRaises(ValueError): ci.run_identity({**ENV, key: value})

    def test_reports_from_another_attempt_fail(self):
        with mock.patch.dict(os.environ, ENV), self.assertRaises(ValueError):
            ci.require_current({**CURRENT, "run_attempt": 2})

    def test_reviewed_nonroot_child_extends_exact_parent(self):
        parent, child = configs()
        ci.compare_config(parent, child, COMMIT, {"expected_image_config": {"User": "1000:1000"}})

    def test_cudnn_preload_change_fails(self):
        parent, child = configs()
        child["Config"]["Env"][1] = "LD_PRELOAD="
        with self.assertRaises(ValueError):
            ci.compare_config(parent, child, COMMIT, {"expected_image_config": {"User": "1000:1000"}})

    def test_reviewed_environment_value_can_override_parent(self):
        parent, child = configs()
        parent["Config"]["Env"].append("HOME=/root")
        child["Config"]["Env"].append("HOME=/work")
        ci.compare_config(parent, child, COMMIT,
                          {"expected_image_config": {"User": "1000:1000"},
                           "expected_environment_additions": {"HOME": "/work"}})

    def test_parent_layer_replacement_fails(self):
        parent, child = configs()
        child["RootFS"]["Layers"][0] = "different-parent"
        with self.assertRaises(ValueError):
            ci.compare_config(parent, child, COMMIT, {"expected_image_config": {"User": "1000:1000"}})

    def test_unreviewed_entrypoint_and_root_fail(self):
        parent, child = configs()
        for field, value in (("Entrypoint", ["unreviewed"]), ("User", "0")):
            changed = copy.deepcopy(child)
            changed["Config"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                ci.compare_config(parent, changed, COMMIT, {"expected_image_config": {"User": "1000:1000"}})


class RuntimeProof(unittest.TestCase):
    def test_matching_installed_proof_passes(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            runtime, proof = reports(directory)
            self.assertEqual(ci.validate_reports(directory, COMMIT), (runtime, proof))

    def test_root_changed_inventory_or_wrong_parent_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for key, value in (("euid", 0), ("inventory_preserved", False), ("parent_image_id", CHILD_ID),
                               ("errors", ["actual failure"]), ("source_commit", "f" * 40)):
                runtime, _ = reports(directory)
                runtime[key] = value
                ci.dump(directory / "runtime-validation.json", runtime)
                with self.subTest(key=key), self.assertRaises(ValueError): ci.validate_reports(directory, COMMIT)

    def test_empty_or_malformed_source_proof_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for files in ({}, {"file.py": "short"}, {"file.py": None}):
                _, proof = reports(directory)
                proof["source_sha256"] = files
                ci.dump(directory / "source-verification.json", proof)
                with self.subTest(files=files), self.assertRaises(ValueError): ci.validate_reports(directory, COMMIT)

    def test_source_hash_or_identity_change_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for key, value in (("source_manifest_sha256", "f" * 64), ("source_commit", "f" * 40),
                               ("parent_image", "other-parent")):
                _, proof = reports(directory)
                proof[key] = value
                ci.dump(directory / "source-verification.json", proof)
                with self.subTest(key=key), self.assertRaises(ValueError): ci.validate_reports(directory, COMMIT)

    def test_missing_failed_or_empty_actual_cpu_checks_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for key, value in (("status", "PENDING"), ("tests_run", 0), ("tests_run", "12"),
                               ("failures", 1), ("errors", 1), ("skipped", 1)):
                runtime, _ = reports(directory)
                runtime["cpu_checks"][key] = value
                ci.dump(directory / "runtime-validation.json", runtime)
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    ci.validate_reports(directory, COMMIT)

    def test_incomplete_cli_profiles_and_source_inventory_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            runtime, _ = reports(directory)
            runtime["cli_profiles"].pop()
            ci.dump(directory / "runtime-validation.json", runtime)
            with self.assertRaises(ValueError): ci.validate_reports(directory, COMMIT)
            _, proof = reports(directory)
            proof["source_sha256"].pop(next(iter(proof["source_sha256"])))
            proof["source_manifest_sha256"] = ci.hash_value(proof["source_sha256"])
            ci.dump(directory / "source-verification.json", proof)
            with self.assertRaises(ValueError): ci.validate_reports(directory, COMMIT)


class Preparation(unittest.TestCase):
    def test_missing_approval_stops_before_registry_or_parent_pull(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(os.environ, ENV), \
                mock.patch.object(ci.ops, "check_seal", side_effect=ValueError("approval missing")), \
                mock.patch.object(ci, "docker") as docker, mock.patch.object(ci, "lookup") as lookup:
            with self.assertRaises(ValueError): ci.prepare(Path(temp))
            docker.assert_not_called()
            lookup.assert_not_called()

    def test_unknown_registry_state_stops_before_build(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(os.environ, ENV), \
                mock.patch.object(ci.ops, "check_seal", return_value=({}, "tree", {"recipe_sha256": "e" * 64, "review_tree": "tree"})), \
                mock.patch.object(ci.ops, "git", side_effect=["worker [h3-image-worker]", COMMIT]), \
                mock.patch.object(ci.ops, "valid_request_message", return_value=True), \
                mock.patch.object(ci, "lookup", return_value="UNKNOWN"), mock.patch.object(ci, "docker") as docker:
            with self.assertRaises(RuntimeError): ci.prepare(Path(temp))
            docker.assert_not_called()

    def test_preparation_is_exclusive_for_this_attempt(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(os.environ, ENV), \
                mock.patch.object(ci.ops, "check_seal", return_value=({}, "tree", {})), \
                mock.patch.object(ci.ops, "git", side_effect=["worker [h3-image-worker]", COMMIT]), \
                mock.patch.object(ci.ops, "valid_request_message", return_value=True):
            ci.dump(Path(temp) / "preparation.json", {})
            with self.assertRaises(ValueError): ci.prepare(Path(temp))

    def test_existing_tag_disables_build_and_pins_existing_digest(self):
        parent, _ = configs()
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(os.environ, ENV), \
                mock.patch.object(ci.ops, "check_seal", return_value=({}, "tree", {"recipe_sha256": "e" * 64, "review_tree": "tree"})), \
                mock.patch.object(ci.ops, "git", side_effect=["worker [h3-image-worker]", COMMIT]), \
                mock.patch.object(ci.ops, "valid_request_message", return_value=True), \
                mock.patch.object(ci, "lookup", return_value="EXISTS"), \
                mock.patch.object(ci, "inspect", return_value=parent), \
                mock.patch.object(ci, "docker", return_value="Digest: " + DIGEST + "\n"):
            output = Path(temp) / "github-output.txt"
            result = ci.prepare(Path(temp), output)
            self.assertTrue(result["reused"])
            self.assertEqual(result["existing_digest"], DIGEST)
            self.assertIn("build_required=false", output.read_text())


class Publication(unittest.TestCase):
    def invoke(self, directory, child, reused=False, pull_error=False, tag_state="ABSENT_CONFIRMED"):
        calls = []
        def docker(*args, **kwargs):
            calls.append(args)
            if args[0] == "push": return "worker: digest: " + DIGEST + " size: 123\n"
            if args[0] == "pull" and pull_error: raise RuntimeError("TLS EOF after completed push")
            return "Digest: " + DIGEST + "\n"
        with mock.patch.dict(os.environ, ENV), mock.patch.object(ci, "inspect", return_value=child), \
                mock.patch.object(ci, "lookup", return_value=tag_state), mock.patch.object(ci, "docker", side_effect=docker):
            result = ci.publish(directory)
        return result, calls

    def test_publishes_once_and_hashes_deployment_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            _, _, _, child = transaction(directory)
            result, calls = self.invoke(directory, child)
            self.assertEqual(result["status"], "PASS_PUBLISHED_AND_PULLED_CPU_VALIDATED")
            self.assertEqual(result["push_count"], 1)
            self.assertEqual(sum(args[0] == "push" for args in calls), 1)
            self.assertIn("deployment-image.txt", result["evidence_sha256"])
            self.assertEqual(result["published_image"], ci.REPOSITORY + "@" + DIGEST)

    def test_existing_publication_is_revalidated_and_reused_without_push(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            _, _, _, child = transaction(directory, reused=True)
            result, calls = self.invoke(directory, child, reused=True)
            self.assertEqual(result["push_count"], 0)
            self.assertTrue(result["reused_existing"])
            self.assertFalse(any(args[0] == "push" for args in calls))

    def test_tail_pull_failure_preserves_actual_published_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            _, _, _, child = transaction(directory)
            with self.assertRaises(RuntimeError): self.invoke(directory, child, pull_error=True)
            record = ci.read(directory / "publication.json")
            self.assertEqual(record["status"], "PUBLISHED_PULL_PENDING")
            self.assertEqual(record["digest"], DIGEST)
            self.assertEqual(record["push_count"], 1)

    def test_second_publish_in_same_transaction_is_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            _, _, _, child = transaction(directory)
            self.invoke(directory, child)
            with self.assertRaises(FileExistsError): self.invoke(directory, child)

    def test_unknown_or_existing_prepush_tag_stops_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            for state in ("UNKNOWN", "EXISTS"):
                directory = Path(temp) / state
                _, _, _, child = transaction(directory)
                with self.subTest(state=state), self.assertRaises(RuntimeError):
                    self.invoke(directory, child, tag_state=state)
                self.assertFalse((directory / "publication.json").exists())

    def test_changed_candidate_identity_stops_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            _, candidate, _, child = transaction(directory)
            candidate["recipe_sha256"] = "f" * 64
            ci.dump(directory / "candidate.json", candidate)
            with self.assertRaises(ValueError): self.invoke(directory, child)
            self.assertFalse((directory / "publication-operation.json").exists())

    def test_pulled_imageid_must_match_tested_image(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            _, _, _, child = transaction(directory)
            wrong = {**child, "Id": "sha256:" + "f" * 64}
            with mock.patch.dict(os.environ, ENV), mock.patch.object(ci, "inspect", side_effect=[child, wrong]), \
                    mock.patch.object(ci, "lookup", return_value="ABSENT_CONFIRMED"), \
                    mock.patch.object(ci, "docker", side_effect=["digest: " + DIGEST, "Digest: " + DIGEST]):
                with self.assertRaises(ValueError): ci.publish(directory)
            self.assertEqual(ci.read(directory / "publication.json")["status"], "PUBLISHED_PULL_PENDING")


class WorkflowRouting(unittest.TestCase):
    def test_worker_push_is_excluded_from_legacy_lite(self):
        text = (ROOT / ".github/workflows/dingo-router-ci.yml").read_text(encoding="utf-8")
        condition = text.split("  omni-gateway-lite:\n", 1)[1].splitlines()[0]
        self.assertIn("!contains(github.event.head_commit.message, '[h3-image-worker]')", condition)

    def test_dedicated_workflow_has_one_unpublished_build_and_terminal_evidence(self):
        text = (ROOT / ".github/workflows/h3-image-worker.yml").read_text(encoding="utf-8")
        self.assertIn("contains(github.event.head_commit.message, '[h3-image-worker]')", text)
        self.assertEqual(text.count("uses: docker/build-push-action@"), 1)
        self.assertIn("push: false", text)
        self.assertIn("load: true", text)
        self.assertIn("driver: docker", text)
        self.assertNotIn("workflow_dispatch:", text)
        self.assertNotIn("matrix:", text)
        self.assertIn("cancel-in-progress: false", text)
        self.assertEqual(text.count("if: always()"), 2)


if __name__ == "__main__":
    unittest.main()
