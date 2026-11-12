"""Offline regressions for publication identity, uncertainty and actual workflow routing."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("seedvr2_bf16_ci", HERE / "seedvr2_bf16_ci.py")
ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ci)
ops = ci.ops
COMMIT = "a" * 40
DIGEST = "sha256:" + "b" * 64
IMAGE_ID = "sha256:" + "c" * 64
ENV = {"GITHUB_REF_NAME": ci.BRANCH, "GITHUB_EVENT_NAME": "push", "GITHUB_SHA": COMMIT,
       "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "2"}
IDENTITY = {"commit": COMMIT, "run_id": "123", "run_attempt": 2}


def temporary_directory():
    parent = ci.ROOT / ".h3-experimental-evidence/seedvr2-bf16/test-tmp"
    parent.mkdir(parents=True, exist_ok=True)
    directory = tempfile.TemporaryDirectory(dir=parent)
    if not Path(directory.name).resolve().is_relative_to(parent.resolve()):
        raise ValueError("Temporary fixture must remain within its workspace evidence directory")
    return directory


def result(code, stderr="", stdout=""):
    return subprocess.CompletedProcess([], code, stdout, stderr)


def parent_config():
    return {"Id": ci.PARENT_ID, "Os": "linux", "Architecture": "amd64",
            "Config": {"User": "1000:1000", "Entrypoint": ["/entrypoint.sh"], "Cmd": ["python3", "-m", "dingo"],
                       "WorkingDir": "/workspace", "Env": ["LD_LIBRARY_PATH=/parent/cuda", "PATH=/parent/bin"],
                       "Labels": {"parent": "retained"}}, "RootFS": {"Layers": ["layer-parent"]}}


def child_config():
    image = deepcopy(parent_config())
    image["Id"] = IMAGE_ID
    image["Config"]["Env"].append("SEEDVR2_DIT_BF16=1")
    image["Config"]["Labels"].update({"org.opencontainers.image.revision": COMMIT,
                                      "org.opencontainers.image.base.name": ci.PARENT})
    image["RootFS"]["Layers"].append("layer-bf16")
    return image


class IdentityTests(unittest.TestCase):
    def test_actual_attempt_two_is_supported(self):
        self.assertEqual(ci.run_identity(ENV), IDENTITY)

    def test_bad_identity_is_rejected(self):
        for key, value in (("GITHUB_EVENT_NAME", "workflow_dispatch"), ("GITHUB_REF_NAME", "other"),
                           ("GITHUB_SHA", "abc123"), ("GITHUB_RUN_ID", "0"), ("GITHUB_RUN_ATTEMPT", "0"),
                           ("GITHUB_RUN_ATTEMPT", "01"), ("GITHUB_RUN_ATTEMPT", "-1")):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                ci.run_identity({**ENV, key: value})

    def test_same_source_other_attempt_cannot_reuse_receipt(self):
        with patch.dict(os.environ, ENV, clear=True), self.assertRaises(ValueError):
            ci.require_current({**IDENTITY, "run_attempt": 1})

    def test_tag_uses_target_repository_and_commit12(self):
        self.assertEqual(ci.image_tag(COMMIT), ops.HARBOR + ":seedvr2-dit-bf16-" + COMMIT[:12])

    def test_short_source_tag_rejected(self):
        with self.assertRaises(ValueError):
            ci.image_tag(COMMIT[:12])

    def test_digest_is_unambiguous(self):
        self.assertEqual(ci.pull_digest("Digest: " + DIGEST + "\n"), DIGEST)
        for output in ("", "digest: " + DIGEST, "Digest: " + DIGEST + "\nDigest: sha256:" + "f" * 64):
            with self.subTest(output=output), self.assertRaises(ValueError):
                ci.pull_digest(output)


class LookupTests(unittest.TestCase):
    def test_missing_manifest_is_confirmed_absent(self):
        for error in ("manifest unknown", "no such manifest", "HTTP 404: manifest unknown"):
            self.assertEqual(ci.classify_manifest_lookup(result(1, error)), "ABSENT_CONFIRMED")

    def test_exact_harbor_missing_artifact_is_confirmed_absent(self):
        image = ci.image_tag(COMMIT)
        error = "unknown: artifact openclaw/ai-dingo-vllm:" + image.split(":")[-1] + " not found"
        self.assertEqual(ci.classify_manifest_lookup(result(1, error), image), "ABSENT_CONFIRMED")
        self.assertEqual(ci.classify_manifest_lookup(result(1, error), ci.image_tag("d" * 40)), "UNKNOWN")

    def test_ambiguous_not_found_is_unknown(self):
        for error in ("not found", "404 not found", "unknown: artifact unrelated not found", ""):
            self.assertEqual(ci.classify_manifest_lookup(result(1, error), ci.image_tag(COMMIT)), "UNKNOWN")

    def test_missing_combined_with_connectivity_or_auth_is_unknown(self):
        for detail in ("timeout", "unauthorized", "status 403", "status code 500", "HTTP/1.1 502", "x509 certificate",
                       "unexpected EOF", "too many requests", "failed to fetch", "network error"):
            self.assertEqual(ci.classify_manifest_lookup(result(1, "manifest unknown\n" + detail)), "UNKNOWN")

    def test_nonstandard_exit_is_unknown(self):
        self.assertEqual(ci.classify_manifest_lookup(result(2, "manifest unknown")), "UNKNOWN")

    def test_exists(self):
        self.assertEqual(ci.classify_manifest_lookup(result(0, stdout="{}")), "EXISTS")

    def test_lookup_timeout_preserves_unknown_evidence(self):
        with temporary_directory() as folder, patch.object(ci, "command", side_effect=subprocess.TimeoutExpired("docker", 120)):
            directory = Path(folder)
            self.assertEqual(ci.lookup(ci.image_tag(COMMIT), directory), "UNKNOWN")
            self.assertEqual(ops.read(directory / "tag-inspect.txt.error.json")["type"], "TimeoutExpired")


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.lock = {"expected_environment_additions": {"SEEDVR2_DIT_BF16": "1"}}

    def test_exact_production_inheritance(self):
        ci.compare_config(parent_config(), child_config(), COMMIT, self.lock)

    def test_startup_user_and_working_directory_must_be_retained(self):
        for key in ci.CONFIG_FIELDS:
            image = child_config()
            image["Config"][key] = "changed"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "configuration changed"):
                ci.compare_config(parent_config(), image, COMMIT, self.lock)

    def test_ld_environment_is_preserved(self):
        image = child_config()
        image["Config"]["Env"][0] = "LD_LIBRARY_PATH=/other"
        with self.assertRaisesRegex(ValueError, "environment"):
            ci.compare_config(parent_config(), image, COMMIT, self.lock)

    def test_unreviewed_environment_is_rejected(self):
        image = child_config()
        image["Config"]["Env"].append("NEW_DEPENDENCY=1")
        with self.assertRaisesRegex(ValueError, "environment"):
            ci.compare_config(parent_config(), image, COMMIT, self.lock)

    def test_source_commit_environment_template_is_expanded(self):
        image = child_config()
        image["Config"]["Env"].append("SEEDVR2_BF16_SOURCE_COMMIT=" + COMMIT)
        self.lock["expected_environment_additions"]["SEEDVR2_BF16_SOURCE_COMMIT"] = "{source_commit}"
        ci.compare_config(parent_config(), image, COMMIT, self.lock)

    def test_parent_identity_is_exact(self):
        parent = parent_config()
        parent["Id"] = "sha256:" + "0" * 64
        with self.assertRaises(ValueError):
            ci.compare_config(parent, child_config(), COMMIT, self.lock)

    def test_child_platform_is_exact(self):
        for key, value in (("Architecture", "arm64"), ("Os", "windows")):
            image = child_config()
            image[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                ci.compare_config(parent_config(), image, COMMIT, self.lock)

    def test_parent_layers_cannot_be_replaced(self):
        image = child_config()
        image["RootFS"]["Layers"][0] = "different-parent"
        with self.assertRaisesRegex(ValueError, "pinned parent layer"):
            ci.compare_config(parent_config(), image, COMMIT, self.lock)

    def test_source_labels_are_required(self):
        image = child_config()
        image["Config"]["Labels"]["org.opencontainers.image.revision"] = "d" * 40
        with self.assertRaisesRegex(ValueError, "labels"):
            ci.compare_config(parent_config(), image, COMMIT, self.lock)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = temporary_directory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.env = patch.dict(os.environ, ENV, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.lock = {"expected_environment_additions": {"SEEDVR2_DIT_BF16": "1"}}

    def prepare(self, state, *, parent=None, message=None):
        def git(*args):
            return COMMIT if args[0] == "rev-parse" else message or ops.MARKER
        def docker(*args, **kwargs):
            return "Digest: " + DIGEST + "\n" if args[0] == "pull" else ""
        with patch.object(ci.ops, "check_seal", return_value=(self.lock, "tree", {"recipe_sha256": "d" * 64, "review_tree": "e" * 40})), \
                patch.object(ci.ops, "git", side_effect=git), patch.object(ci, "lookup", return_value=state), \
                patch.object(ci, "docker", side_effect=docker) as calls, \
                patch.object(ci, "inspect", side_effect=lambda image: parent or parent_config() if image == ci.PARENT else child_config()):
            value = ci.prepare(self.directory, self.directory / "github-output.txt")
            return value, calls

    def test_unknown_preparation_stops_before_parent_pull_or_build(self):
        with self.assertRaisesRegex(RuntimeError, "UNKNOWN"):
            self.prepare("UNKNOWN")
        self.assertFalse((self.directory / "preparation.json").exists())
        self.assertEqual(ops.read(self.directory / "initial-publication-observation.json")["publication_state"], "UNKNOWN")

    def test_confirmed_absence_allows_one_build(self):
        prepared, calls = self.prepare("ABSENT_CONFIRMED")
        self.assertFalse(prepared["reused"])
        self.assertIn("build_required=true", (self.directory / "github-output.txt").read_text())
        self.assertEqual(sum(call.args[0] == "pull" for call in calls.call_args_list), 1)

    def test_existing_tag_locks_immutable_digest_and_suppresses_build(self):
        prepared, calls = self.prepare("EXISTS")
        self.assertTrue(prepared["reused"])
        self.assertEqual(prepared["existing_digest"], DIGEST)
        self.assertIn("build_required=false", (self.directory / "github-output.txt").read_text())
        self.assertFalse(any(call.args[0] == "push" for call in calls.call_args_list))
        self.assertIn(ci.REPOSITORY + "@" + DIGEST, [call.args[-1] for call in calls.call_args_list])

    def test_wrong_actual_parent_imageid_stops_build(self):
        with self.assertRaisesRegex(ValueError, "production parent"):
            self.prepare("ABSENT_CONFIRMED", parent={**parent_config(), "Id": IMAGE_ID})

    def test_mixed_marker_commit_stops_before_build(self):
        with self.assertRaisesRegex(ValueError, "unique BF16"):
            self.prepare("ABSENT_CONFIRMED", message=ops.MARKER + " [h3-image-worker]")


class RuntimeEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = temporary_directory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.files = {"vllm_omni/diffusion/pipelines/seedvr2/pipeline_seedvr2.py": "d" * 64,
                      "vllm_omni/diffusion/pipelines/seedvr2/bf16_runtime.py": "e" * 64}
        self.manifest = {"source_sha256": self.files, "source_manifest_sha256": ops.hash_value(self.files)}
        self.checks = ["real conversion", "rope", "autocast", "velocity bridge", "unsafe reject"]
        self.runtime = {"schema": "seedvr2-bf16-runtime/1", "status": "PASS_CPU_ONLY", "source_commit": COMMIT,
                        "parent_image": ci.PARENT, "parent_image_id": ci.PARENT_ID, "euid": 1000,
                        "inventory_preserved": True, "errors": [], "gpu_status": "NOT_RUN", "cli_status": "PASS",
                        "dependency_status": "PASS_WITH_UPSTREAM_NCCL_OVERRIDE",
                        "tests": {"tests": 7, "errors": 0, "failures": 0, "skipped": 0},
                        "precision_cpu_checks": {"status": "PASS_REAL_CPU_TORCH_HELPERS_ONLY", "gpu_validation": "NOT_RUN",
                                                 "checks": self.checks}, "source_manifest_sha256": ops.hash_value(self.files)}
        self.proof = {"schema": "seedvr2-bf16-source-proof/1", "status": "PASS", "source_commit": COMMIT,
                      "parent_image": ci.PARENT, "parent_image_id": ci.PARENT_ID, **self.manifest}

    def validate(self):
        ops.dump(self.directory / "runtime-validation.json", self.runtime)
        ops.dump(self.directory / "source-verification.json", self.proof)
        original_read = ci.read
        def read(path):
            return self.manifest if Path(path).name == "expected-source-manifest.json" else original_read(path)
        with patch.object(ci, "read", side_effect=read), patch.object(ci.ops, "lock", return_value={"required_precision_cpu_checks": self.checks}):
            return ci.validate_reports(self.directory, COMMIT)

    def test_complete_actual_cpu_contract_passes(self):
        report = self.validate()
        self.assertEqual(report["runtime"], self.runtime)
        self.assertEqual(report["source_proof"], self.proof)

    def test_core_identity_inventory_cli_and_dependency_fail_closed(self):
        for key, value in (("schema", "other"), ("euid", 0), ("inventory_preserved", False), ("cli_status", "NOT_RUN"),
                           ("errors", ["failure"]), ("gpu_status", "PASS"), ("source_commit", "f" * 40),
                           ("parent_image_id", IMAGE_ID), ("dependency_status", "PASS_WITH_NEW_EXCEPTION")):
            original = self.runtime[key]
            self.runtime[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate()
            self.runtime[key] = original

    def test_nonempty_cpu_suite_with_zero_failures_errors_skips_is_required(self):
        for counts in ({"tests": 0, "errors": 0, "failures": 0, "skipped": 0},
                       {"tests": True, "errors": 0, "failures": 0, "skipped": 0},
                       {"tests": 3, "errors": 0, "failures": 0, "skipped": 1}, {"tests": 3}):
            self.runtime["tests"] = counts
            with self.subTest(counts=counts), self.assertRaisesRegex(ValueError, "regressions"):
                self.validate()

    def test_missing_cpu_precision_check_is_rejected(self):
        self.runtime["precision_cpu_checks"]["checks"] = self.checks[:-1]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.validate()

    def test_unproved_gpu_claim_is_rejected(self):
        self.runtime["precision_cpu_checks"]["gpu_validation"] = "PASS"
        with self.assertRaises(ValueError):
            self.validate()

    def test_extra_or_missing_installed_source_rejected(self):
        self.proof["source_sha256"] = {**self.files, "vllm_omni/extra.py": "f" * 64}
        with self.assertRaisesRegex(ValueError, "full map"):
            self.validate()

    def test_runtime_proof_manifest_hashes_must_match(self):
        self.runtime["source_manifest_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "source proof differ"):
            self.validate()

    def test_frozen_manifest_hash_must_match_its_map(self):
        self.manifest["source_manifest_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "malformed"):
            self.validate()


class ActualImageInvocationTests(unittest.TestCase):
    def test_validation_runs_exact_imageid_offline_uid1000_and_evidence_is_bound(self):
        with temporary_directory() as folder, patch.dict(os.environ, ENV, clear=True):
            directory = Path(folder)
            prepared = {**IDENTITY, "image": ci.image_tag(COMMIT), "reused": False, "existing_digest": None,
                        "recipe_sha256": "d" * 64}
            ops.dump(directory / "preparation.json", prepared)
            ops.dump(directory / "parent-inspect.json", parent_config())
            report = {"verified": "actual-runtime-evidence"}
            with patch.object(ci, "inspect", return_value=child_config()), \
                    patch.object(ci.ops, "lock", return_value={"expected_environment_additions": {"SEEDVR2_DIT_BF16": "1"}}), \
                    patch.object(ci, "docker") as docker, patch.object(ci, "validate_reports", return_value=report):
                candidate = ci.validate(directory)
            arguments = docker.call_args.args
            self.assertEqual(arguments[:8], ("run", "--rm", "--network", "none", "--user", "1000:1000", "--entrypoint", "python3"))
            self.assertIn(IMAGE_ID, arguments)
            self.assertNotIn(prepared["image"], arguments)
            self.assertIn(ci.RUNTIME, arguments)
            self.assertEqual(candidate["image_id"], IMAGE_ID)
            self.assertEqual(candidate["runtime_evidence_sha256"], ops.hash_value(report))
            self.assertEqual(candidate["gpu_validation_status"], "NOT_RUN")

    def test_timeout_keeps_partial_command_output(self):
        with temporary_directory() as folder:
            log = Path(folder) / "push.txt"
            error = subprocess.TimeoutExpired("docker", 30, output=b"layer uploaded", stderr=b"connection uncertain")
            with patch.object(ci.subprocess, "run", side_effect=error), self.assertRaises(subprocess.TimeoutExpired):
                ci.command(["docker", "push", "fixture"], log)
            self.assertIn("layer uploaded", log.read_text())
            self.assertIn("unconfirmed", log.read_text())


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = temporary_directory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.report = {"fixture": "runtime contract is supplied by coordinating runtime"}
        self.prepared = {**IDENTITY, "image": ci.image_tag(COMMIT), "reused": False, "existing_digest": None,
                         "recipe_sha256": "d" * 64}
        self.candidate = {**IDENTITY, "status": "PASS_CPU_ONLY", "image": ci.image_tag(COMMIT), "image_id": IMAGE_ID,
                          "recipe_sha256": "d" * 64, "runtime_evidence_sha256": ops.hash_value(self.report)}
        self.env = patch.dict(os.environ, ENV, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.write_records()

    def write_records(self):
        ops.dump(self.directory / "preparation.json", self.prepared)
        ops.dump(self.directory / "candidate.json", self.candidate)

    def publish(self, *, state="ABSENT_CONFIRMED", docker=None, inspected=None):
        operation = docker or (lambda *args, **kwargs: "digest: " + DIGEST if args[0] == "push" else "")
        with patch.object(ci, "lookup", return_value=state), patch.object(ci, "docker", side_effect=operation) as calls, \
                patch.object(ci, "inspect", side_effect=inspected or (lambda image: child_config())), \
                patch.object(ci, "validate_reports", return_value=self.report):
            record = ci.publish(self.directory)
            return record, calls

    def test_one_push_then_same_imageid_digest_pull(self):
        record, calls = self.publish()
        self.assertEqual(record["status"], "PASS_PUBLISHED_AND_PULLED_CPU_VALIDATED")
        self.assertEqual(record["push_count"], 1)
        self.assertEqual(sum(call.args[0] == "push" for call in calls.call_args_list), 1)
        self.assertEqual(record["published_image"], ci.REPOSITORY + "@" + DIGEST)
        self.assertEqual((self.directory / "deployment-image.txt").read_text().strip(), record["published_image"])

    def test_second_publish_cannot_push_again(self):
        self.publish()
        with self.assertRaises(FileExistsError):
            self.publish()

    def test_unknown_and_existing_before_push_do_not_push(self):
        for state in ("UNKNOWN", "EXISTS"):
            with self.subTest(state=state), temporary_directory() as folder:
                original = self.directory
                self.directory = Path(folder)
                self.write_records()
                calls = []
                def operation(*args, **kwargs):
                    calls.append(args)
                    return ""
                with self.assertRaisesRegex(RuntimeError, "confirmed absent"):
                    self.publish(state=state, docker=operation)
                self.assertEqual(calls, [])
                self.directory = original

    def test_existing_successful_tag_is_only_reused(self):
        self.prepared.update(reused=True, existing_digest=DIGEST)
        self.write_records()
        record, calls = self.publish(state="EXISTS")
        self.assertEqual(record["push_count"], 0)
        self.assertTrue(record["reused_existing"])
        self.assertFalse(any(call.args[0] == "push" for call in calls.call_args_list))

    def test_push_network_failure_leaves_claim_and_cannot_repush(self):
        calls = []
        def failing(*args, **kwargs):
            calls.append(args)
            raise RuntimeError("connection lost after push began")
        with self.assertRaisesRegex(RuntimeError, "connection lost"):
            self.publish(docker=failing)
        self.assertEqual(sum(args[0] == "push" for args in calls), 1)
        self.assertEqual(ops.read(self.directory / "publication-operation.json")["operation"], "PUSH")
        with self.assertRaises(FileExistsError):
            self.publish()

    def test_digest_pull_failure_preserves_pending_publication(self):
        def operation(*args, **kwargs):
            if args[0] == "push":
                return "digest: " + DIGEST
            raise RuntimeError("pull unavailable")
        with self.assertRaisesRegex(RuntimeError, "pull unavailable"):
            self.publish(docker=operation)
        record = ops.read(self.directory / "publication.json")
        self.assertEqual(record["status"], "PUBLISHED_PULL_PENDING")
        self.assertEqual(record["digest"], DIGEST)
        with self.assertRaises(FileExistsError):
            self.publish()

    def test_digest_imageid_mismatch_preserves_pending_state(self):
        images = [child_config(), {**child_config(), "Id": "sha256:" + "e" * 64}]
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.publish(inspected=lambda image: images.pop(0))
        self.assertEqual(ops.read(self.directory / "publication.json")["status"], "PUBLISHED_PULL_PENDING")

    def test_changed_candidate_receipt_prevents_push(self):
        for key, value in (("image_id", "sha256:" + "0" * 64), ("recipe_sha256", "0" * 64),
                           ("runtime_evidence_sha256", "0" * 64), ("run_attempt", 1)):
            candidate = {**self.candidate, key: value}
            ops.dump(self.directory / "candidate.json", candidate)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.publish()
            self.assertFalse((self.directory / "publication-operation.json").exists())

    def test_outcome_always_records_actual_unknown_without_push(self):
        with patch.object(ci, "lookup", return_value="UNKNOWN"), patch.object(ci, "docker") as docker:
            record = ci.outcome(self.directory)
        self.assertEqual(record["publication_state"], "UNKNOWN")
        docker.assert_not_called()

    def test_outcome_works_without_preparation(self):
        (self.directory / "preparation.json").unlink()
        with patch.object(ci, "lookup", return_value="EXISTS"):
            record = ci.outcome(self.directory)
        self.assertEqual(record["image"], ci.image_tag(COMMIT))
        self.assertEqual(record["publication_state"], "EXISTS")


def job_conditions(path):
    """Read actual one-line job conditions, rejecting an unsupported folded form."""
    result = {}
    active = None
    in_jobs = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line == "jobs:":
            in_jobs = True
        elif in_jobs:
            job = re.fullmatch(r"  ([A-Za-z0-9_-]+):", line)
            if job:
                active = job[1]
                result[active] = None
            elif line.startswith("    if:"):
                condition = line.split("if:", 1)[1].strip()
                if condition in ("|", ">", "|-", ">-"):
                    raise AssertionError("Multiline conditions must be added to the routing evaluator")
                result[active] = condition
    return result


def evaluate_condition(expression, message, *, event="push", branch=ops.BRANCH, recipe=""):
    if expression is None:
        return True
    expression = expression.removeprefix("${{").removesuffix("}}").strip()
    expression = re.sub(r"contains\(github\.event\.head_commit\.message, '([^']*)'\)",
                        lambda match: repr(match[1] in message), expression)
    for symbol, value in (("github.event_name", event), ("github.ref_name", branch), ("inputs.build_recipe", recipe)):
        expression = expression.replace(symbol, repr(value))
    expression = expression.replace("&&", " and ").replace("||", " or ")
    expression = re.sub(r"!(?!=)", " not ", expression)
    if re.search(r"github\.|inputs\.|contains|[{}]", expression):
        raise AssertionError("Unsupported workflow expression: " + expression)
    return bool(eval(expression.strip(), {"__builtins__": {}}, {}))


class WorkflowRoutingTests(unittest.TestCase):
    def setUp(self):
        self.workflows = ci.ROOT / ".github/workflows"
        self.names = ("dingo-router-ci.yml", "h3-image-worker.yml", "h3-sr-production.yml",
                      "h3-sr-video-gateway.yml", "omni032-h3-sr.yml", ops.WORKFLOW)

    def selected(self, message, **kwargs):
        return [(name, job) for name in self.names for job, condition in job_conditions(self.workflows / name).items()
                if evaluate_condition(condition, message, **kwargs)]

    def test_bf16_marker_routes_to_exactly_one_image_job(self):
        self.assertEqual(self.selected("Repair DiT precision " + ops.MARKER), [(ops.WORKFLOW, "reviewed-image")])

    def test_bf16_plus_any_old_marker_never_triggers_old_image_jobs(self):
        for marker in ("h3-image-worker", "h3-sr-video-gateway", "h3-sr-production", "omni032-h3-sr",
                       "seedvr2-metrics-repair", "seedvr2-optimization", "seedvr2-c14", "sr-native", "h3-hsdp-offload"):
            message = ops.MARKER + " [" + marker + "]"
            with self.subTest(marker=marker):
                self.assertEqual(self.selected(message), [(ops.WORKFLOW, "reviewed-image")])
                self.assertFalse(ops.valid_request_message(message))

    def test_wrong_branch_and_dispatch_cannot_trigger_bf16_job(self):
        self.assertEqual(self.selected(ops.MARKER, branch="other"), [])
        self.assertEqual(self.selected(ops.MARKER, event="workflow_dispatch"), [])

    def test_old_marker_routes_remain_unchanged(self):
        expected = {"h3-image-worker": "h3-image-worker.yml", "h3-sr-production": "h3-sr-production.yml",
                    "h3-sr-video-gateway": "h3-sr-video-gateway.yml", "omni032-h3-sr": "omni032-h3-sr.yml",
                    "seedvr2-c14": "dingo-router-ci.yml", "seedvr2-metrics-repair": "dingo-router-ci.yml",
                    "seedvr2-optimization": "dingo-router-ci.yml", "sr-native": "dingo-router-ci.yml"}
        for marker, workflow in expected.items():
            with self.subTest(marker=marker):
                selected = self.selected("[" + marker + "]")
                self.assertEqual(len(selected), 1)
                self.assertEqual(selected[0][0], workflow)

    def test_pipeline_loads_before_validation_then_one_publication_action(self):
        text = (self.workflows / ops.WORKFLOW).read_text()
        self.assertNotIn("workflow_dispatch:", text)
        self.assertNotIn("matrix:", text)
        self.assertEqual(text.count("push: false"), 1)
        self.assertEqual(text.count("load: true"), 1)
        self.assertEqual(text.count("ci.py publish --directory"), 1)
        self.assertLess(text.index("load: true"), text.index("ci.py validate --directory"))
        self.assertLess(text.index("ci.py validate --directory"), text.index("ci.py publish --directory"))
        self.assertEqual(text.count("if: always()"), 2)
        self.assertIn("ci.py outcome --directory", text)
        self.assertIn("actions/upload-artifact@", text)


class FreezeTests(unittest.TestCase):
    def test_only_one_marker_and_no_skip_ci(self):
        self.assertTrue(ops.valid_request_message("fix " + ops.MARKER))
        for message in (ops.MARKER + ops.MARKER, ops.MARKER + " [skip ci]", ops.MARKER + "\nskip-checks: true",
                        "fix only", ops.MARKER + " [seedvr2-other]"):
            with self.subTest(message=message):
                self.assertFalse(ops.valid_request_message(message))

    def test_review_binds_parent_imageid_and_full_review_tree(self):
        value = {"parent_image": ops.PARENT, "parent_image_id": ops.PARENT_ID}
        recipe = {"recipe_sha256": "d" * 64, "review_tree": "e" * 40}
        review = {"status": "PASS", **value, **recipe}
        ops.check_review(review, value, recipe)
        for key in review:
            with self.subTest(key=key), self.assertRaises(ValueError):
                ops.check_review({**review, key: "wrong"}, value, recipe)

    def test_arbitrary_external_review_cannot_be_imported(self):
        with self.assertRaisesRegex(ValueError, "designated coordinating root"):
            ops.import_review(Path("other-review.json"))

    def test_scope_rejects_outside_and_path_traversal(self):
        for name in ("other.py", "container/seedvr2-bf16-production/../outside.py", "delivery\\file.py"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                ops.check_scope([name])

    def test_targeted_temporary_index_handles_ignored_tracked_files_and_excludes_auxiliary(self):
        with temporary_directory() as folder:
            root = Path(folder)
            def local(*args):
                completed = subprocess.run([ops.GIT, "-C", str(root), *args], capture_output=True)
                self.assertEqual(completed.returncode, 0, completed.stderr.decode(errors="replace"))
                return completed.stdout.decode().strip()
            local("init")
            local("config", "user.name", "Local fixture")
            local("config", "user.email", "local@example.invalid")
            local("config", "core.autocrlf", "false")
            local("remote", "add", "origin", ops.REPOSITORY)
            (root / ".gitignore").write_text("container/\n.h3-experimental-evidence/\n")
            source = root / "container/seedvr2-bf16-production/runtime.py"
            source.parent.mkdir(parents=True)
            source.write_text("before\n")
            local("add", ".gitignore")
            local("add", "--force", "container/seedvr2-bf16-production/runtime.py")
            local("commit", "-m", "fixture base")
            base = local("rev-parse", "HEAD")
            index_before = local("write-tree")
            source.write_text("after\n")
            evidence = root / ".h3-experimental-evidence/local"
            with patch.object(ops, "ROOT", root), patch.object(ops, "EVIDENCE", evidence):
                value = {"recipe_base_commit": base, "parent_image": ops.PARENT, "parent_image_id": ops.PARENT_ID}
                full, recipe = ops.prospective(value)
                self.assertIn("container/seedvr2-bf16-production/runtime.py", recipe["files"])
                self.assertEqual(local("write-tree"), index_before)
                for path in ops.AUXILIARY:
                    ops.dump(root / path, {"status": "temporary"})
                full_with_aux, with_aux = ops.prospective(value)
                self.assertNotEqual(full, full_with_aux)
                self.assertEqual(recipe, with_aux)
                self.assertEqual(local("write-tree"), index_before)
                source.write_text("changed after review\n")
                self.assertNotEqual(recipe["recipe_sha256"], ops.prospective(value)[1]["recipe_sha256"])


if __name__ == "__main__":
    unittest.main()
