"""Known-result retry contracts; no Git/Docker/registry operation executes here."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("_omni032_retry_test_ci", ROOT / ".github/scripts/omni032_h3_sr_ci.py")
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)
COMMIT = "a" * 40


def environment(attempt, run_id="41388420551"):
    return {"GITHUB_SHA": COMMIT, "GITHUB_REF_NAME": ci.BRANCH,
            "GITHUB_EVENT_NAME": "push", "GITHUB_RUN_ID": run_id, "GITHUB_RUN_ATTEMPT": attempt}


class RetryIdentity(unittest.TestCase):
    def test_attempts_two_and_three_are_valid(self):
        for attempt in ("1", "2", "3", "17"):
            with self.subTest(attempt=attempt):
                self.assertEqual(ci.identity(environment(attempt)), COMMIT)

    def test_invalid_attempts_still_fail(self):
        for attempt in ("", "0", "-1", "abc", "1.5", " 2", "+2", None):
            with self.subTest(attempt=attempt), self.assertRaises(ValueError):
                ci.identity(environment(attempt))

    def test_branch_event_and_commit_guards_are_unchanged(self):
        for key, value in (("GITHUB_REF_NAME", "main"), ("GITHUB_EVENT_NAME", "workflow_dispatch"),
                           ("GITHUB_SHA", "short"), ("GITHUB_RUN_ID", "0")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                ci.identity({**environment("2"), key: value})

    def test_working_anchor_can_advance_without_relabelling_the_functional_parent(self):
        value = ci.read(ROOT / "delivery/omni032-h3-sr/source-lock.json")
        value["working_tree_base_commit"] = "b7c33431762291d31616c32dfa5368d9f77639e9"
        ci.validate_lock(value)
        self.assertEqual(value["parent_source_commit"], ci.BASE_COMMIT)
        self.assertEqual(value["recipe_base_commit"], ci.BASE_COMMIT)
        self.assertEqual(value["parent_image"], ci.PARENT)
        for working in ("short", "0" * 39, "A" * 40, None, "../anchor"):
            with self.subTest(working=working), self.assertRaises(ValueError):
                ci.validate_lock({**value, "working_tree_base_commit": working})


class AttemptScopedEvidence(unittest.TestCase):
    def workflow(self):
        import yaml
        return yaml.load((ROOT / ".github/workflows/omni032-h3-sr.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)

    def evidence_path(self, root, env):
        # Expand the actual workflow expression rather than independently
        # inventing a folder pattern that might differ from its real caller.
        template = self.workflow()["jobs"]["reviewed-image"]["env"]["EVIDENCE"]
        return root / template.replace("${{ github.run_id }}", env["GITHUB_RUN_ID"]).replace(
            "${{ github.run_attempt }}", env["GITHUB_RUN_ATTEMPT"])

    def test_prepare_two_and_three_write_separate_evidence_and_preserve_old_files(self):
        value = ci.read(ROOT / "delivery/omni032-h3-sr/source-lock.json")
        recipe = {"recipe_sha256": "b" * 64, "tree": "c" * 40}
        parent = {"Id": ci.PARENT_ID, "Os": "linux", "Architecture": "amd64"}
        def git(*args, **kwargs):
            if args[0] == "log": return "reviewed upgrade " + ci.ops.MARKER
            if args[0] == "rev-parse": return COMMIT
            if args[0] == "diff-tree": return ".github/scripts/omni032_h3_sr_ci.py"
            raise AssertionError("Unexpected Git request in retry fixture: " + repr(args))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / ".h3-experimental-evidence/omni032-h3-sr/ci/preparation.json"
            ci.dump(old, {"old": "completed pre-build failure evidence"})
            old_bytes = old.read_bytes()
            previous = None
            for attempt in ("2", "3"):
                env = environment(attempt)
                directory = self.evidence_path(root, env)
                with mock.patch.dict(os.environ, env), \
                     mock.patch.object(ci.ops, "check_seal", return_value=(value, "full-tree", recipe)), \
                     mock.patch.object(ci.ops, "git", side_effect=git), \
                     mock.patch.object(ci, "command_result", return_value=subprocess.CompletedProcess(
                         ["docker", "manifest", "inspect"], 1, "", "no such manifest: fixture final tag")), \
                     mock.patch.object(ci, "inspect", return_value=parent), \
                     mock.patch.object(ci, "docker", return_value="fixture pull") as tool:
                    record = ci.prepare(directory)
                    self.assertEqual(record["run_attempt"], int(attempt))
                    self.assertEqual(record["run_id"], env["GITHUB_RUN_ID"])
                    self.assertEqual(ci.read(directory / "preparation.json"), record)
                    self.assertEqual(old.read_bytes(), old_bytes)
                    if previous:
                        self.assertNotEqual(previous, directory)
                        self.assertTrue((previous / "preparation.json").is_file())
                    calls = len(tool.call_args_list)
                    with self.assertRaisesRegex(ValueError, "already requested"):
                        ci.prepare(directory)
                    self.assertEqual(len(tool.call_args_list), calls)
                previous = directory

    def test_run_identity_also_isolates_evidence(self):
        root = Path("fixture-root")
        self.assertNotEqual(self.evidence_path(root, environment("2", "41388420551")),
                            self.evidence_path(root, environment("2", "41388420552")))

    def test_existing_or_ambiguous_final_tag_stops_prepare_before_parent_pull(self):
        value = ci.read(ROOT / "delivery/omni032-h3-sr/source-lock.json")
        recipe = {"recipe_sha256": "b" * 64, "tree": "c" * 40}
        def git(*args, **kwargs):
            return "reviewed " + ci.ops.MARKER if args[0] == "log" else COMMIT if args[0] == "rev-parse" else ".github/scripts/omni032_h3_sr_ci.py"
        for result, expected_error in ((subprocess.CompletedProcess([], 0, '{"schemaVersion":2}', ""), ValueError),
                                       (subprocess.CompletedProcess([], 1, "", "TLS EOF"), RuntimeError)):
            with self.subTest(result=result), tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ, environment("2")), \
                 mock.patch.object(ci.ops, "check_seal", return_value=(value, "full-tree", recipe)), \
                 mock.patch.object(ci.ops, "git", side_effect=git), \
                 mock.patch.object(ci, "command_result", return_value=result), \
                 mock.patch.object(ci, "docker") as docker:
                directory = Path(temporary)
                with self.assertRaises(expected_error): ci.prepare(directory)
                docker.assert_not_called()
                self.assertFalse((directory / "preparation.json").exists())

    def test_artifact_and_all_steps_use_the_same_attempt_directory(self):
        workflow = self.workflow()
        job = workflow["jobs"]["reviewed-image"]
        self.assertNotIn("github.run_attempt == 1", job["if"])
        self.assertEqual(workflow["concurrency"]["cancel-in-progress"], "false")
        steps = job["steps"]
        login = next(index for index, step in enumerate(steps) if step.get("uses", "").startswith("docker/login-action@"))
        prepare = next(index for index, step in enumerate(steps) if "omni032_h3_sr_ci.py prepare" in step.get("run", ""))
        build = next(index for index, step in enumerate(steps) if step.get("uses", "").startswith("docker/build-push-action@"))
        self.assertLess(login, prepare)
        self.assertLess(prepare, build)
        artifact = next(step for step in steps if step.get("uses", "").startswith("actions/upload-artifact@"))
        self.assertEqual(artifact["with"]["path"], "${{ env.EVIDENCE }}")
        self.assertIn("${{ github.run_id }}", artifact["with"]["name"])
        self.assertIn("${{ github.run_attempt }}", artifact["with"]["name"])
        for step in steps:
            if "run" in step and "omni032_h3_sr_ci.py" in step["run"]:
                self.assertIn('--directory "${EVIDENCE}"', step["run"])


class SinglePublicationPerAttempt(unittest.TestCase):
    def test_attempt_two_still_claims_publication_once(self):
        env = environment("2")
        image = ci.REPOSITORY + ":omni032-test-" + COMMIT[:12]
        image_id = "sha256:" + "d" * 64
        digest = "sha256:" + "e" * 64
        manifest = "f" * 64
        def docker(*args, **kwargs):
            return "digest: " + digest if args[0] == "push" else "fixture pulled"
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ, env):
            directory = Path(temporary) / (env["GITHUB_RUN_ID"] + "-" + env["GITHUB_RUN_ATTEMPT"])
            ci.dump(directory / "candidate.json", {"status": "PASS_CPU_ONLY", "commit": COMMIT,
                    "image": image, "image_id": image_id, "source_manifest_sha256": manifest,
                    "run_id": env["GITHUB_RUN_ID"], "run_attempt": 2})
            with mock.patch.object(ci, "inspect", return_value={"Id": image_id}), \
                 mock.patch.object(ci, "validate_reports", return_value={"source_manifest_sha256": manifest}), \
                 mock.patch.object(ci, "docker", side_effect=docker) as tool:
                receipt = ci.publish(directory)
                self.assertEqual(receipt["published_image"], ci.REPOSITORY + "@" + digest)
                request = ci.read(directory / "publication-request.json")
                self.assertEqual(request["request_count"], 1)
                for record in (receipt, request):
                    self.assertEqual(record["run_id"], env["GITHUB_RUN_ID"])
                    self.assertEqual(record["run_attempt"], 2)
                with self.assertRaises(FileExistsError): ci.publish(directory)
                self.assertEqual(sum(call.args[0] == "push" for call in tool.call_args_list), 1)


class StageAttemptBinding(unittest.TestCase):
    def test_copied_same_commit_preparation_cannot_validate_in_another_attempt(self):
        env = environment("2")
        for old in ({"run_id": env["GITHUB_RUN_ID"], "run_attempt": 1},
                    {"run_id": "41388420550", "run_attempt": 2},
                    {"run_id": env["GITHUB_RUN_ID"], "run_attempt": "2"}, {}):
            with self.subTest(old=old), tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ, env):
                directory = Path(temporary)
                ci.dump(directory / "preparation.json", {"commit": COMMIT, **old})
                with mock.patch.object(ci, "inspect") as inspect, mock.patch.object(ci, "docker") as docker:
                    with self.assertRaisesRegex(ValueError, "another Actions run/attempt"): ci.validate(directory)
                    inspect.assert_not_called()
                    docker.assert_not_called()

    def test_copied_same_commit_pass_candidate_cannot_publish_in_another_attempt(self):
        env = environment("3")
        for old in ({"run_id": env["GITHUB_RUN_ID"], "run_attempt": 2},
                    {"run_id": "41388420550", "run_attempt": 3},
                    {"run_id": env["GITHUB_RUN_ID"], "run_attempt": "3"}, {}):
            with self.subTest(old=old), tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ, env):
                directory = Path(temporary)
                ci.dump(directory / "candidate.json", {"commit": COMMIT, "status": "PASS_CPU_ONLY", **old})
                with mock.patch.object(ci, "inspect") as inspect, mock.patch.object(ci, "docker") as docker:
                    with self.assertRaisesRegex(ValueError, "another Actions run/attempt"): ci.publish(directory)
                    inspect.assert_not_called()
                    docker.assert_not_called()
                    self.assertFalse((directory / "publication-request.json").exists())

    def test_validated_candidate_records_the_current_actual_attempt(self):
        env = environment("3")
        image_id = "sha256:" + "d" * 64
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ, env):
            directory = Path(temporary)
            ci.dump(directory / "preparation.json", {"commit": COMMIT, "run_id": env["GITHUB_RUN_ID"],
                    "run_attempt": 3, "image": "fixture final tag", "recipe_sha256": "a" * 64})
            ci.dump(directory / "parent-inspect.json", {})
            with mock.patch.object(ci, "inspect", return_value={"Id": image_id}), \
                 mock.patch.object(ci, "compare_config", return_value={"status": "PASS"}), \
                 mock.patch.object(ci, "copy_reports"), \
                 mock.patch.object(ci, "validate_reports", return_value={"source_manifest_sha256": "f" * 64}), \
                 mock.patch.object(ci, "docker", return_value="fixture actual-image validation") as tool:
                record = ci.validate(directory)
                self.assertEqual(record["run_id"], env["GITHUB_RUN_ID"])
                self.assertEqual(record["run_attempt"], 3)
                self.assertEqual(ci.read(directory / "candidate.json"), record)
                self.assertEqual(tool.call_args.args[0], "run")
                self.assertIn(image_id, tool.call_args.args)


class StableTagInspection(unittest.TestCase):
    def result(self, code, stdout="", stderr=""):
        return subprocess.CompletedProcess(["docker", "manifest", "inspect", "fixture tag"], code, stdout, stderr)

    def test_only_explicit_missing_manifest_diagnostics_allow_build(self):
        for stderr in ("no such manifest: fixture tag", "manifest unknown: manifest unknown"):
            with self.subTest(stderr=stderr):
                self.assertEqual(ci.classify_manifest_lookup(self.result(1, stderr=stderr)), "ABSENT_CONFIRMED")
        self.assertEqual(ci.classify_manifest_lookup(self.result(0, stdout='{"schemaVersion":2}')), "EXISTS")

    def test_auth_transport_and_generic_failures_are_unknown(self):
        for stderr in ("unauthorized: authentication required", "denied: access forbidden", "x509: certificate failed",
                       "TLS handshake failed", "unexpected EOF", "dial tcp: connection timed out", "not found",
                       "unknown Docker error", "manifest unknown\nunauthorized", "no such manifest\nnetwork is unreachable",
                       "no such manifest\nfailed to authorize: bad gateway", "manifest unknown\npermission denied"):
            with self.subTest(stderr=stderr):
                self.assertEqual(ci.classify_manifest_lookup(self.result(1, stderr=stderr)), "UNKNOWN")

    def test_rate_limit_and_server_failures_mixed_with_missing_are_unknown(self):
        for detail in ("manifest unknown\nHTTP 500 Internal Server Error",
                       "no such manifest\nTOOMANYREQUESTS: too many requests",
                       "manifest unknown\nHTTP/1.1 503 Service Unavailable",
                       "no such manifest\nHTTP429",
                       "manifest unknown\nresponse status code: 502",
                       "no such manifest\nrate limit exceeded",
                       "manifest unknown\nHTTP 403",
                       "no such manifest\nHTTP 404\nHTTP 500"):
            with self.subTest(detail=detail):
                self.assertEqual(ci.classify_manifest_lookup(self.result(1, stderr=detail)), "UNKNOWN")
        self.assertEqual(ci.classify_manifest_lookup(self.result(-1, stderr="no such manifest")), "UNKNOWN")
        self.assertEqual(ci.classify_manifest_lookup(self.result(1, stderr="HTTP 404: manifest unknown")), "ABSENT_CONFIRMED")

    def test_existing_or_unknown_tag_stops_and_records_evidence(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ, environment("2")):
            directory = Path(temporary)
            for result, expected, error in ((self.result(0, stdout='{"schemaVersion":2}'), "EXISTS", ValueError),
                                           (self.result(1, stderr="unexpected EOF"), "UNKNOWN", RuntimeError)):
                with self.subTest(expected=expected), mock.patch.object(ci, "command_result", return_value=result):
                    with self.assertRaises(error): ci.confirm_tag_absent("fixture final tag", directory)
                    record = ci.read(directory / "tag-inspection.json")
                    self.assertEqual(record["status"], expected)
                    self.assertEqual(record["run_attempt"], 2)

    def test_successful_push_then_failed_pull_is_not_treated_as_unpublished(self):
        env = environment("2")
        image_id, manifest = "sha256:" + "d" * 64, "f" * 64
        image = ci.REPOSITORY + ":fixture-final-tag"
        def docker(*args, **kwargs):
            if args[0] == "push": return "digest: sha256:" + "e" * 64
            if args[0] == "pull": raise RuntimeError("fixture post-push pull TLS failure")
            raise AssertionError(args)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ, env):
            directory = Path(temporary) / "attempt2"
            ci.dump(directory / "candidate.json", {"status": "PASS_CPU_ONLY", "commit": COMMIT, "image": image,
                    "image_id": image_id, "source_manifest_sha256": manifest,
                    "run_id": env["GITHUB_RUN_ID"], "run_attempt": 2})
            with mock.patch.object(ci, "inspect", return_value={"Id": image_id}), \
                 mock.patch.object(ci, "validate_reports", return_value={"source_manifest_sha256": manifest}), \
                 mock.patch.object(ci, "docker", side_effect=docker) as tool:
                with self.assertRaisesRegex(RuntimeError, "post-push pull"): ci.publish(directory)
                self.assertTrue((directory / "publication-request.json").is_file())
                self.assertFalse((directory / "publication.json").exists())
                self.assertEqual(sum(call.args[0] == "push" for call in tool.call_args_list), 1)
            later = Path(temporary) / "attempt3"
            later.mkdir()
            with mock.patch.dict(os.environ, environment("3")), \
                 mock.patch.object(ci, "command_result", return_value=self.result(0, stdout='{"schemaVersion":2}')):
                with self.assertRaisesRegex(ValueError, "already exists"): ci.confirm_tag_absent(image, later)
                self.assertEqual(ci.read(later / "tag-inspection.json")["status"], "EXISTS")


if __name__ == "__main__": unittest.main()
