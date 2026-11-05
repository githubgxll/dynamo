"""Production release regressions: subprocess/API/Docker boundaries are mocked."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("h3_production_ci_under_test", ROOT / ".github/scripts/h3_sr_production_ci.py")
ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ci)
ops = ci.ops
COMMIT = "a" * 40
DIGEST = "sha256:" + "b" * 64
CHILD_ID = "sha256:" + "c" * 64
IMAGE = ci.REPOSITORY + ":h3-sr-production-" + COMMIT[:12]
CURRENT = {"commit": COMMIT, "run_id": "431", "run_attempt": 3}
ENV = {"GITHUB_REF_NAME": ops.BRANCH, "GITHUB_EVENT_NAME": "push", "GITHUB_SHA": COMMIT,
       "GITHUB_RUN_ID": "431", "GITHUB_RUN_ATTEMPT": "3"}


def value():
    return {"repository": ops.REPOSITORY, "remote_branch": ops.BRANCH, "harbor_repository": ops.HARBOR,
            "parent_image": ops.PARENT, "parent_image_id": ops.PARENT_ID, "platform": "linux/amd64",
            "recipe_base_commit": ops.BASE, "tag_prefix": "h3-sr-production", "working_tree_base_commit": ops.BASE,
            "versions": {"vllm-omni": "0.32.0", "torch": "2.13.0+cu130"}}


def configs():
    parent = {"Id": ci.PARENT_ID, "Os": "linux", "Architecture": "amd64", "RootFS": {"Layers": ["layer1"]},
              "Config": {"User": "dynamo", "Env": ["PATH=/usr/bin", "TORCH_CUDA_ARCH_LIST=9.0"],
                         "Entrypoint": ["/entrypoint.sh"], "Cmd": ["serve"], "WorkingDir": "/workspace"}}
    child = copy.deepcopy(parent)
    child["Id"] = CHILD_ID
    child["RootFS"]["Layers"].append("child-layer")
    child["Config"]["Env"] += ["H3_SR_PRODUCTION_SOURCE_COMMIT=" + COMMIT, "H3_SR_PRODUCTION_PARENT_IMAGE=" + ci.PARENT]
    child["Config"]["Labels"] = {"org.opencontainers.image.revision": COMMIT, "org.opencontainers.image.base.name": ci.PARENT}
    return parent, child


def report_files(root):
    files = {"vllm_omni/a.py": "d" * 64, "dingo/video_gateway/gateway.py": "e" * 64}
    expected = {"source_sha256": files, "source_manifest_sha256": ops.hash_value(files)}
    ci.dump(root / "delivery/h3-sr-production/expected-source-manifest.json", expected)
    output = root / "final"
    runtime = {"status": "PASS_CPU_ONLY", "source_commit": COMMIT, "parent_image": ci.PARENT, "euid": 1000,
               "errors": [], "tests": {"tests": 12, "failures": 0, "errors": 0, "skipped": 0},
               "source_manifest_sha256": ops.hash_value(files)}
    proof = {"status": "PASS", "source_commit": COMMIT, **expected}
    ci.dump(output / "runtime-validation.json", runtime)
    ci.dump(output / "source-verification.json", proof)
    return output, runtime, proof


def observation():
    return {"schema": "h3-sr-production-publication-observation/1", **CURRENT, "image": IMAGE,
            "publication_state": "ABSENT_CONFIRMED", "observed_at": "2026-11-05T10:00:00+00:00"}


def getter_fixture(conclusion="failure", status="completed", attempts=("skipped",), changing=False):
    run = {"id": 431, "head_branch": ops.BRANCH, "path": ".github/workflows/" + ops.WORKFLOW,
           "head_sha": COMMIT, "run_attempt": len(attempts), "status": status, "conclusion": conclusion,
           "updated_at": "2026-11-05T10:01:00Z", "html_url": "https://github.com/githubgxll/dynamo/actions/runs/431"}
    reads = 0

    def getter(endpoint):
        nonlocal reads
        if endpoint.startswith("/actions/workflows/"):
            return {"total_count": 1, "workflow_runs": [copy.deepcopy(run)]}
        if endpoint == "/actions/runs/431":
            reads += 1
            result = copy.deepcopy(run)
            if changing and reads > 1:
                result["run_attempt"] += 1
            return result
        match = re.fullmatch(r"/actions/runs/431/attempts/(\d+)/jobs\?per_page=100", endpoint)
        if not match:
            raise AssertionError("Unexpected API endpoint: " + endpoint)
        number = int(match[1])
        steps = [{"name": ops.PUBLISH_STEP, "conclusion": attempts[number - 1]}]
        return {"total_count": 1, "jobs": [{"id": 100 + number, "run_id": 431, "run_attempt": number,
                "head_sha": COMMIT, "status": status, "conclusion": conclusion, "steps": steps}]}
    return getter


class ManifestClassification(unittest.TestCase):
    def classify(self, code, text):
        return ci.classify_manifest_lookup(SimpleNamespace(returncode=code, stdout="", stderr=text), IMAGE)

    def test_success_means_existing(self):
        self.assertEqual(self.classify(0, ""), "EXISTS")

    def test_manifest_unknown_or_exact_harbor_missing_confirms_absence(self):
        self.assertEqual(self.classify(1, "manifest unknown"), "ABSENT_CONFIRMED")
        artifact = IMAGE.split("/", 1)[1]
        self.assertEqual(self.classify(1, "unknown: artifact " + artifact + " not found"), "ABSENT_CONFIRMED")

    def test_wrong_repository_or_generic_not_found_is_unknown(self):
        for text in ("unknown: artifact other/repo:tag not found", "404 not found", "repository not found"):
            with self.subTest(text=text): self.assertEqual(self.classify(1, text), "UNKNOWN")

    def test_eof_auth_rate_limit_and_errors_cannot_confirm_absence(self):
        for text in ("EOF", "manifest unknown; TLS error", "manifest unknown unauthorized", "manifest unknown status: 429",
                     "manifest unknown status code 500", "no such manifest: proxyconnect failed", "manifest unknown rate limit"):
            with self.subTest(text=text): self.assertEqual(self.classify(1, text), "UNKNOWN")

    def test_tool_failure_never_confirms_absence(self):
        self.assertEqual(self.classify(2, "manifest unknown"), "UNKNOWN")

    def test_pull_digest_must_be_exact_and_unambiguous(self):
        self.assertEqual(ci.pull_digest("Digest: " + DIGEST + "\nStatus: pulled"), DIGEST)
        for text in ("", "Digest: sha256:short", "Digest: " + DIGEST + "\nDigest: sha256:" + "d" * 64):
            with self.subTest(text=text), self.assertRaises(ValueError): ci.pull_digest(text)


class SourceAndIdentity(unittest.TestCase):
    def test_attempt_above_one_is_valid(self):
        self.assertEqual(ci.run_identity(ENV), CURRENT)

    def test_wrong_branch_event_or_incomplete_identity_is_rejected(self):
        for key, val in (("GITHUB_REF_NAME", "main"), ("GITHUB_EVENT_NAME", "workflow_dispatch"),
                         ("GITHUB_SHA", "abc"), ("GITHUB_RUN_ATTEMPT", "0"), ("GITHUB_RUN_ID", "")):
            with self.subTest(key=key), self.assertRaises(ValueError): ci.run_identity({**ENV, key: val})

    def test_unique_marker_required_and_skip_ci_forbidden(self):
        self.assertTrue(ops.valid_request_message("production " + ops.MARKER))
        for text in ("no marker", ops.MARKER + ops.MARKER, ops.MARKER + " [omni032-h3-sr]",
                     ops.MARKER + " [omni-gateway-lite]", ops.MARKER + " [seedvr2-c14]",
                     ops.MARKER + " [sr-native]", ops.MARKER + " [SKIP CI]", ops.MARKER + "\nskip-checks: true"):
            with self.subTest(text=text): self.assertFalse(ops.valid_request_message(text))

    def test_lock_rejects_parent_native_upgrade_and_target_drift(self):
        ops.validate_lock(value())
        for key in ("parent_image", "parent_image_id", "remote_branch", "harbor_repository", "recipe_base_commit"):
            item = value(); item[key] = "wrong"
            with self.subTest(key=key), self.assertRaises(ValueError): ops.validate_lock(item)
        item = value(); item["versions"]["torch"] = "new"
        with self.assertRaises(ValueError): ops.validate_lock(item)

    def test_scope_cannot_include_unrelated_changes(self):
        ops.check_scope(["container/h3-sr-production/runtime.py", "dingo/video_gateway/jobs.py", ".github/workflows/h3-sr-production.yml"])
        for path in ("README.md", "container/sr-native/run.py", "../private", ".github/workflows/other.yml"):
            with self.subTest(path=path), self.assertRaises(ValueError): ops.check_scope([path])

    def test_final_reviews_bind_recipe_parent_and_tree(self):
        recipe = {"recipe_sha256": "d" * 64, "review_tree": "e" * 40}
        approved = {"schema": "h3-sr-production-freeze/1", "status": "READY_FOR_BUILD", **recipe,
                    "parent_image": ops.PARENT, "reviews": {"root": "PASS", "k8s": {"status": "PASS"}}}
        ops.check_approval(approved, value(), recipe)
        for key in ("recipe_sha256", "review_tree", "parent_image", "status"):
            with self.subTest(key=key), self.assertRaises(ValueError): ops.check_approval({**approved, key: "wrong"}, value(), recipe)
        with self.assertRaises(ValueError): ops.check_approval({**approved, "reviews": {"root": "PASS"}}, value(), recipe)


class RuntimeProof(unittest.TestCase):
    def test_actual_complete_source_manifest_passes(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(ci, "ROOT", Path(temp)):
            output, _, _ = report_files(Path(temp))
            ci.validate_reports(output, COMMIT)

    def test_zero_skipped_failed_or_wrong_runtime_is_rejected(self):
        for mutate in ({"tests": 0}, {"tests": {"tests": 3, "skipped": 1}}, {"euid": 0}, {"errors": ["bad"]},
                       {"parent_image": "old"}, {"source_commit": "e" * 40}):
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as temp, mock.patch.object(ci, "ROOT", Path(temp)):
                output, runtime, _ = report_files(Path(temp)); ci.dump(output / "runtime-validation.json", {**runtime, **mutate})
                with self.assertRaises(ValueError): ci.validate_reports(output, COMMIT)

    def test_missing_changed_or_hash_only_source_proof_is_rejected(self):
        for mode in ("file", "missing", "hash", "source_commit"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp, mock.patch.object(ci, "ROOT", Path(temp)):
                output, _, proof = report_files(Path(temp))
                if mode == "file": proof["source_sha256"]["extra.py"] = "f" * 64
                elif mode == "missing": proof.pop("source_sha256")
                elif mode == "hash": proof["source_manifest_sha256"] = "bad"
                else: proof["source_commit"] = "f" * 40
                ci.dump(output / "source-verification.json", proof)
                with self.assertRaises(ValueError): ci.validate_reports(output, COMMIT)

    def test_expected_map_canonical_hash_is_checked(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(ci, "ROOT", Path(temp)):
            output, _, _ = report_files(Path(temp))
            ci.dump(Path(temp) / "delivery/h3-sr-production/expected-source-manifest.json",
                    {"source_sha256": {"x.py": "a"}, "source_manifest_sha256": "wrong"})
            with self.assertRaises(ValueError): ci.validate_reports(output, COMMIT)

    def test_child_inherits_all_parent_layers_config_and_environment(self):
        parent, child = configs(); ci.compare_config(parent, child, COMMIT, value())

    def test_native_or_service_identity_drift_is_rejected(self):
        mutations = (lambda c: c["RootFS"]["Layers"].__setitem__(0, "wrong"),
                     lambda c: c.__setitem__("Architecture", "arm64"),
                     lambda c: c["Config"].__setitem__("User", "root"),
                     lambda c: c["Config"].__setitem__("Cmd", ["other"]),
                     lambda c: c["Config"]["Env"].__setitem__(0, "PATH=/wrong"),
                     lambda c: c["Config"]["Env"].append("UNREVIEWED=yes"),
                     lambda c: c["Config"]["Labels"].__setitem__("org.opencontainers.image.revision", "wrong"))
        for mutation in mutations:
            parent, child = configs(); mutation(child)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): ci.compare_config(parent, child, COMMIT, value())


class PublicationTransaction(unittest.TestCase):
    def records(self, folder, reused=False):
        prepared = {**CURRENT, "image": IMAGE, "reused": reused, "existing_digest": DIGEST if reused else None,
                    "recipe_sha256": "d" * 64}
        candidate = {**CURRENT, "status": "PASS_CPU_ONLY", "image_id": CHILD_ID, "source_manifest_sha256": "e" * 64}
        ci.dump(folder / "preparation.json", prepared); ci.dump(folder / "candidate.json", candidate)
        return prepared, candidate

    def test_existing_tag_skips_build_and_retains_exact_digest(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(os.environ, ENV), \
             mock.patch.object(ops, "check_seal", return_value=(value(), "tree", {"recipe_sha256": "d", "review_tree": "t"})), \
             mock.patch.object(ops, "git", side_effect=lambda *args, **kwargs: ops.MARKER if args[0] == "log" else COMMIT if args[0] == "rev-parse" else "container/h3-sr-production/runtime.py"), \
             mock.patch.object(ci, "lookup", return_value="EXISTS"), mock.patch.object(ci, "inspect", return_value=configs()[0]), \
             mock.patch.object(ci, "docker", return_value="Digest: " + DIGEST) as docker:
            directory = Path(temp) / "attempt3"; outputs = Path(temp) / "outputs"
            result = ci.prepare(directory, outputs)
            self.assertTrue(result["reused"]); self.assertEqual(result["existing_digest"], DIGEST)
            self.assertIn("build_required=false", outputs.read_text())
            self.assertFalse(any(call.args[0] == "push" for call in docker.call_args_list))
            with self.assertRaises(ValueError): ci.prepare(directory, outputs)

    def test_unknown_lookup_stops_before_parent_or_image_pull(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(os.environ, ENV), \
             mock.patch.object(ops, "check_seal", return_value=(value(), "tree", {})), \
             mock.patch.object(ops, "git", side_effect=lambda *args, **kwargs: ops.MARKER if args[0] == "log" else COMMIT if args[0] == "rev-parse" else "container/h3-sr-production/runtime.py"), \
             mock.patch.object(ci, "lookup", return_value="UNKNOWN"), mock.patch.object(ci, "docker") as docker:
            with self.assertRaises(RuntimeError): ci.prepare(Path(temp) / "attempt3")
            docker.assert_not_called()

    def test_final_validation_uses_actual_imageid_nonroot_offline(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(os.environ, ENV), \
             mock.patch.object(ops, "lock", return_value=value()), mock.patch.object(ci, "compare_config"), \
             mock.patch.object(ci, "inspect", return_value=configs()[1]), \
             mock.patch.object(ci, "validate_reports", return_value=({"source_manifest_sha256": "e" * 64}, {})), \
             mock.patch.object(ci, "docker", return_value="") as docker:
            folder = Path(temp); self.records(folder); ci.dump(folder / "parent-inspect.json", configs()[0])
            ci.validate(folder)
            args = docker.call_args.args
            self.assertIn(CHILD_ID, args); self.assertNotIn(IMAGE, args)
            self.assertEqual(args[args.index("--network") + 1], "none")
            self.assertEqual(args[args.index("--user") + 1], "1000:0")
            self.assertIn("/tmp/h3-sr-production-evidence", args)

    def test_new_image_pushes_once_and_pulls_same_imageid(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(os.environ, ENV), \
             mock.patch.object(ci, "inspect", return_value={"Id": CHILD_ID}), \
             mock.patch.object(ci, "validate_reports"), mock.patch.object(ci, "lookup", return_value="ABSENT_CONFIRMED"), \
             mock.patch.object(ci, "docker", return_value="digest: " + DIGEST) as docker:
            folder = Path(temp); self.records(folder); result = ci.publish(folder)
            self.assertEqual(result["status"], "PASS_PUBLISHED_AND_PULLED_CPU_VALIDATED")
            self.assertEqual(sum(call.args[0] == "push" for call in docker.call_args_list), 1)
            self.assertEqual((folder / "deployment-image.txt").read_text().strip(), ci.REPOSITORY + "@" + DIGEST)
            with self.assertRaises(FileExistsError): ci.publish(folder)

    def test_existing_artifact_is_revalidated_and_never_pushed(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(os.environ, ENV), \
             mock.patch.object(ci, "inspect", return_value={"Id": CHILD_ID}), \
             mock.patch.object(ci, "validate_reports"), mock.patch.object(ci, "docker", return_value="") as docker:
            folder = Path(temp); self.records(folder, reused=True); result = ci.publish(folder)
            self.assertTrue(result["reused_existing"]); self.assertEqual(result["push_count"], 0)
            self.assertFalse(any(call.args[0] == "push" for call in docker.call_args_list))

    def test_tail_pull_failure_preserves_published_digest(self):
        def docker(*args, **kwargs):
            if args[0] == "push": return "digest: " + DIGEST
            raise RuntimeError("TLS EOF during pull")
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(os.environ, ENV), \
             mock.patch.object(ci, "inspect", return_value={"Id": CHILD_ID}), mock.patch.object(ci, "validate_reports"), \
             mock.patch.object(ci, "lookup", return_value="ABSENT_CONFIRMED"), mock.patch.object(ci, "docker", side_effect=docker):
            folder = Path(temp); self.records(folder)
            with self.assertRaises(RuntimeError): ci.publish(folder)
            record = ci.read(folder / "publication.json")
            self.assertEqual(record["status"], "PUBLISHED_PULL_PENDING"); self.assertEqual(record["digest"], DIGEST)

    def test_pre_push_uncertainty_and_imageid_change_stop_without_push(self):
        for status, image_id in (("UNKNOWN", CHILD_ID), ("EXISTS", CHILD_ID), ("ABSENT_CONFIRMED", "wrong")):
            with self.subTest(status=status, image_id=image_id), tempfile.TemporaryDirectory() as temp, \
                 mock.patch.dict(os.environ, ENV), mock.patch.object(ci, "inspect", return_value={"Id": image_id}), \
                 mock.patch.object(ci, "validate_reports"), mock.patch.object(ci, "lookup", return_value=status), \
                 mock.patch.object(ci, "docker") as docker:
                folder = Path(temp); self.records(folder)
                with self.assertRaises((ValueError, RuntimeError)): ci.publish(folder)
                docker.assert_not_called()

    def test_outcome_checks_actual_registry_even_when_prepare_failed(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(os.environ, ENV), \
             mock.patch.object(ci, "lookup", return_value="UNKNOWN") as lookup:
            folder = Path(temp); result = ci.outcome(folder)
            self.assertEqual(result["publication_state"], "UNKNOWN")
            self.assertEqual(lookup.call_args.args[0], IMAGE)
            self.assertTrue((folder / "publication-observation.json").exists())


class SequentialSubmission(unittest.TestCase):
    def test_no_prior_request_needs_no_observation(self):
        self.assertIsNone(ops.previous_result(None))

    def test_all_attempts_inspected_and_earlier_publication_is_preserved(self):
        result = ops.observe_run(431, getter_fixture(attempts=("success", "failure", "skipped")))
        self.assertEqual(result["run_attempt"], 3)
        self.assertEqual(len(result["history"]), 3)
        self.assertEqual(result["publication_status"], "published")

    def test_active_unknown_or_successful_result_blocks_new_request(self):
        for state in ("queued", "in_progress"):
            with self.subTest(state=state), self.assertRaises(ValueError):
                ops.previous_result({"commit": COMMIT}, getter_fixture(status=state))
        for step in ("success", "failure"):
            with self.subTest(step=step), self.assertRaises(ValueError):
                ops.previous_result({"commit": COMMIT}, getter_fixture(attempts=(step,)))
        with self.assertRaises(ValueError): ops.previous_result({"status": "PUBLISHED", "image": "existing"})

    def test_failure_retries_require_actual_absence_latest_attempt(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "observation.json"; ci.dump(path, observation())
            journal = ops.record_outcome(path, {"commit": COMMIT, "sequence": 1})
            result = ops.previous_result(journal, getter_fixture(attempts=("skipped", "skipped", "failure")))
            self.assertEqual(result["run_attempt"], 3)
            with self.assertRaises(ValueError): ops.previous_result(journal, getter_fixture(attempts=("skipped",) * 4))

    def test_exists_unknown_old_commit_or_changed_receipt_cannot_unblock(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "observation.json"
            for state in ("EXISTS", "UNKNOWN"):
                ci.dump(path, {**observation(), "publication_state": state})
                journal = ops.record_outcome(path, {"commit": COMMIT})
                with self.subTest(state=state), self.assertRaises(ValueError): ops.previous_result(journal, getter_fixture(attempts=("skipped",) * 3))
            ci.dump(path, {**observation(), "commit": "f" * 40})
            with self.assertRaises(ValueError): ops.record_outcome(path, {"commit": COMMIT})
            ci.dump(path, observation()); journal = ops.record_outcome(path, {"commit": COMMIT})
            ci.dump(path, {**observation(), "observed_at": "2026-11-05T11:00:00Z"})
            with self.assertRaises(ValueError): ops.previous_result(journal, getter_fixture(attempts=("skipped",) * 3))

    def test_changed_run_or_incomplete_history_stops_observation(self):
        with self.assertRaises(ValueError): ops.observe_run(431, getter_fixture(changing=True))
        base = getter_fixture()
        def incomplete(endpoint):
            data = base(endpoint)
            if "jobs?" in endpoint: data["total_count"] = 2
            return data
        with self.assertRaises(ValueError): ops.observe_run(431, incomplete)

    def test_zero_or_multiple_matching_runs_are_unknown(self):
        for count in (0, 2):
            with self.subTest(count=count), self.assertRaises(ValueError):
                ops.previous_result({"commit": COMMIT}, lambda endpoint: {"total_count": count, "workflow_runs": [{}] * count})

    def test_submit_requires_explicit_current_authorization(self):
        with self.assertRaises(ValueError), mock.patch.object(ops, "git") as git:
            ops.submit(False)
        git.assert_not_called()

    def test_pending_publication_receipt_cannot_claim_success(self):
        with tempfile.TemporaryDirectory() as temp:
            receipt = Path(temp) / "publication.json"
            ci.dump(receipt, {"status": "PUBLISHED_PULL_PENDING", "commit": COMMIT,
                              "published_image": ci.REPOSITORY + "@" + DIGEST})
            with self.assertRaises(ValueError): ops.record_publication(receipt, {"commit": COMMIT, "recipe_sha256": "d" * 64})

    def test_publication_import_binds_complete_evidence_and_tested_imageid(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            files = {"vllm_omni/example.py": "e" * 64}; manifest = ops.hash_value(files)
            evidence = {"candidate.json": {"status": "PASS_CPU_ONLY", "commit": COMMIT, "image_id": CHILD_ID,
                                           "recipe_sha256": "d" * 64, "source_manifest_sha256": manifest},
                        "preparation.json": {}, "parent-inspect.json": {"Id": ci.PARENT_ID},
                        "final/runtime-validation.json": {"status": "PASS_CPU_ONLY", "euid": 1000, "errors": [],
                            "tests": {"tests": 5, "errors": 0, "failures": 0, "skipped": 0}, "source_commit": COMMIT,
                            "parent_image": ci.PARENT, "source_manifest_sha256": manifest},
                        "final/source-verification.json": {"status": "PASS", "source_commit": COMMIT,
                            "source_sha256": files, "source_manifest_sha256": manifest}}
            for name, data in evidence.items(): ci.dump(folder / name, data)
            payload = {"schema": "h3-sr-production-publication/1", "status": "PASS_PUBLISHED_AND_PULLED_CPU_VALIDATED",
                       "commit": COMMIT, "parent_image": ci.PARENT, "recipe_sha256": "d" * 64, "image": IMAGE,
                       "image_id": CHILD_ID, "digest": DIGEST, "published_image": ci.REPOSITORY + "@" + DIGEST,
                       "source_manifest_sha256": manifest,
                       "evidence_sha256": {name: ci.hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in evidence}}
            receipt = folder / "publication.json"; ci.dump(receipt, payload)
            journal = {"commit": COMMIT, "recipe_sha256": "d" * 64}
            self.assertEqual(ops.record_publication(receipt, journal)["status"], "PUBLISHED")
            (folder / "candidate.json").write_text("{}")
            with self.assertRaises(ValueError): ops.record_publication(receipt, journal)


class LocalChecks(unittest.TestCase):
    def test_explicit_local_dependency_target_only_changes_test_child_environment(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(os.environ, {"H3_SR_CHECK_DEPS": temp, "PYTHONPATH": "old"}), \
             mock.patch.object(ops, "EVIDENCE", Path(temp)), mock.patch.object(ops, "lock", return_value=value()), \
             mock.patch.object(ops, "prospective", return_value=("tree", {"recipe_sha256": "d", "review_tree": "t"})), \
             mock.patch.object(ops, "check_syntax", return_value={"python_syntax_files": 3, "workflow_shell_blocks": 4}), \
             mock.patch.object(ops.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="Ran 7 tests in 0.1s\nOK", stderr="")) as run:
            result = ops.run_local_checks()
            self.assertEqual(result["tests"], {"production-runtime": 7, "production-ci": 7})
            self.assertEqual(run.call_args.kwargs["env"]["PYTHONPATH"], str(Path(temp).resolve()) + os.pathsep + "old")
            self.assertEqual(os.environ["PYTHONPATH"], "old")

    def test_zero_tests_or_skips_are_not_local_pass(self):
        for diagnostic in ("Ran 0 tests in 0.1s\nOK", "Ran 7 tests in 0.1s\nOK (skipped=1)"):
            with self.subTest(diagnostic=diagnostic), tempfile.TemporaryDirectory() as temp, \
                 mock.patch.object(ops, "EVIDENCE", Path(temp)), mock.patch.object(ops, "lock", return_value=value()), \
                 mock.patch.object(ops, "prospective", return_value=("tree", {})), \
                 mock.patch.object(ops.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=diagnostic, stderr="")):
                with self.assertRaises(RuntimeError): ops.run_local_checks()


class WorkflowRouting(unittest.TestCase):
    def conditions(self, path):
        text = path.read_text(encoding="utf-8")
        return re.findall(r"^    if: (.+)$", text, re.M)

    def evaluate(self, expression, message):
        github = SimpleNamespace(event_name="push", ref_name=ops.BRANCH,
                                 event=SimpleNamespace(head_commit=SimpleNamespace(message=message)))
        expression = expression.replace("&&", " and ").replace("||", " or ").replace("!", " not ")
        return eval(expression, {"__builtins__": {}, "github": github, "contains": lambda a, b: b in a,
                                 "inputs": SimpleNamespace(build_recipe="")})

    def test_production_push_routes_to_exactly_one_job(self):
        workflows = ROOT / ".github/workflows"
        old = self.conditions(workflows / "dingo-router-ci.yml") + self.conditions(workflows / "omni032-h3-sr.yml")
        self.assertGreater(len(old), 6)
        self.assertFalse(any(self.evaluate(condition, ops.MARKER) for condition in old))
        new = self.conditions(workflows / "h3-sr-production.yml")
        self.assertEqual(len(new), 1); self.assertTrue(self.evaluate(new[0], ops.MARKER))

    def test_mixed_markers_cannot_trigger_old_builds(self):
        conditions = self.conditions(ROOT / ".github/workflows/dingo-router-ci.yml")
        conditions += self.conditions(ROOT / ".github/workflows/omni032-h3-sr.yml")
        for other in ("[omni032-h3-sr]", "[sr-native]", "[seedvr2-c14]", "[seedvr2-optimization]", "[h3-hsdp-offload]"):
            with self.subTest(other=other):
                self.assertFalse(any(self.evaluate(c, ops.MARKER + other) for c in conditions))
                self.assertFalse(ops.valid_request_message(ops.MARKER + other))

    def test_single_build_loads_then_validates_before_publication(self):
        text = (ROOT / ".github/workflows/h3-sr-production.yml").read_text()
        self.assertNotIn("matrix:", text); self.assertNotIn("workflow_dispatch:", text)
        self.assertEqual(text.count("uses: docker/build-push-action@"), 1)
        self.assertIn("load: true", text); self.assertIn("push: false", text)
        self.assertEqual(text.count("if: steps.recipe.outputs.build_required == 'true'"), 2)
        self.assertIn("group: dingo-runtime-images-${{ github.ref }}", text)
        self.assertIn("cancel-in-progress: false", text)
        self.assertNotIn("github.run_attempt ==", text)
        self.assertLess(text.index("ci.py validate"), text.index("ci.py publish"))
        self.assertLess(text.index("ci.py publish"), text.index("ci.py outcome"))


if __name__ == "__main__":
    unittest.main()
