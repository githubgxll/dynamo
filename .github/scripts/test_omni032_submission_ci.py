"""Sequential submission guards; all GitHub/Git writes are simulated."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("_sequential_ops", ROOT / "delivery/omni032-h3-sr/ops.py")
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)
OLD, COMMIT, TREE = "b" * 40, "c" * 40, "d" * 40
VALUE = {"repository": "https://github.com/githubgxll/dynamo.git", "remote_branch": "Dingorouter-H3-vllm-omni-jirx",
         "working_tree_base_commit": OLD}


def run(attempt=2, status="completed", conclusion="skipped"):
    return {"id": 42, "run_attempt": attempt, "status": status, "conclusion": conclusion,
            "head_sha": OLD, "head_branch": VALUE["remote_branch"], "path": ".github/workflows/omni032-h3-sr.yml",
            "updated_at": "2026-11-02T05:47:37Z", "html_url": "https://github.com/githubgxll/dynamo/actions/runs/42"}


def jobs(steps=None, conclusion="skipped", attempt=2):
    return {"total_count": 1, "jobs": [{"run_id": 42, "run_attempt": attempt, "head_sha": OLD,
            "status": "completed", "conclusion": conclusion, "steps": steps or []}]}


def getter_for(current=None, latest_jobs=None, earlier=None, final=None):
    current = current or run()
    reads = []
    def getter(endpoint):
        if "/attempts/" in endpoint:
            number = int(endpoint.split("/attempts/", 1)[1].split("/")[0])
            if number == current["run_attempt"]: return latest_jobs if latest_jobs is not None else jobs(attempt=number)
            return (earlier or {}).get(number, jobs(attempt=number))
        reads.append(endpoint)
        return final if len(reads) > 1 and final is not None else current
    return getter


def result(attempt=2):
    return ops.observe_run(42, getter_for(run(attempt)))


def contract(count=1):
    return {"build_request_count": count, "submitted_commit": OLD, "previous_build_result": result()}


class PriorObservation(unittest.TestCase):
    def test_latest_skipped_attempt_is_unpublished(self):
        receipt = result()
        self.assertEqual((receipt["run_attempt"], receipt["publication_status"]), (2, "not_started"))
        ops.validate_previous_result(receipt, VALUE, expected_commit=OLD)

    def test_setup_tls_failure_is_unpublished(self):
        current = run(1, conclusion="failure")
        receipt = ops.observe_run(42, getter_for(current, jobs(
            [{"name": "Set up job", "conclusion": "failure"}], "failure", 1)))
        self.assertEqual(receipt["publication_status"], "not_started")

    def test_explicit_publish_step_skipped_is_unpublished(self):
        current = run(conclusion="failure")
        receipt = ops.observe_run(42, getter_for(current, jobs([
            {"name": "Publish the validated image once and verify its pull digest", "conclusion": "skipped"}], "failure")))
        self.assertEqual(receipt["publication_status"], "not_started")

    def test_failed_publication_is_unknown_even_if_run_failed(self):
        current = run(conclusion="failure")
        receipt = ops.observe_run(42, getter_for(current, jobs([
            {"name": "Publish the validated image once and verify its pull digest", "conclusion": "failure"}], "failure")))
        self.assertEqual(receipt["publication_status"], "unknown")
        with self.assertRaises(ValueError): ops.validate_previous_result(receipt, VALUE, expected_commit=OLD)

    def test_successful_publication_cannot_retry(self):
        current = run(conclusion="failure")
        receipt = ops.observe_run(42, getter_for(current, jobs([
            {"name": "Publish the validated image once and verify its pull digest", "conclusion": "success"}], "failure")))
        self.assertEqual(receipt["publication_status"], "published")
        with self.assertRaises(ValueError): ops.validate_previous_result(receipt, VALUE, expected_commit=OLD)

    def test_active_latest_attempt_cannot_retry(self):
        current = run(3, "in_progress", None)
        receipt = ops.observe_run(42, getter_for(current, jobs(attempt=3)))
        with self.assertRaises(ValueError): ops.validate_previous_result(receipt, VALUE, expected_commit=OLD)

    def test_new_attempt_during_observation_rejected(self):
        with self.assertRaisesRegex(ValueError, "changed during"):
            ops.observe_run(42, getter_for(final=run(3, "queued", None)))

    def test_incomplete_job_inventory_rejected(self):
        incomplete = {"total_count": 101, "jobs": jobs()["jobs"]}
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            ops.observe_run(42, getter_for(latest_jobs=incomplete))

    def test_foreign_job_attempt_rejected(self):
        with self.assertRaisesRegex(ValueError, "different run"):
            ops.observe_run(42, getter_for(latest_jobs=jobs(attempt=1)))

    def test_earlier_publication_cannot_be_erased_by_latest_skipped(self):
        for conclusion in ("success", "failure"):
            with self.subTest(conclusion=conclusion):
                prior_jobs = jobs([{"name": "Publish the validated image once and verify its pull digest",
                                    "conclusion": conclusion}], "failure", 1)
                receipt = ops.observe_run(42, getter_for(earlier={1: prior_jobs}))
                self.assertIn(receipt["publication_status"], {"published", "unknown"})
                self.assertEqual(len(receipt["attempt_history"]), 2)
                with self.assertRaises(ValueError): ops.validate_previous_result(receipt, VALUE, expected_commit=OLD)

    def test_wrong_prior_identity_rejected(self):
        for key, value in {"commit": COMMIT, "repository": "other", "run_attempt": True,
                           "run_id": 0, "workflow_path": "wrong"}.items():
            receipt = result()
            receipt[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                ops.validate_previous_result(receipt, VALUE, expected_commit=OLD)


class SequentialHistory(unittest.TestCase):
    def test_no_request_count_cap(self):
        for count in (1, 2, 8, 99):
            with self.subTest(count=count):
                receipt = ops.prior_request(VALUE, contract(count), {"commit": OLD, "request_count": count, "run_id": 42},
                    getter_for())
                self.assertEqual(receipt["run_attempt"], 2)

    def test_newer_terminal_attempt_requires_updated_review(self):
        with self.assertRaisesRegex(ValueError, "newer attempt"):
            ops.prior_request(VALUE, contract(), {"commit": OLD, "request_count": 1, "run_id": 42},
                getter_for(run(3)))

    def test_journal_and_contract_mismatch_blocks(self):
        with self.assertRaisesRegex(ValueError, "disagree"):
            ops.prior_request(VALUE, contract(), {"commit": OLD, "request_count": 2})

    def test_zero_count_does_not_hide_existing_journal(self):
        with self.assertRaises(ValueError): ops.prior_request(VALUE, contract(0), {"commit": OLD})

    def test_api_failure_does_not_authorize_retry(self):
        with self.assertRaises(OSError):
            ops.prior_request(VALUE, contract(), {"commit": OLD, "request_count": 1, "run_id": 42}, mock.Mock(side_effect=OSError("TLS")))

    def test_first_request_has_no_prior_observation(self):
        self.assertIsNone(ops.prior_request(VALUE, contract(0), None))

    def test_different_known_run_cannot_resolve_unknown_journal(self):
        with self.assertRaisesRegex(ValueError, "original journal run"):
            ops.prior_request(VALUE, contract(), {"commit": OLD, "request_count": 1, "run_id": 99, "status": "UNKNOWN"})

    def test_observe_cannot_replace_known_run(self):
        with self.assertRaisesRegex(ValueError, "original journal run"):
            ops.bind_journal_result({"run_id": 99}, result(), allow_discovery=True)

    def test_attempt_cannot_move_backwards(self):
        with self.assertRaisesRegex(ValueError, "backwards"):
            ops.bind_journal_result({"run_id": 42, "run_attempt": 3}, result())

    def test_nested_confirmed_run_is_also_bound(self):
        for journal in ({"confirmed_result": {"run_id": 99}},
                        {"run_id": 42, "confirmed_result": {"run_id": 99}}):
            with self.subTest(journal=journal), self.assertRaises(ValueError):
                ops.bind_journal_result(journal, result(), allow_discovery=True)

    def test_nested_confirmed_attempt_cannot_move_backwards(self):
        with self.assertRaisesRegex(ValueError, "backwards"):
            ops.bind_journal_result({"run_id": 42, "run_attempt": 1,
                                    "confirmed_result": {"run_id": 42, "run_attempt": 3}}, result())

    def test_unbound_journal_requires_explicit_observation(self):
        with self.assertRaisesRegex(ValueError, "not yet bound"):
            ops.bind_journal_result({}, result())

    def test_discovery_requires_unique_original_push_run(self):
        receipt = result()
        with mock.patch.object(ops, "github_json", return_value={"total_count": 1, "workflow_runs": [{"id": 42}]}):
            ops.bind_journal_result({}, receipt, allow_discovery=True)
        with mock.patch.object(ops, "github_json", return_value={"total_count": 2, "workflow_runs": [{"id": 42}, {"id": 99}]}):
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                ops.bind_journal_result({}, receipt, allow_discovery=True)


class Submission(unittest.TestCase):
    def exercise(self, directory, push_error=False):
        journal = Path(directory) / "one-build-journal.json"
        ops.dump(journal, {"commit": OLD, "request_count": 1, "run_id": 42, "status": "CONFIRMED_TERMINAL_UNPUBLISHED"})
        recipe = {"recipe_sha256": "a" * 64}
        reviewed, observed = contract(), result()
        def git(*arguments, **kwargs):
            if arguments == ("rev-parse", "HEAD^{tree}"): return TREE
            if arguments == ("rev-parse", "HEAD"): return COMMIT
            if arguments == ("log", "-1", "--format=%B"): return "control repair " + ops.MARKER
            if arguments[0] == "status": return ""
            if arguments[0] == "push" and push_error: raise RuntimeError("TLS result unknown")
            return ""
        with mock.patch.object(ops, "EVIDENCE", Path(directory)), mock.patch.object(ops, "check", return_value=(VALUE, TREE, recipe)), \
             mock.patch.object(ops, "read", side_effect=lambda path: reviewed if Path(path) == ops.CONTRACT else json.loads(Path(path).read_text())), \
             mock.patch.object(ops.LEGACY, "remote_head", return_value=OLD), mock.patch.object(ops, "prospective", return_value=(TREE, recipe)), \
             mock.patch.object(ops, "check_seal"), mock.patch.object(ops, "observe_run", return_value=observed), \
             mock.patch.object(ops, "git", side_effect=git) as command:
            if push_error:
                with self.assertRaises(RuntimeError): ops.submit(True)
            else:
                ops.submit(True)
                with self.assertRaises(ValueError): ops.submit(True)
            return command.call_args_list

    def test_pushes_fixed_reviewed_commit_once_and_keeps_history(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = self.exercise(directory)
            pushes = [c for c in calls if c.args[0] == "push"]
            self.assertEqual([c.args for c in pushes], [("push", "origin", COMMIT + ":refs/heads/" + VALUE["remote_branch"])])
            current = ops.read(Path(directory) / "one-build-journal.json")
            self.assertEqual(current["request_count"], 2)
            self.assertEqual(ops.read(Path(directory) / "requests/request-0001.json")["commit"], OLD)
            self.assertEqual(current["previous_result"]["run_attempt"], 2)

    def test_push_failure_preserves_unknown_request(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = self.exercise(directory, push_error=True)
            self.assertEqual(sum(c.args[0] == "push" for c in calls), 1)
            current = ops.read(Path(directory) / "one-build-journal.json")
            claim = ops.read(Path(directory) / "requests/request-0002.json")
            self.assertEqual(current, claim)
            self.assertEqual(current["status"], "UNKNOWN")

    def test_exclusive_sequence_claim_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request-0002.json"
            ops.claim_request(path, {"commit": OLD})
            with self.assertRaises(FileExistsError): ops.claim_request(path, {"commit": COMMIT})
            self.assertEqual(ops.read(path), {"commit": OLD})


if __name__ == "__main__": unittest.main()
