"""Reject traversal, incomplete/tampered evidence and mismatched image ancestry."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ops", HERE / "ops.py")
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.lock = {"service_base_commit": "a" * 40, "harbor_repository": "registry.example:8443/openclaw/ai-dingo-vllm"}
        self.parent = self.lock["harbor_repository"] + "@sha256:" + "1" * 64
        self.record = {"schema": "h3-experimental-publication/1", "service_base_commit": "a" * 40,
                       "source_commit": "b" * 40, "digest": "sha256:" + "2" * 64,
                       "image": self.lock["harbor_repository"] + ":omni031rc1-gateway-distilled-ref8shift12-" + "b" * 12,
                       "deployment_image": self.lock["harbor_repository"] + "@sha256:" + "2" * 64,
                       "framework_image": self.parent, "evidence_sha256": {}}
        reports = {
            "experimental/service-smoke.json": "PASS_SERVICE_IMPORTS_AND_HELP_ONLY",
            "experimental/assets-report.json": "PASS_ASSETS_ONLY",
            "experimental/profiles-check.json": "PASS_INSTALLED_PROFILES_CPU_ONLY",
            "experimental/environment-check.json": "PASS_EXPERIMENTAL_ENVIRONMENT_CPU_ONLY",
            "experimental/media-smoke.json": "PASS_REAL_CPU_MEDIA_SMOKE",
            "framework/ref8-patch.json": "PASS_PATCH_APPLIED_CPU_ONLY",
            "framework/ref8-contract.json": "PASS_INSTALLED_CONTRACT_CPU_ONLY",
            "experimental/dependency-license-inventory.json": "METADATA_INVENTORY_ONLY_NOT_LEGAL_CLEARANCE",
        }
        for path, status in reports.items():
            self.add(path, {"status": status})
        for path in ["inputs/profiles.json", "inputs/Dockerfile", "legal/osrb-deps.csv", "legal/osrb.cdx.json", "experimental/dpkg-packages.txt"]:
            self.add(path, {"synthetic": True})
        self.add("inputs/assets.lock.json", {"synthetic": True})
        self.add("inputs/source-lock.json", self.lock)
        self.add("experimental/image-inputs.json", {"source_commit": "b" * 40, "base_image": self.parent,
                 "versions": {"vllm": "0.31.0", "vllm-omni": "0.31.0rc1"},
                 "assets_lock_sha256": self.record["evidence_sha256"]["inputs/assets.lock.json"]})
        (self.root / "deployment-image.txt").write_text(self.record["deployment_image"] + "\n")

    def tearDown(self):
        self.temp.cleanup()

    def add(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        self.record["evidence_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()

    def verify(self):
        (self.root / "image.json").write_text(json.dumps(self.record), encoding="utf-8")
        return ops.verify_publication(self.root, self.lock)

    def test_valid_record_preserves_registry_port(self):
        self.assertEqual(self.verify()["deployment_image"], self.record["deployment_image"])

    def test_missing_gate_is_rejected(self):
        self.record["evidence_sha256"].pop("experimental/media-smoke.json")
        with self.assertRaises(ValueError):
            self.verify()

    def test_failed_gate_even_with_updated_hash_is_rejected(self):
        self.add("framework/ref8-contract.json", {"status": "FAIL"})
        with self.assertRaises(ValueError):
            self.verify()

    def test_tampered_bytes_are_rejected(self):
        (self.root / "inputs/profiles.json").write_text("changed")
        with self.assertRaises(ValueError):
            self.verify()

    def test_different_framework_is_rejected(self):
        self.record["framework_image"] = self.lock["harbor_repository"] + "@sha256:" + "3" * 64
        with self.assertRaises(ValueError):
            self.verify()

    def test_different_harbor_is_rejected(self):
        self.record["deployment_image"] = "registry.invalid/image@" + self.record["digest"]
        with self.assertRaises(ValueError):
            self.verify()

    def test_framework_tag_cannot_be_used_as_final(self):
        self.record['image'] = self.lock['harbor_repository'] + ':omni031rc1-gateway-framework-' + 'b' * 12
        with self.assertRaisesRegex(ValueError, 'final distilled image tag'):
            self.verify()

    def test_framework_digest_cannot_be_used_as_final(self):
        self.record['digest'] = 'sha256:' + '1' * 64
        self.record['deployment_image'] = self.parent
        with self.assertRaisesRegex(ValueError, 'parent cannot be the final'):
            self.verify()

    def test_submitted_input_bytes_must_match_even_if_archive_rehashed(self):
        def committed_bytes(*args, **kwargs):
            return b'actual reviewed Git bytes'
        with patch.object(ops, 'git', side_effect=committed_bytes):
            with self.assertRaisesRegex(ValueError, 'submitted Git bytes'):
                ops.verify_submitted_inputs(self.root, self.record)

    def test_matching_submitted_git_inputs_accepted(self):
        mapping = {'delivery/h3-experimental/source-lock.json': 'inputs/source-lock.json',
                   'container/h3-experimental/assets/assets.lock.json': 'inputs/assets.lock.json',
                   'container/h3-experimental/profiles/profiles.json': 'inputs/profiles.json',
                   'container/h3-experimental/Dockerfile': 'inputs/Dockerfile'}
        def committed_bytes(action, ref, **kwargs):
            self.assertEqual(action, 'show')
            self.assertTrue(kwargs['binary'])
            return (self.root / mapping[ref.split(':', 1)[1]]).read_bytes()
        with patch.object(ops, 'git', side_effect=committed_bytes):
            ops.verify_submitted_inputs(self.root, self.record)

    def test_unsafe_member_paths(self):
        for value in ["../escape", "/absolute", "D:/outside", "a\\b", "a/../../b", ""]:
            with self.subTest(path=value), self.assertRaises(ValueError):
                ops.safe_relative(value)


class BranchSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.lock = {"remote_branch": ops.EXPERIMENT_BRANCH, "service_base_commit": "a" * 40,
                     "remote_baseline": "d" * 40, "allow_remote_branch_creation": True}
        self.calls = []
        self.remote = None
        self.head = "b" * 40
        self.tree = "c" * 40
        self.upstream = self.lock["service_base_commit"]
        self.writer = patch.object(ops, "dump")
        self.record = self.writer.start()
        self.addCleanup(self.writer.stop)

    def fake_git(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if args[0] == "ls-remote":
            return "" if self.remote is None else self.remote + "\trefs/heads/" + ops.EXPERIMENT_BRANCH
        if args == ("rev-parse", "refs/remotes/origin/DingoRouter-video-gateway"):
            return self.upstream
        if args == ("rev-parse", "refs/remotes/origin/" + ops.EXPERIMENT_BRANCH):
            return self.remote
        if args == ("rev-parse", "HEAD"):
            return self.head
        if args == ("rev-parse", "HEAD^{tree}"):
            return self.tree
        if args[0] == "diff":
            return b""
        return ""

    def test_first_creation_requires_successful_empty_lookup(self):
        with patch.object(ops, "git", side_effect=self.fake_git):
            self.assertIsNone(ops.prepare_remote(self.lock))
        fetches = [args for args, _ in self.calls if args[0] == "fetch"]
        self.assertEqual(len(fetches), 1)  # Only service base: do not fetch an absent ref.
        self.assertFalse(any(args[0] == "merge-base" for args, _ in self.calls))

    def test_explicit_pinned_policy_accepts_upstream_advance_without_merging(self):
        self.upstream = "f" * 40
        self.remote = self.head
        original_base = self.lock["service_base_commit"]
        with patch.object(ops, "git", side_effect=self.fake_git):
            self.assertEqual(ops.prepare_remote({**self.lock, "service_base_policy": "pinned-ancestor"}), self.remote)
        self.assertIn((("merge-base", "--is-ancestor", original_base, self.upstream), {}), self.calls)
        self.assertIn((("merge-base", "--is-ancestor", self.remote, "HEAD"), {}), self.calls)
        self.assertFalse(any(args[0] in {"merge", "rebase", "reset", "push", "commit"} for args, _ in self.calls))
        record = self.record.call_args.args[1]
        self.assertEqual(record["service_base_commit"], original_base)
        self.assertEqual(record["observed_upstream_commit"], self.upstream)
        self.assertTrue(record["update_available"])

    def test_pinned_policy_rejects_unverifiable_upstream_history(self):
        self.upstream = "f" * 40
        def rewritten(*args, **kwargs):
            if args == ("merge-base", "--is-ancestor", self.lock["service_base_commit"], self.upstream):
                raise RuntimeError("simulated history rewrite or unavailable history")
            return self.fake_git(*args, **kwargs)
        with patch.object(ops, "git", side_effect=rewritten), self.assertRaisesRegex(ValueError, "Cannot verify"):
            ops.prepare_remote({**self.lock, "service_base_policy": "pinned-ancestor"})
        self.record.assert_not_called()
        self.assertFalse(any(args[0] in {"push", "commit", "ls-remote"} for args, _ in self.calls))

    def test_legacy_strict_policy_still_stops_on_changed_service_head(self):
        self.upstream = "f" * 40
        with patch.object(ops, "git", side_effect=self.fake_git), self.assertRaisesRegex(ValueError, "changed since review"):
            ops.prepare_remote(self.lock)

    def test_unknown_source_policy_and_invalid_commit_are_rejected(self):
        for policy, commit in (("silently-update", "f" * 40), ("pinned-ancestor", "not-a-sha")):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                ops.verify_service_base({**self.lock, "service_base_policy": policy}, commit)

    def test_upstream_fetch_failure_is_not_treated_as_pinned_success(self):
        with patch.object(ops, "git", side_effect=RuntimeError("fetch network failure")), self.assertRaises(RuntimeError):
            ops.prepare_remote({**self.lock, "service_base_policy": "pinned-ancestor"})
        self.record.assert_not_called()

    def test_auth_or_network_failure_is_not_absent_branch(self):
        def fail_lookup(*args, **kwargs):
            if args[0] == "ls-remote":
                raise RuntimeError("simulated authentication/connectivity error")
            return self.fake_git(*args, **kwargs)
        with patch.object(ops, "git", side_effect=fail_lookup), self.assertRaises(RuntimeError):
            ops.prepare_remote(self.lock)
        self.assertFalse(any(args[0] == "push" for args, _ in self.calls))

    def test_creation_can_only_target_exact_designated_branch(self):
        with patch.object(ops, "git") as call, self.assertRaises(ValueError):
            ops.prepare_remote({**self.lock, "remote_branch": "DingoRouter-h3-distilled-20261013"})
        call.assert_not_called()

    def test_absent_branch_requires_explicit_creation_policy(self):
        with patch.object(ops, "git", side_effect=self.fake_git), self.assertRaises(ValueError):
            ops.prepare_remote({**self.lock, "allow_remote_branch_creation": False})

    def test_existing_branch_requires_included_history(self):
        self.remote = "e" * 40
        with patch.object(ops, "git", side_effect=self.fake_git):
            self.assertEqual(ops.prepare_remote(self.lock), self.remote)
        self.assertIn((("merge-base", "--is-ancestor", self.remote, "HEAD"), {}), self.calls)

    def test_diverged_remote_stops_before_push(self):
        self.remote = "e" * 40
        def diverged(*args, **kwargs):
            if args[0] == "merge-base":
                raise RuntimeError("simulated non-ancestor remote")
            return self.fake_git(*args, **kwargs)
        with patch.object(ops, "check", return_value=(self.lock, self.tree)), patch.object(ops, "git", side_effect=diverged), self.assertRaises(RuntimeError):
            ops.submit(True)
        self.assertFalse(any(args[0] == "push" for args, _ in self.calls))

    def test_first_submission_uses_normal_push_and_new_upstream(self):
        with patch.object(ops, "check", return_value=(self.lock, self.tree)), patch.object(ops, "git", side_effect=self.fake_git):
            ops.submit(True)
        pushes = [args for args, _ in self.calls if args[0] == "push"]
        self.assertEqual(pushes, [("push", "--set-upstream", "origin", "HEAD:refs/heads/" + ops.EXPERIMENT_BRANCH)])

    def test_successful_submission_retry_does_not_push_again(self):
        self.remote = self.head
        with patch.object(ops, "check", return_value=(self.lock, self.tree)), patch.object(ops, "git", side_effect=self.fake_git):
            ops.submit(True)
        self.assertFalse(any(args[0] in {"push", "commit"} for args, _ in self.calls))

    def test_malformed_remote_response_is_rejected(self):
        for response in ["bad\trefs/heads/" + ops.EXPERIMENT_BRANCH,
                         "a" * 40 + "\trefs/heads/wrong-branch"]:
            with patch.object(ops, "git", return_value=response), self.assertRaises(ValueError):
                ops.remote_head(ops.EXPERIMENT_BRANCH)


if __name__ == "__main__":
    unittest.main()
