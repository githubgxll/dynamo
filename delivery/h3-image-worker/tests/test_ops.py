"""Frozen-review binding and isolated temporary-index regression checks."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("worker_ops_under_test", ROOT / "delivery/h3-image-worker/ops.py")
ops = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ops)


def recipe():
    return {"recipe_sha256": "a" * 64, "review_tree": "b" * 40}


def review():
    return {"status": "PASS", **recipe(), "parent_image": ops.PARENT}


class ApprovalBinding(unittest.TestCase):
    def test_exact_independent_review_passes(self):
        ops.check_review(review(), {"parent_image": ops.PARENT}, recipe())

    def test_failed_stale_or_different_parent_review_fails(self):
        for key, value in (("status", "PENDING"), ("recipe_sha256", "c" * 64),
                           ("review_tree", "c" * 40), ("parent_image", "different")):
            item = review()
            item[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                ops.check_review(item, {"parent_image": ops.PARENT}, recipe())

    def test_approval_requires_copied_root_pass_bound_to_recipe(self):
        value = {"schema": "h3-image-worker-freeze/1", "status": "READY_FOR_BUILD", **recipe(),
                 "parent_image": ops.PARENT, "reviews": {"root": "PASS"}, "root_review": review()}
        ops.check_approval(value, {"parent_image": ops.PARENT}, recipe())
        for change in ({"root_review": {}}, {"reviews": {}}, {"recipe_sha256": "c" * 64}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                ops.check_approval({**value, **change}, {"parent_image": ops.PARENT}, recipe())

    def test_arbitrary_review_path_cannot_approve(self):
        with tempfile.TemporaryDirectory() as temp, self.assertRaises(ValueError):
            ops.import_review(Path(temp) / "BUILD_REVIEW.json")


class ScopeAndMessage(unittest.TestCase):
    def test_worker_files_and_explicit_ci_files_are_in_scope(self):
        ops.check_scope(["container/h3-image-worker/runtime.py", "delivery/h3-image-worker/ops.py",
                         ".github/scripts/h3_image_worker_ci.py", ".github/workflows/dingo-router-ci.yml"])

    def test_unrelated_and_traversing_paths_are_out_of_scope(self):
        for path in ("README.md", "container/other/runtime.py", "container/h3-image-worker/../private.txt",
                     "container/h3-image-worker/../../private.txt", "/container/h3-image-worker/runtime.py",
                     "container\\h3-image-worker\\runtime.py"):
            with self.subTest(path=path), self.assertRaises(ValueError): ops.check_scope([path])

    def test_only_unique_worker_marker_is_valid(self):
        self.assertTrue(ops.valid_request_message("Native worker " + ops.MARKER))
        for text in ("no marker", ops.MARKER * 2, ops.MARKER + " [sr-native]", ops.MARKER + " [omni-gateway-lite]",
                     ops.MARKER + " [skip ci]", ops.MARKER + "\nskip-checks: true"):
            with self.subTest(text=text): self.assertFalse(ops.valid_request_message(text))

    def test_native_stack_and_publication_target_are_locked(self):
        value = ops.read(ROOT / "delivery/h3-image-worker/release-lock.json")
        ops.validate_lock(value)
        for key in ("harbor_repository", "parent_image", "parent_image_id", "remote_branch", "recipe_base_commit"):
            changed = copy.deepcopy(value)
            changed[key] = "wrong"
            with self.subTest(key=key), self.assertRaises(ValueError): ops.validate_lock(changed)
        changed = copy.deepcopy(value)
        changed["versions"]["torch"] = "wrong"
        with self.assertRaises(ValueError): ops.validate_lock(changed)


class TemporaryIndex(unittest.TestCase):
    def test_prospective_ignores_untracked_ignored_files_and_preserves_actual_index(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with mock.patch.object(ops, "ROOT", directory), mock.patch.object(ops, "EVIDENCE", directory / "evidence"):
                ops.git("init", "--quiet")
                ops.git("remote", "add", "origin", ops.REPOSITORY)
                (directory / ".gitignore").write_text("CLAUDE.md\nevidence/\n", encoding="utf-8")
                (directory / "baseline.txt").write_text("baseline\n", encoding="utf-8")
                ops.git("add", ".gitignore", "baseline.txt")
                ops.git("-c", "user.name=Local Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
                base = ops.git("rev-parse", "HEAD")
                original_index = (directory / ".git/index").read_bytes()
                (directory / "CLAUDE.md").write_text("ignored\n", encoding="utf-8")
                file = directory / "container/h3-image-worker/runtime.py"
                file.parent.mkdir(parents=True)
                file.write_text("VALUE = 1\n", encoding="utf-8", newline="\n")
                value = {"recipe_base_commit": base, "parent_image": ops.PARENT, "parent_image_id": ops.PARENT_ID}
                full, result = ops.prospective(value)
                self.assertEqual((directory / ".git/index").read_bytes(), original_index)
                self.assertEqual(list(result["files"]), ["container/h3-image-worker/runtime.py"])
                self.assertNotIn("CLAUDE.md", ops.git("ls-tree", "-r", "--name-only", full).splitlines())
                self.assertIn("container/h3-image-worker/runtime.py", ops.git("ls-tree", "-r", "--name-only", full).splitlines())

    def test_auxiliary_seal_and_approval_are_removed_from_review_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with mock.patch.object(ops, "ROOT", directory), mock.patch.object(ops, "EVIDENCE", directory / "evidence"):
                ops.git("init", "--quiet")
                (directory / ".gitignore").write_text("evidence/\n", encoding="utf-8")
                ops.git("add", ".gitignore")
                ops.git("-c", "user.name=Local Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
                before = ops.git("rev-parse", "HEAD^{tree}")
                for name in ops.AUXILIARY:
                    path = directory / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text('{}\n', encoding="utf-8")
                ops.git("add", "delivery")
                after = ops.git("write-tree")
                self.assertNotEqual(before, after)
                self.assertEqual(ops.review_tree(after), before)


class CheckEvidence(unittest.TestCase):
    def test_delivery_and_runtime_test_counts_have_distinct_names_and_logs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            here = root / "delivery/h3-image-worker"
            for name in (".github/scripts/test_h3_image_worker_ci.py", "delivery/h3-image-worker/tests/test_ops.py",
                         "container/h3-image-worker/tests/test_worker.py", ".github/scripts/h3_image_worker_ci.py"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("VALUE = 1\n", encoding="utf-8")
            workflow = root / ".github/workflows/h3-image-worker.yml"
            workflow.parent.mkdir(parents=True)
            workflow.write_text("steps:\n        run: |\n          true\n", encoding="utf-8")
            replies = [SimpleNamespace(returncode=0, stdout="", stderr="Ran " + str(n) + " tests\nOK\n")
                       for n in (2, 3, 4)]
            with mock.patch.object(ops, "ROOT", root), mock.patch.object(ops, "HERE", here), \
                    mock.patch.object(ops, "EVIDENCE", root / "evidence"), mock.patch.object(ops, "lock", return_value={}), \
                    mock.patch.object(ops, "prospective", return_value=("full", recipe())), \
                    mock.patch.object(ops.subprocess, "run", side_effect=replies):
                result = ops.check(skip_shell=True)
            self.assertEqual(result["regression_tests"], 9)
            self.assertEqual(len(result["regression_counts"]), 3)
            self.assertEqual(result["regression_counts"]["delivery/h3-image-worker/tests"], 3)
            self.assertEqual(result["regression_counts"]["container/h3-image-worker/tests"], 4)
            logs = list(Path(result["evidence_directory"]).glob("*.txt"))
            self.assertEqual(len(logs), 3)


if __name__ == "__main__":
    unittest.main()
