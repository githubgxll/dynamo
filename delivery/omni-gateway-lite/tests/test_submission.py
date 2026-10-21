"""Exercise real Git indices, ignored tracked inputs, seals and bounded staging."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "ops.py"
spec = importlib.util.spec_from_file_location("srnorm_ops_tests", SCRIPT)
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        temp = SCRIPT.parents[2] / ".h3-experimental-evidence/lite/local/tmp"
        temp.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=temp)
        self.root = Path(self.tmp.name)
        self.saved = (ops.ROOT, ops.HERE, ops.EVIDENCE, ops.LEGACY.ROOT, ops.LEGACY.EVIDENCE)
        ops.ROOT = ops.LEGACY.ROOT = self.root
        ops.HERE = self.root / "delivery/omni-gateway-lite"
        ops.EVIDENCE = ops.LEGACY.EVIDENCE = self.root / ".h3-experimental-evidence/local"
        ops.HERE.mkdir(parents=True)
        self.git("init", "-b", "Dingorouter-H3-vllm-omni-jirx")
        self.git("config", "user.name", "Local Test")
        self.git("config", "user.email", "local-test@example.invalid")
        self.git("config", "core.autocrlf", "false")
        self.git("config", "core.filemode", "false")
        self.git("remote", "add", "origin", "https://github.com/githubgxll/dynamo.git")
        self.put("CLAUDE.md", "inherited ignored tracked file\n")
        self.git("add", "CLAUDE.md")
        self.put(".gitignore", "CLAUDE.md\n.h3-experimental-evidence/\n")
        self.git("add", ".gitignore")
        self.git("update-index", "--chmod=+x", "CLAUDE.md")
        self.git("commit", "-m", "parent")
        self.parent = self.git("rev-parse", "HEAD")
        self.lock = {"recipe_base_commit": self.parent, "parent_image": "registry.invalid/a@sha256:" + "a" * 64,
                     "remote_branch": "Dingorouter-H3-vllm-omni-jirx", "repository": "https://github.com/githubgxll/dynamo.git"}
        self.put("delivery/omni-gateway-lite/source-lock.json", json.dumps(self.lock) + "\n")
        self.put("container/omni-gateway-lite/example.py", "value = 1\n")

    def tearDown(self):
        ops.ROOT, ops.HERE, ops.EVIDENCE, ops.LEGACY.ROOT, ops.LEGACY.EVIDENCE = self.saved
        self.tmp.cleanup()

    def git(self, *args):
        return ops.git(*args)

    def put(self, name, contents):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8", newline="\n")

    def test_seal_is_nonmutating_and_preserves_ignored_mode(self):
        index = (self.root / ".git/index").read_bytes()
        ops.seal()
        _, tree = ops.check_seal()
        self.assertEqual(index, (self.root / ".git/index").read_bytes())
        self.assertEqual(self.git("ls-tree", self.parent, "CLAUDE.md"), self.git("ls-tree", tree, "CLAUDE.md"))

    def test_reject_unrelated_untracked_file(self):
        self.put("personal-notes.txt", "private\n")
        with self.assertRaisesRegex(ValueError, "Unrelated"):
            ops.seal()

    def test_reject_unrelated_tracked_change(self):
        self.put("CLAUDE.md", "changed\n")
        with self.assertRaisesRegex(ValueError, "Unrelated"):
            ops.seal()

    def test_reject_edit_after_seal(self):
        ops.seal()
        self.put("container/omni-gateway-lite/example.py", "value = 2\n")
        with self.assertRaisesRegex(ValueError, "seal"):
            ops.check_seal()

    def test_reject_unsealed_removal(self):
        ops.seal()
        (self.root / "container/omni-gateway-lite/example.py").unlink()
        with self.assertRaisesRegex(ValueError, "seal"):
            ops.check_seal()

    def test_stage_validated_tree_only(self):
        ops.seal()
        _, tree = ops.check_seal()
        ops.LEGACY.stage_validated_tree(tree)
        self.assertEqual(tree, self.git("write-tree"))
        self.git("commit", "-m", "reviewed patch")
        self.assertEqual(tree, ops.check_seal(committed=True)[1])

    def test_submission_requires_explicit_entrypoint(self):
        with self.assertRaisesRegex(ValueError, "02-submit"):
            ops.submit(False)

    def prepare_uploaded_recipe(self):
        ops.seal()
        lock, tree = ops.check_seal()
        ops.LEGACY.stage_validated_tree(tree)
        self.git("commit", "-m", "uploaded recipe")
        return lock, tree, self.git("rev-parse", "HEAD")

    def test_explicit_repeat_build_pushes_empty_commit_without_gh(self):
        lock, tree, old_head = self.prepare_uploaded_recipe()
        actual_git = ops.git
        pushes = []
        def controlled_git(*args, **kwargs):
            if args[0] == "push":
                pushes.append((args, kwargs))
                return ""
            return actual_git(*args, **kwargs)
        with mock.patch.object(ops, "check", return_value=(lock, tree)), \
             mock.patch.object(ops.LEGACY, "remote_head", return_value=old_head), \
             mock.patch.object(ops, "git", side_effect=controlled_git):
            ops.dispatch()
        self.assertNotEqual(self.git("rev-parse", "HEAD"), old_head)
        self.assertEqual(self.git("rev-parse", "HEAD^"), old_head)
        self.assertEqual(self.git("rev-parse", "HEAD^{tree}"), tree)
        self.assertNotIn("[skip ci]", self.git("log", "-1", "--format=%B"))
        self.assertEqual(pushes, [(("push", "origin", "HEAD:refs/heads/" + lock["remote_branch"]), {"network": True})])

    def test_repeat_rejects_unsubmitted_changes_before_committing(self):
        lock, tree, old_head = self.prepare_uploaded_recipe()
        self.put("container/omni-gateway-lite/example.py", "unsubmitted = True\n")
        with mock.patch.object(ops, "check", return_value=(lock, tree)), \
             mock.patch.object(ops.LEGACY, "remote_head", return_value=old_head):
            with self.assertRaisesRegex(ValueError, "02-submit"):
                ops.dispatch()
        self.assertEqual(self.git("rev-parse", "HEAD"), old_head)

    def test_repeat_rejects_remote_drift_before_committing(self):
        lock, tree, old_head = self.prepare_uploaded_recipe()
        with mock.patch.object(ops, "check", return_value=(lock, tree)), \
             mock.patch.object(ops.LEGACY, "remote_head", return_value="0" * 40):
            with self.assertRaisesRegex(ValueError, "02-submit"):
                ops.dispatch()
        self.assertEqual(self.git("rev-parse", "HEAD"), old_head)


if __name__ == "__main__":
    unittest.main()
