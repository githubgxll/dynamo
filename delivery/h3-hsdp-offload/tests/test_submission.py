"""Exercise real Git indices, ignored tracked inputs, seals and bounded staging."""
import importlib.util
import json
import os
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
        temp = SCRIPT.parents[2] / ".h3-experimental-evidence/hsdp/local/tmp"
        temp.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=temp)
        self.root = Path(self.tmp.name)
        self.saved = (ops.ROOT, ops.HERE, ops.EVIDENCE, ops.LEGACY.ROOT, ops.LEGACY.EVIDENCE)
        ops.ROOT = ops.LEGACY.ROOT = self.root
        ops.HERE = self.root / "delivery/h3-hsdp-offload"
        ops.EVIDENCE = ops.LEGACY.EVIDENCE = self.root / ".h3-experimental-evidence/local"
        ops.HERE.mkdir(parents=True)
        self.git("init", "-b", "Dingorouter-H3-vllm-omni-jirx")
        self.git("config", "user.name", "Local Test")
        self.git("config", "user.email", "local-test@example.invalid")
        self.git("config", "core.autocrlf", "false")
        self.git("config", "core.filemode", "false")
        self.git("config", "core.symlinks", "false")
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
        self.put("delivery/h3-hsdp-offload/source-lock.json", json.dumps(self.lock) + "\n")
        self.put("container/h3-hsdp-offload/example.py", "value = 1\n")

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
        self.put("container/h3-hsdp-offload/example.py", "value = 2\n")
        with self.assertRaisesRegex(ValueError, "seal"):
            ops.check_seal()

    def test_reject_unsealed_removal(self):
        ops.seal()
        (self.root / "container/h3-hsdp-offload/example.py").unlink()
        with self.assertRaisesRegex(ValueError, "seal"):
            ops.check_seal()

    def historical_scope_fixture(self):
        # Build A/M/D/T and executable-mode history in this disposable repo.
        # Symlink mode is written into the index without OS symlink privileges.
        for name in ("modified", "deleted", "type", "mode"):
            self.put("outside/" + name, "before\n")
        self.git("add", "outside")
        self.git("commit", "-m", "recipe base with unrelated files")
        self.lock["recipe_base_commit"] = self.git("rev-parse", "HEAD")
        self.put("outside/added", "history addition\n")
        self.put("outside/modified", "history modification\n")
        self.put("outside/type", "link-target\n")
        self.git("add", "outside/added", "outside/modified", "outside/type")
        self.git("rm", "outside/deleted")
        self.git("update-index", "--chmod=+x", "outside/mode")
        blob = self.git("hash-object", "-w", "outside/type")
        self.git("update-index", "--cacheinfo", "120000", blob, "outside/type")
        self.put("container/h3-hsdp-offload/earlier.py", "old_recipe = True\n")
        self.git("add", "container/h3-hsdp-offload/earlier.py")
        self.git("commit", "-m", "already committed unrelated history and prior HSDP recipe")
        self.lock["working_tree_base_commit"] = self.git("rev-parse", "HEAD")
        self.put("delivery/h3-hsdp-offload/source-lock.json", json.dumps(self.lock) + "\n")

    def tree_with_restored_recipe_path(self, name):
        with tempfile.TemporaryDirectory(dir=self.root) as temporary:
            env = dict(os.environ, GIT_INDEX_FILE=str(Path(temporary) / "index"))
            ops.git("read-tree", self.lock["working_tree_base_commit"], env=env)
            original = self.git("ls-tree", self.lock["recipe_base_commit"], "--", name)
            if original:
                mode, kind, rest = original.split(" ", 2)
                blob, path = rest.split("\t", 1)
                ops.git("update-index", "--add", "--cacheinfo", mode, blob, path, env=env)
            else:
                ops.git("update-index", "--force-remove", name, env=env)
            return ops.git("write-tree", env=env)

    def test_committed_unrelated_history_is_excluded_but_full_hsdp_recipe_exports(self):
        self.historical_scope_fixture()
        index = (self.root / ".git/index").read_bytes()
        ops.seal()
        _, tree = ops.check_seal()
        self.assertEqual(index, (self.root / ".git/index").read_bytes())
        sealed = json.loads((self.root / ops.SEAL).read_text(encoding="utf-8"))
        self.assertEqual(sealed["working_tree_base_commit"], self.lock["working_tree_base_commit"])
        self.assertIn("container/h3-hsdp-offload/earlier.py", sealed["delta"])
        self.assertIn("container/h3-hsdp-offload/example.py", sealed["delta"])
        self.assertFalse(any(name.startswith("outside/") for name in sealed["delta"]))
        ops.LEGACY.stage_validated_tree(tree)
        self.git("commit", "-m", "reviewed candidate")
        with tempfile.TemporaryDirectory(dir=ops.EVIDENCE) as destination:
            ops.export_recipe(Path(destination))
            self.assertTrue((Path(destination) / "container/h3-hsdp-offload/earlier.py").is_file())
            self.assertTrue((Path(destination) / "container/h3-hsdp-offload/example.py").is_file())
            self.assertFalse((Path(destination) / "outside").exists())

    def test_historical_reverts_cannot_escape_when_cumulative_delta_disappears(self):
        self.historical_scope_fixture()
        original = ops.raw_delta(self.lock["recipe_base_commit"], self.lock["working_tree_base_commit"])
        expected = {"added": "A", "modified": "M", "deleted": "D", "type": "T", "mode": "M"}
        for suffix, status in expected.items():
            name = "outside/" + suffix
            with self.subTest(name=name, historical_status=status):
                self.assertEqual(original[name]["status"], status)
                candidate = self.tree_with_restored_recipe_path(name)
                self.assertNotIn(name, ops.raw_delta(self.lock["recipe_base_commit"], candidate))
                with self.assertRaisesRegex(ValueError, "Unrelated"):
                    ops.delta_for_tree(candidate, self.lock)

    def test_new_unrelated_commit_after_working_anchor_is_rejected(self):
        self.historical_scope_fixture()
        self.put("outside/added", "new unreviewed edit\n")
        self.git("add", "outside/added")
        self.git("commit", "-m", "unrelated new change")
        with self.assertRaisesRegex(ValueError, "Unrelated"):
            ops.delta_for_tree(self.git("rev-parse", "HEAD^{tree}"), self.lock)

    def test_historical_shared_input_blob_or_mode_drift_is_rejected(self):
        for mode_only in (False, True):
            with self.subTest(mode_only=mode_only):
                name = "container/seedvr2-c14/shared-" + str(mode_only) + ".py"
                self.put(name, "shared = 1\n")
                self.git("add", name)
                self.git("commit", "-m", "shared recipe input")
                lock = dict(self.lock, recipe_base_commit=self.git("rev-parse", "HEAD"))
                if mode_only:
                    self.git("update-index", "--chmod=+x", name)
                else:
                    self.put(name, "shared = 2\n")
                    self.git("add", name)
                self.git("commit", "-m", "historical shared drift")
                lock["working_tree_base_commit"] = self.git("rev-parse", "HEAD")
                with self.assertRaisesRegex(ValueError, "Shared inputs"):
                    ops.delta_for_tree(self.git("rev-parse", "HEAD^{tree}"), lock)

    def test_anchor_order_and_head_ancestry_are_required(self):
        self.historical_scope_fixture()
        tree = self.git("rev-parse", "HEAD^{tree}")
        sibling = self.git("commit-tree", tree, "-p", self.lock["recipe_base_commit"], "-m", "sibling anchor")
        invalid_locks = [
            dict(self.lock, working_tree_base_commit=sibling),
            dict(self.lock, recipe_base_commit=self.lock["working_tree_base_commit"],
                 working_tree_base_commit=self.lock["recipe_base_commit"]),
            dict(self.lock, working_tree_base_commit="HEAD"),
            dict(self.lock, working_tree_base_commit=self.lock["working_tree_base_commit"][:12]),
            dict(self.lock, working_tree_base_commit=tree),
        ]
        for lock in invalid_locks:
            with self.subTest(recipe=lock["recipe_base_commit"], working=lock["working_tree_base_commit"]):
                with self.assertRaises((ValueError, RuntimeError)):
                    ops.delta_for_tree(tree, lock)

    def test_detached_committed_check_still_checks_anchors(self):
        self.historical_scope_fixture()
        ops.seal()
        _, tree = ops.check_seal()
        ops.LEGACY.stage_validated_tree(tree)
        self.git("commit", "-m", "reviewed candidate")
        self.git("checkout", "--detach", "HEAD")
        self.assertEqual(tree, ops.check_seal(committed=True)[1])
        sibling = self.git("commit-tree", tree, "-p", self.lock["recipe_base_commit"], "-m", "bad CI anchor")
        with mock.patch.object(ops, "load_lock", return_value=dict(self.lock, working_tree_base_commit=sibling)):
            with self.assertRaisesRegex(RuntimeError, "merge-base"):
                ops.check_seal(committed=True)

    def test_legacy_lock_keeps_recipe_anchor_scope(self):
        self.assertNotIn("working_tree_base_commit", self.lock)
        ops.seal()
        sealed = json.loads((self.root / ops.SEAL).read_text(encoding="utf-8"))
        self.assertEqual(sealed["working_tree_base_commit"], self.lock["recipe_base_commit"])
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

    def test_skip_markers_cannot_masquerade_as_c14_build_requests(self):
        self.assertTrue(ops.triggers_hsdp("build [h3-hsdp-offload]"))
        self.assertFalse(ops.triggers_hsdp("ordinary lite build"))
        for marker in ("[skip ci]", "[CI SKIP]", "[no ci]", "[skip actions]", "[actions skip]",
                       "\n\nskip-checks: true", "\n\nskip-checks:true"):
            with self.subTest(marker=marker):
                self.assertFalse(ops.triggers_hsdp("build [h3-hsdp-offload] " + marker))

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
        self.assertIn("[h3-hsdp-offload]", self.git("log", "-1", "--format=%B"))
        self.assertEqual(pushes, [(("push", "origin", "HEAD:refs/heads/" + lock["remote_branch"]), {"network": True})])

    def test_repeat_rejects_unsubmitted_changes_before_committing(self):
        lock, tree, old_head = self.prepare_uploaded_recipe()
        self.put("container/h3-hsdp-offload/example.py", "unsubmitted = True\n")
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

    def test_manual_recipe_commit_gets_c14_routing_marker_before_push(self):
        lock, tree, manual_head = self.prepare_uploaded_recipe()
        remote = self.git("rev-parse", "HEAD^")
        self.git("update-ref", "refs/remotes/origin/" + lock["remote_branch"], remote)
        actual_git = ops.git
        pushes = []
        def controlled_git(*args, **kwargs):
            if args[0] == "fetch":
                return ""
            if args[0] == "push":
                pushes.append(args)
                return ""
            return actual_git(*args, **kwargs)
        with mock.patch.object(ops, "check", return_value=(lock, tree)), \
             mock.patch.object(ops.LEGACY, "remote_head", return_value=remote), \
             mock.patch.object(ops, "git", side_effect=controlled_git):
            ops.submit(True)
        self.assertEqual(self.git("rev-parse", "HEAD^"), manual_head)
        self.assertEqual(self.git("rev-parse", "HEAD^{tree}"), tree)
        self.assertIn("[h3-hsdp-offload]", self.git("log", "-1", "--format=%B"))
        self.assertEqual(len(pushes), 1)


if __name__ == "__main__":
    unittest.main()
