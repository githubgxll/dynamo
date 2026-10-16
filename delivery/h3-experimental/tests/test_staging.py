"""Real Git integration tests for ignored upstream files and Windows index modes."""
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("staging_ops", HERE / "ops.py")
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)


class RealGitStagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        subprocess.run([ops.GIT, "init", "-q", str(self.repo)], check=True)
        self.root_patch = patch.object(ops, "ROOT", self.repo)
        self.evidence_patch = patch.object(ops, "EVIDENCE", self.repo / ".evidence")
        self.root_patch.start()
        self.evidence_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.addCleanup(self.evidence_patch.stop)
        ops.git("config", "core.autocrlf", "false")
        ops.git("config", "core.filemode", "false")
        ops.git("config", "core.symlinks", "false")
        ops.git("config", "user.name", "Fixture")
        ops.git("config", "user.email", "fixture@example.invalid")
        (self.repo / ".gitignore").write_text("ignored.txt\nvendor/\n.evidence/\n", encoding="utf-8")
        (self.repo / "tracked.txt").write_text("initial\n", encoding="utf-8")
        ops.git("add", ".gitignore", "tracked.txt")
        ops.git("commit", "-q", "-m", "fixture baseline")
        self.old_tree = ops.git("rev-parse", "HEAD^{tree}")
        (self.repo / "tracked.txt").write_text("reviewed\n", encoding="utf-8")
        (self.repo / "ignored.txt").write_text("upstream tracked and ignored\n", encoding="utf-8")
        (self.repo / "vendor").mkdir()
        (self.repo / "vendor/source.txt").write_text("reviewed upstream\n", encoding="utf-8")
        (self.repo / "run.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        (self.repo / "link").write_text("tracked.txt", encoding="utf-8")
        # Build a main-like tree with tracked ignored files, +x, and a symlink.
        ops.git("add", "-f", "tracked.txt", "ignored.txt", "vendor/source.txt", "run.sh", "link")
        ops.git("update-index", "--chmod=+x", "run.sh")
        link_blob = ops.git("hash-object", "link")
        ops.git("update-index", "--cacheinfo", "120000," + link_blob + ",link")
        self.expected = ops.git("write-tree")
        ops.git("read-tree", self.old_tree)  # Preserve main-like worktree, old branch index.
        (self.repo / "vendor/private-local.txt").write_text("must never be staged\n", encoding="utf-8")

    def test_actual_old_add_failure_is_reproduced(self):
        with self.assertRaisesRegex(RuntimeError, "ignored"):
            ops.git("--literal-pathspecs", "add", "ignored.txt", "vendor/source.txt")

    def test_checked_tree_preserves_ignored_files_exec_and_symlink(self):
        before = {p.relative_to(self.repo).as_posix(): p.read_bytes()
                  for p in self.repo.rglob("*") if p.is_file() and ".git" not in p.parts}
        ops.stage_validated_tree(self.expected)
        self.assertEqual(ops.git("write-tree"), self.expected)
        entries = ops.index_entries()
        self.assertEqual(entries["run.sh"][0], "100755")
        self.assertEqual(entries["link"][0], "120000")
        self.assertIn("ignored.txt", entries)
        self.assertIn("vendor/source.txt", entries)
        self.assertNotIn("vendor/private-local.txt", entries)
        for name, value in before.items():
            self.assertEqual((self.repo / name).read_bytes(), value)

    def test_retry_from_partially_staged_old_attempt(self):
        ops.git("add", "tracked.txt", "run.sh", "link")
        self.assertEqual(ops.index_entries()["link"][0], "100644")
        ops.stage_validated_tree(self.expected)
        self.assertEqual(ops.git("write-tree"), self.expected)

    def test_index_is_backed_up_byte_for_byte(self):
        original = (self.repo / ".git/index").read_bytes()
        backup = ops.stage_validated_tree(self.expected)
        self.assertEqual((backup / "index").read_bytes(), original)
        record = ops.read_json(backup / "record.json")
        self.assertEqual(record["index_sha256"], hashlib.sha256(original).hexdigest())

    def test_staged_only_edits_are_not_discarded(self):
        (self.repo / "tracked.txt").write_text("staged-only user edit\n", encoding="utf-8")
        ops.git("add", "tracked.txt")
        (self.repo / "tracked.txt").write_text("reviewed\n", encoding="utf-8")
        before = ops.git("write-tree")
        with self.assertRaisesRegex(ValueError, "Staged-only content"):
            ops.stage_validated_tree(self.expected)
        self.assertEqual(ops.git("write-tree"), before)

    def test_unreviewed_ignored_staged_file_is_not_discarded_or_published(self):
        ops.git("add", "-f", "vendor/private-local.txt")
        before = ops.git("write-tree")
        with self.assertRaisesRegex(ValueError, "Staged-only content"):
            ops.stage_validated_tree(self.expected)
        self.assertEqual(ops.git("write-tree"), before)


if __name__ == "__main__":
    unittest.main()
