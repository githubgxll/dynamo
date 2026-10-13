"""Exercise prospective-index checks against real isolated Git repositories."""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("distilled_check_local_preflight", Path(__file__).with_name("check_local.py"))
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)
LEGAL = checker.CONTEXT / "legal"


class ProspectiveGitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="h3-distilled-git-test-")
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name).resolve()
        self.root = self.work / "repo"
        self.root.mkdir()
        self.evidence = self.work / "evidence"
        self.evidence.mkdir()
        self.hooks = self.work / "empty-hooks"
        self.hooks.mkdir()
        self.env = dict(os.environ)
        for name in ("GIT_INDEX_FILE", "GIT_DIR", "GIT_WORK_TREE"):
            self.env.pop(name, None)
        self.git("init", "--quiet")
        self.git("config", "core.autocrlf", "false")
        self.git("config", "core.safecrlf", "false")
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                 "-c", "core.hooksPath=" + str(self.hooks), "-c", "commit.gpgsign=false",
                 "commit", "--quiet", "--allow-empty", "-m", "Empty fixture HEAD")
        (self.root / ".gitattributes").write_bytes(b"* text eol=lf\n")
        # A real staged file makes index-preservation checks meaningful. The
        # helper must leave this entry byte-for-byte intact, even on failure.
        (self.root / "already-staged.txt").write_bytes(b"User's staged content\n")
        self.git("add", "--", "already-staged.txt")
        self.index = self.root / ".git/index"
        self.index_before = self.index.read_bytes()
        self.entries_before = self.git("ls-files", "--stage").stdout

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.root, env=self.env,
                              capture_output=True, check=True)

    def assert_index_unchanged(self):
        self.assertEqual(self.index.read_bytes(), self.index_before)
        self.assertEqual(self.git("ls-files", "--stage").stdout, self.entries_before)

    def legal_attributes(self):
        directory = self.root / "legal"
        directory.mkdir(exist_ok=True)
        (directory / ".gitattributes").write_bytes((LEGAL / ".gitattributes").read_bytes())
        return [".gitattributes", "legal/.gitattributes"]

    def test_untracked_trailing_spaces_are_missed_by_old_diff_but_rejected(self):
        (self.root / "bad.txt").write_bytes(b"Untracked trailing whitespace   \n")
        self.assertEqual(self.git("diff", "--check").stdout, b"")
        self.assertEqual(self.git("diff", "--cached", "--check").stdout, b"")
        with self.assertRaisesRegex(ValueError, "git-preview-whitespace failed"):
            checker.check_prospective_index(self.root, [".gitattributes", "bad.txt"], {}, self.evidence)
        self.assertIn(b"bad.txt", (self.evidence / "git-preview-whitespace.txt").read_bytes())
        self.assert_index_unchanged()

    def test_two_upstream_licenses_keep_exact_blob_hashes_without_index_mutation(self):
        paths = self.legal_attributes()
        documents = {}
        for name in ("MiniMax-H3-LICENSE", "fasth3-LICENSE"):
            payload = (LEGAL / name).read_bytes()
            relative = "legal/" + name
            (self.root / relative).write_bytes(payload)
            paths.append(relative)
            documents[relative] = hashlib.sha256(payload).hexdigest()
        report = checker.check_prospective_index(self.root, paths, documents, self.evidence)
        self.assertEqual(report["status"], "PASS_PROSPECTIVE_GIT_ONLY")
        self.assertEqual(report["documents_checked"], 2)
        self.assertTrue(report["real_index_unchanged"])
        self.assert_index_unchanged()

    def test_scoped_license_exception_does_not_allow_unrelated_whitespace(self):
        paths = self.legal_attributes()
        relative = "legal/unrelated.md"
        (self.root / relative).write_bytes(b"Still invalid outside two named licenses   \n")
        with self.assertRaisesRegex(ValueError, "git-preview-whitespace failed"):
            checker.check_prospective_index(self.root, paths + [relative], {}, self.evidence)
        self.assertIn(b"unrelated.md", (self.evidence / "git-preview-whitespace.txt").read_bytes())
        self.assert_index_unchanged()

    def test_pinned_license_crlf_bytes_remain_exact_with_minus_text(self):
        paths = self.legal_attributes()
        relative = "legal/MiniMax-H3-LICENSE"
        payload = b"Pinned upstream line one\r\nPinned upstream line two\r\n"
        (self.root / relative).write_bytes(payload)
        report = checker.check_prospective_index(
            self.root, paths + [relative], {relative: hashlib.sha256(payload).hexdigest()}, self.evidence)
        self.assertEqual(report["status"], "PASS_PROSPECTIVE_GIT_ONLY")
        self.assertEqual((self.evidence / "git-preview-MiniMax-H3-LICENSE.txt").read_bytes(), payload)
        self.assert_index_unchanged()

    def test_missing_minus_text_attribute_detects_git_normalization(self):
        relative = "legal/pinned-document.txt"
        (self.root / "legal").mkdir()
        payload = b"Pinned upstream line one\r\nPinned upstream line two\r\n"
        (self.root / relative).write_bytes(payload)
        with self.assertRaisesRegex(ValueError, "Git would alter a pinned upstream document"):
            checker.check_prospective_index(
                self.root, [".gitattributes", relative],
                {relative: hashlib.sha256(payload).hexdigest()}, self.evidence)
        self.assertEqual((self.evidence / "git-preview-pinned-document.txt.txt").read_bytes(), payload.replace(b"\r\n", b"\n"))
        self.assert_index_unchanged()


if __name__ == "__main__":
    unittest.main()
