import importlib.util
import io
import json
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch
import zipfile

SPEC = importlib.util.spec_from_file_location("distilled_ops", Path(__file__).with_name("ops.py"))
ops = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ops)

CI_SPEC = importlib.util.spec_from_file_location("h3_distilled_ci_operator_test", ops.ROOT / ".github/scripts/h3_distilled_ci.py")
ci = importlib.util.module_from_spec(CI_SPEC)
CI_SPEC.loader.exec_module(ci)


class OperatorSafety(unittest.TestCase):
    def test_binary_and_unrelated_paths_rejected(self):
        for name in ("container/h3-distilled/private.env", "container/h3-distilled/adapter.safetensors", "pyproject.toml", "unrelated.txt"):
            self.assertFalse(ops.allowed(name))
        self.assertTrue(ops.allowed("container/h3-distilled/Dockerfile"))

    def test_staged_changes_not_hidden_by_worktree_cancellation(self):
        class Result:
            stdout = ""
        def run(*args, **kwargs):
            result = Result()
            result.stdout = "unrelated.txt\0" if "--cached" in args else ""
            return result
        with patch.object(ops, "run", run):
            self.assertEqual(ops.changed_paths(), ["unrelated.txt"])

    def test_submit_requires_intent(self):
        with self.assertRaises(ValueError):
            ops.submit(False)

    def test_wrong_branch_cannot_push(self):
        with patch.object(ops, "git", return_value="main"), self.assertRaises(ValueError):
            ops.submit(True)

    def test_archive_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("../image.json", "{}")
            with self.assertRaises(ValueError):
                ops.read_archive(path)

    def test_archive_ambiguous_names_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("a/image.json", "{}")
                archive.writestr("b/image.json", "{}")
            with self.assertRaises(ValueError):
                ops.read_archive(path)

    def test_import_rejects_missing_hashes_before_git_lookup(self):
        record = {"schema": "h3-distilled-publication/1", "platform": "linux/amd64"}
        with patch.object(ops, "read_archive", return_value={"image.json": json.dumps(record).encode()}):
            with self.assertRaisesRegex(ValueError, "fingerprints"):
                ops.import_evidence("unused.zip")

    def test_import_rejects_wrong_platform(self):
        record = {"schema": "h3-distilled-publication/1", "platform": "linux/arm64"}
        with patch.object(ops, "read_archive", return_value={"image.json": json.dumps(record).encode()}):
            with self.assertRaisesRegex(ValueError, "platform"):
                ops.import_evidence("unused.zip")


class PublicationArchiveRoundTrip(unittest.TestCase):
    """Real producer and archive consumer, with synthetic CPU report payloads.

    No Docker or GPU validation is implied: these tests exercise the actual
    publication JSON/hash contract through a real ZIP file. Only the committed
    Git-blob lookup and final local output destination are substituted.
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.export = self.directory / "export"
        self.export.mkdir()
        self.env = {"GITHUB_SHA": "0123456789abcdef0123456789abcdef01234567",
                    "GITHUB_RUN_ID": "7654321", "GITHUB_RUN_ATTEMPT": "1"}
        config = ops.lock()
        asset_sha = ci.sha256_file(ops.CONTEXT / "assets.lock.json")
        image_sha = ci.sha256_file(ops.CONTEXT / "image.lock.json")
        reports = {
            "runtime-source-check.json": {"status": "PASS_INSTALLED_CODE_ONLY", "synthetic_test": True},
            "assets-check.json": {"status": "PASS_ASSETS_ONLY", "lock_sha256": asset_sha,
                                  "synthetic_test": True},
            "bundle-check.json": {
                "status": "PASS_DISTILLED_CPU_ONLY", "source_commit": self.env["GITHUB_SHA"],
                "framework_source_commit": config["recipe_base_commit"], "base_image": config["base_image"],
                "environment_unchanged": True, "asset_lock_sha256": asset_sha,
                "image_lock_sha256": image_sha, "synthetic_test": True,
            },
            "base-environment.json": {"synthetic_test": True, "packages": {"vllm": "0.30.0"}},
            "runtime-environment.json": {"synthetic_test": True, "packages": {"vllm": "0.30.0"}},
        }
        for name, value in reports.items():
            (self.export / name).write_text(json.dumps(value) + "\n", encoding="utf-8")
        ci.copy_inputs(ops.ROOT, self.export)
        values = ci.prepare_values(ops.ROOT, self.env)
        self.manifest = ci.record(ops.ROOT, self.env, values["image"], "sha256:" + "d" * 64, self.export)
        self.payloads = {path.name: path.read_bytes() for path in self.export.iterdir()}
        self.output = self.directory / "imported"

    def archive(self, payloads):
        path = self.directory / "publication.zip"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, value in payloads.items():
                archive.writestr(name, value)
        return path

    def committed_lock_lookup(self, command, **kwargs):
        self.assertEqual(command[:2], ["git", "show"])
        self.assertEqual(kwargs["cwd"], ops.ROOT)
        self.assertTrue(kwargs["check"])
        self.assertTrue(kwargs["capture_output"])
        prefix = self.env["GITHUB_SHA"] + ":container/h3-distilled/"
        self.assertTrue(command[2].startswith(prefix))
        name = command[2][len(prefix):]
        self.assertIn(name, ("image.lock.json", "assets.lock.json"))
        return subprocess.CompletedProcess(command, 0, stdout=(ops.CONTEXT / name).read_bytes(), stderr=b"")

    def import_archive(self, payloads):
        path = self.archive(payloads)
        with patch.object(ops, "OUT", self.output), \
             patch.object(ops.subprocess, "run", side_effect=self.committed_lock_lookup), \
             patch("sys.stdout", new=io.StringIO()):
            ops.import_evidence(path)

    def test_real_producer_zip_and_importer_round_trip(self):
        self.import_archive(self.payloads)
        target = self.output / "published" / self.env["GITHUB_SHA"]
        self.assertEqual(json.loads((target / "image.json").read_bytes()), self.manifest)
        self.assertEqual((target / "deployment-image.txt").read_text().strip(), self.manifest["deployment_image"])
        self.assertIn(":8443/openclaw/ai-dingo-vllm@sha256:", self.manifest["deployment_image"])
        self.assertEqual({p.name for p in target.iterdir()}, set(self.payloads))
        for name in self.manifest["evidence_sha256"]:
            self.assertEqual((target / name).read_bytes(), self.payloads[name])

    def test_tampered_payload_is_rejected(self):
        changed = {**self.payloads, "runtime-source-check.json": b'{"status":"PASS_INSTALLED_CODE_ONLY","changed":true}\n'}
        with self.assertRaisesRegex(ValueError, "Evidence hash mismatch"):
            self.import_archive(changed)
        self.assertFalse(self.output.exists())

    def test_unmanifested_extra_file_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unmanifested extra"):
            self.import_archive({**self.payloads, "unexpected.txt": b"not fingerprinted\n"})
        self.assertFalse(self.output.exists())

    def test_casefold_duplicate_zip_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Ambiguous duplicate"):
            self.import_archive({**self.payloads, "IMAGE.JSON": self.payloads["image.json"]})
        self.assertFalse(self.output.exists())

if __name__ == "__main__":
    unittest.main()
