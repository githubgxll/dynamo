"""Source routing, immutable-wheel acquisition, and full-closure regression tests."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import package_sources as sources_module
import prepare_environment as prep


class SourceTests(unittest.TestCase):
    def test_checked_in_sources_and_supported_official_alternative(self):
        sources = sources_module.load_sources()
        self.assertEqual(sources["python_index"], "https://pypi.tuna.tsinghua.edu.cn/simple")
        self.assertEqual(sources["torch_index"], "https://download.pytorch.org/whl/cu121")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sources.json"
            sources["python_index"] = "https://pypi.org/simple"
            path.write_text(json.dumps(sources))
            self.assertEqual(sources_module.load_sources(path), sources)

    def test_rejects_unreviewed_hosts_credentials_queries_and_cuda_change(self):
        for key, value in (
            ("python_index", "http://pypi.tuna.tsinghua.edu.cn/simple"),
            ("python_index", "https://user:secret@pypi.tuna.tsinghua.edu.cn/simple"),
            ("python_index", "https://pypi.tuna.tsinghua.edu.cn/simple?token=secret"),
            ("python_index", "https://pypi.tuna.tsinghua.edu.cn/simple#fragment"),
            ("python_index", "https://unreviewed.example/simple"),
            ("torch_index", "https://download.pytorch.org/whl/cu130"),
            ("schema_version", True),
        ):
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as directory:
                data = sources_module.load_sources()
                data[key] = value
                path = Path(directory) / "sources.json"
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    sources_module.load_sources(path)

    def test_bootstrap_explicit_source_preserves_versions_and_failure_evidence(self):
        sources = sources_module.load_sources()
        with tempfile.TemporaryDirectory() as directory, patch.object(sources_module.subprocess, "run") as run:
            evidence = Path(directory)
            sources_module.bootstrap(sources, evidence)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--index-url") + 1], sources["python_index"])
            self.assertEqual(command[-4:], ["pip==24.3.1", "setuptools==75.8.0", "wheel==0.45.1", "uv==0.8.22"])
            self.assertIn("--isolated", command)
            self.assertNotIn("--trusted-host", command)
            self.assertEqual(command[command.index("--retries") + 1], "1")
            self.assertEqual(command[command.index("--timeout") + 1], "15")
            self.assertEqual(json.loads((evidence / "package-bootstrap.json").read_text())["status"], "PASSED")
            self.assertEqual(json.loads((evidence / "package-sources.json").read_text()), sources)
            run.side_effect = subprocess.CalledProcessError(1, command)
            with self.assertRaises(subprocess.CalledProcessError):
                sources_module.bootstrap(sources, evidence)
            failed = json.loads((evidence / "package-bootstrap.json").read_text())
            self.assertEqual((failed["status"], failed["returncode"]), ("FAILED", 1))


class OriginalWheelTests(unittest.TestCase):
    PAYLOAD = b"fixture for exact reviewed wheel bytes"

    def fake_download(self, command, output=None):
        self.assertEqual(command[command.index("--index-url") + 1], sources_module.load_sources()["python_index"])
        self.assertIn("--require-hashes", command)
        self.assertIn("--only-binary=:all:", command)
        self.assertIn("--no-deps", command)
        self.assertIn("--isolated", command)
        requirement = Path(command[command.index("-r") + 1]).read_text()
        self.assertEqual(requirement, f"facexlib==0.3.0 --hash=sha256:{prep.ORIGINAL_SHA256}\n")
        destination = Path(command[command.index("--dest") + 1])
        (destination / prep.ORIGINAL_FILENAME).write_bytes(self.PAYLOAD)

    def test_acquisition_uses_selected_index_and_verified_cache(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(prep, "ORIGINAL_SHA256", hashlib.sha256(self.PAYLOAD).hexdigest()), patch.object(prep, "run", side_effect=self.fake_download) as run:
            root = Path(directory)
            result = prep.acquire_original(root)
            self.assertEqual(result.read_bytes(), self.PAYLOAD)
            self.assertEqual(list(root.iterdir()), [result])
            self.assertEqual(prep.acquire_original(root), result)
            self.assertEqual(run.call_count, 1)

    def test_bad_hash_is_rejected_without_poisoning_cache(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(prep, "run", side_effect=self.fake_download):
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                prep.acquire_original(root)
            self.assertEqual(list(root.iterdir()), [])

    def test_partial_download_failure_leaves_no_cached_wheel(self):
        def fail(command, output=None):
            self.fake_download(command)
            raise subprocess.CalledProcessError(1, command)

        with tempfile.TemporaryDirectory() as directory, patch.object(prep, "run", side_effect=fail):
            root = Path(directory)
            with self.assertRaises(subprocess.CalledProcessError):
                prep.acquire_original(root)
            self.assertEqual(list(root.iterdir()), [])

    def test_cached_bad_wheel_rejected_before_any_download(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(prep, "run") as run:
            root = Path(directory)
            (root / prep.ORIGINAL_FILENAME).write_bytes(b"corrupt cached wheel")
            with self.assertRaisesRegex(ValueError, "cached.*SHA256"):
                prep.acquire_original(root)
            run.assert_not_called()


class EnvironmentClosureTests(unittest.TestCase):
    def test_prepare_and_publish_route_both_indexes_and_keep_full_hashed_install(self):
        sources = sources_module.load_sources()
        for phase in ("prepare", "publish"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                inputs, lock, evidence = root / "requirements.in", root / "requirements.lock", root / "evidence"
                inputs.write_text("torch==2.3.1+cu121\n")
                lock.write_text("fixture==1.0 --hash=sha256:fixture\n")
                commands = []

                def capture(command, output=None):
                    commands.append(command)
                    if command[:3] == ["uv", "pip", "compile"]:
                        Path(command[command.index("--output-file") + 1]).write_text(lock.read_text())
                    if output is not None:
                        output.write_text("Fixture checker passed.\n")

                argv = ["prepare_environment.py", "--phase", phase, "--inputs", str(inputs), "--lock", str(lock), "--evidence", str(evidence)]
                with patch.object(prep.sys, "argv", argv), patch.object(prep.sys, "version_info", (3, 10)), patch.object(prep.sys, "platform", "linux"), patch.object(prep.platform, "machine", return_value="x86_64"), patch.object(prep.platform, "platform", return_value="test Linux"), patch.object(prep, "acquire_original") as acquire, patch.object(prep, "patch_facexlib"), patch.object(prep, "run", side_effect=capture):
                    prep.main()
                self.assertEqual(acquire.call_args.args[1], sources)
                installs = [command for command in commands if command[:3] == ["uv", "pip", "install"]]
                self.assertEqual(len(installs), 1)
                install = installs[0]
                self.assertIn("--require-hashes", install)
                self.assertNotIn("--no-deps", install)
                self.assertNotIn("--no-verify-hashes", install)
                for command in commands:
                    if command[:3] in (["uv", "pip", "compile"], ["uv", "pip", "install"]):
                        self.assertEqual(command[command.index("--default-index") + 1], sources["python_index"])
                        self.assertEqual(command[command.index("--index") + 1], sources["torch_index"])
                record = json.loads((evidence / "dependency-build.json").read_text())
                self.assertEqual(record["package_sources"], sources)
                self.assertTrue(record["full_declared_dependency_closure"])
                self.assertIn(["uv", "pip", "check", "--python", sys.executable], commands)


if __name__ == "__main__":
    unittest.main()
