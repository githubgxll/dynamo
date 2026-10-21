"""Offline regression for source ownership, strict identities and complete package checks."""
from __future__ import annotations

import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import sys
import unittest
import zipfile

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lite_runtime", HERE / "runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class RuntimeTests(unittest.TestCase):
    def load_capacity_source(self, relative, name):
        # Local tests use exact vendored source and unchanged repository helpers;
        # the final image test imports the actual installed module instead.
        repo = HERE.parents[1]
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        source_spec = importlib.util.spec_from_file_location(name, HERE / relative)
        module = importlib.util.module_from_spec(source_spec)
        sys.modules[name] = module
        source_spec.loader.exec_module(module)
        return module

    def test_latest_gateway_nine_capacity_cases(self):
        regression = self.load_capacity_source("gateway_regression.py", "lite_capacity_regression")
        module = self.load_capacity_source("gateway/after/artifact_store.py", "lite_after_source")
        report = regression.check_capacity(module)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["case_count"], 9)

    def test_capacity_regression_rejects_previous_source(self):
        regression = self.load_capacity_source("gateway_regression.py", "lite_capacity_regression_before")
        module = self.load_capacity_source("gateway/before/artifact_store.py", "lite_before_source")
        with self.assertRaisesRegex(AssertionError, "ancestor-quota-refresh"):
            regression.check_capacity(module)

    def test_record_changes_exact_owned_row(self):
        old, new = b"old source\n", b"new source\n"
        target = "dingo/video_gateway/artifact_store.py"
        source = f"{target},{runtime.record_digest(old)},{len(old)}\nother.py,sha256=keep,7\n"
        rows = list(csv.reader(io.StringIO(runtime.update_record(source.encode(), target, old, new).decode())))
        self.assertEqual(rows, [[target, runtime.record_digest(new), str(len(new))], ["other.py", "sha256=keep", "7"]])

    def test_record_rejects_wrong_hash(self):
        with self.assertRaisesRegex(ValueError, "original RECORD"):
            runtime.update_record(b"file.py,sha256=wrong,3\n", "file.py", b"old", b"new")

    def test_record_rejects_duplicate(self):
        line = f"file.py,{runtime.record_digest(b'old')},3\n".encode()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            runtime.update_record(line + line, "file.py", b"old", b"new")

    def test_record_rejects_missing(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            runtime.update_record(b"other.py,,\n", "file.py", b"old", b"new")

    def test_path_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            for path in ("../secret", "dingo/../../secret", "dingo\\secret"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    runtime.safe_file(Path(directory), path)

    def test_manifest_rejects_missing_extra_and_changed(self):
        for actual in ({}, {"a": "one", "b": "two"}, {"a": "wrong"}):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                runtime.compare_manifest(actual, {"a": "one"}, "test")

    def gateway_fixture(self, directory):
        root = Path(directory)
        (root / "dingo").mkdir()
        (root / "dingo/core.py").write_bytes(b"content\n")
        (root / "dingo/_version.py").write_bytes(b"version\n")
        expected = {"dingo/core.py": runtime.sha256(b"content\n")}
        lock = {"source_before_sha256": expected, "source_after_sha256": expected,
                "allowed_generated_version_paths": ["dingo/_version.py"]}
        return root, lock

    def test_gateway_explicit_generated_allowance(self):
        with tempfile.TemporaryDirectory() as directory:
            root, lock = self.gateway_fixture(directory)
            result = runtime.gateway_sources(root, lock, patched=False)
            self.assertEqual(result["tracked_python_count"], 1)
            self.assertEqual(set(result["generated_version_sha256"]), {"dingo/_version.py"})

    def test_gateway_rejects_extra_generated_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root, lock = self.gateway_fixture(directory)
            (root / "dingo/extra").mkdir()
            (root / "dingo/extra/_version.py").write_bytes(b"unexpected")
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                runtime.gateway_sources(root, lock, patched=False)

    def test_gateway_rejects_missing_generated(self):
        with tempfile.TemporaryDirectory() as directory:
            root, lock = self.gateway_fixture(directory)
            (root / "dingo/_version.py").unlink()
            with self.assertRaisesRegex(ValueError, "version-file"):
                runtime.gateway_sources(root, lock, patched=True)

    def omni_fixture(self, directory):
        root = Path(directory)
        package = root / "vllm_omni"
        package.mkdir()
        content = {"vllm_omni/nadit.py": b"old rms", "vllm_omni/lora.py": b"native lora", "vllm_omni/data.json": b"{}"}
        wheel = root / "original.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            for name, data in content.items():
                archive.writestr(name, data)
                (root / name).write_bytes(data)
        lock = {"omni_wheel_sha256": runtime.sha256(wheel.read_bytes()), "omni_patch_path": "vllm_omni/nadit.py",
                "native_lora_path": "vllm_omni/lora.py", "native_lora_sha256": runtime.sha256(b"native lora"),
                "omni_patch_after_sha256": runtime.sha256(b"new rms")}
        return root, wheel, lock

    def test_full_omni_includes_non_python_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            root, wheel, lock = self.omni_fixture(directory)
            result = runtime.omni_sources(root, wheel, lock, patched=False)
            self.assertEqual(result["package_file_count"], 3)
            self.assertEqual(result["python_file_count"], 2)

    def test_omni_allows_only_rmsnorm_delta(self):
        with tempfile.TemporaryDirectory() as directory:
            root, wheel, lock = self.omni_fixture(directory)
            (root / "vllm_omni/nadit.py").write_bytes(b"new rms")
            runtime.omni_sources(root, wheel, lock, patched=True)
            (root / "vllm_omni/lora.py").write_bytes(b"distilled patch")
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                runtime.omni_sources(root, wheel, lock, patched=True)

    def test_omni_rejects_missing_resource(self):
        with tempfile.TemporaryDirectory() as directory:
            root, wheel, lock = self.omni_fixture(directory)
            (root / "vllm_omni/data.json").unlink()
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                runtime.omni_sources(root, wheel, lock, patched=False)

    def test_omni_rejects_extra_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root, wheel, lock = self.omni_fixture(directory)
            (root / "vllm_omni/extra.py").write_bytes(b"shadow")
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                runtime.omni_sources(root, wheel, lock, patched=False)

    def test_omni_rejects_wrong_wheel(self):
        with tempfile.TemporaryDirectory() as directory:
            root, wheel, lock = self.omni_fixture(directory)
            lock["omni_wheel_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                runtime.omni_sources(root, wheel, lock, patched=False)

    def test_recipe_locks_consistent(self):
        lock = runtime.read_json(HERE / "runtime.lock.json")
        patch = runtime.read_json(HERE / "rmsnorm-patch.lock.json")
        policy = runtime.read_json(HERE / "environment-policy.json")
        gateway = runtime.read_json(HERE / "gateway.lock.json")
        runtime.validate_inputs(lock, gateway)
        self.assertEqual(lock["parent_image"], patch["parent_image"])
        self.assertEqual(lock["gateway_commit"], gateway["current_commit"])
        self.assertEqual(lock["gateway_commit"], policy["gateway_sha"])
        self.assertEqual(lock["omni_patch_after_sha256"], patch["after_sha256"])
        changes = [name for name in gateway["source_before_sha256"]
                   if gateway["source_before_sha256"][name] != gateway["source_after_sha256"][name]]
        self.assertEqual(changes, [gateway["changed_runtime_file"]])
        replacement = HERE / "gateway/after/artifact_store.py"
        self.assertEqual(runtime.sha256(replacement.read_bytes()), gateway["after_sha256"])

    def test_no_installs_or_assets_in_dockerfile(self):
        text = (HERE / "Dockerfile").read_text()
        self.assertNotIn("pip install", text)
        self.assertNotIn("apt-get", text)
        self.assertNotIn("dnf ", text)
        self.assertNotIn("fetch_assets", text)
        self.assertIn("--network=none", text)
        self.assertIn("USER dynamo", text)


if __name__ == "__main__":
    unittest.main()
