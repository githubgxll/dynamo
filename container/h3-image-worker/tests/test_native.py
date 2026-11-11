"""CPU fixtures for strict active-distribution native checks; no real .so/GPU import."""
import base64
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("candidate3_runtime", Path(__file__).resolve().parents[1] / "runtime.py")
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


class NativeChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.site = self.root / "site"
        self.package = self.root / "package"
        self.package.mkdir()
        self.output = self.root / "evidence"
        self.names = ("cuda_ops", "lmcache_fs", "lmcache_native", "lmcache_redis")
        items, rows = [], []
        for name in self.names:
            relative = f"lmcache/{name}.cpython-312-x86_64-linux-gnu.so"
            data = b"\x7fELF-fixture-only-" + name.encode()
            path = self.site / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            digest = hashlib.sha256(data)
            items.append({"path": relative, "sha256": digest.hexdigest(), "bytes": len(data)})
            rows.append([relative, "sha256=" + base64.urlsafe_b64encode(digest.digest()).decode().rstrip("="), str(len(data))])
        self.lock = {"schema": "h3-image-worker-native-artifacts/1", "distribution": "lmcache",
                     "version": "0.5.6", "gpu_entrypoint": "lmcache.cuda_ops", "files": items}
        (self.package / "native.lock.json").write_text(json.dumps(self.lock), encoding="utf-8")
        self.rows = rows
        self.files = [item["path"] for item in items]
        self.loaded = []

    def record_text(self):
        output = io.StringIO()
        csv.writer(output).writerows(self.rows)
        return output.getvalue()

    def distribution(self, name):
        if name == "lmcache":
            return SimpleNamespace(version="0.5.6", files=self.files,
                                   locate_file=lambda path: self.site / path,
                                   read_text=lambda name: self.record_text() if name == "RECORD" else None)
        if name == "torch":
            return SimpleNamespace(locate_file=lambda path: self.site / path)
        raise AssertionError("unexpected distribution " + name)

    def importer(self, name):
        self.loaded.append(name)
        if name in ("torch", "lmcache"):
            path = self.site / name / "__init__.py"
        else:
            basename = name.split(".")[-1]
            path = self.site / f"lmcache/{basename}.cpython-312-x86_64-linux-gnu.so"
        return SimpleNamespace(__file__=str(path), __spec__=SimpleNamespace(origin=str(path)))

    def run_check(self, importer=None):
        with patch.object(runtime, "HERE", self.package), \
                patch.object(runtime.metadata, "distribution", side_effect=self.distribution), \
                patch.object(runtime.importlib, "import_module", side_effect=importer or self.importer):
            return runtime.check_lmcache_native(self.output)

    def test_all_four_payloads_and_two_actual_native_entrypoints_pass(self):
        report = self.run_check()
        self.assertEqual(report["status"], "PASS_STRICT_NATIVE_CPU_LOADER")
        self.assertEqual(len(report["files"]), 4)
        self.assertEqual(self.loaded, ["torch", "lmcache", "lmcache.lmcache_native", "lmcache.cuda_ops"])
        self.assertNotIn("vllm._C", self.loaded)
        self.assertFalse(report["driver_error_skips"])

    def test_unknown_installed_native_artifact_is_fatal(self):
        self.files.append("lmcache/unknown.cpython-312-x86_64-linux-gnu.so")
        with self.assertRaisesRegex(ValueError, "selection differs"):
            self.run_check()
        self.assertEqual(self.loaded, [])
        self.assertEqual(json.loads((self.output / "native-validation.json").read_text())["status"], "FAILED_NATIVE_CHECK")

    def test_same_size_wrong_payload_hash_is_fatal(self):
        path = self.site / self.lock["files"][0]["path"]
        original = path.read_bytes()
        path.write_bytes(b"X" + original[1:])
        with self.assertRaisesRegex(ValueError, "SHA256 differs"):
            self.run_check()
        self.assertEqual(self.loaded, [])

    def test_unrecorded_native_file_is_fatal(self):
        (self.site / "lmcache/unrecorded_extra.so").write_bytes(b"fixture-only")
        with self.assertRaisesRegex(ValueError, "Actual LMCache native files differ"):
            self.run_check()
        self.assertEqual(self.loaded, [])

    def test_wrong_record_or_duplicate_row_is_fatal(self):
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate):
                original = [list(row) for row in self.rows]
                if duplicate:
                    self.rows.append(list(self.rows[0]))
                else:
                    self.rows[0][1] = "sha256=" + "A" * 43
                with self.assertRaisesRegex(ValueError, "RECORD hash/size"):
                    self.run_check()
                self.rows = original
        self.assertEqual(self.loaded, [])

    def test_native_import_driver_error_is_not_skipped_and_full_failure_is_saved(self):
        marker = "CUDA driver unavailable: full fixture diagnostics " + "x" * 20000
        def fail(name):
            if name == "lmcache.cuda_ops":
                raise ImportError(marker)
            return self.importer(name)
        with self.assertRaises(ImportError) as caught:
            self.run_check(fail)
        self.assertIn(marker, str(caught.exception))
        report = json.loads((self.output / "native-validation.json").read_text())
        self.assertEqual(report["status"], "FAILED_NATIVE_CHECK")
        self.assertIn(marker, report["traceback"])
        self.assertIn(marker, (self.output / "native-failure.txt").read_text())

    def test_native_import_from_other_installation_is_fatal(self):
        def shadow(name):
            result = self.importer(name)
            if name == "lmcache.cuda_ops":
                result.__file__ = str(self.root / "shadow/cuda_ops.so")
            return result
        with self.assertRaisesRegex(ValueError, "locked artifact"):
            self.run_check(shadow)

    def test_native_module_spec_origin_mismatch_is_fatal(self):
        def altered_spec(name):
            result = self.importer(name)
            if name == "lmcache.cuda_ops":
                result.__spec__.origin = str(self.root / "shadow/cuda_ops.so")
            return result
        with self.assertRaisesRegex(ValueError, "locked artifact"):
            self.run_check(altered_spec)


if __name__ == "__main__":
    unittest.main()

