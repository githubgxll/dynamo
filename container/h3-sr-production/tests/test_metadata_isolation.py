"""Real metadata discovery in clean interpreters; no distro-name exceptions."""
from __future__ import annotations

import contextlib
import importlib.metadata as md
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_metadata_isolation_runtime", HERE / "runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
NAME = "h3sr-metadata-isolation-fixture"


def distribution(root, version):
    info = root / (NAME.replace("-", "_") + "-" + version + ".dist-info")
    info.mkdir(parents=True)
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: " + NAME + "\nVersion: " + version + "\n", encoding="utf-8")


class CleanDiscovery(unittest.TestCase):
    def test_import_added_parent_vendor_is_ignored_but_startup_package_is_seen(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary, vendor = root / "installed", root / "import-added-vendor"
            distribution(primary, "1.0")
            distribution(vendor, "2.0")
            path = str(primary) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
            with mock.patch.dict(os.environ, {"PYTHONPATH": path}), mock.patch.object(sys, "path", [str(primary), *sys.path]):
                before = runtime.inventory()
                self.assertEqual([row for row in before if row[0] == NAME], [(NAME, "1.0")])
                sys.path.insert(0, str(vendor))
                in_process = [dist.version for dist in md.distributions() if dist.metadata.get("Name") == NAME]
                self.assertEqual(sorted(in_process), ["1.0", "2.0"])
                self.assertEqual(runtime.inventory(), before)

    def test_real_startup_duplicates_are_preserved_and_real_change_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary, other = root / "installed-a", root / "installed-b"
            distribution(primary, "1.0")
            distribution(other, "2.0")
            with mock.patch.dict(os.environ, {"PYTHONPATH": str(primary)}):
                before = runtime.inventory()
            with mock.patch.dict(os.environ, {"PYTHONPATH": str(primary) + os.pathsep + str(other)}):
                after = runtime.inventory()
            self.assertNotEqual(before, after)
            self.assertEqual([row for row in after if row[0] == NAME], [(NAME, "1.0"), (NAME, "2.0")])

    def test_report_records_real_interpreter_uid_paths_and_origins(self):
        with tempfile.TemporaryDirectory() as temporary:
            value = runtime.fresh_process("metadata-inventory", Path(temporary))
            self.assertEqual(value["python"], sys.executable)
            self.assertEqual(value["euid"], os.geteuid() if hasattr(os, "geteuid") else None)
            self.assertEqual(value["cwd"], os.getcwd())
            self.assertTrue(value["sys_path"])
            self.assertTrue(value["discovered"])
            self.assertTrue(all("metadata_path" in item and "location" in item for item in value["discovered"]))


class FailedProbe(unittest.TestCase):
    def test_nonzero_child_never_uses_an_existing_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            runtime.fresh_process("metadata-inventory", path)
            before = (path / "clean-metadata-context.json").read_bytes()
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "exit 1"):
                runtime.fresh_process("metadata-inventory", path)
            self.assertEqual((path / "clean-metadata-context.json").read_bytes(), before)

    def test_missing_report_is_not_treated_as_success(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(runtime.subprocess, "run", return_value=
                subprocess.CompletedProcess(["python"], 0, "not a report", "")):
            with self.assertRaises(FileNotFoundError):
                runtime.fresh_process("metadata-inventory", Path(temporary))

    def test_timeout_preserves_partial_log_and_fails(self):
        error = subprocess.TimeoutExpired(["python"], 1, output=b"partial probe output", stderr=b"timeout")
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(runtime.subprocess, "run", side_effect=error):
            path = Path(temporary)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "TimeoutExpired"):
                runtime.fresh_process("metadata-inventory", path)
            self.assertIn("partial probe output", (path / "fresh-process.log").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
