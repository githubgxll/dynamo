"""Inventory isolation and strict drift checks; no Torch/image required."""
import contextlib
import importlib.metadata as metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("c14_preservation_under_test", HERE / "runtime.py")
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


def inventory_probe(root):
    probe = root / "inventory.py"
    probe.write_text("import importlib.metadata as m,json\n"
        "rows=[(d.metadata['Name'],d.version,d.read_text('METADATA')) for d in m.distributions()]\n"
        # Match the production snapshot's sort key. Keep raw None and duplicate
        # rows in JSON: only ordering treats missing METADATA as an empty string.
        "rows.sort(key=lambda row:(row[0].lower(),row[1],row[2] or ''))\n"
        "print(json.dumps({'packages':rows}))\n", encoding="utf-8")
    return probe


class InventoryIsolation(unittest.TestCase):
    def test_fresh_process_ignores_import_added_vendor_but_sees_real_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vendor = root / "vendor"
            dist = vendor / "c14_probe_vendor-1.0.dist-info"
            dist.mkdir(parents=True)
            dist_metadata = dist / "METADATA"
            dist_metadata.write_text("Metadata-Version: 2.1\nName: c14-probe-vendor\nVersion: 1.0\n", encoding="utf-8")
            probe = inventory_probe(root)
            before = runtime.json_probe(probe)
            # Equivalent to setuptools importing its bundled _vendor directory:
            # no files were installed, but same-process distributions() expands.
            with patch.object(sys, "path", [*sys.path, str(vendor)]):
                self.assertIn("c14-probe-vendor", [d.metadata["Name"] for d in metadata.distributions()])
                self.assertEqual(runtime.json_probe(probe), before)
            # A real startup-visible installation is still included and tracked.
            with patch.dict(os.environ, {"PYTHONPATH": str(vendor)}):
                visible = runtime.json_probe(probe)
                self.assertNotEqual(visible, before)
                dist_metadata.write_text("Metadata-Version: 2.1\nName: c14-probe-vendor\nVersion: 2.0\n", encoding="utf-8")
                self.assertNotEqual(runtime.json_probe(probe), visible)

    def test_mixed_egg_and_dist_info_preserves_raw_metadata_and_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            site = root / "linux-site"
            for suffix, filename in (("dist-info", "METADATA"), ("egg-info", "PKG-INFO")):
                folder = site / ("c14_probe_system-1.0." + suffix)
                folder.mkdir(parents=True)
                (folder / filename).write_text("Metadata-Version: 2.1\nName: c14-probe-system\nVersion: 1.0\n", encoding="utf-8")
            raw = [(d.metadata["Name"], d.version, d.read_text("METADATA"))
                   for d in metadata.distributions(path=[str(site)])]
            self.assertEqual(len(raw), 2)
            # Reproduce the old fixture's failure with real importlib readers.
            with self.assertRaises(TypeError):
                sorted(raw)
            probe = inventory_probe(root)
            with patch.dict(os.environ, {"PYTHONPATH": str(site)}):
                before = runtime.json_probe(probe)
                actual = [row for row in before["packages"] if row[0] == "c14-probe-system"]
                self.assertEqual(len(actual), 2)
                self.assertEqual(sum(row[2] is None for row in actual), 1)
                self.assertTrue(any(isinstance(row[2], str) and "Name: c14-probe-system" in row[2] for row in actual))
                (site / "c14_probe_system-1.0.dist-info" / "METADATA").write_text(
                    "Metadata-Version: 2.1\nName: c14-probe-system\nVersion: 1.0\nRequires-Dist: changed\n", encoding="utf-8")
                self.assertNotEqual(runtime.json_probe(probe), before)

    def test_probe_failure_is_not_an_empty_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / "fail.py"
            script.write_text("import sys\nprint('private-inventory-content')\n"
                              "print('TypeError: mixed metadata fixture',file=sys.stderr)\nraise SystemExit(9)\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "exit 9") as caught:
                runtime.json_probe(script)
            self.assertIn("TypeError: mixed metadata fixture", str(caught.exception))
            self.assertNotIn("private-inventory-content", str(caught.exception))
            self.assertIsInstance(caught.exception.__cause__, subprocess.CalledProcessError)

    def test_probe_failure_without_stderr_keeps_exit_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / "empty-fail.py"
            script.write_text("raise SystemExit(7)\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "(?s)exit 7.*no stderr"):
                runtime.json_probe(script)

    def test_probe_rejects_wrong_schema_type(self):
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / "list.py"
            script.write_text("print('[]')\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "object"):
                runtime.json_probe(script)

    def test_probe_preserves_interpreter_user_environment_and_disables_bytecode(self):
        with patch.object(runtime.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, '{}', '')) as run:
            runtime.json_probe(HERE / "runtime.py", "snapshot")
        args, kwargs = run.call_args
        self.assertEqual(args[0], [sys.executable, "-B", str(HERE / "runtime.py"), "snapshot"])
        self.assertEqual(kwargs["env"], dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        self.assertTrue(kwargs["check"])
        self.assertEqual(kwargs["timeout"], 120)


class StrictPreservation(unittest.TestCase):
    def baseline(self):
        return {"packages": [{"name": "package-a", "version": "1.0", "metadata": "Requires-Dist: dep-a\n",
            "direct_url": None}], "dpkg": ["lib-a\t1\tamd64"],
            "inherited_files_sha256": {"/legal/NOTICE": "a" * 64}}

    def rejected(self, expected, actual):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()) as output:
            directory = Path(temporary)
            with self.assertRaisesRegex(ValueError, "environment-diff-after-validation"):
                runtime.assert_snapshot(expected, actual, directory, "after-validation")
            report = json.loads((directory / "environment-diff-after-validation.json").read_text())
            return report, output.getvalue()

    def test_unchanged_environment_produces_no_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime.assert_snapshot(self.baseline(), self.baseline(), Path(temporary), "before-validation")
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_package_metadata_drift_is_not_filtered(self):
        before, after = self.baseline(), self.baseline()
        after["packages"][0]["metadata"] = "Requires-Dist: another-dep\n"
        report, _ = self.rejected(before, after)
        self.assertEqual(len(report["changes"]["packages"]["added"]), 1)
        self.assertEqual(len(report["changes"]["packages"]["removed"]), 1)

    def test_direct_url_is_checked_without_logging_credentials(self):
        before, after = self.baseline(), self.baseline()
        secret = "https://user:secret-value@private.example/model.whl"
        after["packages"][0]["direct_url"] = json.dumps({"url": secret})
        report, output = self.rejected(before, after)
        self.assertIn("packages", report["changes"])
        self.assertNotIn("secret-value", json.dumps(report) + output)

    def test_duplicate_metadata_count_still_fails(self):
        before, after = self.baseline(), self.baseline()
        after["packages"].append(dict(after["packages"][0]))
        report, _ = self.rejected(before, after)
        self.assertEqual(len(report["changes"]["packages"]["added"]), 1)

    def test_inherited_file_change_is_named(self):
        before, after = self.baseline(), self.baseline()
        after["inherited_files_sha256"]["/legal/NOTICE"] = "b" * 64
        report, _ = self.rejected(before, after)
        self.assertEqual(report["changes"]["inherited_files_sha256"]["changed"], ["/legal/NOTICE"])

    def test_os_package_change_is_named(self):
        before, after = self.baseline(), self.baseline()
        after["dpkg"].append("lib-b\t2\tamd64")
        report, _ = self.rejected(before, after)
        self.assertEqual(report["changes"]["dpkg"]["added"], ["lib-b\t2\tamd64"])


if __name__ == "__main__":
    unittest.main()
