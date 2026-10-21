# SPDX-License-Identifier: Apache-2.0
"""Local stdlib checks; numerical acceptance executes in the actual image."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import patchlib
import apply_patch as application
import validate_runtime as runtime


def test_temp():
    root = HERE.parents[1] / ".h3-experimental-evidence/srnorm/local/tmp"
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=root)


class PatchChecks(unittest.TestCase):
    def setUp(self):
        self.lock = patchlib.lock_data()
        self.original = (HERE / "tests/fixtures/nadit.before.py.txt").read_bytes()

    def test_exact_reviewed_source_produces_approved_upstream_sha(self):
        actual = patchlib.patch_bytes(self.original, self.lock)
        self.assertEqual(patchlib.sha256(actual), self.lock["after_sha256"])

    def test_different_original_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unpatched source"):
            patchlib.patch_bytes(self.original + b"\n", self.lock)

    def test_crlf_checkout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unpatched source"):
            patchlib.patch_bytes(self.original.replace(b"\n", b"\r\n"), self.lock)

    def test_reapplying_is_rejected_instead_of_fabricating_parent(self):
        with self.assertRaisesRegex(ValueError, "unpatched source"):
            patchlib.patch_bytes(patchlib.patch_bytes(self.original, self.lock), self.lock)

    def test_incorrect_target_hash_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exact approved"):
            patchlib.patch_bytes(self.original, dict(self.lock, after_sha256="0" * 64))

    def test_only_rmsnorm_context_changes(self):
        result = patchlib.patch_bytes(self.original, self.lock)
        reversed_patch = result.replace(patchlib.NEW_DOC, patchlib.OLD_DOC).replace(patchlib.NEW_FORWARD, patchlib.OLD_FORWARD)
        self.assertEqual(reversed_patch, self.original)

    def test_record_updates_only_one_semantic_entry(self):
        old = self.original
        new = patchlib.patch_bytes(old, self.lock)
        rows = [["other.py", "sha256=other", "19"], [patchlib.MODULE_RELATIVE_PATH, patchlib.wheel_record_digest(old), str(len(old))], ["vllm_omni.dist-info/RECORD", "", ""]]
        stream = io.StringIO(newline="")
        csv.writer(stream).writerows(rows)
        actual = list(csv.reader(io.StringIO(patchlib.update_record(stream.getvalue().encode(), old, new).decode())))
        self.assertEqual(actual[0], rows[0])
        self.assertEqual(actual[2], rows[2])
        self.assertEqual(actual[1][1:], [patchlib.wheel_record_digest(new), str(len(new))])

    def test_record_missing_entry_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            patchlib.update_record(b"other.py,,\n", self.original, b"new")

    def test_record_duplicate_entry_rejected(self):
        record = (patchlib.MODULE_RELATIVE_PATH + ",,\n").encode() * 2
        with self.assertRaisesRegex(ValueError, "exactly one"):
            patchlib.update_record(record, self.original, b"new")

    def test_record_stale_digest_rejected(self):
        record = (patchlib.MODULE_RELATIVE_PATH + ",sha256=invalid,7\n").encode()
        with self.assertRaisesRegex(ValueError, "Original RECORD"):
            patchlib.update_record(record, self.original, b"new")

    def test_source_drift_outside_patch_is_rejected(self):
        wrong = dict(self.lock["source_sha256"], **{"vae.py": "0" * 64})
        with self.assertRaisesRegex(ValueError, "vae.py"):
            patchlib.check_sources(wrong, self.lock, patched=False)

    def test_patched_sources_must_match_all_five_files(self):
        good = dict(self.lock["source_sha256"], **{"nadit.py": self.lock["after_sha256"]})
        patchlib.check_sources(good, self.lock, patched=True)
        with self.assertRaises(ValueError):
            patchlib.check_sources(good, self.lock, patched=False)

    def test_upstream_patch_contains_only_expected_two_files(self):
        content = (HERE / "upstream/0708f05e2e27f16fca8b48e7c5a085277d1cafda.patch").read_text()
        headers = [line for line in content.splitlines() if line.startswith("diff --git ")]
        self.assertEqual(len(headers), 2)
        self.assertIn("b/" + patchlib.MODULE_RELATIVE_PATH, headers[1])
        self.assertIn("b/tests/diffusion/models/seedvr2/test_seedvr2_norm.py", headers[0])

    def test_source_and_record_transaction_in_fixture_installation(self):
        with test_temp() as temporary:
            root = Path(temporary)
            module = root / patchlib.MODULE_RELATIVE_PATH
            module.parent.mkdir(parents=True)
            module.write_bytes(self.original)
            for name in ("vae.py", "pipeline_seedvr2.py", "config.py", "long_video.py"):
                (module.parent / name).write_text("# fixture " + name, encoding="utf-8")
            fixture_lock = dict(self.lock, source_sha256=patchlib.source_hashes(module))
            record = root / "vllm_omni-0.31.0rc1.dist-info/RECORD"
            record.parent.mkdir()
            record.write_text(f"{patchlib.MODULE_RELATIVE_PATH},{patchlib.wheel_record_digest(self.original)},{len(self.original)}\n", encoding="utf-8")
            fake_dist = type("Distribution", (), {"read_text": lambda _, name: '{"url":"original-wheel"}'})()
            compiled = module.parent / "__pycache__/nadit.cpython-312.pyc"
            compiled.parent.mkdir()
            compiled.write_bytes(b"stale-bytecode")
            report = root / "evidence/patch-application.json"
            with patch.object(application, "lock_data", return_value=fixture_lock), \
                 patch.object(application, "locate_installed", return_value=(fake_dist, module, record)), \
                 patch.object(application.metadata, "version", side_effect=lambda name: fixture_lock["required_versions"][name]), \
                 patch.object(application, "inventory", return_value=[{"name": "vllm-omni", "version": "0.31.0rc1"}]):
                result = application.apply_installed(report)
            self.assertEqual(result["status"], "PASS")
            self.assertFalse(compiled.exists())
            self.assertEqual(patchlib.sha256(module.read_bytes()), self.lock["after_sha256"])
            self.assertTrue((report.parent / "omni-RECORD.before.csv").is_file())
            self.assertEqual(json.loads(report.read_text())["record_after_sha256"], patchlib.sha256(record.read_bytes()))

    def test_upstream_results_require_six_passes(self):
        with test_temp() as temporary:
            path = Path(temporary) / "results.xml"
            case = '<testcase name="test_rms_norm_large_finite_inputs[x]"/>'
            path.write_text("<testsuites><testsuite>" + case * 6 + "</testsuite></testsuites>")
            self.assertEqual(runtime.check_pytest_xml(path), 6)
            path.write_text("<testsuites><testsuite>" + case * 5 + "</testsuite></testsuites>")
            with self.assertRaises(ValueError):
                runtime.check_pytest_xml(path)

    def test_upstream_skipped_test_is_not_success(self):
        with test_temp() as temporary:
            path = Path(temporary) / "results.xml"
            case = '<testcase name="test_rms_norm_large_finite_inputs[x]"/>'
            path.write_text("<testsuites><testsuite>" + case * 5 + '<testcase name="test_rms_norm_large_finite_inputs[x]"><skipped/></testcase></testsuite></testsuites>')
            with self.assertRaises(ValueError):
                runtime.check_pytest_xml(path)


if __name__ == "__main__":
    unittest.main()
