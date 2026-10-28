"""Reject path ambiguities before an Actions ZIP can escape its fresh evidence root."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
try:
    spec = importlib.util.spec_from_file_location("hsdp_import_tests", HERE / "import_evidence.py")
    importer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(importer)
finally:
    sys.path.pop(0)

fixture_spec = importlib.util.spec_from_file_location("hsdp_ci_report_fixtures",
    HERE.parents[1] / ".github/scripts/test_h3_hsdp_offload_ci.py")
fixtures = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(fixtures)
ci = fixtures.ci


class ImportEvidenceTests(unittest.TestCase):
    def test_current_repair_reports_use_publication_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixtures.reports(root / "build")
            fixtures.reports(root / "final")
            record = {"dit_entry_status": "PASS", "patched_source_sha256": ci.PATCHED_SOURCE_SHA256}
            importer.validate_repair_reports(ci, root, record, fixtures.COMMIT)

    def test_old_runtime_reports_cannot_be_imported_as_dit_repair(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixtures.reports(root / "build")
            fixtures.reports(root / "final")
            runtime = ci.read(root / "final/runtime-validation.json")
            del runtime["dit_entry_status"]
            ci.dump(root / "final/runtime-validation.json", runtime)
            record = {"dit_entry_status": "PASS", "patched_source_sha256": ci.PATCHED_SOURCE_SHA256}
            with self.assertRaisesRegex(ValueError, "dit_entry_status"):
                importer.validate_repair_reports(ci, root, record, fixtures.COMMIT)

    def test_missing_or_wrong_publication_repair_identity_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixtures.reports(root / "build")
            fixtures.reports(root / "final")
            for record in ({}, {"dit_entry_status": "PASS"},
                           {"dit_entry_status": "PASS", "patched_source_sha256": {ci.PIPELINE: "f" * 64}}):
                with self.subTest(record=record), self.assertRaisesRegex(ValueError, "Publication does not identify"):
                    importer.validate_repair_reports(ci, root, record, fixtures.COMMIT)

    def test_bad_pipeline_cannot_be_imported_with_valid_receipt_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixtures.reports(root / "build")
            fixtures.reports(root / "final")
            for phase in ("build", "final"):
                sources = ci.read(root / phase / "source-verification.json")["source_sha256"]
                sources[ci.PIPELINE] = "f" * 64
                fixtures.rewrite_manifest(root / phase, sources)
            record = {"dit_entry_status": "PASS", "patched_source_sha256": ci.PATCHED_SOURCE_SHA256}
            with self.assertRaisesRegex(ValueError, "Required installed repair source"):
                importer.validate_repair_reports(ci, root, record, fixtures.COMMIT)

    def test_normal_relative_evidence_path_is_preserved(self):
        self.assertEqual(str(importer.safe_name("final/rmsnorm/runtime-validation.json")),
                         "final/rmsnorm/runtime-validation.json")

    def test_unsafe_or_ambiguous_windows_paths_are_rejected(self):
        for name in ("../escape", "/escape", "C:/escape", "a\\escape", "a//b", "a/./b",
                     "a/b. ", "a/b.", "a/CON.txt", "LPT1", "x/aux", "x/control\x00"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                importer.safe_name(name)

    def test_duplicate_and_symlink_members_never_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for kind in ("duplicate", "symlink", "escape"):
                archive = root / (kind + ".zip")
                with zipfile.ZipFile(archive, "w") as bundle:
                    if kind == "duplicate":
                        bundle.writestr("A.txt", "one")
                        bundle.writestr("a.txt", "two")
                    elif kind == "symlink":
                        member = zipfile.ZipInfo("link")
                        member.create_system = 3
                        member.external_attr = 0o120777 << 16
                        bundle.writestr(member, "../../elsewhere")
                    else:
                        bundle.writestr("../outside", "unsafe")
                with patch.object(importer.ops, "check_seal"), \
                     patch.object(importer.ops, "EVIDENCE", root / (kind + "-evidence")):
                    with self.subTest(kind=kind), self.assertRaises(ValueError):
                        importer.import_archive(archive)
            self.assertFalse((root / "outside").exists())


if __name__ == "__main__":
    unittest.main()
