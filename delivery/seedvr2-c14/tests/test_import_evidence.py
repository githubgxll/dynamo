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
    spec = importlib.util.spec_from_file_location("c14_import_tests", HERE / "import_evidence.py")
    importer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(importer)
finally:
    sys.path.pop(0)


class ImportEvidenceTests(unittest.TestCase):
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
