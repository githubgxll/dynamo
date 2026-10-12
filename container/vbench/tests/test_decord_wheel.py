"""Decord metadata repair keeps native bytes and strict dependency checking."""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from packaging import tags

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import decord_wheel as decord


class DecordPatchTests(unittest.TestCase):
    def make_wheel(self, root, tag=decord.ORIGINAL_TAG):
        files = {
            "decord/__init__.py": b"from ._ffi import base\n",
            "decord/libdecord.so": b"\x7fELF\x02\x01native-library-fixture",
            "decord.libs/libavcodec-fixture.so": b"\x7fELF\x02\x01codec-fixture",
            decord.OLD_INFO + "/LICENSE": b"upstream license fixture\n",
            decord.OLD_INFO + "/top_level.txt": b"decord\n",
            decord.OLD_INFO + "/METADATA": b"Metadata-Version: 2.1\nName: decord\nVersion: 0.6.0\nRequires-Dist: numpy (>=1.14.0)\n\n",
            decord.OLD_INFO + "/WHEEL": f"Wheel-Version: 1.0\nGenerator: bdist_wheel (0.36.2)\nRoot-Is-Purelib: false\nTag: {tag}\n\n".encode(),
            decord.OLD_INFO + "/RECORD": f"{decord.OLD_INFO}/top_level.txt,sha256=stale,0\n{decord.OLD_INFO}/RECORD,,\n".encode(),
        }
        original = root / decord.ORIGINAL_FILENAME
        with zipfile.ZipFile(original, "w") as archive:
            for name, data in files.items():
                archive.writestr(name, data)
        return original, files

    def test_deterministic_repair_preserves_payload_licenses_and_valid_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, files = self.make_wheel(root)
            with patch.object(decord, "ORIGINAL_SHA256", decord.sha256(original)):
                result = decord.patch_decord(original, root / "out", root / "evidence")
                first = result.read_bytes()
                result = decord.patch_decord(original, root / "out", root / "evidence")
                self.assertEqual(first, result.read_bytes())
            with zipfile.ZipFile(result) as wheel:
                self.assertTrue(all(info.create_system == 3 for info in wheel.infolist()))
                self.assertTrue(all(info.compress_type == zipfile.ZIP_STORED for info in wheel.infolist()))
                for name, data in files.items():
                    if name.rsplit("/", 1)[-1] not in ("METADATA", "WHEEL", "RECORD"):
                        new_name = name.replace(decord.OLD_INFO + "/", decord.NEW_INFO + "/", 1)
                        self.assertEqual(wheel.read(new_name), data)
                self.assertNotIn("cp36", wheel.read(decord.NEW_INFO + "/WHEEL").decode())
                self.assertIn("Tag: " + decord.CORRECTED_TAG, wheel.read(decord.NEW_INFO + "/WHEEL").decode())
                self.assertEqual(wheel.read(decord.NEW_INFO + "/METADATA"), files[decord.OLD_INFO + "/METADATA"].replace(b"Version: 0.6.0\n", b"Version: 0.6.0+vbench1\n"))
                record_rows = list(csv.reader(io.StringIO(wheel.read(decord.NEW_INFO + "/RECORD").decode())))
                self.assertEqual({r[0] for r in record_rows}, set(wheel.namelist()))
                for name, digest, size in record_rows:
                    if name.endswith("/RECORD"):
                        self.assertEqual((digest, size), ("", ""))
                        continue
                    data = wheel.read(name)
                    self.assertEqual(digest, "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode())
                    self.assertEqual(int(size), len(data))
            audit = json.loads((root / "evidence/decord-patch.json").read_text())
            self.assertFalse(audit["code_changed"])
            self.assertFalse(audit["native_binaries_changed"])
            self.assertTrue(audit["licenses_preserved"])
            self.assertEqual(audit["original_record_audit"]["mismatched_entries"], [decord.OLD_INFO + "/top_level.txt"])
            self.assertEqual(audit["payload_sha256"]["decord/libdecord.so"], hashlib.sha256(files["decord/libdecord.so"]).hexdigest())
            self.assertIn(decord.OLD_INFO + "/LICENSE", audit["unchanged_file_sha256"])
            self.assertTrue((root / "evidence/decord-WHEEL.patch").is_file())

    def test_original_hash_and_unexpected_tag_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, _ = self.make_wheel(root, "cp39-cp39-linux_aarch64")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                decord.patch_decord(original, root / "out", root / "evidence")
            with patch.object(decord, "ORIGINAL_SHA256", decord.sha256(original)):
                with self.assertRaisesRegex(ValueError, "WHEEL tag"):
                    decord.patch_decord(original, root / "out", root / "evidence")
            self.assertFalse((root / "out").exists())

    def test_corrected_tag_matches_target_cpython310_but_not_windows_or_arm(self):
        target = set(tags.cpython_tags((3, 10), ["cp310"], ["manylinux2010_x86_64"]))
        target.update(tags.compatible_tags((3, 10), "cp310", ["manylinux2010_x86_64"]))
        self.assertFalse(tags.parse_tag(decord.ORIGINAL_TAG) & target)
        self.assertTrue(tags.parse_tag(decord.CORRECTED_TAG) & target)
        for platform in ("win_amd64", "manylinux2014_aarch64"):
            with self.subTest(platform=platform):
                incompatible = set(tags.compatible_tags((3, 10), "cp310", [platform]))
                self.assertFalse(tags.parse_tag(decord.CORRECTED_TAG) & incompatible)


class DecordAcquisitionTests(unittest.TestCase):
    DATA = b"exact-wheel-fixture"
    SOURCES = {"python_index": "https://pypi.tuna.tsinghua.edu.cn/simple"}

    def fake_download(self, command, check):
        self.assertTrue(check)
        self.assertIn("--require-hashes", command)
        self.assertIn("--no-deps", command)
        self.assertIn("--only-binary=:all:", command)
        self.assertEqual(command[command.index("--index-url") + 1], self.SOURCES["python_index"])
        self.assertEqual(command[command.index("--timeout") + 1], "15")
        self.assertEqual(command[command.index("--retries") + 1], "1")
        self.assertEqual(Path(command[command.index("-r") + 1]).read_text(), f"decord==0.6.0 --hash=sha256:{decord.ORIGINAL_SHA256}\n")
        (Path(command[command.index("--dest") + 1]) / decord.ORIGINAL_FILENAME).write_bytes(self.DATA)

    def test_reviewed_download_and_revalidated_cache(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(decord, "ORIGINAL_SHA256", hashlib.sha256(self.DATA).hexdigest()), patch.object(decord.subprocess, "run", side_effect=self.fake_download) as run:
            root = Path(directory)
            result = decord.acquire_original(root, self.SOURCES)
            self.assertEqual(result.read_bytes(), self.DATA)
            self.assertEqual(decord.acquire_original(root, self.SOURCES), result)
            self.assertEqual(run.call_count, 1)
            result.write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "cached.*SHA256"):
                decord.acquire_original(root, self.SOURCES)
            self.assertEqual(run.call_count, 1)

    def test_failed_download_does_not_create_cached_original(self):
        def fail(command, check):
            self.fake_download(command, check)
            raise subprocess.CalledProcessError(1, command)
        for side_effect in (self.fake_download, fail):
            with self.subTest(side_effect=side_effect), tempfile.TemporaryDirectory() as directory, patch.object(decord.subprocess, "run", side_effect=side_effect):
                root = Path(directory)
                with self.assertRaises((ValueError, subprocess.CalledProcessError)):
                    decord.acquire_original(root, self.SOURCES)
                self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
