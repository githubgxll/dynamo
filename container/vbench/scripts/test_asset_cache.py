"""Cache behavior under interrupted downloads and untrusted/corrupt receipts."""
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("asset_cache", Path(__file__).with_name("asset_cache.py"))
cache = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache)
URL = "https://official.example/model.bin"
NAMESPACE = "a" * 64
BODY = b"official model bytes"
SHA = hashlib.sha256(BODY).hexdigest()


class AssetCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / "cache"
        self.destination = self.root / "download" / "model.bin"
        self.calls = []
        self.events = []

    def download(self, url, destination, **options):
        self.calls.append((url, options))
        destination.write_bytes(BODY)
        # The helper must check bytes instead of trusting this optional metadata.
        return {"sha256": "f" * 64, "bytes": 1}

    def acquire(self, **options):
        arguments = dict(cache_dir=self.cache, namespace=NAMESPACE,
                         download_fn=self.download, validate_fn=lambda path: None,
                         event=self.events.append)
        arguments.update(options)
        return cache.acquire_cached(arguments.pop("url", URL), arguments.pop("destination", self.destination), **arguments)

    def files(self):
        return next(self.cache.glob("*.blob")), next(self.cache.glob("*.receipt.json"))

    def test_hit_uses_zero_network_and_independent_copy(self):
        expected = {"sha256": SHA, "bytes": len(BODY)}
        self.assertEqual(self.acquire(), expected)
        self.assertEqual(self.acquire(expected_sha256=SHA, expected_bytes=len(BODY)), expected)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual([item["event"] for item in self.events], ["miss", "stored", "hit"])
        blob, _ = self.files()
        self.destination.write_bytes(b"consumer changed its own copy")
        self.assertEqual(blob.read_bytes(), BODY)

    def test_one_asset_failure_preserves_earlier_asset(self):
        self.acquire()
        class DownloadFailure(RuntimeError):
            pass
        def broken(url, destination, **kwargs):
            destination.write_bytes(b"incomplete")
            raise DownloadFailure("network failed")
        with self.assertRaises(DownloadFailure):
            self.acquire(url="https://official.example/next.bin", destination=self.root / "next.bin", download_fn=broken)
        self.assertFalse((self.root / "next.bin").exists())
        self.assertEqual(len(list(self.cache.glob("*.blob"))), 1)
        self.acquire()
        self.assertEqual(len(self.calls), 1)

    def test_corrupt_blob_or_receipt_is_redownloaded(self):
        for mutation in ("blob", "json", "receipt_hash", "receipt_size", "receipt_identity", "receipt_schema", "missing_blob", "missing_receipt"):
            with self.subTest(mutation=mutation):
                self.acquire()
                blob, receipt = self.files()
                if mutation == "blob":
                    blob.write_bytes(b"corrupt cached content")
                elif mutation == "json":
                    receipt.write_text('{"secret":"do not log this"')
                elif mutation == "missing_blob":
                    blob.unlink()
                elif mutation == "missing_receipt":
                    receipt.unlink()
                else:
                    record = json.loads(receipt.read_text())
                    if mutation == "receipt_hash":
                        record["sha256"] = "f" * 64
                    elif mutation == "receipt_size":
                        record["bytes"] += 1
                    elif mutation == "receipt_identity":
                        record["url"] = "https://unreviewed.example/evil"
                    elif mutation == "receipt_schema":
                        record["schema_version"] = True
                    receipt.write_text(json.dumps(record))
                before = len(self.calls)
                self.acquire(expected_sha256=SHA)
                self.assertEqual(len(self.calls), before + 1)
                self.assertEqual(self.destination.read_bytes(), BODY)
        self.assertNotIn("secret", json.dumps(self.events))

    def test_current_publish_hash_cannot_be_replaced_by_receipt(self):
        self.acquire()
        with self.assertRaisesRegex(cache.CacheError, "SHA256 mismatch"):
            self.acquire(expected_sha256="0" * 64)
        self.assertEqual(len(self.calls), 2)
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.cache.iterdir()), [])
        self.assertEqual(self.calls[-1][1]["expected_sha256"], "0" * 64)

    def test_current_size_and_limit_are_enforced_on_hit_and_download(self):
        for options in ({"expected_bytes": len(BODY) + 1}, {"max_bytes": len(BODY) - 1}):
            with self.subTest(options=options):
                self.acquire()
                before = len(self.calls)
                with self.assertRaises(cache.CacheError):
                    self.acquire(**options)
                self.assertEqual(len(self.calls), before + 1)
                self.assertEqual(list(self.cache.iterdir()), [])

    def test_catalog_or_url_change_has_separate_cache_entry(self):
        self.acquire()
        self.acquire(namespace="b" * 64)
        self.acquire(url="https://official.example/another.bin")
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(len(list(self.cache.glob("*.blob"))), 3)
        self.acquire()
        self.assertEqual(len(self.calls), 3)

    def test_fresh_validation_error_is_not_cached_and_is_preserved(self):
        def reject(path):
            raise ValueError("invalid model")
        with self.assertRaisesRegex(ValueError, "invalid model"):
            self.acquire(validate_fn=reject)
        self.assertEqual(list(self.cache.iterdir()), [])
        self.assertFalse(self.destination.exists())

    def test_cached_validation_error_redownloads_once_but_does_not_hide_second_error(self):
        self.acquire()
        validations = []
        def reject(path):
            validations.append(path)
            raise ValueError("bad content")
        with self.assertRaisesRegex(ValueError, "bad content"):
            self.acquire(validate_fn=reject)
        self.assertEqual(len(validations), 2)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_validator_cannot_modify_downloaded_bytes(self):
        def mutate(path):
            path.write_bytes(b"rewritten")
        with self.assertRaisesRegex(cache.CacheError, "changed during validation"):
            self.acquire(validate_fn=mutate)
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_no_cache_still_checks_bytes_and_validation(self):
        validations = []
        self.acquire(cache_dir=None, expected_sha256=SHA, validate_fn=validations.append)
        self.acquire(cache_dir=None, expected_sha256=SHA)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(validations, [self.destination])
        self.assertFalse(self.cache.exists())

    def test_only_safe_parameters_reach_download(self):
        for options in ({"namespace": "not-a-sha"}, {"url": "https://example.com/model?secret=x"},
                        {"url": "https://user:password@example.com/model"}, {"url": "http://example.com/model"},
                        {"expected_bytes": True}, {"max_bytes": 0}, {"expected_sha256": "wrong"}):
            with self.subTest(options=options), self.assertRaises(cache.CacheError):
                self.acquire(**options)
        self.assertEqual(self.calls, [])

    def test_duplicate_receipt_fields_are_not_accepted(self):
        self.acquire()
        _, receipt = self.files()
        record = receipt.read_text().rstrip()
        receipt.write_text(record[:-1] + ',"bytes":' + str(len(BODY)) + "}")
        self.acquire()
        self.assertEqual(len(self.calls), 2)

    def test_symlink_cache_entry_is_removed_without_touching_target(self):
        self.acquire()
        blob, _ = self.files()
        outside = self.root / "outside.bin"
        outside.write_bytes(b"preserve")
        blob.unlink()
        try:
            blob.symlink_to(outside)
        except OSError:
            self.skipTest("OS does not permit symlink creation")
        self.acquire()
        self.assertFalse(blob.is_symlink())
        self.assertEqual(outside.read_bytes(), b"preserve")
        self.assertEqual(len(self.calls), 2)

    def test_symlink_root_and_destination_are_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        linked = self.root / "linked"
        try:
            linked.symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest("OS does not permit symlink creation")
        for options in ({"cache_dir": linked}, {"destination": linked / "model.bin"}):
            with self.subTest(options=options), self.assertRaises(cache.CacheError):
                self.acquire(**options)
        self.assertEqual(list(outside.iterdir()), [])

    def test_hashing_uses_streaming_reads(self):
        with patch.object(Path, "read_bytes", side_effect=AssertionError("no whole-file reads")):
            self.acquire()
            self.acquire(expected_sha256=SHA)
        self.assertEqual(len(self.calls), 1)

    def test_link_and_windows_reparse_guards_without_symlink_privilege(self):
        original_lstat = Path.lstat
        for kind in ("symlink", "junction"):
            with self.subTest(kind=kind):
                def fake_lstat(path):
                    if path == self.cache:
                        return SimpleNamespace(st_mode=stat.S_IFLNK if kind == "symlink" else stat.S_IFDIR,
                                               st_file_attributes=1024 if kind == "junction" else 0)
                    return original_lstat(path)
                with patch.object(Path, "lstat", fake_lstat), self.assertRaisesRegex(cache.CacheError, "link or reparse"):
                    self.acquire()
        self.assertEqual(self.calls, [])

    def test_interrupted_receipt_commit_does_not_trust_orphan_blob(self):
        with patch.object(cache, "_write_receipt", side_effect=OSError("simulated interrupted commit")):
            with self.assertRaises(OSError):
                self.acquire()
        self.assertEqual(len(list(self.cache.glob("*.blob"))), 1)
        self.assertEqual(list(self.cache.glob("*.receipt.json")), [])
        self.acquire(expected_sha256=SHA)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.destination.read_bytes(), BODY)


if __name__ == "__main__":
    unittest.main()
