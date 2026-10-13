"""Small synthetic payloads exercise acquisition integrity, not real GPU loading."""
import contextlib
import copy
import hashlib
import http.client
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fetch_assets import HTTPSRedirectHandler, acquire, download
from verify_assets import (AssetError, canonical_header_digest, check_header,
                           load_lock, safe_path, verify)


def fixture():
    header = {"__metadata__": {"format": "test-only"},
              "weight": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 8]}}
    raw = json.dumps(header).encode()
    data = struct.pack("<Q", len(raw)) + raw + b"12345678"
    revision = "a" * 40
    asset = {"id": "synthetic", "repo": "test/model", "revision": revision,
             "filename": "adapter.safetensors", "relative_path": "test/adapter.safetensors",
             "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
             "url": f"https://huggingface.co/test/model/resolve/{revision}/adapter.safetensors",
             "header_contract": {"metadata": header["__metadata__"], "tensor_count": 1,
                                 "dtype_counts": {"BF16": 1}, "header_json_sha256": canonical_header_digest(header)}}
    return data, asset


class Response:
    def __init__(self, data, status=200, headers=None, failure_after=None):
        self.stream = io.BytesIO(data)
        self.status = status
        self.headers = headers or {}
        self.failure_after = failure_after

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stream.close()

    def read(self, count):
        if self.failure_after is not None:
            if self.stream.tell() >= self.failure_after:
                raise ConnectionResetError("untrusted secret signed URL must not be printed")
            count = min(count, self.failure_after - self.stream.tell())
        return self.stream.read(count)


class Opener:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append(request)
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.data, self.asset = fixture()
        self.lock = {"schema": "h3-distilled-assets/1", "assets": [self.asset], "documents": []}
        self.lock_path = self.root / "assets.lock.json"

    def tearDown(self):
        self.tmp.cleanup()

    def partial(self):
        return self.cache / (self.asset["sha256"] + ".partial")

    def full(self):
        return self.cache / (self.asset["sha256"] + ".safetensors")

    def run_download(self, opener, attempts=4):
        with contextlib.redirect_stdout(io.StringIO()):
            return download(self.asset, self.cache, opener, attempts=attempts, sleep=lambda _: None)

    def write_lock(self):
        self.lock_path.write_text(json.dumps(self.lock), encoding="utf-8")

    def test_download_promotes_only_checked_payload(self):
        result = self.run_download(Opener(Response(self.data)))
        self.assertEqual(result.read_bytes(), self.data)
        self.assertFalse(self.partial().exists())

    def test_cached_payload_is_rehashed_without_network(self):
        self.full().write_bytes(self.data)
        opener = Opener()
        self.assertEqual(self.run_download(opener), self.full())
        self.assertEqual(opener.requests, [])

    def test_corrupt_cache_is_replaced(self):
        self.full().write_bytes(b"x" * len(self.data))
        result = self.run_download(Opener(Response(self.data)))
        self.assertEqual(result.read_bytes(), self.data)

    def test_interrupted_response_resumes_exact_range(self):
        offset = 21
        first = Response(self.data, failure_after=offset)
        second = Response(self.data[offset:], 206, {"Content-Range": f"bytes {offset}-{len(self.data)-1}/{len(self.data)}"})
        opener = Opener(first, second)
        result = self.run_download(opener)
        self.assertEqual(result.read_bytes(), self.data)
        self.assertEqual(opener.requests[1].get_header("Range"), f"bytes={offset}-")

    def test_truncated_eof_resumes(self):
        offset = 23
        opener = Opener(Response(self.data[:offset]), Response(self.data[offset:], 206,
                        {"Content-Range": f"bytes {offset}-{len(self.data)-1}/{len(self.data)}"}))
        self.assertEqual(self.run_download(opener).read_bytes(), self.data)

    def test_range_mismatch_cannot_append_or_promote(self):
        self.partial().write_bytes(self.data[:10])
        opener = Opener(Response(self.data[9:], 206, {"Content-Range": f"bytes 9-{len(self.data)-1}/{len(self.data)}"}))
        with self.assertRaises(AssetError):
            self.run_download(opener, attempts=1)
        self.assertFalse(self.full().exists())
        self.assertFalse(self.partial().exists())

    def test_server_ignores_range_overwrites_partial(self):
        self.partial().write_bytes(self.data[:10])
        self.assertEqual(self.run_download(Opener(Response(self.data))).read_bytes(), self.data)

    def test_wrong_hash_never_becomes_complete(self):
        wrong = self.data[:-1] + b"x"
        with self.assertRaises(AssetError):
            self.run_download(Opener(Response(wrong)), attempts=1)
        self.assertFalse(self.full().exists())
        self.assertFalse(self.partial().exists())

    def test_complete_partial_is_verified_without_network(self):
        self.partial().write_bytes(self.data)
        self.assertEqual(self.run_download(Opener()).read_bytes(), self.data)

    def test_partial_is_retained_after_exhausted_transient_retry(self):
        with self.assertRaises(AssetError):
            self.run_download(Opener(Response(self.data, failure_after=20)), attempts=1)
        self.assertEqual(self.partial().read_bytes(), self.data[:20])

    def test_wrong_content_length_fails(self):
        with self.assertRaises(AssetError):
            self.run_download(Opener(Response(self.data, headers={"Content-Length": "1234567"})), attempts=1)
        self.assertFalse(self.full().exists())

    def test_oversized_payload_is_rejected(self):
        with self.assertRaises(AssetError):
            self.run_download(Opener(Response(self.data + b"extra")), attempts=1)
        self.assertFalse(self.full().exists())

    def test_http_error_does_not_expose_signed_url(self):
        opener = Opener(urllib.error.HTTPError("https://secret/?token=SECRET", 403, "SECRET", {}, None))
        with self.assertRaises(AssetError) as caught:
            self.run_download(opener)
        self.assertNotIn("SECRET", str(caught.exception))
        self.assertEqual(len(opener.requests), 1)

    def test_header_metadata_contract_is_enforced(self):
        path = self.root / "adapter"
        path.write_bytes(self.data)
        self.asset["header_contract"]["metadata"] = {"format": "wrong"}
        with self.assertRaises(AssetError):
            check_header(path, self.asset)

    def test_verify_document_without_vendored_source(self):
        out = self.root / "output"
        dest = safe_path(out, self.asset["relative_path"])
        dest.parent.mkdir(parents=True)
        dest.write_bytes(self.data)
        doc = b"notice text"
        (out / "NOTICE").write_bytes(doc)
        self.lock["documents"] = [{"relative_path": "NOTICE", "sha256": hashlib.sha256(doc).hexdigest(),
                                   "local_source": "does-not-exist"}]
        self.assertEqual(verify(out, self.lock)["status"], "PASS_ASSETS_ONLY")
        (out / "NOTICE").write_bytes(b"tampered")
        with self.assertRaises(AssetError):
            verify(out, self.lock)

    def test_path_traversal_rejected_before_output_or_network(self):
        self.asset["relative_path"] = "../outside.safetensors"
        self.write_lock()
        with self.assertRaises(AssetError):
            acquire(self.lock_path, self.cache, self.root / "out")
        self.assertFalse((self.root / "out").exists())
        self.assertFalse((self.root / "outside.safetensors").exists())

    def test_absolute_and_windows_paths_rejected(self):
        for value in ("/tmp/escape", "C:/escape", "..\\escape", "valid/../escape", "valid//file"):
            with self.subTest(value=value), self.assertRaises(AssetError):
                safe_path(self.root, value)

    def test_url_cannot_change_source_or_remove_revision(self):
        self.asset["url"] = "https://huggingface.co/test/model/resolve/main/adapter.safetensors"
        self.write_lock()
        with self.assertRaises(AssetError):
            load_lock(self.lock_path)

    def test_duplicate_output_rejected(self):
        other = copy.deepcopy(self.asset)
        other["id"] = "second"
        self.lock["assets"].append(other)
        self.write_lock()
        with self.assertRaises(AssetError):
            load_lock(self.lock_path)

    def test_https_redirect_cannot_downgrade(self):
        with self.assertRaises(AssetError):
            HTTPSRedirectHandler().redirect_request(None, None, 302, "", {}, "http://cdn.example/file")

    def test_shipped_lock_and_vendored_document_hashes(self):
        from verify_assets import check_file
        folder = Path(__file__).resolve().parents[1]
        lock = load_lock(folder / "assets.lock.json")
        self.assertEqual(len(lock["assets"]), 4)
        self.assertEqual(sum(a["size_bytes"] for a in lock["assets"]), 5636659576)
        for doc in lock["documents"]:
            check_file(safe_path(folder, doc["local_source"]), doc)


if __name__ == "__main__":
    unittest.main()
