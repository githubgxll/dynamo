"""Fault-injection tests for bounded downloads; no real requests or sleeping."""
import hashlib
import http.client
import importlib.util
import json
from pathlib import Path
import socket
import ssl
import tempfile
import unittest
from unittest.mock import Mock, patch
import urllib.error

SPEC = importlib.util.spec_from_file_location("asset_download_under_test", Path(__file__).with_name("prepare_assets.py"))
assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assets)


class Response:
    def __init__(self, chunks, length=None, status=200, advance=None):
        self.chunks = list(chunks)
        self.headers = {} if length is None else {"Content-Length": str(length)}
        self.status = status
        self.advance = advance
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def getcode(self):
        return self.status

    def read(self, size):
        if self.advance:
            self.advance()
        value = self.chunks.pop(0) if self.chunks else b""
        if isinstance(value, BaseException):
            raise value
        return value

    read1 = read


class DownloadTests(unittest.TestCase):
    URL = "https://official.example/model"
    PAYLOAD = b"complete immutable checkpoint bytes"
    SECRET = "https://signed.example/file?token=must-not-appear"

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.destination = Path(temporary.name) / "model.pth"
        self.partial = self.destination.with_suffix(".pth.partial")
        self.events = []
        self.now = 0
        self.sleeps = []
        self.addCleanup(patch.stopall)
        patch.object(assets.time, "monotonic", side_effect=lambda: self.now).start()
        patch.object(assets.time, "sleep", side_effect=self.sleep).start()

    def sleep(self, delay):
        self.sleeps.append(delay)
        self.now += delay

    def response(self, **kwargs):
        return Response([self.PAYLOAD, b""], len(self.PAYLOAD), **kwargs)

    def download(self, effects, **kwargs):
        self.opener = Mock()
        self.opener.open.side_effect = effects
        with patch.object(assets.urllib.request, "build_opener", return_value=self.opener):
            return assets.download(self.URL, self.destination,
                expected_sha256=kwargs.pop("expected_sha256", hashlib.sha256(self.PAYLOAD).hexdigest()),
                event=self.events.append, **kwargs)

    def assert_clean(self):
        self.assertFalse(self.partial.exists())
        self.assertFalse(self.destination.exists())

    def test_tls_eof_recovers_same_url_no_range_and_safe_events(self):
        failure = urllib.error.URLError(ssl.SSLEOFError(8, self.SECRET))
        result = self.download([failure, self.response()])
        self.assertEqual(result, {"sha256": hashlib.sha256(self.PAYLOAD).hexdigest(), "bytes": len(self.PAYLOAD)})
        self.assertEqual(self.destination.read_bytes(), self.PAYLOAD)
        self.assertEqual(self.sleeps, [2])
        self.assertEqual([item["event"] for item in self.events], ["start", "retry", "start", "success"])
        self.assertNotIn("must-not-appear", json.dumps(self.events))
        self.assertNotIn("signed.example", json.dumps(self.events))
        for call in self.opener.open.call_args_list:
            self.assertEqual(call.args[0].full_url, self.URL)
            self.assertFalse(call.args[0].has_header("Range"))
        self.assertFalse(self.partial.exists())

    def test_partial_disconnect_restarts_from_zero(self):
        partial = Response([b"wrong prefix", ConnectionResetError(104, self.SECRET)], len(self.PAYLOAD))
        self.download([partial, self.response()])
        self.assertTrue(partial.closed)
        self.assertEqual(self.destination.read_bytes(), self.PAYLOAD)
        self.assertEqual(self.sleeps, [2])

    def test_truncated_length_and_incomplete_read_retry(self):
        truncated = Response([b"short", b""], len(self.PAYLOAD))
        incomplete = Response([http.client.IncompleteRead(b"partial", len(self.PAYLOAD))], len(self.PAYLOAD))
        self.download([truncated, incomplete, self.response()])
        self.assertEqual(self.sleeps, [2, 5])
        self.assertEqual(self.opener.open.call_count, 3)
        self.assertEqual(self.destination.read_bytes(), self.PAYLOAD)

    def test_transient_http_and_temporary_dns_retry(self):
        failures = [urllib.error.HTTPError(self.SECRET, code, self.SECRET, {}, None)
                    for code in (408, 429, 500, 502, 503, 504)]
        failures.append(urllib.error.URLError(socket.gaierror(socket.EAI_AGAIN, self.SECRET)))
        for failure in failures:
            with self.subTest(kind=type(failure).__name__, code=getattr(failure, "code", None)):
                self.download([failure, self.response()])
                self.assertEqual(self.opener.open.call_count, 2)
                self.destination.unlink()

    def test_exhaustion_stops_at_three_and_cleans_partial(self):
        failures = [urllib.error.URLError(TimeoutError(self.SECRET))] * 4
        self.partial.write_bytes(b"stale")
        with self.assertRaises(assets.AssetError) as error:
            self.download(failures)
        self.assertEqual(self.opener.open.call_count, 3)
        self.assertEqual(self.sleeps, [2, 5])
        self.assertEqual(self.events[-1]["event"], "failure")
        self.assertNotIn("must-not-appear", str(error.exception))
        self.assert_clean()

    def test_cert_permissions_not_found_dns_permanent_fail_without_retry(self):
        failures = [urllib.error.HTTPError(self.SECRET, code, self.SECRET, {}, None) for code in (401, 403, 404)]
        failures.extend([
            urllib.error.URLError(ssl.SSLCertVerificationError(1, self.SECRET)),
            urllib.error.URLError(socket.gaierror(socket.EAI_NONAME, self.SECRET)),
            urllib.error.URLError(ssl.SSLError(1, self.SECRET)),
            urllib.error.URLError(self.SECRET),
        ])
        for failure in failures:
            with self.subTest(kind=type(failure).__name__, code=getattr(failure, "code", None)):
                with self.assertRaises(assets.AssetError) as error:
                    self.download([failure, self.response()])
                self.assertNotIn("must-not-appear", str(error.exception))
                self.assertEqual(self.opener.open.call_count, 1)
                self.assertEqual(self.sleeps, [])
                self.assert_clean()

    def test_hash_size_empty_and_non200_are_not_accepted_or_retried(self):
        cases = [
            (self.response(), {"expected_sha256": "0" * 64}),
            (self.response(), {"max_bytes": 2}),
            (Response([b"oversize", b""], 1), {}),
            (Response([b"", b""], 0), {}),
            (Response([self.PAYLOAD], status=206), {}),
            (Response([self.PAYLOAD], length="invalid"), {}),
        ]
        for response, kwargs in cases:
            with self.subTest(status=response.status, headers=response.headers, kwargs=kwargs):
                with self.assertRaises(assets.AssetError):
                    self.download([response, self.response()], **kwargs)
                self.assertEqual(self.opener.open.call_count, 1)
                self.assertEqual(self.sleeps, [])
                self.assert_clean()

    def test_returned_retryable_non200_response_retries(self):
        self.download([Response([], status=503), self.response()])
        self.assertEqual(self.opener.open.call_count, 2)
        self.assertEqual(self.destination.read_bytes(), self.PAYLOAD)

    def test_shared_deadline_caps_next_attempt_timeout_and_stops(self):
        def failure(request, timeout):
            self.now += 4
            raise TimeoutError(self.SECRET)

        self.opener = Mock()
        self.opener.open.side_effect = failure
        with patch.object(assets.urllib.request, "build_opener", return_value=self.opener):
            with self.assertRaisesRegex(assets.AssetError, "elapsed-time"):
                assets.download(self.URL, self.destination, timeout=90, deadline=10, event=self.events.append)
        self.assertEqual(self.opener.open.call_count, 2)
        self.assertEqual([call.kwargs["timeout"] for call in self.opener.open.call_args_list], [10, 4])
        self.assertEqual(self.sleeps, [2])
        self.assert_clean()

    def test_deadline_during_body_cleans_and_preserves_existing_destination(self):
        self.destination.write_bytes(b"previous verified checkpoint")
        def advance():
            self.now += 3
        with self.assertRaisesRegex(assets.AssetError, "elapsed-time"):
            self.download([self.response(advance=advance)], deadline=2)
        self.assertEqual(self.opener.open.call_count, 1)
        self.assertEqual(self.destination.read_bytes(), b"previous verified checkpoint")
        self.assertFalse(self.partial.exists())


if __name__ == "__main__":
    unittest.main()
