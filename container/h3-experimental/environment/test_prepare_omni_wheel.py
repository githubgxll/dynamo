"""Exercise wheel acquisition failures without network or GPU installation."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import URLError

from prepare_omni_wheel import prepare_wheel


class WheelPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.data = b"Synthetic wheel bytes for checksum testing only"
        self.url = "https://files.pythonhosted.org/packages/fixture/vllm_omni-0.31.0rc1-py3-none-any.whl"
        self.policy = {"omni_wheel": self.url + "#sha256=" + hashlib.sha256(self.data).hexdigest()}

    def tearDown(self):
        self.temp.cleanup()

    def test_download_hash_receipt_and_reuse_without_network(self):
        opener = Mock(return_value=io.BytesIO(self.data))
        receipt = prepare_wheel(self.policy, self.directory, opener=opener)
        self.assertEqual(Path(receipt["wheel_path"]).read_bytes(), self.data)
        self.assertEqual(receipt["url"], self.url)
        self.assertEqual(receipt, json.loads((self.directory/"omni-wheel-download.json").read_text()))
        self.assertEqual(receipt, prepare_wheel(self.policy, self.directory, opener=opener))
        self.assertEqual(opener.call_count, 1)

    def test_wrong_download_hash_never_becomes_installable_or_gets_a_receipt(self):
        with self.assertRaisesRegex(ValueError, "SHA256 differs"):
            prepare_wheel(self.policy, self.directory, opener=lambda *a, **k: io.BytesIO(b"wrong"))
        self.assertFalse((self.directory/"omni-wheel-download.json").exists())
        self.assertEqual(list((self.directory/"omni-wheels").iterdir()), [])

    def test_existing_changed_wheel_is_rejected(self):
        receipt = prepare_wheel(self.policy, self.directory, opener=lambda *a, **k: io.BytesIO(self.data))
        Path(receipt["wheel_path"]).write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "Existing retained Omni wheel checksum differs"):
            prepare_wheel(self.policy, self.directory)

    def test_transient_download_retry_keeps_same_url_and_cleans_partial_files(self):
        opener = Mock(side_effect=[URLError("transient"), io.BytesIO(self.data)])
        with patch("prepare_omni_wheel.time.sleep"):
            receipt = prepare_wheel(self.policy, self.directory, opener=opener)
        self.assertEqual([call.args[0] for call in opener.call_args_list], [self.url, self.url])
        self.assertEqual(list((self.directory/"omni-wheels").iterdir()), [Path(receipt["wheel_path"])])

    def test_missing_hash_or_unapproved_origin_is_rejected(self):
        for requirement in (self.url, self.policy["omni_wheel"].replace("files.pythonhosted.org", "unreviewed.example")):
            with self.subTest(requirement=requirement), self.assertRaises(ValueError):
                prepare_wheel({"omni_wheel": requirement}, self.directory)


if __name__ == "__main__":
    unittest.main()
