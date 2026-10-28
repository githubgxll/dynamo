import hashlib
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import download
import fetch_source


class Response(io.BytesIO):
    status = 200
    url = 'https://example.org/data'
    headers = {}


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.target = self.root / 'asset'
        self.data = b'known frozen content'
        self.digest = hashlib.sha256(self.data).hexdigest()

    def tearDown(self):
        self.temp.cleanup()

    def test_cache_hit_needs_no_network(self):
        self.target.write_bytes(self.data)
        with patch('urllib.request.urlopen', side_effect=AssertionError('network')):
            self.assertEqual(download.acquire('https://example.org/a', self.target, self.digest), self.target)

    def test_foreign_existing_file_is_preserved(self):
        self.target.write_bytes(b'old')
        with self.assertRaises(ValueError):
            download.acquire('https://example.org/a', self.target, self.digest)
        self.assertEqual(self.target.read_bytes(), b'old')

    def test_resume_ignored_replaces_partial(self):
        self.target.with_suffix('.partial').write_bytes(b'known')
        with patch('urllib.request.urlopen', return_value=Response(self.data)):
            download.acquire('https://example.org/a', self.target, self.digest, len(self.data))
        self.assertEqual(self.target.read_bytes(), self.data)

    def test_resume_range_is_checked(self):
        self.target.with_suffix('.partial').write_bytes(self.data[:5])
        response = Response(self.data[5:])
        response.status = 206
        response.headers = {'Content-Range': f'bytes 5-{len(self.data)-1}/{len(self.data)}'}
        with patch('urllib.request.urlopen', return_value=response):
            download.acquire('https://example.org/a', self.target, self.digest, len(self.data))
        self.assertEqual(self.target.read_bytes(), self.data)

    def test_bad_range_never_publishes(self):
        response = Response(self.data)
        response.status = 206
        response.headers = {'Content-Range': 'bytes 10-20/21'}
        with patch('urllib.request.urlopen', return_value=response), self.assertRaises(ValueError):
            download.acquire('https://example.org/a', self.target, self.digest)
        self.assertFalse(self.target.exists())

    def test_corrupt_download_not_published(self):
        with patch('urllib.request.urlopen', return_value=Response(b'bad')), self.assertRaises(RuntimeError):
            download.acquire('https://example.org/a', self.target, self.digest, attempts=1)
        self.assertFalse(self.target.exists())

    def test_full_partial_is_reused(self):
        self.target.with_suffix('.partial').write_bytes(self.data)
        with patch('urllib.request.urlopen', side_effect=AssertionError('network')):
            download.acquire('https://example.org/a', self.target, self.digest)
        self.assertEqual(self.target.read_bytes(), self.data)

    def test_unsafe_urls_paths(self):
        for url in ['http://example.org/a', 'https://user:pass@example.org/a']:
            with self.assertRaises(ValueError):
                download.acquire(url, self.target, self.digest)
        for name in ['/tmp/a', '../a', 'a/../../b', 'C:/a', 'a\\b']:
            with self.assertRaises(ValueError):
                download.relative(name)

    def archive(self, names):
        path = self.root / 'source.tar.gz'
        with tarfile.open(path, 'w:gz') as bundle:
            for name, link in names:
                entry = tarfile.TarInfo(name)
                if link:
                    entry.type = tarfile.SYMTYPE
                    entry.linkname = '/etc/passwd'
                    bundle.addfile(entry)
                else:
                    entry.size = 3
                    entry.mode = 0o755
                    bundle.addfile(entry, io.BytesIO(b'abc'))
        return path

    def test_extract_preserves_content(self):
        path = self.archive([('root/pkg/run.py', False)])
        fetch_source.extract(path, self.root/'out')
        self.assertEqual((self.root/'out/pkg/run.py').read_bytes(), b'abc')

    def test_extract_rejects_traversal_and_links(self):
        for name, link in [('root/../../escape', False), ('root/link', True)]:
            with self.assertRaises(ValueError):
                fetch_source.extract(self.archive([(name, link)]), self.root/'out')


if __name__ == '__main__':
    unittest.main()
