"""Exercise real prepare/publish orchestration with small inert asset fixtures."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit
import zipfile

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import asset_cache

SPEC = importlib.util.spec_from_file_location("asset_pipeline_under_test", SCRIPTS / "prepare_assets.py")
assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assets)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class PipelineTests(unittest.TestCase):
    SOURCE_URL = "https://official.example/source/" + "1" * 40 + ".zip"
    FIRST_URL = "https://official.example/first.pth"
    SECOND_URL = "https://official.example/second.pth"

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / "download-cache"
        self.catalog_path = self.root / "catalog.json"
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
            output.writestr("Repo/LICENSE", "fixture source license\n")
            output.writestr("Repo/vbench/__init__.py", "# Fixture text; never executed.\n")
            output.writestr("Repo/example.mp4", "excluded sample")
        self.source_zip = archive.getvalue()
        # Only the format signature is inspected; these are not valid models and
        # must never be deserialized by asset acquisition or verification.
        self.first = b"\x80\x02inert first checkpoint fixture"
        self.second = b"\x80\x02inert second checkpoint fixture"
        self.payloads = {self.SOURCE_URL: self.source_zip, self.FIRST_URL: self.first, self.SECOND_URL: self.second}
        self.catalog = {
            "schema_version": 1,
            "sources": [{"id": "vbench-source", "source_kind": "vbench", "root": "source",
                "url": self.SOURCE_URL, "revision": "1" * 40, "archive_prefix": "Repo", "sha256": sha(self.source_zip)}],
            "weights": [
                {"id": "first-weight", "url": self.FIRST_URL, "path": "vbench/first.pth", "sha256": sha(self.first)},
                {"id": "second-weight", "url": self.SECOND_URL, "path": "vbench/second.pth"},
            ],
            "links": [],
        }
        assets.write_json(self.catalog_path, self.catalog)

    def args(self, label, phase="prepare", lock=None):
        return argparse.Namespace(phase=phase, catalog=str(self.catalog_path), lock=str(lock) if lock else None,
            output=str(self.root / (label + "-assets")), vbench_source=str(self.root / (label + "-source")),
            evidence=str(self.root / (label + "-evidence")), cache_dir=str(self.cache), timeout=1, deadline=10)

    def downloader(self, allowed, fail=None):
        def fake_download(url, destination, **kwargs):
            self.assertIn(url, allowed, "unexpected network request despite verified cache")
            destination = Path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            event = kwargs.get("event")
            common = {"host": urlsplit(url).hostname, "attempt": 1, "max_attempts": 3}
            if event:
                event({"event": "start", **common})
            if url == fail:
                destination.write_bytes(b"partial transport payload")
                if event:
                    event({"event": "failure", "reason": "injected transport failure", **common})
                raise assets.AssetError("injected transport failure")
            data = self.payloads[url]
            result = {"sha256": sha(data), "bytes": len(data)}
            if kwargs.get("expected_sha256"):
                self.assertEqual(kwargs["expected_sha256"], result["sha256"])
            self.assertLessEqual(len(data), kwargs.get("max_bytes", 2 * 1024**3))
            destination.write_bytes(data)
            if event:
                event({"event": "success", **common, **result})
            return result
        return fake_download

    def events(self, args):
        return [json.loads(line) for line in (Path(args.evidence) / "asset-acquisition.jsonl").read_text().splitlines()]

    def assert_fixture_outputs(self, args):
        self.assertEqual((Path(args.output) / "vbench/first.pth").read_bytes(), self.first)
        self.assertEqual((Path(args.output) / "vbench/second.pth").read_bytes(), self.second)
        lock = Path(args.evidence) / "assets.lock.json"
        manifest = assets.read_json(lock)
        self.assertEqual({(item["root"], item["path"]) for item in manifest["files"]}, {
            ("source", "LICENSE"), ("source", "vbench/__init__.py"),
            ("assets", "vbench/first.pth"), ("assets", "vbench/second.pth"),
        })
        self.assertFalse(any("cache" in item["path"] or "receipt" in item["path"] for item in manifest["files"]))
        assets.verify_manifest(manifest, args.output, args.vbench_source)
        return lock

    def test_failed_third_download_resumes_verified_cache_and_publishes_without_network(self):
        failed = self.args("failed")
        with patch.object(assets, "download", side_effect=self.downloader(set(self.payloads), fail=self.SECOND_URL)) as download:
            with self.assertRaisesRegex(assets.AssetError, "injected transport failure"):
                assets.run(failed)
        self.assertEqual([call.args[0] for call in download.call_args_list], [self.SOURCE_URL, self.FIRST_URL, self.SECOND_URL])
        self.assertFalse((Path(failed.evidence) / "assets.lock.json").exists())
        blobs = list(self.cache.glob("*.blob"))
        self.assertEqual({sha(path.read_bytes()) for path in blobs}, {sha(self.source_zip), sha(self.first)})
        self.assertEqual(len(list(self.cache.glob("*.receipt.json"))), 2)
        saved_cache = {path.name: path.read_bytes() for path in self.cache.iterdir()}

        resumed = self.args("resumed")
        with patch.object(assets, "download", side_effect=self.downloader({self.SECOND_URL})) as download, patch.object(asset_cache, "_fingerprint", wraps=asset_cache._fingerprint) as fingerprints:
            assets.run(resumed)
        self.assertEqual([call.args[0] for call in download.call_args_list], [self.SECOND_URL])
        self.assertEqual({record["asset_id"] for record in self.events(resumed) if record["event"] == "hit"}, {"vbench-source", "first-weight"})
        checked_blobs = {Path(call.args[0]).name for call in fingerprints.call_args_list if Path(call.args[0]).suffix == ".blob"}
        self.assertTrue({path.name for path in blobs}.issubset(checked_blobs), "cache reuse must rehash stored bytes")
        for name, data in saved_cache.items():
            self.assertEqual((self.cache / name).read_bytes(), data)
        lock = self.assert_fixture_outputs(resumed)

        publish = self.args("published", "publish", lock)
        with patch.object(assets, "download", side_effect=AssertionError("publish must use the verified warm cache")) as download:
            assets.run(publish)
        download.assert_not_called()
        published_lock = self.assert_fixture_outputs(publish)
        self.assertEqual(lock.read_bytes(), published_lock.read_bytes())

    def test_rejected_cache_validation_staging_cannot_leave_ghost_source_files(self):
        warm = self.args("warm")
        with patch.object(assets, "download", side_effect=self.downloader(set(self.payloads))):
            assets.run(warm)
        real_extract = assets.extract_archive
        stages = []

        def failing_first_extract(archive, destination, **kwargs):
            if kwargs.get("source_kind") == "vbench":
                stages.append(Path(destination))
                if len(stages) == 1:
                    Path(destination).mkdir(parents=True, exist_ok=True)
                    (Path(destination) / "ghost.py").write_text("partial untrusted extraction")
                    raise assets.AssetError("injected cache validation failure after partial extraction")
            return real_extract(archive, destination, **kwargs)

        recovered = self.args("recovered")
        with patch.object(assets, "extract_archive", side_effect=failing_first_extract), patch.object(assets, "download", side_effect=self.downloader({self.SOURCE_URL})) as download:
            assets.run(recovered)
        self.assertEqual([call.args[0] for call in download.call_args_list], [self.SOURCE_URL])
        self.assertEqual(len(stages), 2)
        self.assertNotEqual(stages[0], stages[1], "revalidation needs a new extraction directory")
        self.assertFalse(any(Path(recovered.vbench_source).rglob("ghost.py")))
        self.assert_fixture_outputs(recovered)
        self.assertTrue(any(event["asset_id"] == "vbench-source" and event["event"] == "invalid" for event in self.events(recovered)))

    def test_each_prepare_resolves_hf_main_but_publish_uses_the_reviewed_revision(self):
        repository = "fixture/AMT"
        api_url = "https://huggingface.co/api/models/fixture/AMT/revision/main"
        revisions = ("2" * 40, "3" * 40)
        self.catalog["weights"] = [{"id": "amt-s", "hf_repository": repository,
            "hf_filename": "amt-s.pth", "path": "vbench/amt_model/amt-s.pth"}]
        assets.write_json(self.catalog_path, self.catalog)
        for number, revision in enumerate(revisions):
            weight_url = f"https://huggingface.co/{repository}/resolve/{revision}/amt-s.pth"
            self.payloads[api_url] = json.dumps({"sha": revision}).encode()
            self.payloads[weight_url] = self.first if number == 0 else self.second
            allowed = {api_url, weight_url} | ({self.SOURCE_URL} if number == 0 else set())
            args = self.args("hf-" + str(number))
            with patch.object(assets, "download", side_effect=self.downloader(allowed)) as download, patch.object(assets, "resolve_hf_revision", wraps=assets.resolve_hf_revision) as resolve:
                assets.run(args)
            resolve.assert_called_once()
            urls = [call.args[0] for call in download.call_args_list]
            self.assertEqual(urls.count(api_url), 1)
            self.assertEqual(urls.count(weight_url), 1)
            lock = Path(args.evidence) / "assets.lock.json"
            self.assertEqual(assets.read_json(lock)["revisions"]["amt-s"], revision)
        receipts = [assets.read_json(path) for path in self.cache.glob("*.receipt.json")]
        self.assertFalse(any(item["url"] == api_url for item in receipts), "mutable main metadata cannot be reused as an asset cache entry")

        publish = self.args("hf-publish", "publish", lock)
        with patch.object(assets, "resolve_hf_revision", side_effect=AssertionError("publish cannot resolve main")) as resolve, patch.object(assets, "download", side_effect=AssertionError("all locked assets are already cached")) as download:
            assets.run(publish)
        resolve.assert_not_called()
        download.assert_not_called()
        self.assertEqual((Path(publish.output) / "vbench/amt_model/amt-s.pth").read_bytes(), self.second)
        self.assertEqual(lock.read_bytes(), (Path(publish.evidence) / "assets.lock.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
