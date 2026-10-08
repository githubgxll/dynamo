"""Small fixture tests. No network requests or model downloads."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import ssl
import stat
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import zipfile

SPEC = importlib.util.spec_from_file_location("prepare_assets", Path(__file__).with_name("prepare_assets.py"))
assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assets)


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def archive(self, entries):
        path = self.root / "fixture.zip"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, content in entries.items():
                archive.writestr(name, content)
        return path

    def test_source_only_extraction_keeps_licenses_and_runtime_config(self):
        path = self.archive({"Repo/LICENSE": "license", "Repo/vbench/__init__.py": "pass", "Repo/vbench/third_party/amt/LICENSE": "AMT", "Repo/vbench/third_party/amt/cfgs/AMT-S.yaml": "network: test", "Repo/vbench/demo.mp4": "video", "Repo/vbench/preview.png": "png", "Repo/private/credentials.txt": "excluded", "Repo/vbench2_beta_long/utils.py": "pass", "Repo/.git/config": "excluded"})
        output = self.root / "source"
        assets.extract_archive(path, output, prefix="Repo", source_kind="vbench")
        actual = {x.relative_to(output).as_posix() for x in output.rglob("*") if x.is_file()}
        self.assertEqual(actual, {"LICENSE", "vbench/__init__.py", "vbench/third_party/amt/LICENSE", "vbench/third_party/amt/cfgs/AMT-S.yaml", "vbench2_beta_long/utils.py"})

    def test_dino_keeps_root_python_and_license_only(self):
        path = self.archive({"Repo/hubconf.py": "pass", "Repo/vision_transformer.py": "pass", "Repo/utils.py": "pass", "Repo/LICENSE": "license", "Repo/imgs/demo.png": "excluded", "Repo/notebooks/demo.ipynb": "excluded"})
        output = self.root / "dino"
        assets.extract_archive(path, output, prefix="Repo", source_kind="dino")
        self.assertEqual(len(list(output.iterdir())), 4)

    def test_raft_extracts_only_selected_weight(self):
        path = self.archive({"models/raft-things.pth": "desired", "models/raft-kitti.pth": "excluded"})
        destination = self.root / "weights" / "raft-things.pth"
        assets.extract_archive(path, destination, member="models/raft-things.pth")
        self.assertEqual(destination.read_text(), "desired")
        self.assertEqual(list(destination.parent.iterdir()), [destination])

    def test_traversal_is_rejected_even_if_member_unselected(self):
        path = self.archive({"models/raft-things.pth": "desired", "../escape": "bad"})
        with self.assertRaises(assets.AssetError):
            assets.extract_archive(path, self.root / "weight", member="models/raft-things.pth")
        self.assertFalse((self.root / "weight").exists())

    def test_zip_links_rejected(self):
        path = self.root / "link.zip"
        with zipfile.ZipFile(path, "w") as archive:
            info = zipfile.ZipInfo("Repo/link")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "../../secret")
        with zipfile.ZipFile(path) as archive, self.assertRaises(assets.AssetError):
            assets.checked_members(archive)

    def test_zip_expansion_limit(self):
        path = self.archive({"large.txt": "x" * 10000})
        with zipfile.ZipFile(path) as archive, self.assertRaises(assets.AssetError):
            assets.checked_members(archive, max_total=100)

    def test_zip_compression_ratio_limit(self):
        path = self.archive({"large.txt": "x" * 10000})
        with zipfile.ZipFile(path) as archive, self.assertRaises(assets.AssetError):
            assets.checked_members(archive, max_ratio=2)

    def manifest_fixture(self):
        asset_root, source_root = self.root / "assets", self.root / "source"
        asset_root.mkdir()
        source_root.mkdir()
        (asset_root / "weights.pth").write_bytes(b"weights")
        (source_root / "runtime.py").write_bytes(b"pass\n")
        manifest = {"schema_version": 1, "status": "COMPLETE_ASSET_LOCK", "files": assets.manifest_entries(asset_root, source_root)}
        return manifest, asset_root, source_root

    def test_runtime_verification_detects_corruption(self):
        manifest, asset_root, source_root = self.manifest_fixture()
        self.assertEqual(assets.verify_manifest(manifest, asset_root, source_root)["files"], 2)
        (asset_root / "weights.pth").write_bytes(b"changed")
        with self.assertRaisesRegex(assets.AssetError, "manifest mismatch"):
            assets.verify_manifest(manifest, asset_root, source_root)

    def test_runtime_verification_rejects_extra_file(self):
        manifest, asset_root, source_root = self.manifest_fixture()
        (source_root / "unexpected.py").write_text("pass")
        with self.assertRaisesRegex(assets.AssetError, "unexpected.py"):
            assets.verify_manifest(manifest, asset_root, source_root)

    def test_runtime_verification_rejects_missing_hash(self):
        manifest, asset_root, source_root = self.manifest_fixture()
        manifest["files"][0]["sha256"] = None
        with self.assertRaises(assets.AssetError):
            assets.verify_manifest(manifest, asset_root, source_root)

    def test_checkpoint_rejects_html_without_deserializing(self):
        file = self.root / "weight.pth"
        file.write_bytes(b"<!doctype html><html>Error</html>")
        with self.assertRaisesRegex(assets.AssetError, "signature"):
            assets.validate_checkpoint(file)

    def test_network_error_does_not_expose_signed_url(self):
        error = urllib.error.URLError("https://cdn.example/file?token=secret-value")
        with patch.object(assets.urllib.request.OpenerDirector, "open", side_effect=error):
            with self.assertRaises(assets.AssetError) as raised:
                assets.download("https://example.com/weight", self.root / "download")
        self.assertNotIn("secret-value", str(raised.exception))
        self.assertNotIn("token", str(raised.exception))
        self.assertIn("example.com", str(raised.exception))

    def test_network_errors_report_reason_type_and_numeric_code_only(self):
        secret = "https://cdn.example/file?token=secret-value"
        errors = [
            (urllib.error.URLError(socket.gaierror(-2, secret)), "gaierror", "errno=-2"),
            (urllib.error.URLError(TimeoutError(secret)), "TimeoutError", None),
            (urllib.error.URLError(ssl.SSLError(1, secret)), "SSLError", "errno=1"),
            (urllib.error.HTTPError(secret, 403, secret, {}, None), "HTTP 403", None),
        ]
        for error, kind, code in errors:
            with self.subTest(kind=kind):
                destination = self.root / "download"
                partial = self.root / "download.partial"
                partial.write_bytes(b"partial-data")
                with patch.object(assets.urllib.request.OpenerDirector, "open", side_effect=error), self.assertRaises(assets.AssetError) as raised:
                    assets.download("https://example.com/weight", destination)
                detail = str(raised.exception)
                self.assertIn(kind, detail)
                if code:
                    self.assertIn(code, detail)
                self.assertNotIn("secret-value", detail)
                self.assertNotIn("cdn.example", detail)
                self.assertFalse(partial.exists())
                self.assertFalse(destination.exists())

    def test_archive_checkpoint_preserves_bytes_in_prepare_and_publish(self):
        weight = b"\x80\x02fake-checkpoint-never-deserialized"
        checksum = hashlib.sha256(weight).hexdigest()
        archive = self.archive({"Repo/weight.pth": weight, "Repo/other.pth": b"excluded"}).read_bytes()
        archive_sha = hashlib.sha256(archive).hexdigest()
        catalog = {"schema_version": 1, "sources": [], "weights": [{
            "id": "linear-head", "url": "https://codeload.github.com/example/repo/zip/" + "1" * 40,
            "revision": "1" * 40, "archive_member": "Repo/weight.pth", "path": "model/weight.pth",
            "sha256": archive_sha, "checkpoint_sha256": checksum,
        }], "links": []}
        catalog_path = self.root / "catalog.json"
        assets.write_json(catalog_path, catalog)

        def fake_download(url, destination, **kwargs):
            self.assertEqual(kwargs["expected_sha256"], archive_sha)
            Path(destination).write_bytes(archive)
            return {"sha256": archive_sha, "bytes": len(archive)}

        lock = self.root / "prepare-evidence" / "assets.lock.json"
        for phase in ("prepare", "publish"):
            args = argparse.Namespace(phase=phase, catalog=str(catalog_path), lock=str(lock),
                output=str(self.root / (phase + "-assets")), vbench_source=str(self.root / (phase + "-source")),
                evidence=str(self.root / (phase + "-evidence")), timeout=1, deadline=1)
            with patch.object(assets, "download", side_effect=fake_download):
                assets.run(args)
            self.assertEqual((Path(args.output) / "model/weight.pth").read_bytes(), weight)
            self.assertFalse((Path(args.output) / "model/other.pth").exists())
        manifest = assets.read_json(lock)
        self.assertEqual(manifest["downloads"][0]["sha256"], archive_sha)
        self.assertEqual(manifest["files"][0]["sha256"], checksum)
        self.assertEqual(lock.read_bytes(), (self.root / "publish-evidence/assets.lock.json").read_bytes())

        catalog["weights"][0]["checkpoint_sha256"] = "0" * 64
        assets.write_json(catalog_path, catalog)
        args = argparse.Namespace(**{**vars(args), "phase": "prepare", "output": str(self.root / "bad-assets"),
            "vbench_source": str(self.root / "bad-source"), "evidence": str(self.root / "bad-evidence")})
        with patch.object(assets, "download", side_effect=fake_download), self.assertRaisesRegex(assets.AssetError, "checkpoint SHA256 mismatch"):
            assets.run(args)
        self.assertFalse((Path(args.evidence) / "assets.lock.json").exists())

    def test_prepare_publish_roundtrip_and_no_mutable_resolution_in_publish(self):
        source_zip = self.archive({"Repo/LICENSE": "license", "Repo/vbench/__init__.py": "pass"}).read_bytes()
        weight = b"\x80\x02fake-model-never-deserialized"
        catalog = {"schema_version": 1, "sources": [{"id": "vbench-source", "source_kind": "vbench", "root": "source", "url": "https://example.com/source.zip", "revision": "1" * 40, "archive_prefix": "Repo"}], "weights": [{"id": "amt-s", "hf_repository": "example/AMT", "hf_filename": "amt-s.pth", "path": "vbench/amt_model/amt-s.pth"}], "links": []}
        catalog_path = self.root / "catalog.json"
        assets.write_json(catalog_path, catalog)
        args = argparse.Namespace(phase="prepare", catalog=str(catalog_path), lock=None, output=str(self.root / "prepare-assets"), vbench_source=str(self.root / "prepare-source"), evidence=str(self.root / "prepare-evidence"), timeout=1, deadline=1)

        def fake_download(url, destination, **kwargs):
            data = source_zip if url.endswith("source.zip") else weight
            digest = hashlib.sha256(data).hexdigest()
            if kwargs.get("expected_sha256") and kwargs["expected_sha256"] != digest:
                raise assets.AssetError("download SHA256 mismatch")
            Path(destination).write_bytes(data)
            return {"sha256": digest, "bytes": len(data)}

        with patch.object(assets, "download", side_effect=fake_download), patch.object(assets, "resolve_hf_revision", return_value="2" * 40) as resolve:
            assets.run(args)
            self.assertEqual(resolve.call_count, 1)
        lock = Path(args.evidence) / "assets.lock.json"
        publish = argparse.Namespace(**{**vars(args), "phase": "publish", "lock": str(lock), "output": str(self.root / "publish-assets"), "vbench_source": str(self.root / "publish-source"), "evidence": str(self.root / "publish-evidence")})
        with patch.object(assets, "download", side_effect=fake_download), patch.object(assets, "resolve_hf_revision", side_effect=AssertionError("main must never resolve in publish")):
            assets.run(publish)
        self.assertEqual(lock.read_bytes(), (Path(publish.evidence) / "assets.lock.json").read_bytes())

    def test_publish_fails_without_lock_before_network(self):
        path = self.root / "catalog.json"
        assets.write_json(path, {"schema_version": 1})
        args = argparse.Namespace(phase="publish", catalog=str(path), lock=None, output=str(self.root / "assets"), vbench_source=str(self.root / "source"), evidence=str(self.root / "evidence"))
        with patch.object(assets, "download", side_effect=AssertionError("network forbidden")), self.assertRaisesRegex(assets.AssetError, "publish requires"):
            assets.run(args)


if __name__ == "__main__":
    unittest.main()
