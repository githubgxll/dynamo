import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("h3_image_worker", Path(__file__).resolve().parents[1] / "worker.py")
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class WorkerAssets(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = {"schema_version": 1, "models": [
            {"id": worker.IMAGE_REPOSITORY, "revision": "image-fixed", "files": []},
            {"id": worker.BASE_REPOSITORY, "revision": "base-fixed", "files": []}]}
        self.add_asset("transformer/shard.safetensors", b"frozen weight", 0)
        self.add_asset("text_encoder/config.json", b'{"model_type":"qwen"}', 1, git=True)

    def add_asset(self, name, data, model, git=False, required=True):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self.manifest["models"][model]["files"].append({
            "path": name, "bytes": len(data), "required": required,
            "sha256": None if git else hashlib.sha256(data).hexdigest(),
            "git_blob_sha1": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() if git else None})

    def test_complete_payload_hashes_weights_and_git_blob_configs(self):
        result = worker.verify_assets(self.root, self.manifest)
        self.assertEqual(result["status"], "PASS_MODEL_ASSETS")
        self.assertEqual(result["required_files"], 2)
        self.assertEqual(result["files"]["text_encoder/config.json"]["algorithm"], "sha1")

    def test_same_size_content_change_is_rejected(self):
        (self.root / "transformer/shard.safetensors").write_bytes(b"changed bytes")
        with self.assertRaisesRegex(ValueError, "digest differs"):
            worker.verify_assets(self.root, self.manifest)

    def test_missing_asset_is_rejected(self):
        (self.root / "text_encoder/config.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            worker.verify_assets(self.root, self.manifest)

    def test_optional_docs_are_not_required(self):
        self.add_asset("README.md", b"old documentation", 0, required=False)
        (self.root / "README.md").unlink()
        self.assertEqual(worker.verify_assets(self.root, self.manifest)["required_files"], 2)

    def test_unsafe_and_noncanonical_paths_are_rejected(self):
        for name in ("../outside", "/absolute", "text_encoder\\file", "C:/escape", "./file"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                worker.relative_path(name)

    def test_duplicate_asset_is_rejected(self):
        item = dict(self.manifest["models"][0]["files"][0])
        self.manifest["models"][0]["files"].append(item)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            worker.required_assets(self.manifest)

    def test_locked_index_shards_are_required(self):
        index = json.dumps({"weight_map": {"tensor": "shard.safetensors"}}).encode()
        self.add_asset("transformer/model.safetensors.index.json", index, 0, git=True)
        self.assertEqual(worker.verify_assets(self.root, self.manifest)["required_files"], 3)

    def test_index_cannot_reference_unlocked_file(self):
        index = json.dumps({"weight_map": {"tensor": "unlocked.safetensors"}}).encode()
        self.add_asset("transformer/model.safetensors.index.json", index, 0, git=True)
        with self.assertRaisesRegex(ValueError, "unlocked shard"):
            worker.verify_assets(self.root, self.manifest)

    def test_flat_and_generation_layouts_discover_mount_sources(self):
        for layout in ("", "generation", "H3_Image_models", "H3_Image_models/generation"):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                image = root / layout / "MiniMax-H3-Image"
                image.mkdir(parents=True)
                (image / "model_index.json").write_text("{}")
                (root / layout / "MiniMax-H3/text_encoder").mkdir(parents=True)
                result = worker.discover_models(root)
                self.assertEqual(result["source_roots"][worker.IMAGE_REPOSITORY], str(image.resolve()))

    def test_multiple_layouts_are_ambiguous(self):
        for layout in ("", "generation"):
            image = self.root / layout / "MiniMax-H3-Image"
            image.mkdir(parents=True)
            (image / "model_index.json").write_text("{}")
            (self.root / layout / "MiniMax-H3/text_encoder").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "found 2"):
            worker.discover_models(self.root)


class WorkerProfiles(unittest.TestCase):
    def setUp(self):
        self.profiles = worker.read_json(Path(__file__).resolve().parents[1] / "profiles.json")

    def test_all_profiles_use_native_entrypoint_and_canonical_root(self):
        for profile in self.profiles["profiles"]:
            with self.subTest(profile=profile["id"]):
                argv = worker.build_argv(self.profiles, profile["id"])
                self.assertEqual(argv[:3], ["vllm", "serve", str(worker.MODEL_ROOT)])
                self.assertIn("--omni", argv)

    def test_model_path_and_port_are_literal_argv_values(self):
        root = Path(tempfile.gettempdir()).resolve() / "model; echo injected"
        argv = worker.build_argv(self.profiles, "h100-4-tp4", root, 8123)
        self.assertEqual(argv[2], str(root))
        self.assertEqual(argv[argv.index("--port") + 1], "8123")
        self.assertEqual(self.profiles["profiles"][2]["argv"][2], self.profiles["model_root"])

    def test_bad_profiles_relative_roots_and_ports_fail(self):
        with self.assertRaises(ValueError):
            worker.build_argv(self.profiles, "unknown")
        with self.assertRaises(ValueError):
            worker.build_argv(self.profiles, "h100-4-tp4", Path("relative"))
        for port in (0, 65536, -1):
            with self.subTest(port=port), self.assertRaises(ValueError):
                worker.build_argv(self.profiles, "h100-4-tp4", port=port)


class SharedModelView(unittest.TestCase):
    add_asset = WorkerAssets.add_asset

    def setUp(self):
        WorkerAssets.setUp(self)
        if os.name != "nt":
            return
        # Windows lacks CreateSymbolicLink privilege here. Tiny copied fixture
        # files simulate link metadata; Linux executes real symbolic links.
        links = {}
        native_resolve = Path.resolve
        native_rename = Path.rename
        native_readlink = os.readlink

        def key(path):
            return os.path.normcase(os.path.abspath(os.fspath(path)))

        def symlink(path, target, target_is_directory=False):
            if target_is_directory:
                raise AssertionError("These fixtures only simulate file links")
            path.write_bytes(Path(target).read_bytes())
            links[key(path)] = os.fspath(target)

        def resolve(path, *args, **kwargs):
            target = links.get(key(path))
            return native_resolve(Path(target) if target is not None else path, *args, **kwargs)

        def readlink(path, *args, **kwargs):
            target = links.get(key(path))
            return target if target is not None else native_readlink(path, *args, **kwargs)

        def rename(path, target):
            before, after = key(path), key(target)
            result = native_rename(path, target)
            moved = {after + name[len(before):]: value for name, value in links.items()
                     if name == before or name.startswith(before + os.sep)}
            for name in list(links):
                if name == before or name.startswith(before + os.sep):
                    del links[name]
            links.update(moved)
            return result

        patches = (mock.patch.object(Path, "symlink_to", symlink),
                   mock.patch.object(Path, "is_symlink", lambda path: key(path) in links and path.exists()),
                   mock.patch.object(Path, "resolve", resolve),
                   mock.patch.object(Path, "rename", rename),
                   mock.patch.object(os, "readlink", readlink))
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
    def split_sources(self, layout=""):
        self.add_asset("model_index.json", b'{"text_encoder": "base"}', 0, git=True)
        shared = self.root / "shared"
        image = shared / layout / "MiniMax-H3-Image"
        base = shared / layout / "MiniMax-H3"
        for model in self.manifest["models"]:
            destination = image if model["id"] == worker.IMAGE_REPOSITORY else base
            for entry in model["files"]:
                path = destination / entry["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((self.root / entry["path"]).read_bytes())
        return shared, image, base

    def test_shared_sources_build_view_without_writing_source_roots(self):
        shared, image, base = self.split_sources()
        before = sorted(str(path) for path in shared.rglob("*"))
        report = worker.verify_sources(shared, self.manifest)
        view = worker.prepare_view(report, self.root / "work")
        self.assertEqual(sorted(str(path) for path in shared.rglob("*")), before)
        self.assertTrue((view / "model_index.json").is_symlink())
        self.assertEqual((view / "text_encoder/config.json").resolve(), (base / "text_encoder/config.json").resolve())
        self.assertFalse((image / "text_encoder").exists())

    def test_repeated_owned_view_is_reused(self):
        shared, _, _ = self.split_sources("generation")
        report = worker.verify_sources(shared, self.manifest)
        first = worker.prepare_view(report, self.root / "work")
        second = worker.prepare_view(report, self.root / "work")
        self.assertEqual(first, second)
        worker.validate_view(first, worker.view_identity(report)[1])

    def test_canonical_mode_rejects_external_symlink(self):
        outside = self.root / "outside.dat"
        outside.write_bytes(b"frozen weight")
        original = self.root / "transformer/shard.safetensors"
        original.unlink()
        original.symlink_to(outside)
        # It remains within the canonical root and is allowed there.
        self.assertEqual(worker.verify_assets(self.root, self.manifest)["required_files"], 2)
        with tempfile.TemporaryDirectory() as other:
            external = Path(other) / "outside.dat"
            external.write_bytes(b"frozen weight")
            original.unlink()
            original.symlink_to(external)
            with self.assertRaisesRegex(ValueError, "outside the model root"):
                worker.verify_assets(self.root, self.manifest)

    def test_shared_source_symlink_cannot_escape_its_locked_root(self):
        shared, image, _ = self.split_sources()
        weight = image / "transformer/shard.safetensors"
        weight.unlink()
        weight.symlink_to(self.root / "transformer/shard.safetensors")
        with self.assertRaisesRegex(ValueError, "outside the model root"):
            worker.verify_sources(shared, self.manifest)

    def test_existing_view_with_altered_link_is_rejected(self):
        shared, _, base = self.split_sources()
        report = worker.verify_sources(shared, self.manifest)
        view = worker.prepare_view(report, self.root / "work")
        link = view / "model_index.json"
        link.unlink()
        link.symlink_to(base / "text_encoder/config.json")
        with self.assertRaisesRegex(ValueError, "altered link"):
            worker.prepare_view(report, self.root / "work")

    def test_existing_unowned_directory_is_rejected(self):
        shared, _, _ = self.split_sources()
        report = worker.verify_sources(shared, self.manifest)
        name, _ = worker.view_identity(report)
        view = self.root / "work/model-view" / name
        view.mkdir(parents=True)
        (view / "unowned.txt").write_text("keep this")
        with self.assertRaisesRegex(ValueError, "ownership"):
            worker.prepare_view(report, self.root / "work")
        self.assertEqual((view / "unowned.txt").read_text(), "keep this")

    def test_existing_regular_file_cannot_replace_verified_link(self):
        shared, _, _ = self.split_sources()
        report = worker.verify_sources(shared, self.manifest)
        view = worker.prepare_view(report, self.root / "work")
        link = view / "model_index.json"
        link.unlink()
        link.write_bytes(b"replacement")
        # Remove simulated metadata when the Windows fixture deletes its link.
        if os.name == "nt":
            with mock.patch.object(Path, "is_symlink", return_value=False):
                with self.assertRaisesRegex(ValueError, "unowned regular file"):
                    worker.prepare_view(report, self.root / "work")
        else:
            with self.assertRaisesRegex(ValueError, "unowned regular file"):
                worker.prepare_view(report, self.root / "work")


if __name__ == "__main__":
    unittest.main()
