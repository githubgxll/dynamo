"""Compose reviewed parent bytes; no installed Linux/GPU claims."""
from __future__ import annotations

import csv
import importlib.util
import io
import importlib.metadata
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_production_recipe_under_test", HERE / "runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def inputs():
    installed = HERE / "inputs"
    return installed if installed.is_dir() else HERE.parents[1] / "delivery/h3-sr-production"


def fixtures():
    values = {p.relative_to(HERE / "tests/fixtures/peak").as_posix(): p.read_bytes()
              for p in (HERE / "tests/fixtures/peak").rglob("*.py")}
    for p in (HERE / "tests/fixtures/sr-timing").glob("*.py"):
        values["vllm_omni/diffusion/models/seedvr2/" + p.name] = p.read_bytes()
    for p in (HERE / "tests/fixtures/duration").glob("*.py"):
        values["vllm_omni/diffusion/models/seedvr2/" + p.name] = p.read_bytes()
    return values


def populate(root):
    for relative, data in fixtures().items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


class SourceComposition(unittest.TestCase):
    def test_guarded_gateway_payload_matches_lock_and_effective_sources(self):
        policy = runtime.read(inputs() / "source-lock.json")
        self.assertTrue(policy["dingo_changes"])
        payload = HERE / "dingo-overlay" if (HERE / "dingo-overlay").is_dir() else inputs() / "dingo-overlay"
        repository = HERE.parents[1]
        local = (repository / "dingo/vllm/omni/production_handler.py").is_file()
        installed = None if local else Path(importlib.metadata.distribution("ai-dingo").locate_file(""))
        for relative, entry in policy["dingo_changes"].items():
            data = (payload / relative).read_bytes()
            self.assertEqual(runtime.sha(data), entry["after_sha256"], relative)
            compile(data, relative, "exec")
            effective = (repository if local else installed) / relative
            self.assertEqual(effective.read_bytes(), data, relative)

    def test_complete_plan_matches_frozen_manifest_and_does_not_modify_inputs(self):
        parent = runtime.read(inputs() / "parent-source-manifest.json")["source_sha256"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            populate(root)
            before = {name: runtime.sha((root / name).read_bytes()) for name in fixtures()}
            changes, plan = runtime.plan_native(root, parent)
            self.assertEqual(plan, runtime.read(inputs() / "expected-source-manifest.json"))
            self.assertEqual(len(changes), 13)
            self.assertEqual({name: runtime.sha((root / name).read_bytes()) for name in fixtures()}, before)
            # Window transport must retain both earlier timing and job ownership.
            window = changes["vllm_omni/diffusion/models/seedvr2/long_video.py"]
            self.assertIn(b"_h3sr_window_headers(authorization)", window)
            lock = runtime.read(HERE / "window-headers-patches.lock.json")
            self.assertEqual(runtime.sha(window), lock["changes"]["vllm_omni/diffusion/models/seedvr2/long_video.py"]["after_sha256"])
            for name, digest in parent.items():
                if name not in changes:
                    self.assertEqual(plan["source_sha256"][name], digest)

    def test_corrupt_first_input_rejected_before_any_write(self):
        parent = runtime.read(inputs() / "parent-source-manifest.json")["source_sha256"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            populate(root)
            path = root / "vllm_omni/diffusion/data.py"
            data = path.read_bytes() + b"\n# corruption\n"
            path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "published parent"):
                runtime.plan_native(root, parent)
            self.assertEqual(path.read_bytes(), data)

    def test_fixture_cannot_be_relabelled_as_another_parent(self):
        parent = runtime.read(inputs() / "parent-source-manifest.json")["source_sha256"]
        parent["vllm_omni/diffusion/data.py"] = "0" * 64
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            populate(root)
            with self.assertRaisesRegex(ValueError, "published parent"):
                runtime.plan_native(root, parent)


class RecordAndPaths(unittest.TestCase):
    def test_record_updates_sources_and_preserves_self_and_other_rows(self):
        data = b"pkg/one.py,sha256=old,3\npkg/two.py,sha256=unchanged,4\npkg.dist-info/RECORD,,\n"
        rows = {row[0]: row[1:] for row in csv.reader(io.StringIO(runtime.update_record(
            data, {"pkg/one.py": b"new contents", "pkg/new.py": b"added"}).decode()))}
        self.assertEqual(rows["pkg.dist-info/RECORD"], ["", ""])
        self.assertEqual(rows["pkg/two.py"], ["sha256=unchanged", "4"])
        self.assertEqual(rows["pkg/one.py"][1], "12")
        self.assertEqual(rows["pkg/new.py"][1], "5")

    def test_duplicate_or_malformed_record_fails(self):
        for data in (b"pkg.py,,\npkg.py,,\n", b"pkg.py,\n"):
            with self.subTest(data=data), self.assertRaises(ValueError):
                runtime.update_record(data, {})

    def test_unsafe_cross_platform_targets_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("../outside.py", "/outside.py", "C:/outside.py", "pkg\\outside.py", "pkg/data.json"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    runtime.target(root, name)


if __name__ == "__main__":
    unittest.main()
