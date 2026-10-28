# SPDX-License-Identifier: Apache-2.0
"""Exercise apply/RECORD/source verification on temporary package fixtures.

Only installation logic is real here; parent runtime, Torch and dependencies
are simulated. The Docker validator checks actual installed distributions.
"""
from contextlib import ExitStack
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parents[1]
CONTAINER = HERE.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InstallFixture(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.temp = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.root = self.temp / "site-packages"
        self.runtime = load("_dit_install_runtime", HERE / "runtime.py")
        self.patch = load("_dit_install_patch", HERE / "source_patches.py")
        # Use the retained implementations, both in the repository and image.
        lite_path = CONTAINER / "omni-gateway-lite/runtime.py"
        c14_path = CONTAINER / "seedvr2-c14/runtime.py"
        self.lite = load("_dit_install_lite", lite_path)
        self.c14 = load("_dit_install_c14", c14_path)
        self.reference = "vllm_omni/diffusion/distributed/hsdp.py"
        self.original = {}
        for name in (self.runtime.TARGET, self.runtime.PIPELINE, self.reference):
            body = (HERE / "vendor/original" / name).read_bytes()
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
            self.original[name] = body
        self.expected_before = {name: hashlib.sha256(body).hexdigest() for name, body in self.original.items()}
        self.record = self.root / "vllm_omni.dist-info/RECORD"
        self.record.parent.mkdir()
        rows = [[name, self.lite.record_digest(body), str(len(body))] for name, body in self.original.items()]
        output = io.StringIO(newline="")
        csv.writer(output, lineterminator="\n").writerows(rows)
        self.record.write_text(output.getvalue(), encoding="utf-8", newline="\n")
        self.dist = SimpleNamespace(locate_file=lambda _: self.root)
        parent = self.temp / "parent"
        (parent / "build-info").mkdir(parents=True)
        (parent / "build-info/source-before.json").write_text(
            json.dumps({"gateway": {"generated_version_sha256": "fixture"}}), encoding="utf-8")
        (parent / "gateway.lock.json").write_text("{}", encoding="utf-8")
        self.c14.source_expected = lambda *args: dict(self.expected_before)
        self.c14.verify_installed = lambda: {"status": "PASS", "source_sha256": dict(self.expected_before)}
        self.c14.assert_snapshot = lambda *args: None
        self.c14.NADIT = self.reference
        self.c14.GATEWAY = "fixture"
        self.lite.installed_root = lambda name: (self.dist, self.root)
        self.lite.distribution_record = lambda dist: self.record
        self.lite.gateway_sources = lambda *args, **kwargs: {"generated_version_sha256": "fixture"}
        for name, value in {"contract": lambda: (self.c14, self.lite, {}),
                            "INFO": self.temp / "build-info", "C14": parent, "LITE": parent,
                            "snapshot_environment": lambda: {}, "archive_parent": lambda *args: None}.items():
            self.stack.enter_context(mock.patch.object(self.runtime, name, value))
        self.stack.enter_context(mock.patch.object(os, "geteuid", return_value=0, create=True))
        self.stack.enter_context(mock.patch.dict(os.environ, {"H3_HSDP_SOURCE_COMMIT": "a" * 40}))
        self.stack.enter_context(mock.patch.dict(sys.modules, {
            "source_patches": self.patch, "torch_contract": SimpleNamespace(verify=lambda: {"status": "PASS"})}))

    def test_apply_updates_both_sources_record_and_receipt(self):
        self.runtime.apply(self.runtime.INFO)
        proof = self.runtime.verify_installed()
        self.assertEqual(proof["changed_sources"], [self.runtime.TARGET, self.runtime.PIPELINE])
        self.assertEqual(proof["added_sources"], [self.runtime.HELPER])
        self.assertEqual(proof["dit_entry_status"], "PASS")
        self.assertEqual((self.root / self.reference).read_bytes(), self.original[self.reference])
        self.assertEqual((self.root / self.runtime.PIPELINE).read_bytes(),
                         self.patch.transform_pipeline(self.original[self.runtime.PIPELINE]))
        self.c14.check_record(self.lite, self.dist, proof["source_sha256"])
        with self.assertRaisesRegex(ValueError, "create-once"):
            self.runtime.apply(self.runtime.INFO)

    def test_bad_pipeline_record_fails_before_any_source_write(self):
        rows = list(csv.reader(io.StringIO(self.record.read_text(encoding="utf-8"))))
        rows = [row for row in rows if row[0] != self.runtime.PIPELINE]
        output = io.StringIO(newline="")
        csv.writer(output, lineterminator="\n").writerows(rows)
        self.record.write_text(output.getvalue(), encoding="utf-8", newline="\n")
        with self.assertRaisesRegex(ValueError, "RECORD row"):
            self.runtime.apply(self.runtime.INFO)
        self.assertFalse(self.runtime.INFO.exists())
        self.assertFalse((self.root / self.runtime.HELPER).exists())
        for name, body in self.original.items():
            self.assertEqual((self.root / name).read_bytes(), body)

    def test_changed_pipeline_bytes_after_apply_are_rejected(self):
        self.runtime.apply(self.runtime.INFO)
        (self.root / self.runtime.PIPELINE).write_bytes(self.original[self.runtime.PIPELINE])
        with self.assertRaisesRegex(ValueError, "source mismatch"):
            self.runtime.verify_installed()

    def test_old_receipt_without_pipeline_change_is_rejected(self):
        self.runtime.apply(self.runtime.INFO)
        receipt = self.runtime.read(self.runtime.INFO / "application.json")
        receipt["changed_sources"] = [self.runtime.TARGET]
        self.runtime.dump(self.runtime.INFO / "application.json", receipt)
        with self.assertRaisesRegex(ValueError, "DiT entry changes"):
            self.runtime.verify_installed()


if __name__ == "__main__":
    unittest.main()
