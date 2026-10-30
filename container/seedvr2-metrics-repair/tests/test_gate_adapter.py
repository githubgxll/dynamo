# SPDX-License-Identifier: Apache-2.0
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_test_sr_metrics_gate_adapter", HERE / "gpu_roundtrip.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class ChildGateAdapter(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        child, parent = root / "child", root / "parent"
        (child / "inputs").mkdir(parents=True)
        parent.mkdir()
        body = b"def main():\n import runtime\n return runtime.verify_installed()\n"
        (parent / "gpu_roundtrip.py").write_bytes(body)
        torch_body = b"def verify():\n return {'status':'PASS'}\n"
        (parent / "torch_contract.py").write_bytes(torch_body)
        (child / "runtime.py").write_text("def verify_installed():\n return {'full_child_identity':'verified'}\n")
        (child / "source_patches.py").write_text("SCOPE = 'complete child delta'\n")
        (child / "inputs/source-lock.json").write_text(json.dumps({"parent_gpu_gate_sha256": hashlib.sha256(body).hexdigest(),
            "parent_torch_contract_sha256": hashlib.sha256(torch_body).hexdigest()}))
        return child, parent

    def test_runs_unchanged_gate_with_explicit_child_identity(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules), patch.object(sys, "path", list(sys.path)):
            child, parent = self.fixture(temp)
            for name in ("runtime", "source_patches", "torch_contract"):
                sys.modules.pop(name, None)
            with patch.object(adapter, "HERE", child), patch.object(adapter, "PARENT", parent):
                gate = adapter.resolve_gate()
                self.assertEqual(gate.main(), {"full_child_identity": "verified"})
                self.assertEqual(Path(gate.main.__code__.co_filename), parent / "gpu_roundtrip.py")
                self.assertEqual(Path(sys.modules["torch_contract"].__file__), parent / "torch_contract.py")

    def test_changed_parent_gate_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            child, parent = self.fixture(temp)
            (parent / "gpu_roundtrip.py").write_text("raise RuntimeError('must not run')")
            with patch.object(adapter, "HERE", child), patch.object(adapter, "PARENT", parent), self.assertRaisesRegex(ValueError, "differs"):
                adapter.resolve_gate()

    def test_unrelated_cached_runtime_rejected(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, runtime=SimpleNamespace(__file__="/unrelated/runtime.py")):
            child, parent = self.fixture(temp)
            with patch.object(adapter, "HERE", child), patch.object(adapter, "PARENT", parent), self.assertRaisesRegex(ValueError, "already bound"):
                adapter.resolve_gate()

    def test_changed_torch_contract_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            child, parent = self.fixture(temp)
            (parent / "torch_contract.py").write_text("raise RuntimeError('wrong contract')")
            with patch.object(adapter, "HERE", child), patch.object(adapter, "PARENT", parent), self.assertRaisesRegex(ValueError, "Torch contract differs"):
                adapter.resolve_gate()


if __name__ == "__main__":
    unittest.main()
