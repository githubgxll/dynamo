# SPDX-License-Identifier: Apache-2.0
"""Exercise installed CPU media/control code; no model/GPU acceptance claim."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent


class InstalledIdentity(unittest.TestCase):
    def test_imports_use_installed_exact_sources(self):
        from source_patches import lock
        root = Path(importlib.metadata.distribution("vllm-omni").locate_file("")).resolve()
        patch = lock()
        expected = {name: value["after_sha256"] for name, value in patch["changes"].items()}
        expected.update(patch["additions"])
        for name, value in expected.items():
            with self.subTest(name=name):
                module = importlib.import_module(name[:-3].replace("/", "."))
                self.assertEqual(Path(module.__file__).resolve(), (root / name).resolve())
                self.assertEqual(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(), value)

    def test_default_observation_is_disabled(self):
        from types import SimpleNamespace
        module = importlib.import_module("vllm_omni.diffusion.models.seedvr2.sr_telemetry")
        self.assertFalse(module.window(SimpleNamespace(sampling_params=SimpleNamespace(extra_args={})), "cpu").enabled)
        self.assertNotIn("SEEDVR2_METRICS_DIR", os.environ)

    def test_child_gpu_adapter_resolves_exact_inherited_gate(self):
        from unittest.mock import patch
        spec = importlib.util.spec_from_file_location("_cpu_sr_gpu_adapter", HERE / "gpu_roundtrip.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.dict(sys.modules), patch.object(sys, "path", list(sys.path)):
            sys.modules.pop("runtime", None)
            gate = module.resolve_gate()
            self.assertEqual(Path(gate.main.__code__.co_filename).resolve(), Path("/opt/h3-hsdp-offload/gpu_roundtrip.py"))
            self.assertEqual(Path(sys.modules["runtime"].__file__).resolve(), HERE / "runtime.py")
            self.assertEqual(Path(sys.modules["torch_contract"].__file__).resolve(), Path("/opt/h3-hsdp-offload/torch_contract.py"))
            with patch.dict(os.environ, SEEDVR2_METRICS_DIR="/tmp/sr-gate-observations"):
                self.assertEqual(sys.modules["runtime"].verify_installed()["status"], "PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["SR_OPT_REQUIRE_RUNTIME"] = "1"
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(InstalledIdentity)])
    suite.addTests(unittest.defaultTestLoader.discover(str(HERE / "tests"), pattern="test_*.py"))
    with (args.output_dir / "tests.txt").open("w", encoding="utf-8") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = {"schema": "seedvr2-optimization-cpu/1", "status": "PASS" if result.wasSuccessful() and not result.skipped else "FAIL",
              "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
              "scope": "Actual installed imports, CPU media contracts and simulated CUDA control fixtures; no H100 model inference claim"}
    (args.output_dir / "regression.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        print((args.output_dir / "tests.txt").read_text(encoding="utf-8")[-24000:])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
