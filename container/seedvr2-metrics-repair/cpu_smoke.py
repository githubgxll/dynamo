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
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


class InstalledIdentity(unittest.TestCase):
    def test_actual_installed_cpu_postprocess_never_uses_cuda(self):
        from unittest.mock import patch
        import torch
        pipeline = importlib.import_module("vllm_omni.diffusion.models.seedvr2.pipeline_seedvr2")
        telemetry = importlib.import_module("vllm_omni.diffusion.models.seedvr2.sr_telemetry")
        video = torch.arange(5 * 48 * 80 * 3, dtype=torch.int32).to(torch.uint8).reshape(1, 5, 48, 80, 3)
        marker = {"schema": telemetry.SCHEMA, "measurement_mode": "basic", "sr_request_id": "installed-cpu-postprocess",
            "model_request_id": "installed-window", "actual_model_request_id": "actual-cpu-request", "window_index": 0,
            "window_start": 0, "input_frame_count": 5, "padded_frame_count": 5, "new_frame_count": 5, "overlap_frames": 0,
            "configured_window": 17, "selected_window": 17, "model_width": 80, "model_height": 48, "delivery_width": 80, "delivery_height": 48}
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, SEEDVR2_METRICS_DIR=temp), \
             patch.object(torch.cuda, "Event", side_effect=AssertionError("CPU postprocess created CUDA event")), \
             patch.object(torch.cuda, "get_device_properties", side_effect=AssertionError("CPU postprocess queried CUDA")):
            output = pipeline._seedvr2_post_process({"payload": {"video": video}, "metadata": {telemetry.KEY: marker}})
            receipt = json.loads(next(Path(temp).rglob("postprocess-*.json")).read_text())
        actual = output["payload"]["video"]
        self.assertTrue(torch.equal(torch.from_numpy(actual), video))
        self.assertEqual(actual.__array_interface__["data"][0], video.data_ptr())
        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual(receipt["execution_scope"], "api_cpu_postprocess")
        self.assertEqual(receipt["stages"][0]["phase"], "cpu_video_numpy")
        self.assertIsNone(receipt["stages"][0]["cuda_stream_elapsed_ms"])
        self.assertEqual(receipt["local_d2h"], "not_executed_here")

    def test_actual_installed_unmarked_cpu_ipc_roundtrip_unchanged(self):
        import torch
        from vllm_omni.diffusion.data import DiffusionOutput
        from vllm_omni.diffusion import ipc
        video = torch.arange(1_200_000, dtype=torch.int32).reshape(100, 100, 120)
        expected = video.clone()
        output = DiffusionOutput(output={"payload": {"video": video}, "metadata": {"unmarked_model": "control"}})
        packed = ipc.pack_diffusion_output_shm(output)
        result = ipc.unpack_diffusion_output_shm(packed)
        self.assertTrue(torch.equal(result.output["payload"]["video"], expected))
        self.assertEqual(result.output["metadata"], {"unmarked_model": "control"})

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
    os.environ["SR_METRICS_REQUIRE_RUNTIME"] = "1"
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(InstalledIdentity)])
    suite.addTests(unittest.defaultTestLoader.discover(str(HERE / "tests"), pattern="test_*.py"))
    with (args.output_dir / "tests.txt").open("w", encoding="utf-8") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = {"schema": "seedvr2-metrics-repair-cpu/1", "status": "PASS" if result.wasSuccessful() and not result.skipped else "FAIL",
              "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
              "scope": "Actual installed imports, CPU media contracts and simulated CUDA control fixtures; no H100 model inference claim"}
    (args.output_dir / "regression.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        print((args.output_dir / "tests.txt").read_text(encoding="utf-8")[-24000:])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
