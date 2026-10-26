# SPDX-License-Identifier: Apache-2.0
"""Actual image CPU gates, without models/network/GPU; never a C14 quality test."""
from __future__ import annotations
import argparse
import ast
import importlib
import json
import os
from pathlib import Path
import sys
import unittest
import zipfile

HERE = Path(__file__).resolve().parent


class DefaultOffNumerics(unittest.TestCase):
    def test_finish_video_matches_original_wheel(self):
        import torch
        native = importlib.import_module("vllm_omni.diffusion.models.seedvr2.pipeline_seedvr2")
        with zipfile.ZipFile("/opt/dynamo/build-info/omni-wheels/vllm_omni-0.31.0rc1-py3-none-any.whl") as wheel:
            source = wheel.read("vllm_omni/diffusion/models/seedvr2/pipeline_seedvr2.py").decode()
        function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "finish_video")
        scope = {"torch": torch, "correct_video_color": native.correct_video_color}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "retained-wheel-finish-video", "exec"), scope)
        # Boundary/clamp values and equal-colour regions exercise quantization/LAB.
        decoded = torch.linspace(-1.3, 1.3, 1 * 3 * 5 * 16 * 16, dtype=torch.float32).reshape(1, 3, 5, 16, 16)
        decoded[:, :, :, :4, :4] = 0
        sample = decoded.flip(-1).clone()
        for method in ("none", "lab", "wavelet"):
            with self.subTest(method=method), torch.inference_mode():
                before = scope["finish_video"](decoded, sample, 5, method)
                after = native.finish_video(decoded, sample, 5, method)
                self.assertEqual(after.dtype, torch.uint8)
                self.assertTrue(torch.equal(before, after))

    def test_installed_capture_is_disabled(self):
        module = importlib.import_module("vllm_omni.diffusion.models.seedvr2.c14_capture")
        self.assertNotIn("SEEDVR2_C14_PROFILE", os.environ)
        self.assertIsNone(module.profile())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Required rather than skipped on a CPU builder: these libraries already
    # belong to the complete successfully published parent environment.
    import torch
    import safetensors
    import av
    os.environ["C14_REQUIRE_RUNTIME_TESTS"] = "1"
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(DefaultOffNumerics)])
    suite.addTests(unittest.defaultTestLoader.discover(str(HERE / "tests"), pattern="test_*.py"))
    with (args.output_dir / "tests.txt").open("w", encoding="utf-8") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = {"schema": "seedvr2-c14-diagnostics-cpu/1", "status": "PASS" if result.wasSuccessful() and not result.skipped else "FAIL",
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
        "versions": {"torch": torch.__version__, "safetensors": safetensors.__version__, "av": av.__version__},
        "scope": "Actual installed CPU tensors, baseline gates and default-off colour output; no SP8/model/video quality proof"}
    (args.output_dir / "diagnostics-regression.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        print((args.output_dir / "tests.txt").read_text(encoding="utf-8")[-20000:])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
