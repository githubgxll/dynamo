#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Mandatory final-image CPU gates; target imports and historical fixtures differ.

Each group gets a fresh interpreter so historical fixture imports cannot rebind
the new runtime's source verifier. No group invokes a GPU gate or model loader.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import unittest

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
GROUPS = ("migration", "installed", "hsdp", "c14", "sr-media", "sr-metrics")
SCOPES = {
    "migration": "Exact official-0.32 source/AST/lock and synthetic adapter regressions",
    "installed": "Actual installed 0.32 imports, complete source proof, CPU IPC/postprocess/colour quantization; no GPU",
    "hsdp": "Actual installed CPU alias/native-parameter/fastpath/diffuse-entry cases plus frozen historical source/control fixtures",
    "c14": "Retained CPU Torch/PyAV diagnostic fixtures and default-off contracts; no C14 GPU/quality acceptance",
    "sr-media": "Actual installed long-video route on synthetic CPU frames/audio/PTS/cancellation; no model inference",
    "sr-metrics": "Actual installed v2 producer/receiver plus strict fake-CUDA/stream/failure fixtures; no real D2H",
}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def suite_for(group):
    loader = unittest.TestLoader()
    if group == "migration":
        return loader.discover(str(HERE / "tests"), pattern="test_*.py")
    if group == "installed":
        module = load("_omni032_actual_runtime_cases", HERE / "tests/test_runtime_media.py")
        return module.installed_suite()
    if group == "hsdp":
        # Keep the exact native Torch converter implementation/policy; selected
        # 0.32 vLLM still requires the parent's Torch 2.13.0+cu130 stack.
        adapter = load("_omni032_hsdp_cpu_adapter", HERE / "gpu_roundtrip.py")
        adapter.resolve_gate()  # Import/provenance only: never calls gate.main().
        native = load("_omni032_actual_hsdp_cases", PARENT / "h3-hsdp-offload/cpu_smoke.py")
        suite = unittest.TestSuite([loader.loadTestsFromTestCase(native.InstalledCPU)])
        suite.addTests(loader.discover(str(PARENT / "h3-hsdp-offload/tests"), pattern="test_*.py"))
        return suite
    if group == "c14":
        os.environ["C14_REQUIRE_RUNTIME_TESTS"] = "1"
        return loader.discover(str(PARENT / "seedvr2-c14/tests"), pattern="test_*.py")
    if group == "sr-media":
        os.environ["SR_OPT_REQUIRE_RUNTIME"] = "1"
        module = load("_omni032_actual_sr_media_cases", PARENT / "seedvr2-optimization/tests/test_long_video_media.py")
        cases = [test for cls in (module.LongVideoPatch, module.LongVideoMedia)
                 for test in loader.loadTestsFromTestCase(cls)]
        # This /1 diagnostic count was already superseded in the successful b315
        # gate. Its replacement is the complete mandatory /2 receiver suite.
        old_count = "test_metrics_missing_stage_and_d2h_cannot_pass_complete"
        return unittest.TestSuite(test for test in cases if test._testMethodName != old_count)
    if group == "sr-metrics":
        os.environ["SR_METRICS_REQUIRE_RUNTIME"] = "1"
        sys.modules["telemetry"] = importlib.import_module("vllm_omni.diffusion.models.seedvr2.sr_telemetry")
        path = PARENT / "seedvr2-metrics-repair/tests"
        suite = unittest.TestSuite()
        for pattern in ("test_completeness.py", "test_telemetry_repair.py", "test_transfer_patch.py",
                        "test_transfer_runtime.py", "test_producer_receiver_integration.py"):
            suite.addTests(loader.discover(str(path), pattern=pattern))
        return suite
    raise ValueError("Unknown CPU test group: " + group)


def run_group(group, output):
    output.mkdir(parents=True, exist_ok=True)
    log = output / "tests.txt"
    report = {"schema": "omni032-h3-sr-cpu-group/1", "group": group, "scope": SCOPES[group],
              "status": "FAIL", "tests_run": 0, "failures": 0, "errors": 0, "skipped": 0,
              "gpu_validation_status": "NOT_RUN", "model_inference_status": "NOT_RUN"}
    with log.open("w", encoding="utf-8") as stream:
        try:
            suite = suite_for(group)
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
            report.update(tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped))
            report["status"] = "PASS" if result.wasSuccessful() and result.testsRun > 0 and not result.skipped else "FAIL"
        except BaseException:
            traceback.print_exc(file=stream)
            report.update(errors=1, initialization_failed=True)
    (output / "regression.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "PASS":
        print(log.read_text(encoding="utf-8")[-24000:], flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group", choices=GROUPS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.group:
        raise SystemExit(0 if run_group(args.group, args.output_dir)["status"] == "PASS" else 1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for group in GROUPS:
        target = args.output_dir / group
        target.mkdir(exist_ok=True)
        command = [sys.executable, "-B", str(HERE / "cpu_smoke.py"), "--group", group, "--output-dir", str(target)]
        with (target / "process.log").open("w", encoding="utf-8") as log:
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False, timeout=1200)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = -1
        path = target / "regression.json"
        if path.is_file():
            report = json.loads(path.read_text(encoding="utf-8"))
        else:
            report = {"group": group, "status": "FAIL", "tests_run": 0, "failures": 0,
                      "errors": 1, "skipped": 0, "scope": SCOPES[group], "report_missing": True}
        if code != 0:
            report["status"] = "FAIL"
            print((target / "process.log").read_text(encoding="utf-8")[-24000:], flush=True)
        report["process_exit_code"] = code
        reports.append(report)
    success = all(item["status"] == "PASS" and item["tests_run"] > 0
                  and not any(item.get(key) for key in ("errors", "failures", "skipped")) for item in reports)
    report = {"schema": "omni032-h3-sr-cpu/1", "status": "PASS" if success else "FAIL",
              "tests_run": sum(item["tests_run"] for item in reports),
              "failures": sum(item["failures"] for item in reports),
              "errors": sum(item["errors"] for item in reports),
              "skipped": sum(item["skipped"] for item in reports), "groups": reports,
              "gpu_validation_status": "NOT_RUN", "model_inference_status": "NOT_RUN",
              "rmsnorm_six_numerical_cases": "SEPARATE_REQUIRED_GATE_RUN_BY_RUNTIME_VALIDATOR",
              "scope": "Fresh-process source and real installed CPU checks; synthetic stream fixtures do not establish H100/D2H/model acceptance"}
    (args.output_dir / "regression.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
