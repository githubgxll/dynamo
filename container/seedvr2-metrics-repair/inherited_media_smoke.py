# SPDX-License-Identifier: Apache-2.0
"""Run unchanged parent synthetic media/PTS/cancellation cases on installed code."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["SR_OPT_REQUIRE_RUNTIME"] = "1"
    path = Path("/opt/seedvr2-optimization/tests/test_long_video_media.py")
    spec = importlib.util.spec_from_file_location("_metrics_inherited_media_cases", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    superseded = "test_metrics_missing_stage_and_d2h_cannot_pass_complete"
    cases = [test for case in (module.LongVideoPatch, module.LongVideoMedia)
             for test in unittest.defaultTestLoader.loadTestsFromTestCase(case)]
    suite = unittest.TestSuite(test for test in cases if test._testMethodName != superseded)
    with (args.output_dir / "tests.txt").open("w", encoding="utf-8") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = {"schema": "sr-metrics-repair-inherited-media/1", "status": "PASS" if result.wasSuccessful() and not result.skipped else "FAIL",
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
        "superseded_schema_case": {"test": superseded, "reason": "Old /1 error-count assertion replaced by mandatory /2 completeness regression suite"},
        "scope": "Parent CPU synthetic frames/audio/encoding/PTS/cancellation cases against actual installed long-video code; no model inference"}
    (args.output_dir / "regression.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        print((args.output_dir / "tests.txt").read_text()[-16000:])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
