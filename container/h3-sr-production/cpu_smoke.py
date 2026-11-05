"""Run production contracts in the actual image; never a GPU acceptance."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import unittest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    suite = unittest.defaultTestLoader.discover(str(here / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    value = {"status": "PASS" if result.wasSuccessful() and not result.skipped else "FAIL",
             "tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
             "skipped": len(result.skipped), "gpu_status": "NOT_RUN"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "regression.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if value["status"] == "PASS" and result.testsRun else 1)

if __name__ == "__main__":
    main()
