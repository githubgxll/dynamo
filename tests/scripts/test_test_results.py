# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the oneclick runner's former false-success cases."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "test_results", Path(__file__).with_name("test_results.py")
)
results = importlib.util.module_from_spec(spec)
spec.loader.exec_module(results)


class ResultTests(unittest.TestCase):
    def report(self, kind, commands):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            rows = []
            for i, (exit_code, text, xml, omitted) in enumerate(commands):
                log = root / f"{i}.log"
                log.write_text(text)
                report = root / f"{i}.xml"
                if xml is not None:
                    report.write_text(xml)
                rows.append(
                    dict(
                        name=str(i),
                        exit_code=exit_code,
                        log=str(log),
                        xml=str(report),
                        omitted=omitted,
                    )
                )
            (root / "commands.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            with contextlib.redirect_stdout(io.StringIO()):
                code = results.summarize(root, kind)
            return code, json.loads((root / "summary.json").read_text())

    def test_compile_error_is_failure_without_failed_assertions(self):
        code, data = self.report(
            "rust", [(101, "error: package does not exist", None, False)]
        )
        self.assertEqual(code, 1)
        self.assertEqual(data["totals"]["command_failures"], 1)

    def test_all_rust_binaries_are_counted(self):
        code, data = self.report(
            "rust",
            [
                (
                    0,
                    "test result: ok. 3 passed; 0 failed; 1 ignored;\ntest result: ok. 2 passed; 0 failed; 0 ignored;",
                    None,
                    False,
                )
            ],
        )
        self.assertEqual(code, 0)
        self.assertEqual(data["totals"]["passed"], 5)

    def test_pytest_collection_error_is_counted(self):
        code, data = self.report(
            "python",
            [
                (
                    1,
                    "",
                    '<testsuites><testsuite><testcase><error message="import"/></testcase><testcase/></testsuite></testsuites>',
                    False,
                )
            ],
        )
        self.assertEqual(code, 1)
        self.assertEqual(data["totals"]["errors"], 1)
        self.assertEqual(data["totals"]["passed"], 1)

    def test_empty_pytest_group_cannot_pass(self):
        for exit_code in (0, 5):
            with self.subTest(exit_code=exit_code):
                code, _ = self.report(
                    "python", [(exit_code, "3 deselected", "<testsuites/>", False)]
                )
                self.assertEqual(code, 1)

    def test_omissions_are_incomplete_not_pass(self):
        code, data = self.report(
            "python",
            [
                (
                    0,
                    "",
                    "<testsuites><testsuite><testcase/></testsuite></testsuites>",
                    False,
                ),
                (0, "missing backend", None, True),
            ],
        )
        self.assertEqual(code, 2)
        self.assertEqual(data["totals"]["not_run_groups"], 1)

    def test_crash_keeps_partial_outcomes_without_counting_summary_twice(self):
        log = "test_x.py::test_ok PASSED [ 20%]\ntest_x.py::test_bad FAILED [ 40%]\nFAILED test_x.py::test_bad\nFatal Python error: Segmentation fault"
        for xml in (None, "<testsuites>"):
            code, data = self.report("python", [(139, log, xml, False)])
            self.assertEqual(code, 1)
            self.assertEqual(data["totals"]["passed"], 1)
            self.assertEqual(data["totals"]["failed"], 1)
            self.assertEqual(data["totals"]["partial_groups"], 1)
            self.assertFalse(data["commands"][0]["report_complete"])

    def test_missing_xml_cannot_pass_even_with_exit_zero(self):
        code, data = self.report(
            "python", [(0, "x.py::test_ok PASSED [100%]", None, False)]
        )
        self.assertEqual(code, 1)
        self.assertEqual(data["totals"]["partial_groups"], 1)


if __name__ == "__main__":
    unittest.main()
