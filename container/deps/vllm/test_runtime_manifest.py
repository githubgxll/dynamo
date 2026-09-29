"""CPU-only tests for the image dependency gate; no package installation."""

from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from collect_runtime_manifest import audit_dependencies, main, review_uv_check


class Distribution:
    def __init__(self, name, version="1.0", requires=(), requires_python=None):
        self.metadata = {"Name": name}
        if requires_python is not None:
            self.metadata["Requires-Python"] = requires_python
        self.version = version
        self.requires = list(requires)


def audit(*distributions):
    return audit_dependencies(
        distributions, python_version="3.12.9", environment={"sys_platform": "linux"}
    )


def known_packages(*, kvbm="1.3.0", nixl="1.3.1", requirement="nixl[cu13]==1.0.1"):
    return [
        Distribution("kvbm", kvbm, [requirement]),
        Distribution("nixl", nixl, ['nixl-cu13==1.3.1; extra == "cu13"']),
        Distribution("nixl-cu13", "1.3.1"),
    ]


KNOWN_UV_OUTPUT = """Using Python 3.12.9 environment at: /usr
Checked 300 packages in 2ms
Found 1 incompatibility
The package `kvbm` requires `nixl[cu13]==1.0.1`, but `1.3.1` is installed
"""


class DependencyAuditTests(unittest.TestCase):
    def test_compatible_environment(self):
        report = audit(Distribution("app", requires=["dependency>=1"]), Distribution("dependency"))
        self.assertEqual(report["status"], "PASS")

    def test_only_exact_known_metadata_exception(self):
        report = audit(*known_packages())
        review_uv_check(report, 1, KNOWN_UV_OUTPUT)
        self.assertEqual(report["status"], "PASS_WITH_KNOWN_METADATA_EXCEPTION")
        self.assertEqual(report["active_extras"]["nixl"], ["cu13"])

    def test_different_exception_versions_fail(self):
        for overrides in ({"kvbm": "1.3.1"}, {"nixl": "1.3.2"}, {"requirement": "nixl==1.0.1"}):
            with self.subTest(overrides=overrides):
                self.assertEqual(audit(*known_packages(**overrides))["status"], "FAIL")

    def test_missing_package_fails(self):
        report = audit(Distribution("transitive", requires=["absent>=1"]))
        self.assertTrue(any("missing absent" in error for error in report["errors"]))

    def test_transitive_conflict_not_hidden_by_exception(self):
        report = audit(*known_packages(), Distribution("library", requires=["other==2"]), Distribution("other"))
        review_uv_check(report, 1, KNOWN_UV_OUTPUT)
        self.assertEqual(report["status"], "FAIL")

    def test_recursive_extras_reach_fixpoint(self):
        report = audit(
            Distribution("third", requires=['missing==1; extra == "codec"']),
            Distribution("second", requires=['third[codec]>=1; extra == "media"']),
            Distribution("first", requires=["second[media]>=1"]),
        )
        self.assertEqual(report["active_extras"]["third"], ["codec"])
        self.assertTrue(any("missing missing" in error for error in report["errors"]))

    def test_inactive_markers_and_unrequested_extras_skipped(self):
        report = audit(Distribution("app", requires=[
            'windows-only; sys_platform == "win32"',
            'developer-only; extra == "dev"',
            'old-python; python_version < "3.12"',
        ]))
        self.assertEqual(report["status"], "PASS")

    def test_extra_cycle_terminates(self):
        report = audit(
            Distribution("first", requires=["second[media]"]),
            Distribution("second", requires=['first[codec]; extra == "media"']),
        )
        self.assertEqual(report["status"], "PASS")

    def test_requires_python_failure(self):
        report = audit(Distribution("app", requires_python=">=3.13"))
        self.assertTrue(any("Requires-Python" in error for error in report["errors"]))

    def test_invalid_metadata_fails(self):
        for distribution in (
            Distribution("bad", requires=["not valid !!!"]),
            Distribution("bad", requires_python="not valid"),
            Distribution("bad", version="not valid"),
        ):
            with self.subTest(distribution=distribution.__dict__):
                self.assertEqual(audit(distribution)["status"], "FAIL")

    def test_uv_tool_failure_fails(self):
        report = audit(*known_packages())
        review_uv_check(report, 2, "error: Python environment not found")
        self.assertEqual(report["status"], "FAIL")

    def test_unknown_uv_failure_not_hidden_by_exception(self):
        for output in (KNOWN_UV_OUTPUT + "error: unexpected failure\n", "error: unknown failure", KNOWN_UV_OUTPUT.replace("Found 1 incompatibility", "Found 2 incompatibilities")):
            with self.subTest(output=output):
                report = audit(*known_packages())
                review_uv_check(report, 1, output)
                self.assertEqual(report["status"], "FAIL")

    def test_uv_nonzero_without_exception_fails(self):
        report = audit(Distribution("app"))
        review_uv_check(report, 1, KNOWN_UV_OUTPUT)
        self.assertEqual(report["status"], "FAIL")

    def test_disagreement_with_clean_uv_fails(self):
        report = audit(*known_packages())
        review_uv_check(report, 0, "All installed packages are compatible")
        self.assertEqual(report["status"], "FAIL")

    def test_duplicate_package_fails(self):
        report = audit(Distribution("some-package"), Distribution("some_package"))
        self.assertTrue(any("duplicate installed" in error for error in report["errors"]))

    def test_main_writes_audit_before_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "uv-pip-check.exit-code").write_text("2\n")
            (directory / "uv-pip-check.txt").write_text("error: failed to find Python")
            with patch("collect_runtime_manifest.Path", return_value=directory), patch(
                "collect_runtime_manifest.md.distributions", return_value=[
                    Distribution("vllm", "0.30.0"),
                    Distribution("vllm-omni", "0.30.0"),
                ]
            ):
                with self.assertRaisesRegex(RuntimeError, "uv pip check failed as a tool"):
                    main()
            report = json.loads((directory / "dependency-audit.json").read_text())
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["uv_pip_check_exit_code"], 2)


if __name__ == "__main__":
    unittest.main()
