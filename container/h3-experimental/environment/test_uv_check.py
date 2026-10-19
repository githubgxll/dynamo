"""Regress the final non-root audit's cache failure without a container.

Mocked process boundaries verify isolation and fail-closed handling, not Linux
permissions or the real image. Dependency fixtures keep the existing gate strict.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import audit_environment as environment
from dependency_audit import review_uv_check
import test_dependency_audit as fixtures


class UvCheckIsolationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def context(self):
        return json.loads((self.directory / "uv-pip-check-context.json").read_text(encoding="utf-8"))

    def test_root_owned_cache_and_hostile_config_are_not_inherited(self):
        inherited = {
            "UV_CACHE_DIR": "/opt/uv/cache",
            "UV_CONFIG_FILE": "/root/private/uv.toml",
            "UV_PYTHON": "/wrong/interpreter",
            "UV_OFFLINE": "false",
            "UV_NO_CACHE": "false",
            "UV_INDEX_URL": "https://user:private-value@example.invalid/simple",
            "uv_project_environment": "/wrong/venv",
            "HOME": "/root",
            "TMPDIR": "/root/not-writable",
            "TMP": "/root/not-writable",
            "TEMP": "/root/not-writable",
            "NO_COLOR": "0",
            "H3_TEST_KEEP_ME": "preserved",
        }
        observed = []

        def run(command, **kwargs):
            env = kwargs["env"]
            self.assertFalse(any(name.upper().startswith("UV_") for name in env))
            self.assertEqual(env["HOME"], inherited["HOME"])
            self.assertEqual(env["H3_TEST_KEEP_ME"], "preserved")
            self.assertEqual(env["NO_COLOR"], "1")
            temporary = Path(env["TMPDIR"])
            self.assertEqual(env["TMP"], str(temporary))
            self.assertEqual(env["TEMP"], str(temporary))
            self.assertEqual(temporary.parent, self.directory)
            self.assertTrue(temporary.is_dir())
            # Prove the process receives a usable location, not just a new name.
            (temporary / "interpreter-cache-probe").write_text("writable", encoding="utf-8")
            observed.append(temporary)
            return subprocess.CompletedProcess(command, 0, "Checked 3 packages in 1ms\n")

        with patch.dict(os.environ, inherited), patch.object(environment.subprocess, "run", side_effect=run):
            parent = dict(os.environ)
            result = environment.run_uv_check(self.directory)
            self.assertEqual(dict(os.environ), parent)
        self.assertEqual(result.returncode, 0)
        self.assertFalse(observed[0].exists())
        context = self.context()
        self.assertEqual(context["temporary_directory"], str(observed[0]))
        self.assertTrue(context["temporary_directory_removed"])
        self.assertNotIn("private-value", json.dumps(context))

    def test_check_uses_current_interpreter_and_forbids_config_cache_and_downloads(self):
        with patch.object(environment.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "")) as runner:
            environment.run_uv_check(self.directory)
        command = runner.call_args.args[0]
        for option in ("--no-config", "--no-cache", "--offline", "--no-python-downloads"):
            self.assertIn(option, command)
        self.assertEqual(command[-4:], ["pip", "check", "--python", sys.executable])
        self.assertEqual(runner.call_args.kwargs["stdout"], subprocess.PIPE)
        self.assertEqual(runner.call_args.kwargs["stderr"], subprocess.STDOUT)
        self.assertTrue(runner.call_args.kwargs["text"])
        self.assertFalse(runner.call_args.kwargs["check"])
        self.assertEqual(runner.call_args.kwargs["timeout"], 120)
        self.assertEqual(self.context()["command"], command)

    def test_repeated_checks_never_reuse_a_previous_temporary_cache(self):
        paths = []

        def run(command, **kwargs):
            path = Path(kwargs["env"]["TMPDIR"])
            self.assertTrue(path.is_dir())
            self.assertFalse(any(path.iterdir()))
            paths.append(path)
            (path / "root-created-or-stale-cache").write_text("fixture", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "")

        with patch.object(environment.subprocess, "run", side_effect=run):
            environment.run_uv_check(self.directory)
            environment.run_uv_check(self.directory)
        self.assertEqual(len(set(paths)), 2)
        self.assertTrue(all(not path.exists() for path in paths))

    def test_real_exit_and_diagnostics_are_preserved_even_for_tool_failure(self):
        for code, text in ((0, "Checked 1 package in 1ms\n"), (1, fixtures.KNOWN_UV_OUTPUT),
                           (2, "error: failed to initialize cache: Permission denied (os error 13)\n")):
            with self.subTest(code=code), patch.object(environment.subprocess, "run",
                    return_value=subprocess.CompletedProcess(["uv"], code, text)):
                result = environment.run_uv_check(self.directory)
                self.assertEqual(result.returncode, code)
                self.assertEqual(result.stdout, text)
                context = self.context()
                self.assertEqual(context["exit_code"], code)
                self.assertFalse(Path(context["temporary_directory"]).exists())

    def test_missing_or_unexecutable_uv_is_a_diagnostic_tool_failure(self):
        for error in (FileNotFoundError(2, "uv not found"), PermissionError(13, "uv is not executable")):
            with self.subTest(error=type(error).__name__), patch.object(environment.subprocess, "run", side_effect=error):
                result = environment.run_uv_check(self.directory)
                self.assertEqual(result.returncode, 2)
                self.assertIn(type(error).__name__, result.stdout)
                self.assertIn(str(error), result.stdout)
                context = self.context()
                self.assertEqual(context["error_type"], type(error).__name__)
                self.assertFalse(Path(context["temporary_directory"]).exists())

    def test_timeout_preserves_partial_diagnostics_and_cleans_temporary_data(self):
        for partial in (b"partial diagnostics\xff\n", "partial diagnostics\n", None):
            with self.subTest(partial=partial), patch.object(environment.subprocess, "run",
                    side_effect=subprocess.TimeoutExpired(["uv"], 120, output=partial)):
                result = environment.run_uv_check(self.directory)
                self.assertEqual(result.returncode, 2)
                self.assertIn("timed out after 120 seconds", result.stdout)
                if partial:
                    self.assertIn("partial diagnostics", result.stdout)
                self.assertIsInstance(result.stdout, str)
                context = self.context()
                self.assertEqual(context["error_type"], "TimeoutExpired")
                self.assertFalse(Path(context["temporary_directory"]).exists())

    def test_isolation_does_not_turn_exit_two_into_a_metadata_exception(self):
        with patch.object(environment.subprocess, "run",
                return_value=subprocess.CompletedProcess([], 2, fixtures.KNOWN_UV_OUTPUT)):
            result = environment.run_uv_check(self.directory)
        report = fixtures.audit(*fixtures.official_stack(), evidence=fixtures.evidence())
        review_uv_check(report, result.returncode, result.stdout)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("uv pip check failed as a tool (exit 2)", report["errors"])

    def test_isolation_preserves_only_the_verified_exact_nccl_exception(self):
        with patch.object(environment.subprocess, "run",
                return_value=subprocess.CompletedProcess([], 1, fixtures.KNOWN_UV_OUTPUT)):
            result = environment.run_uv_check(self.directory)
        report = fixtures.audit(*fixtures.official_stack(), evidence=fixtures.evidence())
        review_uv_check(report, result.returncode, result.stdout)
        self.assertEqual(report["status"], "PASS_WITH_UPSTREAM_NCCL_OVERRIDE")
        self.assertEqual(report["errors"], [])

    def test_new_dependency_conflict_still_fails_after_isolation(self):
        conflict = fixtures.KNOWN_UV_OUTPUT.replace("Found 1 incompatibility", "Found 2 incompatibilities")
        conflict += "The package `example` requires `dependency==2`, but `1` is installed\n"
        with patch.object(environment.subprocess, "run",
                return_value=subprocess.CompletedProcess([], 1, conflict)):
            result = environment.run_uv_check(self.directory)
        report = fixtures.audit(*fixtures.official_stack(), evidence=fixtures.evidence())
        review_uv_check(report, result.returncode, result.stdout)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("not exactly the verified upstream" in item for item in report["errors"]))


if __name__ == "__main__":
    unittest.main()
