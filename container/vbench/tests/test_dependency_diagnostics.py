"""Installed-environment failure evidence must be visible without weakening gates."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_environment as prep


class DependencyDiagnosticTests(unittest.TestCase):
    def test_both_checkers_report_and_any_failure_still_stops(self):
        for pip_code, uv_code in ((0, 0), (1, 0), (0, 1), (1, 1)):
            with self.subTest(pip=pip_code, uv=uv_code), tempfile.TemporaryDirectory() as directory:
                root, stdout, calls = Path(directory), io.StringIO(), []

                def checker(command, output):
                    calls.append(command)
                    code = pip_code if output.name == 'pip-check.txt' else uv_code
                    output.write_text('fixture-package requires example>=2, installed 1\n' if code else 'No broken requirements found.\n')
                    if code:
                        raise subprocess.CalledProcessError(code, command)

                with patch.object(prep, 'run', side_effect=checker), contextlib.redirect_stdout(stdout):
                    if pip_code or uv_code:
                        with self.assertRaisesRegex(RuntimeError, 'dependency checks failed'):
                            prep.check_dependencies(root)
                    else:
                        prep.check_dependencies(root)
                self.assertEqual(len(calls), 2)
                summary = json.loads((root/'dependency-checks.json').read_text())
                self.assertEqual(summary['status'], 'FAIL' if pip_code or uv_code else 'PASS')
                self.assertEqual([x['exit_code'] for x in summary['checks']], [pip_code, uv_code])
                for name in ('pip-check.txt', 'uv-pip-check.txt'):
                    self.assertIn('=== '+name, stdout.getvalue())
                    self.assertIn((root/name).read_text().strip(), stdout.getvalue())

    def test_checker_launch_failure_is_recorded_and_other_checker_still_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def checker(command, output):
                if output.name == 'pip-check.txt':
                    raise FileNotFoundError('missing checker')
                output.write_text('second checker ran\n')
            with patch.object(prep, 'run', side_effect=checker) as run, contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                prep.check_dependencies(root)
            self.assertEqual(run.call_count, 2)
            checks = json.loads((root/'dependency-checks.json').read_text())['checks']
            self.assertEqual(checks[0]['error_type'], 'FileNotFoundError')
            self.assertEqual(checks[1]['exit_code'], 0)

    def test_diagnostic_url_credentials_and_queries_are_not_saved_or_printed(self):
        with tempfile.TemporaryDirectory() as directory:
            root, stdout = Path(directory), io.StringIO()
            def checker(command, output):
                output.write_text('example @ https://user:secret@cdn.example/package?token=signed-secret\n')
                raise subprocess.CalledProcessError(1, command)
            with patch.object(prep, 'run', side_effect=checker), contextlib.redirect_stdout(stdout), self.assertRaises(RuntimeError):
                prep.check_dependencies(root)
            for text in (stdout.getvalue(), (root/'pip-check.txt').read_text(), (root/'uv-pip-check.txt').read_text()):
                self.assertIn('cdn.example', text)
                self.assertNotIn('secret', text)
                self.assertNotIn('user:', text)


if __name__ == '__main__':
    unittest.main()
