"""Regression coverage for narrowly correcting the pyIQA notice declaration."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import license_report


class PyiqaNoticeTests(unittest.TestCase):
    def test_current_repository_generator_format_is_supported(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from compliance.generators.common import Component, write_notices
        components = [Component('python', 'pyiqa', '0.1.13', 'Apache-2.0',
                                'https://pypi.org/project/pyiqa/', 'Original bundled full license\n')]
        with tempfile.TemporaryDirectory() as directory:
            file = write_notices('python', components, Path(directory))
            license_report.correct_pyiqa_notice(file)
            corrected = file.read_text(encoding='utf-8')
        self.assertIn('License: ' + license_report.PYIQA_REVIEWED_SPDX, corrected)
        self.assertIn('Original bundled full license\n', corrected)

    def test_only_reviewed_header_changes_and_full_text_is_preserved(self):
        before = (b'NOTICES header\n\n## other-before 1\n\nLicense: Apache-2.0\n'
                  b'\n```\nOriginal other license text\n```\n\n')
        header = b'## pyiqa 0.1.13\n\nLicense: Apache-2.0\n'
        after = (b'Source: https://pypi.org/project/pyiqa/\n\n```\n'
                 b'IQA-PyTorch full original noncommercial license text.\n'
                 b'License: Apache-2.0 may occur inside unrelated text.\n```\n\n'
                 b'## another-package 2\n\nLicense: MIT\n\n```\nFull other terms\n```\n')
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'NOTICES-Python.txt'
            file.write_bytes(before + header + after)
            license_report.correct_pyiqa_notice(file)
            changed = file.read_bytes()
        replacement = (b'## pyiqa 0.1.13\n\nLicense: ' + license_report.PYIQA_REVIEWED_SPDX.encode() +
                       b'\nBundled license evidence: pyiqa-bundled-licenses.txt (LICENSE and LICENSE-S-Lab)\n')
        self.assertEqual(changed, before + replacement + after)

    def test_unexpected_or_duplicate_component_fails_without_modifying_file(self):
        bad_inputs = [
            b'## pyiqa 0.1.14\n\nLicense: Apache-2.0\n',
            b'## pyiqa 0.1.13\n\nLicense: MIT\n',
            b'## pyiqa 0.1.13\r\n\nLicense: Apache-2.0\r\n',
            b'## other 1\n\nLicense: Apache-2.0\n',
            b'## pyiqa 0.1.13\n\nLicense: Apache-2.0\n' * 2,
            b'## pyiqa 0.1.13\n\nLicense: Apache-2.0\n## pyiqa 0.1.14\n',
        ]
        for original in bad_inputs:
            with self.subTest(original=original), tempfile.TemporaryDirectory() as directory:
                file = Path(directory) / 'NOTICES-Python.txt'
                file.write_bytes(original)
                with self.assertRaisesRegex(ValueError, 'Unexpected pyIQA'):
                    license_report.correct_pyiqa_notice(file)
                self.assertEqual(file.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
