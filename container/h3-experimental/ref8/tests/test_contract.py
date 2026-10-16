"""Meaningful source, scope and failure-path regressions; no fake GPU acceptance."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import apply_patch as repair
import check_contract
import native_contract
import schedule_gate


class Ref8ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = (ROOT / 'fixtures/lora.py').read_bytes()
        cls.pipeline = (ROOT / 'fixtures/pipeline_minimax_h3.py').read_bytes()
        cls.repaired = repair.transform(cls.original)

    def test_independent_031_version_and_inventory(self):
        lock = repair.load_lock()
        self.assertEqual(lock['omni_version'], '0.31.0rc1')
        self.assertEqual(lock['omni_commit'], '6dd0d1f9310f7598b773c796b434c8000c2816ec')
        self.assertEqual(set(lock['fixtures']), set(repair.SOURCE_MAP))
        self.assertEqual(len(lock['fixtures']), 6)
        self.assertNotIn('parent_image', lock)

    def test_patch_accepts_only_exact_original_and_one_application(self):
        self.assertEqual(repair.sha(self.repaired), repair.AFTER)
        for changed in (self.original + b'\n', self.repaired,
                        self.original.replace(b'_TURBO_AUDIO_SHIFT = 3.0', b'_TURBO_AUDIO_SHIFT = 6.0')):
            with self.subTest(sha=repair.sha(changed)), self.assertRaises(ValueError):
                repair.transform(changed)

    def test_only_ref8_exact_filename_changes(self):
        before = native_contract.native_functions(self.original, self.pipeline)
        after = native_contract.native_functions(self.repaired, self.pipeline)
        for filename in [repair.ARTIFACT,
                         'minimax_h3_ref2v_turbo_8step_v1.1_768p_bf16.safetensors',
                         'minimax_h3_ref2v_turbo_4step_v1.0_768p_bf16.safetensors',
                         'minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors']:
            old, new = before['parse_turbo_filename'](filename), after['parse_turbo_filename'](filename)
            self.assertEqual(new.video_shift, 12 if filename == repair.ARTIFACT else old.video_shift)
            self.assertEqual(new.audio_shift, old.audio_shift)
            self.assertEqual(new.alpha, old.alpha)
            self.assertEqual(new.denoise_steps, old.denoise_steps)

    def test_actual_validation_rejects_nine_wrong_task_wrong_audio(self):
        report = check_contract.check_native(self.repaired, self.pipeline, ref_shift=12)
        self.assertTrue(all(row['passed'] for row in report['checks']))
        labels = [row['check'] for row in report['checks'] if not row['accepted']]
        self.assertTrue(any('reject-steps9' in label for label in labels))
        self.assertTrue(any('wrong-task' in label for label in labels))
        self.assertTrue(any('reject-audio-task' in label for label in labels))
        self.assertTrue(any('audio6' in label for label in labels))

    def test_031_first_pass_forwarding_and_selection(self):
        self.assertEqual(schedule_gate.check_static_path(self.pipeline)['status'], 'PASS_STATIC_PATH_ONLY')
        self.assertEqual(check_contract.check_schedule_selection(self.pipeline)['num_inference_steps'], 8)

    def test_reject_shift_not_forwarded(self):
        broken = self.pipeline.replace(b'"video_shift": float(extra.get("flow_shift", self.default_video_shift)),',
                                       b'"video_shift": self.default_video_shift,', 1)
        self.assertNotEqual(broken, self.pipeline)
        with self.assertRaisesRegex(ValueError, 'Request field'):
            schedule_gate.check_static_path(broken)

    def test_reject_video_audio_wiring_swap(self):
        for before, after in [(b'shift_scale=video_shift,', b'shift_scale=audio_shift,'),
                              (b'shift_scale=audio_shift,', b'shift_scale=video_shift,')]:
            with self.subTest(before=before), self.assertRaisesRegex(ValueError, 'Sigma call'):
                schedule_gate.check_static_path(self.pipeline.replace(before, after, 1))

    def test_numeric_gate_rejects_shift_ignored(self):
        def bad(*, num_steps, shift_scale, base_schedule):
            return schedule_gate.reference_sigmas(num_steps, 6 if shift_scale == 12 else shift_scale)
        with self.assertRaisesRegex(ValueError, 'author-grid'):
            schedule_gate.check_numbers(bad)

    def test_numeric_gate_rejects_n_minus_one_boundaries(self):
        def bad(*, num_steps, shift_scale, base_schedule):
            return schedule_gate.reference_sigmas(num_steps, shift_scale)[:-1]
        with self.assertRaisesRegex(ValueError, 'boundary count'):
            schedule_gate.check_numbers(bad)

    def test_installed_actions_fail_on_windows(self):
        with patch.object(sys, 'platform', 'win32'):
            with self.assertRaisesRegex(ValueError, 'Linux image construction'):
                repair.apply_installed({})
            with self.assertRaisesRegex(ValueError, 'Linux non-root'):
                check_contract.installed_check({})

    def test_fixture_and_tool_tampering_fail_closed(self):
        with tempfile.TemporaryDirectory(prefix='contract-test-') as directory:
            root = Path(directory)
            (root / 'fixtures').mkdir()
            for name in repair.TOOLS | {'patch.lock.json'}:
                (root / name).write_bytes((ROOT / name).read_bytes())
            for name in repair.SOURCE_MAP:
                (root / 'fixtures' / name).write_bytes((ROOT / 'fixtures' / name).read_bytes())
            with patch.object(repair, 'HERE', root):
                repair.load_lock()
                selected = root / 'fixtures/sigma_schedule.py'
                original = selected.read_bytes()
                selected.write_bytes(original + b'\n')
                with self.assertRaisesRegex(ValueError, 'Changed fixture'):
                    repair.load_lock()
                selected.write_bytes(original)
                with (root / 'native_contract.py').open('ab') as stream:
                    stream.write(b'\n')
                with self.assertRaisesRegex(ValueError, 'Changed tool'):
                    repair.load_lock()

    def test_local_report_does_not_claim_installed_torch(self):
        result = check_contract.local_check(repair.load_lock())
        self.assertFalse(result['native_cpu_torch_executed'])
        self.assertFalse(result['gpu_validated'])


if __name__ == '__main__':
    unittest.main()
