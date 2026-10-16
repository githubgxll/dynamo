from pathlib import Path
import copy
import json
import sys
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import check_profiles
import launch


class ProfilesTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((HERE / 'profiles.json').read_bytes())

    def profile(self, key):
        return next(p for p in self.data['profiles'] if p['id'] == key)

    def test_six_profiles_and_four_assets(self):
        result = check_profiles.validate(self.data)
        self.assertEqual(result['status'], 'PASS_LOCAL_PROFILES_ONLY')
        self.assertEqual(len(result['profiles']), 6)
        self.assertFalse(result['gpu_validated'])

    def test_ref8_shift6_rejected(self):
        self.profile('turbo-ref8')['video_shift'] = 6
        with self.assertRaisesRegex(ValueError, 'contract drift'):
            check_profiles.validate(self.data)

    def test_base_cannot_activate_lora(self):
        self.profile('base-fl50')['request_contract']['lora'] = {'name': 'unexpected'}
        with self.assertRaisesRegex(ValueError, 'Base must have no adapter'):
            check_profiles.validate(self.data)

    def test_fasth3_cannot_take_dynamic_lora(self):
        self.profile('fasth3-dense4')['request_contract']['lora'] = {'name': 'unexpected'}
        with self.assertRaisesRegex(ValueError, 'fusion cannot use request'):
            check_profiles.validate(self.data)

    def test_fasth3_base_revision_cannot_float(self):
        self.profile('fasth3-dense4')['base_revision'] = 'main'
        with self.assertRaisesRegex(ValueError, 'teacher revision'):
            check_profiles.validate(self.data)

    def test_fasth3_cannot_serve_fl2va(self):
        self.profile('fasth3-dense4')['allowed_tasks'].append('fl2va')
        with self.assertRaisesRegex(ValueError, 'T2VA only'):
            check_profiles.validate(self.data)

    def test_request_turbo_must_match_asset(self):
        self.profile('turbo-fl8')['request_contract']['lora']['path'] = '/elsewhere'
        with self.assertRaisesRegex(ValueError, 'activate exact LoRA'):
            check_profiles.validate(self.data)

    def test_command_is_native_no_shell_and_model_override_keeps_contract(self):
        p = self.profile('fasth3-dense4')
        args = launch.command(p, self.data, '/readonly/pinned/FL2VA')
        self.assertEqual(args[:3], ['vllm-omni', 'serve', '/readonly/pinned/FL2VA'])
        self.assertEqual(args[args.index('--lora-path') + 1], p['adapter_path'])
        self.assertNotIn('--lora-backend', args)
        self.assertNotIn('--cpu-offload', args)

    def test_base_command_no_lora(self):
        args = launch.command(self.profile('base-ref50'), self.data)
        self.assertFalse(any('lora' in x for x in args))
        self.assertEqual(args[args.index('--task-type') + 1], 'ref2va')

    def test_no_extra_profile(self):
        self.data['profiles'].append(copy.deepcopy(self.data['profiles'][0]))
        with self.assertRaisesRegex(ValueError, 'six unique'):
            check_profiles.validate(self.data)

    def test_relative_model_override_rejected_before_execution(self):
        with self.assertRaisesRegex(ValueError, 'absolute Linux path'):
            launch.command(self.profile('fasth3-dense4'), self.data, 'relative/FL2VA')


if __name__ == '__main__':
    unittest.main()
