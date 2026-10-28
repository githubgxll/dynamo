import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('native_ops', HERE/'ops.py')
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)


class SubmissionTests(unittest.TestCase):
    def test_native_marker(self):
        self.assertTrue(ops.triggers_native('build(sr): official [sr-native]'))
        self.assertFalse(ops.triggers_native('build [h3-hsdp-offload]'))

    def test_github_skip_markers_override_recipe(self):
        for message in ['[sr-native] [skip ci]', '[sr-native]\nskip-checks: true']:
            self.assertFalse(ops.triggers_native(message))

    def test_scope_preserves_other_work(self):
        ops.check_scope(['container/sr-native/seedvr2/Dockerfile', 'delivery/sr-native/README.md', '.github/workflows/dingo-router-ci.yml'])
        with self.assertRaises(ValueError):
            ops.check_scope(['container/h3-hsdp-offload/runtime.py'])

    def test_source_lock_separates_environments(self):
        lock = ops.load_lock()
        self.assertEqual(lock['remote_branch'], 'Dingorouter-H3-vllm-omni-jirx')
        self.assertEqual(set(lock['models']), {'seedvr2', 'flashvsr-plus'})
        for model in lock['models'].values():
            self.assertRegex(model['base_image'], r'^nvidia/cuda@sha256:[0-9a-f]{64}$')


if __name__ == '__main__':
    unittest.main()
