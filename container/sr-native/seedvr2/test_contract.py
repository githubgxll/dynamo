import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('seedvr_native_run', HERE / 'run.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class Contract(unittest.TestCase):
    def test_first_cohort_has_no_temporal_padding(self):
        for variant in ('3b', '7b'):
            for width, height in ((1840, 1104), (1920, 1152)):
                runtime.validate_geometry(width, height, 17, 4, variant)

    def test_65_frame_followup_is_allowed(self):
        runtime.validate_geometry(1920, 1152, 65, 4, '7b')

    def test_sp8_is_not_shared_reference_topology(self):
        with self.assertRaises(ValueError):
            runtime.validate_geometry(1920, 1152, 17, 8, '3b')

    def test_no_hidden_padding(self):
        for frames in (0, 1, 16, 18, 32):
            with self.assertRaises(ValueError):
                runtime.validate_geometry(1920, 1152, frames, 4, '7b')

    def test_invalid_canvas(self):
        for width, height in ((1800, 1080), (0, 16), (-16, 16)):
            with self.assertRaises(ValueError):
                runtime.validate_geometry(width, height, 17, 4, '3b')

    def test_standard_seven_b_assets(self):
        names = {x['filename'] for x in runtime.selected_assets('7b')}
        self.assertEqual(names, {'seedvr2_ema_7b.pth', 'ema_vae.pth', 'pos_emb.pt', 'neg_emb.pt'})
        self.assertFalse(any('sharp' in n for n in names))

    def test_official_assets_are_frozen(self):
        rows = json.loads((HERE / 'assets.lock.json').read_text())['assets']
        self.assertEqual(len(rows), 5)
        for row in rows:
            self.assertEqual(len(row['sha256']), 64)
            self.assertEqual(len(row['revision']), 40)
            self.assertIn(row['revision'], row['url'])
            self.assertGreater(row['size'], 0)

    def test_missing_weights_fail_before_loading(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            with self.assertRaises(RuntimeError):
                runtime.verify_assets(Path(d), '3b')

    def test_wrong_original_bytes_rejected(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            (Path(d) / 'model.pth').write_bytes(b'changed')
            row = {'filename': 'model.pth', 'size': 7, 'sha256': '0' * 64, 'revision': 'a' * 40}
            with patch.object(runtime, 'selected_assets', return_value=[row]):
                with self.assertRaisesRegex(RuntimeError, 'SHA256 differs'):
                    runtime.verify_assets(Path(d), '3b')

    def test_source_contract_covers_official_full_chain(self):
        rows = json.loads((HERE / 'source-tree.json').read_text())['files']
        for path in ('projects/inference_seedvr2_3b.py', 'projects/inference_seedvr2_7b.py',
                     'projects/video_diffusion_sr/infer.py', 'configs_3b/main.yaml',
                     'configs_7b/main.yaml', 'common/distributed/basic.py'):
            self.assertEqual(len(rows[path]), 64)

    def test_flash_sdist_source_and_no_abi_wheel(self):
        lock = json.loads((HERE / 'sources.lock.json').read_text())['sources']
        self.assertTrue(next(x for x in lock if x['name'] == 'flash-attn')['cutlass_in_sdist'])
        self.assertFalse(any(x['url'].endswith('.whl') for x in lock))


if __name__ == '__main__':
    unittest.main()
