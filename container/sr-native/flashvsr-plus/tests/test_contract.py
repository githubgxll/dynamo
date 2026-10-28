"""Local tests for admission and fixed recipe; no Torch/GPU claim."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime_common as common
import run


class Admission(unittest.TestCase):
    def test_target_preserves_full_frame(self):
        self.assertIsNone(common.validate_geometry(1280,768,1920,1152)['crop'])

    def test_wrong_aspect_rejected(self):
        with self.assertRaises(ValueError): common.validate_geometry(1280,768,1920,1080)

    def test_previous_diagnostic_canvas_rejected(self):
        with self.assertRaises(ValueError): common.validate_geometry(1280,768,1840,1104)

    def test_nonpositive_dimensions_rejected(self):
        with self.assertRaises(ValueError): common.validate_geometry(0,768,1920,1152)

    def test_frozen_timeline(self):
        common.validate_timeline({'frames':17,'fps':'24','timestamps':[i/24 for i in range(17)]},17,24)

    def test_partial_decode_rejected(self):
        with self.assertRaises(ValueError):
            common.validate_timeline({'frames':17,'fps':'24','timestamps':[i/24 for i in range(16)]},17,24)

    def test_vfr_rejected(self):
        with self.assertRaises(ValueError):
            common.validate_timeline({'frames':3,'fps':'24','timestamps':[0,1/24,0.2]},3,24)

    def test_fps_change_rejected(self):
        with self.assertRaises(ValueError):
            common.validate_timeline({'frames':17,'fps':'30','timestamps':[i/30 for i in range(17)]},17,24)

    def test_digest_not_short(self):
        with self.assertRaises(ValueError): common.require_digest('abc123')

    def test_kernel_only_needs_no_input(self):
        args=run.parse_args(['--kernel-only','--output-dir','unused'])
        self.assertTrue(args.kernel_only)

    def test_actual_inference_requires_frozen_file(self):
        with self.assertRaises(SystemExit): run.parse_args(['--output-dir','unused'])

    def test_source_hash_gate_detects_mutation(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            (root/'module.py').write_text('before',encoding='utf-8')
            lock={'revision':'fixed','files':{'module.py':common.sha256(root/'module.py')}}
            (root/'upstream-files.json').write_text(json.dumps(lock),encoding='utf-8')
            with patch.object(common,'HERE',root):
                common.verify_source(root)
                (root/'module.py').write_text('after',encoding='utf-8')
                with self.assertRaises(RuntimeError):common.verify_source(root)

    def test_asset_wrong_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); (root/'weight').write_bytes(b'abc')
            lock={'assets':[{'filename':'weight','size':3,'sha256':'0'*64}]}
            (root/'assets.lock.json').write_text(json.dumps(lock),encoding='utf-8')
            with patch.object(common,'HERE',root):
                with self.assertRaises(RuntimeError):common.verify_assets(root)

    def test_source_and_submodule_pins(self):
        lock=json.loads((common.HERE/'source.lock.json').read_text())
        self.assertEqual(len(lock['sources']),3)
        self.assertEqual(lock['sources'][2]['revision'],'a75b4ac483166189a45290783cb0a18af5ff0ea5')
        for item in lock['sources']:common.require_digest(item['sha256'])

    def test_only_required_base_weights(self):
        lock=json.loads((common.HERE/'assets.lock.json').read_text())
        self.assertEqual({i['filename'] for i in lock['assets']},
                         {'ckpts/base/dit.safetensors','ckpts/base/lq_proj_in.safetensors','ckpts/TCDecoder.ckpt'})


if __name__=='__main__':unittest.main()
