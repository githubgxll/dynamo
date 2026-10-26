"""Stdlib checks for fail-closed replay admission and distributed local phases."""
from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import replay_common as common
import replay_decode as decode


class ReplayContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.window = {"schema": "seedvr2-c14-window/1", "window_id": "job:156", "start": 156,
                       "valid_frames": 17, "padded_frames": 17, "width": 1840, "height": 1104,
                       "target_frames": 362, "fps": 24, "seed": 7723, "color_correction_method": "lab",
                       "luminance_weight": .8, "expected_world_size": 8}
        self.config = {"vae_use_tiling": True, "vae_parallel_mode": "spatial_shard_height",
                       "sp_world_size": 8, "sp_ranks": list(range(8)), "vae_weight_dtype": "torch.float16",
                       "autocast_enabled": False, "autocast_dtype": "torch.float16", "cudnn_benchmark": False,
                       "cudnn_allow_tf32": True, "matmul_allow_tf32": False}
        self.tensors = {"rgb": {"shape": [1, 3, 1, 2, 4], "dtype": "torch.float32", "stride": [24, 8, 8, 4, 1]}}
        self.write("window.json", self.window)
        for rank in range(8):
            self.write(f"rank-{rank}.json", {"status": "complete", "rank": rank, "world_size": 8,
                       "window_id": self.window["window_id"], "actual_config": self.config, "tensors": self.tensors})
    def tearDown(self):
        self.temp.cleanup()
    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value), encoding="utf-8")
    def test_complete_rank_context(self):
        self.assertEqual(common.validate_decode_context(common.load_window(self.root)), self.config)
    def test_rank_wrong_window_rejected(self):
        value = common.read_json(self.root / "rank-4.json")
        value["window_id"] = "another:156"
        self.write("rank-4.json", value)
        with self.assertRaises(ValueError): common.load_window(self.root)
    def test_rank_missing_or_partial_rejected(self):
        (self.root / "rank-7.json").unlink()
        with self.assertRaises(FileNotFoundError): common.load_window(self.root)
    def test_rank_layout_difference_rejected(self):
        value = common.read_json(self.root / "rank-7.json")
        value["tensors"]["rgb"]["dtype"] = "torch.float16"
        self.write("rank-7.json", value)
        with self.assertRaises(ValueError): common.load_window(self.root)
    def test_rank_numeric_difference_rejected(self):
        value = common.read_json(self.root / "rank-3.json")
        value["actual_config"]["autocast_enabled"] = True
        self.write("rank-3.json", value)
        with self.assertRaises(ValueError): common.validate_decode_context(common.load_window(self.root))
    def test_geometry_not_inferred(self):
        self.window["width"] = 1800
        self.write("window.json", self.window)
        with self.assertRaises(ValueError): common.load_window(self.root)
    def test_no_tolerance_expansion_by_default(self):
        self.assertEqual(common.tolerance(self.window, None)["absolute"], 0)
    def test_unsealed_tolerance_rejected(self):
        self.write("policy.json", {"absolute": 1, "relative": 1})
        with self.assertRaises(ValueError): common.tolerance(self.window, self.root / "policy.json")
    def test_declared_tolerance_requires_reason_and_hash(self):
        self.write("policy.json", {"schema": "seedvr2-c14-tolerance/1", "absolute": 1e-6, "relative": 0,
                                   "reason": "Measured repeated same-backend baseline variation"})
        self.window["tolerance_policy_sha256"] = common.sha256(self.root / "policy.json")
        self.assertEqual(common.tolerance(self.window, self.root / "policy.json")["absolute"], 1e-6)
    def test_overlapping_or_huge_stride_rejected(self):
        for shape, stride in (([2, 3], [1, 1]), ([3, 4], [1000, 1]), ([3, 4], [4, -1])):
            with self.assertRaises(ValueError): common.validate_stride(shape, stride)
        common.validate_stride([1, 3, 5, 8, 9], [1080, 1, 216, 27, 3])
    def test_output_exclusive(self):
        window = common.load_window(self.root)
        target = self.root / "replay"
        common.begin_output(target, "unit", window, {})
        with self.assertRaises(FileExistsError): common.begin_output(target, "unit", window, {})
    def test_failed_evidence_is_not_complete(self):
        target = self.root / "failed"
        target.mkdir()
        common.save_failure(target, ValueError("bad input"))
        self.assertFalse((target / "COMPLETE.json").exists())
        self.assertEqual(common.read_json(target / "FAILED.json")["status"], "FAIL_NO_CANDIDATE_CONCLUSION")
    def test_escape_rejected(self):
        with self.assertRaises(ValueError): common.contained(self.root, "../other")
    def test_synchronized_local_failure_stops_every_peer(self):
        class Group:
            def all_gather_object(self, output, error, group):
                output[:] = [None, None, "load: disk failure", None, error, None, None, None]
        with self.assertRaisesRegex(RuntimeError, "rank2"):
            decode.synchronized_local(Group(), "gloo", "load", lambda: 42)
    def test_synchronized_own_failure_reported(self):
        class Group:
            def all_gather_object(self, output, error, group): output[:] = [error] + [None] * 7
        with self.assertRaisesRegex(RuntimeError, "ValueError: broken"):
            decode.synchronized_local(Group(), "gloo", "write", lambda: (_ for _ in ()).throw(ValueError("broken")))
    def test_synchronized_success_retains_result(self):
        class Group:
            def all_gather_object(self, output, error, group): output[:] = [None] * 8
        self.assertEqual(decode.synchronized_local(Group(), "gloo", "load", lambda: 42), 42)


if __name__ == "__main__":
    unittest.main()
