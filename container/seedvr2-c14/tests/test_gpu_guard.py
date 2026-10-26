"""Admission/ownership unit tests; no GPU or subprocess is started."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpu_guard
import replay_common as common
import sr_only


class GuardContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.window = self.root / "window"
        self.window.mkdir()
        self.run = self.root / "parent"
        self.run.mkdir()
        (self.run / "manifest.json").write_text("{}", encoding="utf-8")
        self.output = self.root / "result"
        self.guard = self.root / "guard.json"
        self.nonce = "a" * 64
        self.value = {"schema": "seedvr2-c14-replay-guard/1", "phase": "color", "nonce": self.nonce,
                      "window": str(self.window), "output": str(self.output), "parent_run": str(self.run),
                      "supervisor": {"pid": 1234, "pgid": 1234, "start_ticks": 1},
                      "parent_manifest_sha256": common.sha256(self.run / "manifest.json")}
        self.guard.write_text(json.dumps(self.value), encoding="utf-8")
    def tearDown(self):
        self.temp.cleanup()
    def environment(self):
        return patch.dict(os.environ, {"SEEDVR2_C14_REPLAY_GUARD": str(self.guard), "SEEDVR2_C14_REPLAY_NONCE": self.nonce})
    def test_missing_guard_rejected_before_gpu_import(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "gpu_guard"):
            common.require_guard(self.window, self.output, "color")
    def test_live_guard_exact_binding(self):
        with self.environment(), patch.object(sr_only, "same_process", return_value=True):
            self.assertEqual(common.require_guard(self.window, self.output, "color")["nonce"], self.nonce)
    def test_dead_guard_rejected(self):
        with self.environment(), patch.object(sr_only, "same_process", return_value=False), self.assertRaises(ValueError):
            common.require_guard(self.window, self.output, "color")
    def test_guard_different_phase_rejected(self):
        with self.environment(), patch.object(sr_only, "same_process", return_value=True), self.assertRaises(ValueError):
            common.require_guard(self.window, self.output, "decode")
    def test_guard_different_output_rejected(self):
        with self.environment(), patch.object(sr_only, "same_process", return_value=True), self.assertRaises(ValueError):
            common.require_guard(self.window, self.root / "another", "color")
    def test_guard_changed_parent_rejected(self):
        (self.run / "manifest.json").write_text('{"edited":true}', encoding="utf-8")
        with self.environment(), patch.object(sr_only, "same_process", return_value=True), self.assertRaises(ValueError):
            common.require_guard(self.window, self.output, "color")
    def test_child_environment_drops_stale_distributed_and_capture(self):
        with patch.dict(os.environ, {"SEEDVR2_C14_PROFILE": "/old/profile", "WORLD_SIZE": "16", "TORCHELASTIC_RUN_ID": "old",
                                     "PYTHONPATH": "/old", "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}):
            env = gpu_guard.environment(self.root, self.nonce)
        for key in ("SEEDVR2_C14_PROFILE", "WORLD_SIZE", "TORCHELASTIC_RUN_ID", "PYTHONPATH"):
            self.assertNotIn(key, env)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1,2,3,4,5,6,7")
        self.assertEqual(env["SEEDVR2_C14_REPLAY_NONCE"], self.nonce)
    def test_decode_exact_supervised_command(self):
        command = gpu_guard.replay_command("decode", self.window, self.output, "/readonly/model", "/logs")
        self.assertIn("torch.distributed.run", command)
        self.assertIn("--nproc-per-node=8", command)
        self.assertIn("--max-restarts=0", command)
        self.assertIn(str(Path("/readonly/model") / "ema_vae_fp16.safetensors"), command)
        self.assertNotIn("serve", command)
    def test_color_does_not_start_model_server(self):
        command = gpu_guard.replay_command("color", self.window, self.output, "/readonly/model", "/logs")
        self.assertIn("cuda:0", command)
        self.assertNotIn("torch.distributed.run", command)
        self.assertNotIn("serve", command)
    def test_unknown_phase_rejected(self):
        with self.assertRaises(ValueError):
            gpu_guard.replay_command("h3", self.window, self.output, "/model", "/logs")
    def test_insufficient_replay_space_rejected(self):
        class Disk: free = 1
        with patch.object(common.shutil, "disk_usage", return_value=Disk()), self.assertRaises(RuntimeError):
            common.begin_output(self.output, "A1-replay-color", {}, {})
        self.assertFalse(self.output.exists())
    def test_mismatched_rank_owner_not_signalled(self):
        folder = self.root / "rank-owners"
        folder.mkdir()
        (folder / "rank-0.json").write_text(json.dumps({"schema": "seedvr2-c14-replay-rank/1", "phase": "color",
            "nonce": "b" * 64, "owner": self.value["supervisor"]}), encoding="utf-8")
        with patch.object(gpu_guard, "cleanup_engine"), patch.object(gpu_guard.os, "killpg", create=True) as signal_call:
            with self.assertRaises(ValueError): gpu_guard.cleanup_replay(self.root, None, None)
            signal_call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
