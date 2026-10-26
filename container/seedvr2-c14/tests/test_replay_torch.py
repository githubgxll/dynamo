"""Real CPU Torch/PyAV fixtures. Mandatory in image CI; no CUDA/SP8 quality claim."""
from __future__ import annotations
import os
from pathlib import Path
import sys
import tempfile
import subprocess
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import replay_common as common
import replay_chain as chain
import replay_encoding as encoding
import replay_decode as decode
import gpu_guard
import sr_only

try:
    import torch
    import av
    import numpy as np
    from safetensors.torch import save_file
    from vllm_omni.diffusion.models.seedvr2.pipeline_seedvr2 import correct_video_color
    from vllm_omni.diffusion.models.seedvr2.vae import SeedVR2VAE
    AVAILABLE = True
except ImportError:
    if os.environ.get("C14_REQUIRE_RUNTIME_TESTS") == "1":
        raise
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, "Actual inherited Torch/Omni/PyAV is required in image CI")
class ActualReplayFixtures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        torch.set_num_threads(1)
    def tearDown(self):
        self.temp.cleanup()
    def test_equal_rgb_lab_is_finite_and_repeatable(self):
        source = torch.full((1, 3, 1, 32, 40), .25)
        original = source.clone()
        a = correct_video_color(source, source, "lab", .8)
        b = correct_video_color(source, source, "lab", .8)
        self.assertTrue(torch.isfinite(a).all())
        self.assertTrue(torch.equal(source, original))
        self.assertTrue(torch.equal(a, b))
    def test_ties_and_near_ties_not_declared_quality_pass(self):
        source = torch.full((1, 3, 1, 32, 40), .1)
        reference = source.clone()
        source[..., ::2] += 1e-5
        reference[..., 1::2] += .05
        outputs = [correct_video_color(source, reference, method, .8) for method in ("none", "wavelet", "lab")]
        for value in outputs:
            self.assertEqual(value.shape, source.shape)
            self.assertTrue(torch.isfinite(value).all())
        self.assertFalse(common.compare(outputs[0], outputs[2], {"absolute": 0., "relative": 0.})["pass"])
    def test_tensor_roundtrip_restores_actual_noncontiguous_stride(self):
        value = torch.arange(120, dtype=torch.float16).reshape(2, 3, 4, 5).permute(3, 0, 1, 2).unsqueeze(0)
        path = self.root / "tensor.safetensors"
        save_file({"tensor": value.contiguous()}, str(path))
        meta = {"shape": list(value.shape), "stride": list(value.stride()), "dtype": str(value.dtype)}
        with patch.object(common, "tensor_metadata", return_value=(path, meta)):
            loaded, _ = common.load_tensor(self.root, "tensor", restore_stride=True)
        self.assertEqual(loaded.stride(), value.stride())
        self.assertTrue(torch.equal(loaded, value))
    def test_baseline_gate_catches_single_pixel_failure(self):
        expected = torch.zeros((1, 3, 1, 8, 10))
        actual = expected.clone()
        actual[0, 0, 0, 3, 3] = 1
        result = common.compare(actual, expected, {"absolute": 0., "relative": 0.})
        self.assertFalse(result["pass"])
        self.assertEqual(result["mismatched_values"], 1)
    def test_nonfinite_baseline_rejected(self):
        value = torch.tensor([float("nan")])
        with self.assertRaises(ValueError): common.compare(value, value, {"absolute": 0, "relative": 0})
    def test_display_is_full_frame_then_roi_no_autocontrast(self):
        value = torch.full((1, 3, 1, 16, 20), .25)
        record = common.export_rgb_frame(value, self.root, "fixture", 162)
        from PIL import Image
        with Image.open(self.root / record["full"]) as pic:
            self.assertTrue(np.all(np.asarray(pic) == 64))
        with Image.open(self.root / record["roi"]) as pic:
            self.assertTrue(np.all(np.asarray(pic) == 64))
    def test_native_vae_new_context_between_complete_calls(self):
        # Tiny real architecture fixture; not a substitute for FP16/SP8/real-weight replay.
        torch.manual_seed(7)
        model = SeedVR2VAE(channels=(4, 4, 4, 4), groups=1).eval()
        model.use_tiling = True
        latent = torch.arange(1 * 16 * 5 * 4 * 4, dtype=torch.float32).reshape(1, 16, 5, 4, 4) / 1000
        with torch.inference_mode():
            first = model.decode(latent, chunk_size=2)
            whole = model.decode(latent, chunk_size=5)
            again = model.decode(latent, chunk_size=2)
        self.assertEqual(tuple(first.shape), (1, 3, 17, 32, 32))
        self.assertTrue(torch.equal(first, again))
        self.assertTrue(torch.isfinite(whole).all())
        self.assertEqual(whole.shape, first.shape)
    def test_native_meta_constructor_does_not_draw_rng(self):
        before = torch.get_rng_state().clone()
        with torch.device("meta"):
            model = SeedVR2VAE(channels=(4, 4, 4, 4), groups=1)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(next(model.parameters()).device.type, "meta")
    def test_strict_vae_weight_loading_from_bare_keys(self):
        from vllm_omni.diffusion.models.seedvr2 import vae
        original_class = vae.SeedVR2VAE
        source = original_class(channels=(4, 4, 4, 4), groups=1).half()
        path = self.root / "vae.safetensors"
        save_file({name: value.contiguous() for name, value in source.state_dict().items()}, str(path))
        def fixture():
            model = original_class(channels=(4, 4, 4, 4), groups=1)
            model.set_parallel_size = lambda *args, **kwargs: None  # CPU fixture only; no SP claim.
            return model
        with patch.object(vae, "SeedVR2VAE", side_effect=fixture):
            loaded = decode.load_vae(path, "cpu")
        for name, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, loaded.state_dict()[name]))
        self.assertTrue(loaded.use_tiling)
    def test_vae_loading_rejects_missing_weight(self):
        from vllm_omni.diffusion.models.seedvr2 import vae
        original_class = vae.SeedVR2VAE
        source = original_class(channels=(4, 4, 4, 4), groups=1).half().state_dict()
        source.pop(next(iter(source)))
        path = self.root / "missing.safetensors"
        save_file({name: value.contiguous() for name, value in source.items()}, str(path))
        with patch.object(vae, "SeedVR2VAE", side_effect=lambda: original_class(channels=(4, 4, 4, 4), groups=1)):
            with self.assertRaises(RuntimeError): decode.load_vae(path, "cpu")
    def test_vae_loading_rejects_float32_weights(self):
        path = self.root / "wrong-dtype.safetensors"
        save_file({"decoder.conv_in.weight": torch.zeros(1)}, str(path))
        with self.assertRaises(ValueError): decode.load_vae(path, "cpu")
    def test_ffv1_readback_uses_exact_rgb(self):
        frames = np.arange(4 * 16 * 20 * 3, dtype=np.uint8).reshape(4, 16, 20, 3)
        video = self.root / "roundtrip.mkv"
        encoding.write_lossless_rgb(video, frames)
        result = chain.lossless_readback(video, iter(frames), count=4)
        self.assertTrue(result["pass"])
        self.assertTrue(chain.compare_video(video, video, count=4)["pass"])
    def test_streaming_compare_rejects_missing_frames(self):
        frames = np.zeros((3, 16, 20, 3), dtype=np.uint8)
        video = self.root / "short.mkv"
        encoding.write_lossless_rgb(video, frames)
        with self.assertRaises(ValueError): chain.lossless_readback(video, iter(frames), count=4)
    def test_nut_lossless_intermediate_preserves_exact_24fps(self):
        frames = np.arange(4 * 16 * 20 * 3, dtype=np.uint8).reshape(4, 16, 20, 3)
        video = self.root / "exact-timebase.nut"
        encoding.write_lossless_rgb(video, frames, container_format="nut")
        self.assertTrue(chain.lossless_readback(video, iter(frames), count=4, exact_timeline=True)["pass"])
    def test_guard_reaps_actual_separate_rank_session_cpu_only(self):
        # torchrun may own separate rank sessions. Test with harmless CPU sleepers.
        launcher = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        rank = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        try:
            owner = sr_only.proc(launcher.pid)
            rank_owner = sr_only.proc(rank.pid)
            self.assertEqual(rank_owner["pgid"], rank.pid)
            nonce = "a" * 64
            common.write_json(self.root / "guard.json", {"nonce": nonce, "phase": "decode"})
            directory = self.root / "rank-owners"
            directory.mkdir()
            common.write_json(directory / "rank-0.json", {"schema": "seedvr2-c14-replay-rank/1", "phase": "decode",
                                                       "nonce": nonce, "owner": rank_owner})
            gpu_guard.cleanup_replay(self.root, launcher, owner)
            self.assertIsNotNone(launcher.poll())
            rank.wait(timeout=5)
            self.assertFalse(sr_only.members(rank_owner["pgid"]))
        finally:
            for process in (launcher, rank):
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
    def test_decode_frame_budget(self):
        frames = np.zeros((4, 16, 20, 3), dtype=np.uint8)
        video = self.root / "bounded.mkv"
        encoding.write_lossless_rgb(video, frames)
        with self.assertRaises(ValueError): encoding.decode_rgb(video, max_frames=3)


if __name__ == "__main__":
    unittest.main()
