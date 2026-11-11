#!/usr/bin/env python3
"""Exercise installed H3-Image contracts with small CPU fixtures."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


MODULE_NAMES = {
    "h3i": "vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3_image",
    "image_vae": "vllm_omni.diffusion.models.minimax_h3.image_vae",
    "transformer": "vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer",
}
RELEASE_PDD = {
    "num_inference_steps": 6,
    "sigmas": [1.0, 0.9675236940383911, 0.8875485062599182,
               0.75, 0.5327739119529724, 0.23200887441635132, 0.0],
}


class InstalledH3ImageChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch = importlib.import_module("torch")
        cls.Image = importlib.import_module("PIL.Image")
        cls.client_error = importlib.import_module("vllm_omni.errors").OmniClientError
        for name, module_name in MODULE_NAMES.items():
            setattr(cls, name, importlib.import_module(module_name))

    def pipeline(self):
        pipeline = self.h3i.MiniMaxH3ImagePipeline.__new__(self.h3i.MiniMaxH3ImagePipeline)
        self.torch.nn.Module.__init__(pipeline)
        pipeline.num_steps = 6
        pipeline.supported_tasks = frozenset({"t2i", "edit"})
        pipeline.min_generation_area = 2048 * 2048
        return pipeline

    def request(self, prompt="A red bicycle", **changes):
        sampling = SimpleNamespace(
            num_inference_steps=6, guidance_scale_provided=True,
            guidance_scale=1.0, num_outputs_per_prompt=1,
            width=2048, height=2048, width_not_provided=False,
            height_not_provided=False, extra_args={}, generator_device="cpu",
            seed=42, output_type="pil",
        )
        for name, value in changes.items():
            setattr(sampling, name, value)
        return SimpleNamespace(
            prompts=[prompt], sampling_params=sampling, is_dummy_run=lambda: False,
        )

    def test_local_encoder_resolution_never_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            release = Path(directory) / "release"
            base = Path(directory) / "base"
            (release / "text_encoder").mkdir(parents=True)
            (base / "text_encoder").mkdir(parents=True)
            index = {"text_encoder": ["transformers", "Qwen3VLForConditionalGeneration", {
                "pretrained_model_name_or_path": "MiniMaxAI/MiniMax-H3",
                "subfolder": "text_encoder",
            }]}
            with patch.object(self.h3i, "download_weights_from_hf_specific",
                              side_effect=AssertionError("unexpected model download")) as download:
                pipeline = self.h3i.MiniMaxH3ImagePipeline
                self.assertEqual(pipeline._resolve_model_root(str(release), None), release)
                self.assertEqual(pipeline._resolve_text_encoder_root(index, release), str(release))
                (release / "text_encoder").rmdir()
                index["text_encoder"][2]["pretrained_model_name_or_path"] = str(base)
                self.assertEqual(pipeline._resolve_text_encoder_root(index, release), str(base))
                index["text_encoder"][2]["subfolder"] = "wrong_encoder"
                with self.assertRaises(ValueError):
                    pipeline._resolve_text_encoder_root(index, release)
                download.assert_not_called()

    def test_six_step_sigma_grid_and_fp32_round_trip(self):
        grid = self.h3i.minimax_h3_image_pdd_grid(RELEASE_PDD)
        sigmas = self.h3i.minimax_h3_image_run_sigmas(grid)
        self.assertEqual(len(sigmas), 7)
        self.assertEqual((sigmas[0], sigmas[-1]), (1.0, 0.0))
        self.assertTrue(all(left > right for left, right in zip(sigmas, sigmas[1:])))
        timesteps = self.torch.tensor([1.0 - sigma for sigma in sigmas[:-1]], dtype=self.torch.float32)
        self.assertEqual((1.0 - timesteps).tolist(), sigmas[:-1])
        for bad in ([1.0, 0.5], [1.0, 1.0, 0.0], [0.9, 0.5, 0.0]):
            with self.subTest(sigmas=bad), self.assertRaises(ValueError):
                self.h3i.minimax_h3_image_pdd_grid({"num_inference_steps": 2, "sigmas": bad})

    def test_generation_and_edit_requests_use_release_contract(self):
        pipeline = self.pipeline()
        for steps in (None, 6):
            result = pipeline._parse_request(self.request(num_inference_steps=steps))
            self.assertEqual(result["steps"], 6)
            self.assertEqual(result["canvas"], (2048, 2048))
            self.assertEqual(result["generator_device"], "cpu")
        image = self.Image.new("RGB", (48, 64), (20, 40, 60))
        result = pipeline._parse_request(self.request({
            "prompt": "Change the bicycle to blue", "multi_modal_data": {"image": image},
        }))
        self.assertEqual(len(result["references"]), 1)
        self.assertEqual(result["references"][0].size, (48, 64))

    def test_invalid_request_parameters_are_rejected(self):
        pipeline = self.pipeline()
        for changes in (
            {"num_inference_steps": 4}, {"num_inference_steps": 8},
            {"guidance_scale": 2.0}, {"num_outputs_per_prompt": 2},
        ):
            with self.subTest(changes=changes), self.assertRaises(self.client_error):
                pipeline._parse_request(self.request(**changes))
        with self.assertRaises(self.client_error):
            pipeline._parse_request(self.request({"prompt": "a bicycle", "negative_prompt": "red"}))
        request = self.request()
        request.prompts.append("another image")
        with self.assertRaises(self.client_error):
            pipeline._parse_request(request)

    def test_generation_canvas_preserves_explicit_target(self):
        canvas = self.h3i.minimax_h3_image_generation_canvas
        self.assertEqual(canvas(1024, 1024, 4194304), (2048, 2048))
        self.assertEqual(canvas(2048, 2048, 4194304), (2048, 2048))
        self.assertEqual(canvas(4096, 4096, 4194304), (4096, 4096))
        with self.assertRaises(self.client_error):
            self.h3i.minimax_h3_validate_min_generation_area(True)

    def test_tensor_parallel_architecture_divisibility(self):
        model = self.transformer.MiniMaxH3DiTModel.__new__(self.transformer.MiniMaxH3DiTModel)
        arch = SimpleNamespace(
            num_attention_heads=56, ffn_hidden_size=14336,
            hidden_size=5376, attention_head_dim=96,
        )
        for degree in (1, 2, 4):
            model._validate_tp_config(arch=arch, tp_size=degree)
        for degree in (-1, 0, 3):
            with self.subTest(degree=degree), self.assertRaises(ValueError):
                model._validate_tp_config(arch=arch, tp_size=degree)

    def test_offload_plan_keeps_image_vae_resident(self):
        plan = self.h3i.MiniMaxH3ImagePipeline._offload_plan
        self.assertEqual(plan.encoder_block_attrs, {"text_encoder": ("vision.blocks", "text_model.layers")})
        self.assertIn("transformer", plan.resident_dit_paths)
        self.assertIn("text_encoder", plan.on_demand_component_paths)
        self.assertNotIn("vae", plan.on_demand_component_paths)

    def test_cfg_parallel_and_hsdp_are_rejected_before_loading(self):
        with patch.object(self.h3i, "get_local_device", return_value=self.torch.device("cpu")):
            for parallel in (
                SimpleNamespace(cfg_parallel_size=2, use_hsdp=False),
                SimpleNamespace(cfg_parallel_size=1, use_hsdp=True),
            ):
                config = SimpleNamespace(parallel_config=parallel)
                with self.subTest(parallel=vars(parallel)), self.assertRaises(ValueError):
                    self.h3i.MiniMaxH3ImagePipeline(od_config=config)

    def test_vae_loader_and_decoder_preserve_native_precision(self):
        torch = self.torch
        config = {
            "_class_name": "MiniMaxH3ImageVAE", "latent_channels": 24,
            "latents_mean": [0.1] * 24, "latents_std": [2.0] * 24,
            "vae_tile_size": 32, "vae_tile_overlap_min": 16,
            "vit_decoder_2d": True,
            "vit_decoder_kwargs": {
                "dim_head": 16, "heads": 2, "num_layers": 1,
                "rope_dim_ratio": 0.5, "rope_theta": 100.0,
                "norm_type": "rms_norm", "qk_norm_type": "rms_norm",
            },
        }
        # Keep the released VAE constructor, loader, decoder and casts. Only
        # replace its fixed large CNN encoder with a tiny checkpoint fixture.
        def tiny_encoder(channels):
            return torch.nn.Conv3d(3, 2 * channels, 1)

        save_file = importlib.import_module("safetensors.torch").save_file
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            with tempfile.TemporaryDirectory() as directory, patch.object(
                self.image_vae, "_EncoderFCN3d", side_effect=tiny_encoder,
            ):
                path = Path(directory)
                model = self.image_vae.MiniMaxH3ImageVAE(config)
                (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
                save_file({name: value.contiguous() for name, value in model.state_dict().items()},
                          str(path / "diffusion_pytorch_model.safetensors"))
                torch.set_default_dtype(torch.bfloat16)
                loaded = self.image_vae.MiniMaxH3ImageVAE.from_pretrained(path)
                self.assertTrue(all(value.dtype == torch.float32 for value in loaded.parameters()))
                self.assertEqual(loaded.latents_std.dtype, torch.float32)
                self.assertEqual(loaded.pixel_std.dtype, torch.float32)
                ropes = [module for module in loaded.decoder.modules()
                         if isinstance(module, self.image_vae._RotaryEmbedding2d)]
                self.assertTrue(ropes)
                original = [module.inv_freq.clone() for module in ropes]
                loaded.prepare_decoder_weights(torch.bfloat16)
                self.assertTrue(all(value.dtype == torch.bfloat16 for value in loaded.decoder.parameters()))
                self.assertTrue(all(value.dtype == torch.float32 for value in loaded.encoder.parameters()))
                for module, expected in zip(ropes, original):
                    self.assertEqual(module.inv_freq.dtype, torch.float32)
                    torch.testing.assert_close(module.inv_freq, expected, rtol=0, atol=0)
                sentinel = object()
                loaded._decode_graphs = sentinel
                loaded.prepare_decoder_weights(torch.bfloat16)
                self.assertIs(loaded._decode_graphs, sentinel)
        finally:
            torch.set_default_dtype(previous)

    def test_vae_tiles_reassemble_without_reordering(self):
        tensor = self.torch.arange(3 * 64 * 80, dtype=self.torch.float32).reshape(1, 3, 64, 80)
        y_start, y_len, y_overlap = self.image_vae._split_tiles(64, 32, 16)
        x_start, x_len, x_overlap = self.image_vae._split_tiles(80, 32, 16)
        tiles = [[tensor[..., y:y + height, x:x + width]
                  for x, width in zip(x_start, x_len)]
                 for y, height in zip(y_start, y_len)]
        result = self.image_vae._assemble_tiles(tiles, y_overlap, x_overlap)
        self.torch.testing.assert_close(result, tensor, rtol=1e-6, atol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(InstalledH3ImageChecks)
    test_ids = [test.id() for test in suite]
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    origins = {}
    for name, module_name in MODULE_NAMES.items():
        module = getattr(InstalledH3ImageChecks, name, None)
        if module is not None:
            path = Path(module.__file__).resolve()
            origins[module_name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    passed = result.wasSuccessful() and not result.skipped and result.testsRun == len(test_ids)
    report = {
        "status": "PASS_CPU_RUNTIME_ONLY" if passed else "FAIL",
        "tests_run": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped),
        "test_ids": test_ids, "module_origins": origins,
        "failure_details": [{"test": test.id(), "traceback": trace} for test, trace in result.failures],
        "error_details": [{"test": test.id(), "traceback": trace} for test, trace in result.errors],
        "skip_details": [{"test": test.id(), "reason": reason} for test, reason in result.skipped],
        "python": sys.version, "executable": sys.executable,
        "euid": os.geteuid() if hasattr(os, "geteuid") else None,
        "scope": "Installed module CPU contracts and tiny VAE loader/decoder precision fixtures. "
                 "The large VAE CNN encoder is stubbed only in its checkpoint fixture. "
                 "No real model, GPU, TP communication, DLO transfer, or media acceptance.",
    }
    (args.output_dir / "cpu-checks.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "tests_run", "failures", "errors", "skipped")}))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
