"""Real CPU Torch helper checks; no CUDA, weights, model import or GPU claim.

Run with the parent image's Linux Python/Torch 2.13. Missing Torch is an explicit
NOT_RUN (exit 2); this script never installs a package. No runtime probe I/O.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


def load_runtime(path):
    spec = importlib.util.spec_from_file_location("seedvr2_production_cpu_selfcheck", path)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    return runtime


def run_checks(runtime_path):
    import torch
    runtime = load_runtime(runtime_path)
    if not runtime.configuration()["enabled"]:
        raise RuntimeError("CPU helper selfcheck requires SEEDVR2_DIT_BF16=1 (or unset)")
    checks = []

    class Rope(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("freqs", torch.tensor([0.123456791, 0.000123456789], dtype=torch.float32))
            self._axis_cache = {"stale": torch.ones(1)}

    class DiT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3).to(dtype=torch.float16)
            self.rope = Rope()

    model = DiT()
    vae = torch.nn.Linear(3, 3).to(dtype=torch.float16)
    original_freqs = model.rope.freqs.clone()
    original_vae = [parameter.clone() for parameter in vae.parameters()]
    text = torch.ones((3, 3), dtype=torch.float16)
    rng = torch.get_rng_state().clone()
    converted_text, rope_names = runtime._convert_transformer(model, text, torch)
    assert all(parameter.dtype == torch.bfloat16 for parameter in model.parameters())
    assert converted_text.dtype == torch.bfloat16 and text.dtype == torch.float16
    assert model.rope.freqs.dtype == torch.float32 and torch.equal(model.rope.freqs, original_freqs)
    assert not model.rope._axis_cache and rope_names == ["rope"]
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(parameter.dtype == torch.float16 and torch.equal(parameter, original)
               for parameter, original in zip(vae.parameters(), original_vae))
    checks.append("real_cpu_parameters_bf16_rope_exact_fp32_cache_cleared_vae_and_rng_unchanged")

    class AutocastCheck:
        def __call__(self):
            assert not torch.is_autocast_enabled("cpu")
            return SimpleNamespace(vid_sample=torch.ones(3, dtype=torch.bfloat16))

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert torch.is_autocast_enabled("cpu")
        native = runtime.run_transformer(SimpleNamespace(device=torch.device("cpu"), transformer=AutocastCheck()))
        assert native.dtype == torch.bfloat16 and torch.is_autocast_enabled("cpu")
    checks.append("real_cpu_dit_only_autocast_scope_and_outer_context_restored")
    pipeline = SimpleNamespace(device=torch.device("cpu"), _srbf16_configured=True, _srbf16_group=None)
    good = torch.tensor([0.0, 65280.0, -65280.0], dtype=torch.bfloat16)
    bridged = runtime.velocity_to_fp16(pipeline, good)
    assert bridged.dtype == torch.float16 and torch.isfinite(bridged).all().item()
    assert good.dtype == torch.bfloat16
    checks.append("real_cpu_bf16_65280_explicit_finite_fp16_bridge")
    for label, values, dtype in (("65536", [65536.0], torch.bfloat16),
                                 ("nan", [float("nan")], torch.bfloat16),
                                 ("inf", [float("inf")], torch.bfloat16),
                                 ("negative_inf", [-float("inf")], torch.bfloat16),
                                 ("wrong_dtype", [1.0], torch.float16),
                                 ("empty", [], torch.bfloat16)):
        try:
            runtime.velocity_to_fp16(pipeline, torch.tensor(values, dtype=dtype))
        except RuntimeError as error:
            assert "all-rank precision gate failed: velocity_native_bf16" in str(error)
        else:
            raise AssertionError("Unsafe velocity was accepted: " + label)
        checks.append("real_cpu_reject_" + label)
    view = torch.arange(513 * 1027, dtype=torch.float32).to(torch.bfloat16).reshape(513, 1027).transpose(0, 1)
    assert not view.is_contiguous() and view.numel() > runtime._CHUNK_ELEMENTS
    blocks = list(runtime._blocks(view, runtime._CHUNK_ELEMENTS))
    assert sum(block.numel() for block in blocks) == view.numel()
    assert max(block.numel() for block in blocks) <= runtime._CHUNK_ELEMENTS
    stats = runtime._numeric_stats(view, torch)
    assert stats["all_finite"] and stats["finite_absmax"] == view.float().abs().max().item()
    checks.append("real_cpu_large_noncontiguous_bounded_numeric_stats")
    bad = view.clone()
    bad[0, 0] = float("nan")
    stats = runtime._numeric_stats(bad, torch)
    assert not stats["all_finite"] and stats["finite_absmax"] is None
    checks.append("real_cpu_nonfinite_across_bounded_blocks_rejected")
    assert torch.equal(torch.get_rng_state(), rng)
    checks.append("real_cpu_precision_conversion_and_bridge_do_not_consume_rng")
    return {"schema": "seedvr2-dit-bf16-production-cpu-selfcheck/1",
            "status": "PASS_REAL_CPU_TORCH_HELPERS_ONLY", "torch_version": torch.__version__,
            "checks": checks, "cuda_calls": 0,
            "distributed_validation": "NOT_RUN", "gpu_validation": "NOT_RUN",
            "boundary": "Toy CPU tensors; not actual NaDiT/VAE, SP, CUDA BF16, media, quality or performance acceptance."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=Path(__file__).resolve().parents[1] / "core" / "bf16_runtime.py")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report, status = run_checks(args.runtime), 0
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        report, status = {"schema": "seedvr2-dit-bf16-production-cpu-selfcheck/1",
                          "status": "NOT_RUN_TORCH_UNAVAILABLE", "gpu_validation": "NOT_RUN"}, 2
    except Exception as error:
        report, status = {"schema": "seedvr2-dit-bf16-production-cpu-selfcheck/1",
                          "status": "FAILED_CPU_HELPER_CHECK", "error": repr(error),
                          "gpu_validation": "NOT_RUN"}, 1
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        print("SEEDVR2_BF16_CPU_SELFCHECK=" + str(args.output.resolve()))
    else:
        print(encoded, end="")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
