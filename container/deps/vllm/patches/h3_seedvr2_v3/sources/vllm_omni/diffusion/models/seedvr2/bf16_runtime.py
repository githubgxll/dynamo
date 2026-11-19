"""SeedVR2-only DiT BF16 boundary; no probes, import hooks or diagnostic I/O.

Installed beside pipeline_seedvr2.py. Only that pipeline calls these helpers;
no global torch defaults, other models, weight files or RNG state are changed.
SEEDVR2_DIT_BF16 is frozen on first use (default 1); restart to change it.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import sys

_FROZEN = None
_CHUNK_ELEMENTS = 262144
_LOGGER = logging.getLogger(__name__)
HELPER_VERSION = "seedvr2-dit-bf16-production-v1"
REPORT_PREFIX = "SEEDVR2_BF16_PRECISION "


def configuration():
    """Return a copy of the process's immutable, strictly parsed feature flag."""
    global _FROZEN
    if _FROZEN is None:
        value = os.environ.get("SEEDVR2_DIT_BF16", "1")
        if value not in ("0", "1"):
            raise ValueError("SEEDVR2_DIT_BF16 must be exactly 0 or 1")
        _FROZEN = value == "1"
    return {"enabled": _FROZEN}


def _dtype_counts(module, *, buffers=False):
    rows = module.named_buffers() if buffers else module.named_parameters()
    result = {}
    for _name, value in rows:
        if value.is_floating_point():
            bucket = result.setdefault(str(value.dtype), {"tensors": 0, "numel": 0})
            bucket["tensors"] += 1
            bucket["numel"] += value.numel()
    return result


def _rank_identity(pipeline):
    """Local communicator metadata only; never import Torch or communicate."""
    identity = {"pid": os.getpid(), "global_rank": None, "sp_rank": None,
                "sp_world_size": None, "configured_sp_size": None}
    try:
        identity["configured_sp_size"] = int(pipeline.od_config.parallel_config.sequence_parallel_size)
        torch = sys.modules.get("torch")
        dist = getattr(torch, "distributed", None)
        if dist is not None and dist.is_available() and dist.is_initialized():
            identity["global_rank"] = dist.get_rank()
            group = getattr(pipeline, "_srbf16_group", None)
            if group is not None:
                identity["sp_rank"] = dist.get_rank(group)
                identity["sp_world_size"] = dist.get_world_size(group)
            elif identity["configured_sp_size"] == 1:
                identity.update(sp_rank=0, sp_world_size=1)
    except Exception as error:
        identity["rank_metadata_error"] = type(error).__name__
    return identity


def precision_report(pipeline):
    """Small metadata-only report; no tensor reductions, files or collectives."""
    cached = getattr(pipeline, "_srbf16_precision_report", None)
    if cached is not None:
        return copy.deepcopy(cached)
    enabled = configuration()["enabled"]
    configured = bool(getattr(pipeline, "_srbf16_configured", False))
    report = {
        "schema": "seedvr2-dit-bf16-production/1",
        "helper_version": HELPER_VERSION,
        "helper_path": str(Path(__file__).resolve()),
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        **_rank_identity(pipeline),
        "enabled": enabled,
        "configured": configured,
        "mode": "dit_bf16_vae_fp16" if enabled else "original_call_passthrough",
        "dit_parameters": _dtype_counts(pipeline.transformer),
        "dit_buffers": _dtype_counts(pipeline.transformer, buffers=True),
        "vae_parameters": _dtype_counts(pipeline.vae),
        "text_dtype": str(pipeline.text.dtype),
        "rope_fp32_buffers": list(getattr(pipeline, "_srbf16_rope_names", ())),
        "rope_values": ("PRECONVERSION_RUNTIME_VALUES_EXACTLY_PRESERVED" if configured
                        else "UNMODIFIED_BY_BF16_HELPER"),
        "noise_posterior_solver": "UNCHANGED_FP16",
        "gpu_validation": "CONFIGURATION_ONLY_NOT_INFERENCE_ACCEPTANCE",
    }
    pipeline._srbf16_precision_report = report
    return copy.deepcopy(report)


def _emit_report(pipeline):
    # Observability must never alter a numerical gate or collective sequence.
    try:
        if getattr(pipeline, "_srbf16_report_attempted", False):
            return
        pipeline._srbf16_report_attempted = True
        _LOGGER.info(REPORT_PREFIX + "%s", json.dumps(precision_report(pipeline), sort_keys=True))
    except Exception:
        pass


def _group(pipeline, torch):
    expected = int(pipeline.od_config.parallel_config.sequence_parallel_size)
    if expected < 1:
        raise RuntimeError("SRBF16 invalid configured SP size")
    dist = torch.distributed
    if expected == 1:
        return None
    if not dist.is_initialized():
        raise RuntimeError("SRBF16 precision gate requires an initialized SP group")
    from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
    group = get_sp_group().device_group
    if dist.get_world_size(group) != expected:
        raise RuntimeError("SRBF16 actual SP group does not match configured group")
    return group


def _agree(pipeline, local_ok, stage, detail, torch, group):
    """One MAX gate per boundary: every healthy SP rank raises or continues."""
    failed = torch.tensor([0 if local_ok else 1], dtype=torch.int32, device=pipeline.device)
    if group is not None:
        torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX, group=group)
    if failed.item():
        raise RuntimeError("SRBF16 all-rank precision gate failed: " + stage + "; local=" + str(detail))


def _check_fp_dtype(module, dtype):
    for name, value in module.named_parameters():
        if value.is_floating_point() and (value.dtype != dtype or value.is_meta):
            raise RuntimeError("SRBF16 parameter dtype/materialization mismatch: " + name)


def _convert_transformer(transformer, text, torch):
    """Preserve runtime RoPE values before conversion; never access the VAE/RNG."""
    ropes = []
    for name, module in transformer.named_modules():
        if hasattr(module, "freqs") and hasattr(module, "_axis_cache"):
            frequency = module.freqs
            if not isinstance(frequency, torch.Tensor) or not frequency.is_floating_point():
                raise RuntimeError("SRBF16 invalid RoPE frequency buffer")
            # clone BEFORE module.to(): FP32 values are not rebuilt from BF16.
            ropes.append((name, module, frequency.detach().clone().float()))
    if not ropes:
        raise RuntimeError("SRBF16 no expected RoPE buffers found")
    transformer.to(dtype=torch.bfloat16)
    for _name, module, original in ropes:
        module.freqs = original
        module._axis_cache.clear()
        if module.freqs.dtype != torch.float32 or not torch.equal(module.freqs, original):
            raise RuntimeError("SRBF16 RoPE value preservation failed")
    _check_fp_dtype(transformer, torch.bfloat16)
    for name, value in transformer.named_buffers():
        if value.is_floating_point() and value.dtype not in (torch.bfloat16, torch.float32):
            raise RuntimeError("SRBF16 unexpected transformer buffer dtype: " + name)
    return text.to(dtype=torch.bfloat16), [name for name, _module, _value in ropes]


def configure_once(pipeline):
    """Called first in SeedVR2.forward, before VAE/RNG or model collectives."""
    if not configuration()["enabled"]:
        _emit_report(pipeline)
        return
    if getattr(pipeline, "_srbf16_configured", False):
        return
    torch = importlib.import_module("torch")
    group = _group(pipeline, torch)
    error = None
    try:
        if str(pipeline.device).split(":", 1)[0] != "cuda":
            raise RuntimeError("SRBF16 model configuration is CUDA-only; CPU checks use pure helpers")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("SRBF16 CUDA device does not report BF16 support")
        _check_fp_dtype(pipeline.transformer, torch.float16)
        _check_fp_dtype(pipeline.vae, torch.float16)
        if pipeline.text.dtype != torch.float16:
            raise RuntimeError("SRBF16 expects the original FP16 condition buffer")
    except Exception as problem:
        error = repr(problem)
    _agree(pipeline, error is None, "configuration_precheck", error, torch, group)
    error, rope_names = None, None
    try:
        pipeline.text, rope_names = _convert_transformer(pipeline.transformer, pipeline.text, torch)
        _check_fp_dtype(pipeline.vae, torch.float16)
    except Exception as problem:
        error = repr(problem)
    _agree(pipeline, error is None, "configuration_postcheck", error, torch, group)
    pipeline._srbf16_group = group
    pipeline._srbf16_rope_names = tuple(rope_names)
    pipeline._srbf16_configured = True
    # A metadata inspection before first forward must not leave a stale report.
    pipeline._srbf16_precision_report = None
    _emit_report(pipeline)


def _blocks(value, limit):
    """Recursively narrow BEFORE reshape, bounding noncontiguous copies."""
    if value.numel() <= limit:
        yield value.reshape(-1)
        return
    dimension = max(range(value.ndim), key=lambda axis: value.shape[axis])
    half = value.shape[dimension] // 2
    yield from _blocks(value.narrow(dimension, 0, half), limit)
    yield from _blocks(value.narrow(dimension, half, value.shape[dimension] - half), limit)


def _numeric_stats(value, torch):
    """Same finite/absolute-range semantics as the validated diagnostic helper."""
    if not isinstance(value, torch.Tensor) or value.numel() == 0 or not value.is_floating_point():
        raise ValueError("SRBF16 velocity must be a nonempty floating tensor")
    finite, maximum = True, 0.0
    for original in _blocks(value.detach(), _CHUNK_ELEMENTS):
        part = original.float()
        part_finite = bool(torch.isfinite(part).all().item())
        finite = finite and part_finite
        if part_finite:
            maximum = max(maximum, part.abs().max().item())
    return {"numel": value.numel(), "shape": list(value.shape), "dtype": str(value.dtype),
            "all_finite": finite, "finite_absmax": maximum if finite else None}


def run_transformer(pipeline, *args, **kwargs):
    """Isolate ambient autocast only around the enabled SeedVR2 DiT call."""
    if not configuration()["enabled"]:
        return pipeline.transformer(*args, **kwargs).vid_sample
    torch = importlib.import_module("torch")
    with torch.autocast(device_type=pipeline.device.type, enabled=False):
        return pipeline.transformer(*args, **kwargs).vid_sample


def velocity_to_fp16(pipeline, velocity):
    """Agree on native BF16 finite/range, cast, then agree on FP16 finite."""
    if not configuration()["enabled"]:
        return velocity
    torch = importlib.import_module("torch")
    group = getattr(pipeline, "_srbf16_group", None)
    if not getattr(pipeline, "_srbf16_configured", False):
        # A configuration-state error also reaches the native boundary gate.
        group = _group(pipeline, torch)
    error, native = None, None
    try:
        if not getattr(pipeline, "_srbf16_configured", False):
            raise RuntimeError("SRBF16 velocity bridge entered before configuration")
        native = _numeric_stats(velocity, torch)
        if (velocity.dtype != torch.bfloat16 or not native["all_finite"]
                or native["finite_absmax"] > torch.finfo(torch.float16).max):
            raise ValueError("native BF16 velocity is nonfinite, wrong dtype, or outside FP16 range")
    except Exception as problem:
        error = repr(problem)
    _agree(pipeline, error is None, "velocity_native_bf16", {"stats": native, "error": error}, torch, group)
    error, converted, converted_stats = None, None, None
    try:
        converted = velocity.to(dtype=torch.float16)
        converted_stats = _numeric_stats(converted, torch)
        if not converted_stats["all_finite"]:
            raise ValueError("velocity became nonfinite after FP16 cast")
    except Exception as problem:
        error = repr(problem)
    _agree(pipeline, error is None, "velocity_fp16_bridge", {"stats": converted_stats, "error": error}, torch, group)
    return converted
