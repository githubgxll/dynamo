"""Strict, bounded I/O shared by C14 offline replay tools; no framework imports at startup."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path

SCHEMA = "seedvr2-c14-replay/1"
ROI = (0.22, 0.30, 0.42, 0.62)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError(f"Oversized metadata: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path.name}")
    return value


def require_guard(window: Path, output: Path, phase: str) -> dict:
    """Low-level GPU entrypoints require a live owner holding A0's shared lock."""
    from contracts import plain
    from sr_only import same_process
    name = os.environ.get("SEEDVR2_C14_REPLAY_GUARD", "")
    nonce = os.environ.get("SEEDVR2_C14_REPLAY_NONCE", "")
    if not name or not re.fullmatch(r"[0-9a-f]{64}", nonce):
        raise ValueError("Use gpu_guard.py run; an unsupervised GPU replay is not admitted")
    value = read_json(plain(name))
    if (value.get("schema"), value.get("phase"), value.get("nonce")) != (
        "seedvr2-c14-replay-guard/1", phase, nonce
    ) or not same_process(value.get("supervisor", {})):
        raise ValueError("Replay guard is stale, mismatched, or not alive")
    if Path(value["window"]).resolve(strict=True) != window.resolve(strict=True) or Path(value["output"]).resolve() != output.resolve():
        raise ValueError("Replay guard binds a different window/output")
    require_hash(Path(value["parent_run"]) / "manifest.json", value["parent_manifest_sha256"])
    return value


def register_guard_process(guard: dict) -> None:
    """torchrun may create separate sessions; attest every actual rank before CUDA."""
    from sr_only import proc
    owner = proc(os.getpid())
    if owner is None:
        raise ValueError("Cannot record this replay process identity")
    rank = int(os.environ.get("RANK", "0")) if guard["phase"] == "decode" else 0
    if rank not in range(8):
        raise ValueError("Invalid replay rank identity")
    directory = Path(os.environ["SEEDVR2_C14_REPLAY_GUARD"]).parent / "rank-owners"
    directory.mkdir(exist_ok=True)
    write_json(directory / f"rank-{rank}.json", {"schema": "seedvr2-c14-replay-rank/1", "rank": rank,
               "phase": guard["phase"], "nonce": guard["nonce"], "owner": owner})


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def contained(root: Path, relative: str) -> Path:
    candidate = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Evidence paths must be relative and contained")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve(strict=True)) or candidate.is_symlink():
        raise ValueError("Evidence path escapes its capture directory")
    return resolved


def require_hash(path: Path, expected: str) -> str:
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"Missing/invalid SHA256 for {path.name}")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch: {path.name}")
    return actual


def load_window(directory: Path) -> dict:
    directory = directory.resolve(strict=True)
    value = read_json(directory / "window.json")
    if value.get("schema") != "seedvr2-c14-window/1":
        raise ValueError("Unsupported window capture schema")
    if value.get("start") not in (130, 143, 156, 169, 182):
        raise ValueError("First C14 batch accepts only the five declared windows")
    if value.get("valid_frames") != 17 or value.get("padded_frames") != 17:
        raise ValueError("Replay requires the complete captured 17-frame window")
    if (value.get("width"), value.get("height"), value.get("fps"), value.get("target_frames")) != (1840, 1104, 24, 362):
        raise ValueError("Replay window geometry/timebase differs from frozen C14")
    if value.get("seed") != 7723 or value.get("color_correction_method") != "lab":
        raise ValueError("Replay input must be the frozen LAB/seed7723 baseline")
    if value.get("luminance_weight") != 0.8 or value.get("expected_world_size") != 8:
        raise ValueError("Replay input has unexpected luminance/SP topology")
    # Partial captures must not accidentally become trusted baseline material.
    actuals = []
    tensors = None
    for rank in range(8):
        record = read_json(directory / f"rank-{rank}.json")
        if (record.get("status"), record.get("rank"), record.get("world_size"), record.get("window_id")) != (
            "complete", rank, 8, value.get("window_id")
        ) or not value.get("window_id"):
            raise ValueError(f"Incomplete/mismatched rank attestation {rank}")
        if tensors is None:
            tensors = record.get("tensors")
        if not isinstance(tensors, dict) or record.get("tensors") != tensors:
            raise ValueError("Rank tensor shape/dtype/stride inventories differ")
        actuals.append(record.get("actual_config", {}))
    value["rank_actual_configs"] = actuals
    value["window_manifest_sha256"] = sha256(directory / "window.json")
    return value


def validate_decode_context(window: dict) -> dict:
    configs = window["rank_actual_configs"]
    for config in configs:
        expected = {
            "vae_use_tiling": True, "vae_parallel_mode": "spatial_shard_height",
            "sp_world_size": 8, "sp_ranks": list(range(8)), "vae_weight_dtype": "torch.float16",
        }
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(f"Unsupported captured VAE context: {key}")
        for key in ("autocast_enabled", "cudnn_benchmark", "cudnn_allow_tf32", "matmul_allow_tf32"):
            if type(config.get(key)) is not bool:
                raise ValueError(f"Captured numeric context missing: {key}")
        if config["autocast_enabled"] and config.get("autocast_dtype") not in ("torch.float16", "torch.bfloat16"):
            raise ValueError("Unsupported autocast dtype")
    keys = ("vae_use_tiling", "vae_parallel_mode", "sp_world_size", "sp_ranks", "vae_weight_dtype",
            "autocast_enabled", "autocast_dtype", "cudnn_benchmark", "cudnn_allow_tf32", "matmul_allow_tf32")
    baseline = {key: configs[0].get(key) for key in keys}
    if any({key: config.get(key) for key in keys} != baseline for config in configs[1:]):
        raise ValueError("Rank numeric/parallel contexts differ")
    return baseline


def verify_sources(window: dict, modules: tuple[str, ...]) -> dict:
    import importlib
    expected = window.get("native_source_hashes", {})
    checked = {}
    for name in modules:
        module = importlib.import_module(name)
        relative = name.replace(".", "/") + ".py"
        path = Path(module.__file__).resolve(strict=True)
        checked[relative] = require_hash(path, expected.get(relative))
    return checked


def tensor_metadata(directory: Path, name: str) -> tuple[Path, dict]:
    if not re.fullmatch(r"[a-z0-9_]+", name):
        raise ValueError("Invalid tensor identifier")
    path = contained(directory, f"tensors/{name}.safetensors")
    meta = read_json(contained(directory, f"tensors/{name}.json"))
    if meta.get("schema") != "seedvr2-c14-tensor/1" or meta.get("coverage") != "complete_at_pipeline_boundary":
        raise ValueError("Replay requires an attested complete tensor; rank-local shards are not accepted")
    if meta.get("rank") != 0 or meta.get("world_size") != 8:
        raise ValueError("Unexpected tensor writer/topology")
    window = read_json(directory / "window.json")
    rank = read_json(directory / "rank-0.json")
    attested = rank.get("tensors", {}).get(name)
    if meta.get("name") != name or meta.get("window_id") != window.get("window_id") or not attested:
        raise ValueError("Tensor is not bound to this captured window")
    if any(meta.get(key) != attested.get(key) for key in ("shape", "stride", "dtype")):
        raise ValueError("Tensor metadata differs from all-rank attested inventory")
    if path.stat().st_size > 2 * 1024**3 + 1024 * 1024:
        raise ValueError("Single tensor exceeds the first-batch 2 GiB admission bound")
    require_hash(path, meta.get("sha256"))
    return path, meta


def validate_stride(shape: list[int], stride: list[int]) -> None:
    if len(shape) != len(stride) or not shape or any(type(i) is not int or i <= 0 for i in shape):
        raise ValueError("Invalid tensor dimensions")
    if any(type(i) is not int or i < 0 for i in stride):
        raise ValueError("Invalid tensor strides")
    extent = 1
    for size, step in sorted(zip(shape, stride), key=lambda pair: pair[1]):
        if size == 1:
            continue
        if step < extent:
            raise ValueError("Overlapping tensor strides are not supported")
        extent += (size - 1) * step
    if extent > math.prod(shape) * 4:
        raise ValueError("Unbounded tensor storage implied by strides")


def load_tensor(directory: Path, name: str, device: str = "cpu", restore_stride: bool = False):
    import torch
    from safetensors.torch import load_file
    path, meta = tensor_metadata(directory, name)
    values = load_file(str(path), device="cpu")
    if set(values) != {"tensor"}:
        raise ValueError("Tensor archive must contain exactly the tensor key")
    value = values["tensor"]
    if list(value.shape) != meta.get("shape") or str(value.dtype) != meta.get("dtype"):
        raise ValueError("Tensor archive/metadata mismatch")
    if value.numel() * value.element_size() > 2 * 1024**3:
        raise ValueError("Single tensor exceeds the first-batch 2 GiB admission bound")
    if restore_stride:
        validate_stride(meta["shape"], meta["stride"])
        restored = torch.empty_strided(tuple(value.shape), tuple(meta["stride"]), dtype=value.dtype, device=device)
        restored.copy_(value)
        value = restored
    else:
        value = value.to(device=device)
    return value, meta


def tolerance(window: dict, policy_path: Path | None) -> dict:
    if policy_path is None:
        return {"absolute": 0.0, "relative": 0.0, "reason": "Exact baseline equality; no tolerance expansion"}
    require_hash(policy_path, window.get("tolerance_policy_sha256"))
    policy = read_json(policy_path)
    if policy.get("schema") != "seedvr2-c14-tolerance/1" or len(policy.get("reason", "").strip()) < 16:
        raise ValueError("Tolerance policy must be declared in the capture profile and explain its basis")
    for key in ("absolute", "relative"):
        if isinstance(policy.get(key), bool) or not isinstance(policy.get(key), (float, int)) or not math.isfinite(policy[key]) or policy[key] < 0:
            raise ValueError("Tolerance values must be finite and nonnegative")
    return policy


def roi_box(height: int, width: int) -> tuple[int, int, int, int]:
    return int(width * ROI[0]), int(height * ROI[1]), math.ceil(width * ROI[2]), math.ceil(height * ROI[3])


def compare(actual, expected, policy: dict) -> dict:
    import torch
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("Baseline tensor geometry/dtype changed")
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("Nonfinite baseline values: no quality conclusion is admissible")
    # Float32 preserves differences between fp16 values and uint8 pixels exactly.
    difference = (actual.float() - expected.float()).abs()
    allowed = policy["absolute"] + policy["relative"] * expected.float().abs()
    failures = int((difference > allowed).sum().item())
    result = {"pass": failures == 0, "mismatched_values": failures, "values": actual.numel(),
              "max_abs": float(difference.max().item()), "mean_abs": float(difference.mean().item())}
    if actual.ndim == 5 and actual.shape[1] == 3:
        x0, y0, x1, y1 = roi_box(actual.shape[-2], actual.shape[-1])
        roi = difference[..., y0:y1, x0:x1]
        result["roi_max_abs"] = float(roi.max().item())
        result["roi_mean_abs"] = float(roi.mean().item())
    return result


def save_tensor(path: Path, value) -> dict:
    from safetensors.torch import save_file
    if path.exists():
        raise FileExistsError(path)
    save_file({"tensor": value.detach().cpu().contiguous()}, str(path))
    result = {"sha256": sha256(path), "shape": list(value.shape), "dtype": str(value.dtype)}
    write_json(path.with_suffix(".json"), result)
    return result


def begin_output(path: Path, action: str, window: dict, profile: dict) -> dict:
    minimum = {"A1-replay-color": 8, "A2-replay-decode": 4,
               "A3-replay-single-window-encoding": 4, "A3-fusion-and-full-encoding-chain": 10}.get(action, 0) * 1024**3
    existing = path.absolute().parent
    while not existing.exists():
        existing = existing.parent
    if shutil.disk_usage(existing).free < minimum:
        raise RuntimeError(f"Replay requires at least {minimum // 1024**3} GiB free before starting")
    path.mkdir(parents=True, exist_ok=False)
    record = {"schema": SCHEMA, "action": action, "window_start": window["start"],
              "parent_window_sha256": window["window_manifest_sha256"], "profile": profile,
              "minimum_free_bytes_at_admission": minimum,
              "scope": "Diagnostic replay only; no C14 quality or production acceptance claim"}
    write_json(path / "request.json", record)
    return record


def configure_numeric_context(actual: dict) -> dict:
    """Use the captured CUDA math settings; never silently relax replay fidelity."""
    import torch
    torch.backends.cudnn.benchmark = actual["cudnn_benchmark"]
    torch.backends.cudnn.allow_tf32 = actual["cudnn_allow_tf32"]
    torch.backends.cuda.matmul.allow_tf32 = actual["matmul_allow_tf32"]
    return {"enabled": actual["autocast_enabled"],
            "dtype": getattr(torch, (actual.get("autocast_dtype") or "torch.float16").removeprefix("torch."))}


def save_failure(output: Path, error: BaseException) -> None:
    if output.is_dir() and not (output / "FAILED.json").exists():
        write_json(output / "FAILED.json", {"status": "FAIL_NO_CANDIDATE_CONCLUSION", "error_type": type(error).__name__,
                                           "error": str(error)})


def export_rgb_frame(value, directory: Path, stem: str, global_frame: int) -> dict:
    """Display-only exports AFTER full-frame processing, no per-frame normalization."""
    import torch
    from PIL import Image
    if value.ndim != 5 or value.shape[:3] != (1, 3, 1):
        raise ValueError("Display export requires one complete BCTHW RGB frame")
    if not torch.isfinite(value).all():
        raise ValueError("Cannot export nonfinite RGB")
    rgb = (value[0, :, 0].float().clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    picture = Image.fromarray(rgb)
    full = directory / f"{stem}-g{global_frame:04d}.png"
    crop = directory / f"{stem}-g{global_frame:04d}-roi.png"
    if full.exists() or crop.exists():
        raise FileExistsError("Refusing to overwrite display evidence")
    picture.save(full)
    picture.crop(roi_box(picture.height, picture.width)).save(crop)
    return {"global_frame": global_frame, "full": full.name, "full_sha256": sha256(full),
            "roi": crop.name, "roi_sha256": sha256(crop), "normalized_roi": list(ROI),
            "quantization": "round(clamp(rgb,0,1)*255); no auto contrast or normalization"}
