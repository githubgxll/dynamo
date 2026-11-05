# SPDX-License-Identifier: Apache-2.0
"""Opt-in, synchronous native-path evidence. No model or numerical defaults change.

Installed as seedvr2_c14.capture. Only a trusted operator-owned profile enables
capture. All ranks execute hooks in the same order; rank zero writes complete
pipeline-boundary tensors. This is diagnostic code, not a C14 quality repair.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time

SCHEMA = "seedvr2-c14-profile/1"
PROFILE_ENV = "SEEDVR2_C14_PROFILE"
SELECTED = [130, 143, 156, 169, 182]
_active = threading.local()
_local_lock = threading.Lock()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_profile(value):
    if not isinstance(value, dict):
        raise ValueError("C14 profile must be an object")
    if value.get("schema") != SCHEMA:
        raise ValueError("Unsupported C14 profile schema")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", value.get("run_id", "")):
        raise ValueError("Unsafe C14 run_id")
    if not Path(value.get("output_root", "")).is_absolute():
        raise ValueError("C14 output_root must be absolute")
    if value.get("capture_window_starts") != SELECTED or value.get("expected_world_size") != 8:
        raise ValueError("First C14 batch requires the five frozen windows and SP8")
    for name in ("parent_base_sha256",):
        if not re.fullmatch(r"[0-9a-f]{64}", value.get(name, "")):
            raise ValueError("Missing C14 " + name)
    if value.get("model_revision") != "37255ff8cccfb01071b87f635a5948ca8d53117c":
        raise ValueError("C14 model revision differs")
    models = value.get("model_sha256", {})
    for name in ("seedvr2_ema_3b_fp16.safetensors", "ema_vae_fp16.safetensors", "pos_emb.pt"):
        if not re.fullmatch(r"[0-9a-f]{64}", models.get(name, "")):
            raise ValueError("Missing actual mounted model hash: " + name)
    for name in ("max_bytes", "max_tensor_bytes", "min_free_bytes"):
        if not isinstance(value.get(name), int) or isinstance(value[name], bool) or value[name] <= 0:
            raise ValueError("Positive C14 budget required: " + name)
    if value["max_tensor_bytes"] > value["max_bytes"]:
        raise ValueError("Tensor budget exceeds total budget")
    if value.get("tolerance_policy_sha256") is not None and not re.fullmatch(r"[0-9a-f]{64}", value["tolerance_policy_sha256"]):
        raise ValueError("Invalid tolerance policy identity")
    return value


def profile():
    path = os.environ.get(PROFILE_ENV)
    if not path:
        return None
    value = validate_profile(json.loads(Path(path).read_text(encoding="utf-8")))
    value = dict(value)
    value["profile_sha256"] = digest(path)
    return value


def root_for(value):
    root = Path(value["output_root"]) / value["run_id"]
    no_links(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def no_links(path):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        if item.is_symlink() or (item.exists() and getattr(item.lstat(), "st_file_attributes", 0) & 0x400):
            raise ValueError("Symlink/reparse evidence path refused: " + str(item))
    return path


@contextmanager
def locked(root):
    with _local_lock, (root / ".capture.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            stream.write(b"0")
            stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def reserve(value, size):
    root = root_for(value)
    with locked(root):
        ledger = root / "budget.json"
        previous = json.loads(ledger.read_text()) if ledger.exists() else {"reserved_bytes": 0}
        total = previous["reserved_bytes"] + size
        if total > value["max_bytes"] or shutil.disk_usage(root).free - size < value["min_free_bytes"]:
            raise RuntimeError("C14 evidence disk budget exceeded; capture stopped, no silent omission")
        previous.update(reserved_bytes=total, max_bytes=value["max_bytes"])
        ledger.write_text(json.dumps(previous) + "\n", encoding="utf-8")


def json_once(path, data):
    path = no_links(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")


def mapping(start, valid, target=362, overlap=4):
    if start < 0 or valid < 1 or start + valid > target:
        raise ValueError("Invalid actual window mapping")
    result = []
    last = start + valid == target
    for local in range(valid):
        first_overlap = start > 0 and local < overlap
        final_overlap = not last and local >= valid - overlap
        weight = (local + 1) / (overlap + 1) if first_overlap else (valid - local) / (overlap + 1) if final_overlap else 1.0
        result.append({"global_frame": start + local, "local_frame": local, "blend_weight": weight,
                       "valid": True, "padding": False,
                       "output_owner": "blend_with_previous" if first_overlap else "held_for_next" if final_overlap else "this_window"})
    for local in range(valid, valid + (1-valid) % 4):
        result.append({"global_frame": None, "local_frame": local, "blend_weight": 0.0,
                       "valid": False, "padding": True, "output_owner": "discarded_padding",
                       "repeats_local_frame": valid - 1})
    return result


def window_context(job, start, valid, width, height, target, seed, method):
    value = profile()
    if value is None:
        return None
    if (width, height, target, seed, method) != (1840, 1104, 362, 7723, "lab"):
        raise ValueError("C14 capture requires frozen 1080-tier A0 settings")
    if start not in range(0, 352, 13) or valid != min(17, target-start):
        raise ValueError("C14 actual long-video window/stride differs from frozen A0")
    source = Path(job) / "input.mp4"
    if digest(source) != value["parent_base_sha256"]:
        raise ValueError("C14 source bytes differ from frozen Base")
    root = root_for(value)
    binding = root / "native-job.json"
    with locked(root):
        if binding.exists():
            previous = json.loads(binding.read_text())
            if previous["job_id"] != Path(job).name or previous["profile_sha256"] != value["profile_sha256"]:
                raise ValueError("C14 run already bound to another native job; use a new run_id")
        else:
            json_once(binding, {"job_id": Path(job).name, "profile_sha256": value["profile_sha256"], "input_sha256": value["parent_base_sha256"]})
    record = {"schema": "seedvr2-c14-window/1", "window_id": f"{Path(job).name}:{start}", "start": start,
              "valid_frames": valid, "padded_frames": valid + (1-valid) % 4, "target_frames": target,
              "width": width, "height": height, "seed": seed, "color_correction_method": method,
              "luminance_weight": 0.8, "frame_map": mapping(start, valid, target),
              "expected_world_size": value["expected_world_size"], "profile_sha256": value["profile_sha256"],
              "native_source_hashes": value.get("native_source_hashes", {}), "model_sha256": value["model_sha256"],
              "model_revision": value["model_revision"], "parent_base_sha256": value["parent_base_sha256"],
              "parent_image": value.get("parent_image"), "deployment_image": value.get("deployment_image"),
              "source_commit": value.get("source_commit"),
              "tolerance_policy_sha256": value.get("tolerance_policy_sha256"),
              "selected": start in value["capture_window_starts"], "fps": 24,
              "input_encoding": {"container": "matroska", "codec": "libx264", "pixel_format": "yuv420p", "options": {"preset": "ultrafast", "qp": "0"}},
              "window_output_options": {"preset": "ultrafast", "qp": "0"},
              "encoding_command_status": "codec options recorded; generic encoder defaults must be separately audited"}
    json_once(root / "windows" / f"{start:06d}" / "window.json", record)
    return {key: record[key] for key in ("window_id", "start", "valid_frames", "profile_sha256")}


def context_dir(value, context):
    if context["profile_sha256"] != value["profile_sha256"]:
        raise ValueError("Child C14 profile differs from API profile")
    path = no_links(root_for(value) / "windows" / f"{int(context['start']):06d}")
    no_links(path / "window.json")
    record = json.loads((path / "window.json").read_text())
    if any(context[key] != record[key] for key in ("window_id", "start", "valid_frames", "profile_sha256")):
        raise ValueError("Untrusted/mismatched C14 child identity")
    return path, record


def child_options(context, original):
    return original if context is None else {**original, "seedvr2_c14": context}


def media(context, name, data):
    if context is None:
        return
    value = profile()
    path, record = context_dir(value, context)
    if not record["selected"]:
        return
    if name not in ("window-input.mkv", "window-output.mp4"):
        raise ValueError("Unexpected C14 window media name")
    reserve(value, len(data))
    with (path / name).open("xb") as stream:
        stream.write(data)
    json_once(path / (name + ".json"), {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})


def _rank():
    import torch.distributed as dist
    if not dist.is_initialized():
        return 0, 1
    return dist.get_rank(), dist.get_world_size()


def _sync_error(error):
    """All hook callers participate; a writer failure cannot strand other ranks."""
    import torch.distributed as dist
    if dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
        error = next((item for item in errors if item), None)
    if error:
        raise RuntimeError(error)


def stats(tensor):
    import torch
    flat = tensor.reshape(-1)
    # Exact nonfinite/saturation counts in bounded blocks; quantiles are explicitly sampled.
    finite_count = nan = posinf = neginf = low = high = 0
    lo, hi = None, None
    for chunk in flat.split(1024 * 1024):
        x = chunk.float()
        mask = torch.isfinite(x)
        finite = x[mask]
        finite_count += finite.numel()
        nan += int(torch.isnan(x).sum())
        posinf += int(torch.isposinf(x).sum())
        neginf += int(torch.isneginf(x).sum())
        low += int((x <= 0).sum())
        high += int((x >= 1).sum())
        if finite.numel():
            a, b = float(finite.min()), float(finite.max())
            lo, hi = a if lo is None else min(lo, a), b if hi is None else max(hi, b)
    step = max(1, (flat.numel() + 999999) // 1000000)
    sample = flat[::step].float()
    sample = sample[torch.isfinite(sample)]
    quantiles = torch.quantile(sample, torch.tensor([0., .01, .5, .99, 1.])).tolist() if sample.numel() else None
    return {"finite": finite_count, "nan": nan, "positive_inf": posinf, "negative_inf": neginf,
            "min": lo, "max": hi, "at_or_below_zero": low, "at_or_above_one": high,
            "sampled_quantiles": quantiles, "quantile_sample_stride": step,
            "elements": flat.numel(), "finite_fraction": finite_count / flat.numel() if flat.numel() else None,
            "at_or_below_zero_fraction": low / flat.numel() if flat.numel() else None,
            "at_or_above_one_fraction": high / flat.numel() if flat.numel() else None,
            "threshold_note": "0/1 statistics are normalized-RGB thresholds, not dtype saturation bounds"}


class Capture:
    def __init__(self, value, context, pipeline, prepared):
        self.value, self.path, self.record = value, *context_dir(value, context)
        self.rank, self.world = _rank()
        self.started = time.monotonic()
        if self.world != value["expected_world_size"]:
            raise ValueError("C14 capture requires all eight ranks")
        import torch
        import torch.distributed as dist
        vae = pipeline.vae
        group = vae._spatial_group
        self.config = {"vae_use_tiling": vae.use_tiling, "vae_parallel_mode": pipeline.od_config.parallel_config.vae_parallel_mode,
                       "sp_world_size": dist.get_world_size(group) if group is not None else 1,
                       "sp_ranks": dist.get_process_group_ranks(group) if group is not None else [self.rank],
                       "vae_weight_dtype": str(next(vae.parameters()).dtype),
                       "autocast_enabled": torch.is_autocast_enabled("cuda"),
                       "autocast_dtype": str(torch.get_autocast_dtype("cuda")),
                       "cudnn_benchmark": torch.backends.cudnn.benchmark,
                       "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                       "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                       "pipeline_dtype": str(pipeline.od_config.dtype),
                       "enforce_eager": pipeline.od_config.enforce_eager,
                       "cache_backend": pipeline.od_config.cache_backend,
                       "quantized": pipeline.od_config.quantization_config is not None,
                       "ulysses_degree": pipeline.od_config.parallel_config.ulysses_degree,
                       "vae_patch_parallel_size": pipeline.od_config.parallel_config.vae_patch_parallel_size,
                       "ring_degree": pipeline.od_config.parallel_config.ring_degree,
                       "tensor_parallel_size": pipeline.od_config.parallel_config.tensor_parallel_size}
        if not vae.use_tiling or self.config["sp_world_size"] != 8 or self.config["vae_parallel_mode"] != "spatial_shard_height":
            raise ValueError("C14 actual VAE/SP configuration differs from frozen profile")
        if (self.config["pipeline_dtype"], self.config["vae_weight_dtype"], self.config["enforce_eager"],
            self.config["cache_backend"], self.config["quantized"], self.config["ulysses_degree"],
            self.config["vae_patch_parallel_size"], self.config["ring_degree"], self.config["tensor_parallel_size"]) != (
                "torch.float16", "torch.float16", True, "none", False, 8, 8, 1, 1):
            raise ValueError("C14 actual model/dtype/parallel settings differ from frozen A0")
        if str(pipeline.device).split(":")[0] != "cuda" or self.config["sp_ranks"] != list(range(8)):
            raise ValueError("C14 requires CUDA and the complete rank0..7 SP group")
        if prepared.frame_count != self.record["valid_frames"]:
            raise ValueError("Child decoded frame count differs from native window identity")
        self.tensors = {}
        self.chunks = []
        self.geometry = {"input_shape": list(prepared.frames.shape),
                         "input_layout": "THWC" if prepared.frames.dtype == torch.uint8 else "TCHW",
                         "source_fps": prepared.source.fps if prepared.source is not None else None,
                         "source_time_base": str(prepared.source.time_base) if prepared.source is not None else None,
                         "source_pts": list(prepared.source.pts) if prepared.source is not None else None,
                         "output_width": self.record["width"], "output_height": self.record["height"],
                         "preprocessing": "native prepare_video; bicubic target resize, normalize [-1,1], repeat last frame padding"}
        _active.capture = self

    def tensor(self, name, tensor, layout, range_hint=None):
        if not re.fullmatch(r"[a-z0-9_]+", name):
            raise ValueError("Unsafe tensor name")
        self.tensors[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "stride": list(tensor.stride())}
        error = None
        try:
            if self.rank == 0 and self.record["selected"]:
                from safetensors.torch import save_file
                size = tensor.numel() * tensor.element_size()
                if size > self.value["max_tensor_bytes"]:
                    raise RuntimeError("C14 individual tensor exceeds host copy budget")
                reserve(self.value, size + 65536)
                directory = self.path / "tensors"
                no_links(directory)
                directory.mkdir(exist_ok=True)
                destination = directory / (name + ".safetensors")
                if destination.exists():
                    raise ValueError("Refusing to overwrite C14 tensor")
                cpu = tensor.detach().to("cpu").contiguous()
                pending = destination.with_suffix(".partial")
                save_file({"tensor": cpu}, str(pending))
                pending.replace(destination)
                json_once(destination.with_suffix(".json"), {
                    "schema": "seedvr2-c14-tensor/1", "name": name, "sha256": digest(destination),
                    **self.tensors[name], "saved_contiguous_stride": list(cpu.stride()), "layout": layout, "device": str(tensor.device),
                    "coverage": "complete_at_pipeline_boundary", "rank": self.rank, "world_size": self.world,
                    "range_hint": range_hint, "stats": stats(cpu), "window_id": self.record["window_id"]})
                del cpu
        except Exception as exc:
            error = f"C14 capture {name} failed: {type(exc).__name__}: {exc}"
        _sync_error(error)

    def finish(self):
        error = None
        try:
            import torch
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            resources = {"cuda_device": str(torch.cuda.current_device()),
                         "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                         "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                         "process_host_highwater_bytes": int(usage.ru_maxrss) * 1024,
                         "window_with_capture_seconds": time.monotonic() - self.started,
                         "scope": "Cumulative process highwater; GPU counters were not reset. Includes diagnostic overhead; not production performance."}
            json_once(self.path / f"rank-{self.rank}.json", {"schema": "seedvr2-c14-rank/1", "status": "complete",
                "rank": self.rank, "world_size": self.world, "window_id": self.record["window_id"],
                "actual_config": self.config, "actual_geometry": self.geometry, "resources": resources,
                "tensors": self.tensors,
                "coverage_note": "Shapes/dtypes attested on each rank; complete tensors saved only on rank0 after native gathers; cross-rank tensor-value equality not tested."})
            json_once(self.path / f"decode-chunks-rank-{self.rank}.json", {"schema": "seedvr2-c14-decode/1", "chunks": self.chunks})
        except Exception as exc:
            error = f"C14 rank evidence failed: {type(exc).__name__}: {exc}"
        finally:
            _active.capture = None
        _sync_error(error)


def begin(request, pipeline, prepared):
    if not os.environ.get(PROFILE_ENV):
        return None
    capture, error = None, None
    try:
        value = profile()
        params = request.sampling_params
        context = params.extra_args.get("seedvr2_c14")
        if context is None:
            if not request.is_dummy_run():
                raise ValueError("Diagnostic server accepts only native long-window contexts")
        else:
            if (params.height, params.width, params.seed, params.extra_args.get("color_correction_method")) != (1104, 1840, 7723, "lab"):
                raise ValueError("Actual child request differs from frozen capture settings")
            capture = Capture(value, context, pipeline, prepared)
    except Exception as exc:
        error = f"C14 initialization failed: {type(exc).__name__}: {exc}"
    try:
        _sync_error(error)
    except Exception:
        _active.capture = None
        raise
    return capture


@contextmanager
def session(request, pipeline, prepared):
    capture = begin(request, pipeline, prepared)
    try:
        yield capture
        if capture is not None:
            capture.finish()
    except Exception as exc:
        if capture is not None:
            # Do not add a collective on an exceptional model path: other ranks
            # may already be inside a failing NCCL call. The process supervisor
            # terminates the whole group; absent complete records reject replay.
            try:
                json_once(capture.path / f"failed-rank-{capture.rank}.json",
                          {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
            except Exception:
                pass
        raise
    finally:
        _active.capture = None


def decode_chunk(start, stop, decoded):
    capture = getattr(_active, "capture", None)
    if capture is not None:
        previous = sum(item["rgb_frames"] for item in capture.chunks)
        capture.chunks.append({"latent_start": start, "latent_stop": stop, "rgb_start": previous,
                               "rgb_stop": previous + decoded.shape[2], "rgb_frames": decoded.shape[2],
                               "local_shape_before_height_gather": list(decoded.shape)})


def color_frame(capture, index, color_input, color_reference, corrected):
    if capture is None:
        return
    capture.tensor(f"color_input_f{index:03d}", color_input, "BCTHW", "RGB [0,1]")
    capture.tensor(f"color_reference_f{index:03d}", color_reference, "BCTHW", "RGB [0,1]")
    capture.tensor(f"color_output_f{index:03d}", corrected, "BCTHW", "RGB [0,1]")


def quantized(capture, tensor):
    if capture is not None:
        capture.tensor("quantized_rgb", tensor, "BTHWC", "uint8 RGB [0,255]")


def fused_frame(job, index, array):
    value = profile()
    if value is None:
        return
    if not 0 <= index < 362:
        raise ValueError("Unexpected native fused frame index")
    import torch
    from safetensors.torch import save_file
    root = root_for(value)
    with locked(root):
        binding = json.loads((root / "native-job.json").read_text())
        if binding["job_id"] != Path(job).name:
            raise ValueError("C14 fusion evidence job identity differs")
    directory = root / "fusion"
    no_links(directory)
    directory.mkdir(exist_ok=True)
    path = directory / f"frame-{index:06d}.safetensors"
    if path.exists():
        raise ValueError("Refusing to overwrite C14 fusion frame")
    reserve(value, array.nbytes + 65536)
    save_file({"tensor": torch.from_numpy(array).contiguous()}, str(path))
    json_once(path.with_suffix(".json"), {"schema": "seedvr2-c14-fusion/1", "global_frame": index,
        "sha256": digest(path), "shape": list(array.shape), "dtype": str(array.dtype), "layout": "HWC",
        "encoding_evidence": "../long-encoding.json"})


def long_encoder(job, stream):
    value = profile()
    if value is None:
        return
    import av
    context = stream.codec_context
    json_once(root_for(value) / "long-encoding.json", {
        "schema": "seedvr2-c14-long-encoding/1", "job_id": Path(job).name,
        "implementation": "PyAV", "pyav_version": av.__version__,
        "codec": context.name, "pixel_format": stream.pix_fmt,
        "width": stream.width, "height": stream.height,
        "rate": str(stream.average_rate or context.framerate), "options": dict(stream.options),
        "colorspace": int(context.colorspace), "color_range": int(context.color_range),
        "color_primaries": int(context.color_primaries), "color_trc": int(context.color_trc),
        "stage": "arguments_before_encoder_open"})


def run_mux(job, run, arguments, **kwargs):
    value = profile()
    if value is None:
        return run(arguments, **kwargs)
    record = {"schema": "seedvr2-c14-audio-mux/1", "job_id": Path(job).name,
              "argv": list(arguments), "status": "started"}
    try:
        result = run(arguments, **kwargs)
        record.update(status="completed", returncode=result.returncode)
        return result
    except Exception as exc:
        record.update(status="failed", error=str(exc))
        raise
    finally:
        json_once(root_for(value) / "long-audio-mux.json", record)


@contextmanager
def encoding_context(extra_params, codec_options, converter, options):
    context = (extra_params or {}).get("seedvr2_c14")
    value = profile() if context is not None else None
    if value is None:
        yield
        return
    path, record = context_dir(value, context)
    _active.encoding = {"path": path, "selected": record["selected"], "record": {
        "schema": "seedvr2-c14-encoding/1", "video_codec_options": codec_options,
        "enable_borrowed_frames": options.get("enable_borrowed_frames", False),
        "frame_conversion_workers": converter.max_workers if converter is not None else 1,
        "audio_present": options.get("audio") is not None,
        "audio_sample_rate": options.get("audio_sample_rate"), "status": "started"}}
    try:
        yield
        _active.encoding["record"]["status"] = "completed"
    finally:
        item = _active.encoding
        _active.encoding = None
        if item["selected"]:
            json_once(item["path"] / "encoding.json", item["record"])


def encoding_route(selected_path, frames, frame_shape, dtype, fps, audio_sample_rate, workers, reason):
    item = getattr(_active, "encoding", None)
    if item is not None:
        item["record"].update(selected_path=selected_path, frames=len(frames), frame_shape=list(frame_shape),
                              dtype=str(dtype), fps=fps, effective_audio_sample_rate=audio_sample_rate,
                              effective_frame_conversion_workers=workers, fallback_reason=reason)


def muxer_settings(stream, codec, fps, options):
    """Capture actual PyAV arguments after defaults are merged, before encode."""
    item = getattr(_active, "encoding", None)
    if item is not None:
        import av
        context = stream.codec_context
        item["record"]["muxer"] = {
            "implementation": "PyAV", "pyav_version": av.__version__,
            "codec": codec, "pixel_format": stream.pix_fmt,
            "width": stream.width, "height": stream.height, "fps": fps,
            "rate": str(stream.average_rate), "options": dict(options),
            "colorspace": int(context.colorspace), "color_range": int(context.color_range),
            "color_primaries": int(context.color_primaries), "color_trc": int(context.color_trc),
            "stage": "arguments_before_encoder_open", "ffmpeg_cli": None,
            "note": "PyAV encodes in-process. These are the exact passed settings, including merged CRF and QP; output media must be probed separately."}


def verify_window(path):
    path = no_links(path)
    for member in path.rglob("*"):
        no_links(member)
    record = json.loads((path / "window.json").read_text())
    if record["schema"] != "seedvr2-c14-window/1":
        raise ValueError("Unexpected window schema")
    if any(path.glob("failed-rank-*.json")):
        raise ValueError("Window has failed rank evidence")
    ranks = [json.loads((path / f"rank-{i}.json").read_text()) for i in range(record["expected_world_size"])]
    for rank, item in enumerate(ranks):
        if (item["rank"], item["world_size"], item["window_id"], item["status"]) != (rank, 8, record["window_id"], "complete"):
            raise ValueError("Incomplete/misbound rank evidence")
        if item["tensors"] != ranks[0]["tensors"] or item["actual_config"] != ranks[0]["actual_config"]:
            raise ValueError("Rank tensor layout/config differs")
        chunks = json.loads((path / f"decode-chunks-rank-{rank}.json").read_text())["chunks"]
        if sum(chunk["rgb_frames"] for chunk in chunks) != record["padded_frames"]:
            raise ValueError("Actual decode chunks do not cover padded RGB frames")
        latent_frames = (record["padded_frames"]-1)//4 + 1
        expected = [(start, min(start+2,latent_frames)) for start in range(0,latent_frames,2)]
        if [(c["latent_start"],c["latent_stop"]) for c in chunks] != expected:
            raise ValueError("Actual decode latent chunks differ from native baseline2")
        cursor = 0
        for index, chunk in enumerate(chunks):
            count = 4 * (chunk["latent_stop"]-chunk["latent_start"]) - (3 if index==0 else 0)
            if (chunk["rgb_start"],chunk["rgb_stop"],chunk["rgb_frames"]) != (cursor,cursor+count,count):
                raise ValueError("Actual decode RGB continuation differs")
            cursor += count
    if record["selected"]:
        required = ["input_decoded", "prepared_sample", "posterior_mean", "posterior_logvar", "posterior_sample",
                    "rng_before", "rng_after", "dit_condition", "dit_noise", "dit_velocity", "vae_decode_input", "decoded_rgb", "quantized_rgb"]
        required += [f"{stage}_f{frame:03d}" for frame in range(record["valid_frames"])
                     for stage in ("color_input", "color_reference", "color_output")]
        if set(required) != set(ranks[0]["tensors"]):
            raise ValueError("Missing or unexpected selected-window tensor capture")
        for name in required:
            tensor = path / "tensors" / (name + ".safetensors")
            metadata = json.loads(tensor.with_suffix(".json").read_text())
            if digest(tensor) != metadata["sha256"] or metadata["coverage"] != "complete_at_pipeline_boundary":
                raise ValueError("Corrupt or incomplete captured tensor: " + name)
        for name in ("window-input.mkv", "window-output.mp4"):
            if digest(path / name) != json.loads((path / (name + ".json")).read_text())["sha256"]:
                raise ValueError("Corrupt native window media")
        encoding = json.loads((path / "encoding.json").read_text())
        if encoding["status"] != "completed" or "selected_path" not in encoding or "muxer" not in encoding:
            raise ValueError("Actual encoding path missing")
    return {"status": "PASS_CAPTURE_STRUCTURE_ONLY", "window_id": record["window_id"], "selected": record["selected"]}


def verify_run(root):
    root = no_links(root)
    for member in root.rglob("*"):
        no_links(member)
    starts = list(range(0, 352, 13))
    actual = sorted(path.name for path in (root / "windows").iterdir() if path.is_dir())
    if actual != [f"{start:06d}" for start in starts]:
        raise ValueError("Expected exactly 28 native windows at stride13")
    results = [verify_window(root / "windows" / f"{start:06d}") for start in starts]
    for index in range(362):
        tensor = root / "fusion" / f"frame-{index:06d}.safetensors"
        metadata = json.loads(tensor.with_suffix(".json").read_text())
        if digest(tensor) != metadata["sha256"]:
            raise ValueError("Corrupt/missing fusion pixel evidence")
        if (metadata.get("global_frame"),metadata.get("layout"),metadata.get("dtype"),metadata.get("shape")) != (
            index,"HWC","uint8",[1104,1840,3]):
            raise ValueError("Incorrect fusion pixel identity/layout")
    if not (root / "long-encoding.json").is_file():
        raise ValueError("Native long encoder evidence missing")
    mux = json.loads((root / "long-audio-mux.json").read_text())
    if mux.get("status") != "completed" or mux.get("returncode") != 0:
        raise ValueError("Native long audio mux did not complete")
    return {"status": "PASS_CAPTURE_STRUCTURE_ONLY", "windows": results,
            "scope": "Complete hashes/layouts/rank attestations; no quality verdict or cross-rank numerical proof"}
