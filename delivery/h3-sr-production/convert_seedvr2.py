#!/usr/bin/env python3
"""CPU-only, offline, fail-closed official SeedVR2-3B -> Omni FP16 conversion.

Source files are never modified. Run once in an init container; mount the
published output directory read-only in the worker. No CUDA/model inference is
performed here. Conversion verification is NOT Omni/GPU compatibility proof.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import sys
import tempfile
import time


SCHEMA = "h3-seedvr2-fp16-conversion/1"
CONTRACT_VERSION = 1
SOURCE_REPO = "ByteDance-Seed/SeedVR2-3B"
SOURCE_REVISION = "37255ff8cccfb01071b87f635a5948ca8d53117c"
SOURCE_LOCK = {
    "seedvr2_ema_3b.pth": {
        "bytes": 13566090228,
        "sha256": "6bcc5ac59447e97b100477480aebb01be2ec724c8340bb83faae21f64848604b",
    },
    "ema_vae.pth": {
        "bytes": 1002691902,
        "sha256": "c7df8a67e68b7f9aca3d5d2153d2ce8ab4373687741a0f9ce87cb356ace51cac",
    },
    "pos_emb.pt": {
        "bytes": 595100,
        "sha256": "fa07a14844314772266b66c3b95deb0027696d8fe7065721263db5176f45d799",
    },
}
CHECKPOINTS = {
    "seedvr2_ema_3b.pth": "seedvr2_ema_3b_fp16.safetensors",
    "ema_vae.pth": "ema_vae_fp16.safetensors",
}
MODEL_INDEX = {"_class_name": "SeedVR2Pipeline"}
MANIFEST_NAME = "conversion-manifest.json"
MIN_FREE_BYTES = 20 * 1024**3
CHUNK_ELEMENTS = 1024 * 1024


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def event(name, **fields):
    print(json.dumps({"event": name, "at": utc_now(), **fields}), flush=True)


def _reject_links(path: Path):
    """Check lexical components before resolve(), including dangling links."""
    for candidate in [path, *path.parents]:
        if candidate.is_symlink():
            raise ValueError(f"Symlink path is not allowed: {candidate}")
        if candidate.exists():
            attrs = getattr(candidate.lstat(), "st_file_attributes", 0)
            if attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
                raise ValueError(f"Reparse path is not allowed: {candidate}")


def checked_path(value, *, directory=False, must_exist=True):
    path = Path(os.path.abspath(os.fspath(value)))
    _reject_links(path)
    if must_exist:
        if directory and not path.is_dir():
            raise ValueError(f"Directory not found: {path}")
        if not directory and not path.is_file():
            raise ValueError(f"Regular file not found: {path}")
    if path.exists() and not directory and not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"Not a regular file: {path}")
    return path


def sha256_file(path):
    path = checked_path(path)
    digest = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size, after.st_mtime_ns, after.st_ino
    ):
        raise ValueError(f"File changed while hashing: {path}")
    return {"bytes": after.st_size, "sha256": digest.hexdigest()}


def verify_source(source: Path):
    result = {}
    for name, expected in SOURCE_LOCK.items():
        path = checked_path(source / name)
        if path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Official source size mismatch: {name}")
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"Official source SHA256 mismatch: {name}")
        result[name] = observed
        event("source_verified", file=name, **observed)
    return result


def _json_write_new(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_file(path):
    # Called for our new staging files only; Windows fsync needs write access.
    with path.open("rb+") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path):
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _load_dependencies():
    import torch
    import safetensors
    from safetensors import safe_open
    from safetensors.torch import save_file
    # CPU operations only. Never import or instantiate the Omni CUDA pipeline.
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    return torch, safetensors, safe_open, save_file


def _require_finite(torch, tensor, name):
    if tensor.is_floating_point():
        flat = tensor.reshape(-1)
        for offset in range(0, flat.numel(), CHUNK_ELEMENTS):
            if not torch.isfinite(flat[offset:offset + CHUNK_ELEMENTS]).all().item():
                raise ValueError(f"Non-finite tensor / FP16 overflow: {name}")


def _validate_tensor(torch, tensor, name):
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"Checkpoint entry is not a tensor: {name}")
    if tensor.device.type != "cpu" or tensor.layout != torch.strided:
        raise ValueError(f"Only dense CPU tensors are supported: {name}")
    if tensor.is_complex() or tensor.is_quantized:
        raise ValueError(f"Complex/quantized tensor is unsupported: {name}")
    _require_finite(torch, tensor, name)


def _convert_tensor(torch, tensor, name):
    _validate_tensor(torch, tensor, name)
    if tensor.is_floating_point():
        converted = tensor.detach().to(dtype=torch.float16).contiguous()
    else:
        # Preserve integer/bool buffers; own storage avoids shared-storage saves.
        converted = tensor.detach().clone().contiguous()
    _require_finite(torch, converted, name)
    return converted


def convert_checkpoint(source_path, output_path, torch, safe_open, save_file):
    """No key rewriting, no unsafe pickle fallback, no per-rank conversion."""
    state = torch.load(source_path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or not state:
        raise ValueError("Official checkpoint must be a nonempty flat state_dict")
    if any(not isinstance(key, str) or not key for key in state):
        raise ValueError("Checkpoint keys must be nonempty strings")
    converted = {}
    tensor_records = {}
    for name in sorted(state):
        original = state[name]
        value = _convert_tensor(torch, original, name)
        converted[name] = value
        tensor_records[name] = {
            "shape": list(original.shape),
            "numel": original.numel(),
            "source_dtype": str(original.dtype),
            "output_dtype": str(value.dtype),
        }
    save_file(converted, str(output_path), metadata={
        "format": "pt", "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "conversion": "floating tensors to float16; integer buffers unchanged",
    })
    _fsync_file(output_path)
    del converted, value
    gc.collect()
    with safe_open(str(output_path), framework="pt", device="cpu") as reopened:
        if set(reopened.keys()) != set(state):
            raise ValueError("Saved tensor key set differs from source")
        for name in sorted(state):
            actual = reopened.get_tensor(name)
            expected = _convert_tensor(torch, state[name], name)
            if actual.shape != expected.shape or actual.dtype != expected.dtype:
                raise ValueError(f"Saved tensor shape/dtype mismatch: {name}")
            if not torch.equal(actual, expected):
                raise ValueError(f"FP16 round-trip value mismatch: {name}")
            _require_finite(torch, actual, name)
            del actual, expected
    del state, original
    gc.collect()
    return {"tensor_count": len(tensor_records), "tensors": tensor_records,
            "verification": "keys_shapes_dtypes_exact_cast_values_and_finite"}


def validate_embedding(path, torch):
    tensor = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    _validate_tensor(torch, tensor, "pos_emb.pt")
    if tensor.ndim != 2 or tensor.shape[0] <= 0 or tensor.shape[1] != 5120:
        raise ValueError("pos_emb.pt must be a nonempty [L,5120] tensor")
    if not tensor.is_floating_point():
        raise ValueError("pos_emb.pt must be floating point")
    _convert_tensor(torch, tensor, "pos_emb.pt runtime FP16 cast")
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "copied_without_modification": True}


def verify_output(output: Path):
    """Offline, read-only integrity check; imports no torch or safetensors."""
    output = checked_path(output, directory=True)
    manifest_path = checked_path(output / MANIFEST_NAME)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema") != SCHEMA or manifest.get("status") != "VERIFIED"
            or manifest.get("contract_version") != CONTRACT_VERSION):
        raise ValueError("No compatible VERIFIED conversion manifest")
    if manifest.get("source") != {"repo": SOURCE_REPO, "revision": SOURCE_REVISION,
                                  "files": SOURCE_LOCK}:
        raise ValueError("Conversion source identity mismatch")
    expected_names = {*CHECKPOINTS.values(), "pos_emb.pt", "model_index.json"}
    outputs = manifest.get("outputs", {})
    if set(outputs) != expected_names:
        raise ValueError("Unexpected conversion output inventory")
    if {p.name for p in output.iterdir()} != expected_names | {MANIFEST_NAME}:
        raise ValueError("Extra/missing files in published model directory")
    for name in sorted(expected_names):
        if sha256_file(output / name) != outputs[name]:
            raise ValueError(f"Converted output hash mismatch: {name}")
    if json.loads((output / "model_index.json").read_text()) != MODEL_INDEX:
        raise ValueError("model_index.json does not select SeedVR2Pipeline")
    if outputs["pos_emb.pt"] != SOURCE_LOCK["pos_emb.pt"]:
        raise ValueError("pos_emb.pt is not the original pinned embedding")
    return manifest


def _peak_memory():
    try:
        import resource
        return {"ru_maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "unit": "bytes" if sys.platform == "darwin" else "KiB"}
    except ImportError:
        return {"ru_maxrss": None, "unit": "unavailable"}


def prepare(source_dir, output_dir, *, verify_only=False):
    source = checked_path(source_dir, directory=True)
    output = checked_path(output_dir, directory=True, must_exist=False)
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output directories must be disjoint")
    if output == Path(output.anchor) or output.parent == output:
        raise ValueError("Output must be a dedicated model directory")
    checked_path(output.parent, directory=True)
    if verify_only:
        manifest = verify_output(output)
        event("PASS_FP16_OFFLINE_VERIFY", output=str(output))
        return manifest
    if output.exists():
        manifest = verify_output(output)
        # Same command may reuse only a complete, hash-verified publication.
        event("PASS_FP16_REUSED", output=str(output))
        return manifest
    lock = output.with_name(output.name + ".conversion.lock")
    _reject_links(lock)
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    staging = None
    started = time.monotonic()
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "host": socket.gethostname(),
                       "started_at": utc_now(), "output": str(output)}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if output.exists():
            raise ValueError("Output appeared during lock acquisition; stop")
        if shutil.disk_usage(output.parent).free < MIN_FREE_BYTES:
            raise ValueError("Conversion requires at least 20 GiB free disk space")
        sources = verify_source(source)
        torch, safetensors, safe_open, save_file = _load_dependencies()
        staging = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
        event("conversion_started", source=str(source), staging=str(staging))
        tensors = {}
        for old, new in CHECKPOINTS.items():
            tensors[new] = convert_checkpoint(source / old, staging / new, torch, safe_open, save_file)
            event("checkpoint_converted_and_roundtrip_verified", file=new,
                  tensor_count=tensors[new]["tensor_count"], peak_memory=_peak_memory())
        embedding = validate_embedding(source / "pos_emb.pt", torch)
        shutil.copyfile(source / "pos_emb.pt", staging / "pos_emb.pt")
        _fsync_file(staging / "pos_emb.pt")
        _json_write_new(staging / "model_index.json", MODEL_INDEX)
        # Rehash all source files after conversion to detect concurrent changes.
        if verify_source(source) != sources:
            raise ValueError("Source changed during conversion")
        outputs = {p.name: sha256_file(p) for p in staging.iterdir()}
        manifest = {
            "schema": SCHEMA, "contract_version": CONTRACT_VERSION, "status": "VERIFIED",
            "source": {"repo": SOURCE_REPO, "revision": SOURCE_REVISION, "files": sources},
            "outputs": outputs, "tensor_validation": tensors, "embedding": embedding,
            "libraries": {"torch": torch.__version__, "safetensors": safetensors.__version__},
            # Kubernetes projected ConfigMap scripts can themselves be symlinks.
            "script_sha256": sha256_file(Path(__file__).resolve())["sha256"],
            "completed_at": utc_now(), "elapsed_seconds": time.monotonic() - started,
            "cpu_peak_memory": _peak_memory(),
            "validation_scope": "CPU format/numerics only; Omni strict load and GPU/media acceptance pending",
        }
        _json_write_new(staging / MANIFEST_NAME, manifest)
        verify_output(staging)
        _fsync_directory(staging)
        _reject_links(output)
        if output.exists():
            raise ValueError("Refusing to replace an existing output directory")
        os.rename(staging, output)
        _fsync_directory(output.parent)
        event("PASS_FP16_CONVERSION", output=str(output), peak_memory=_peak_memory())
        return manifest
    except BaseException as exc:
        if staging is not None and staging.is_dir():
            with contextlib.suppress(Exception):
                _json_write_new(staging / "FAILED.json", {
                    "status": "FAILED", "error_type": type(exc).__name__,
                    "error": str(exc), "at": utc_now(), "peak_memory": _peak_memory(),
                })
        event("CONVERSION_STOPPED", staging=str(staging) if staging else None,
              error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        # Only our newly acquired lock; no recursive cleanup or source writes.
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        prepare(args.source_dir, args.output_dir, verify_only=args.verify_only)
    except Exception as exc:
        print(f"STOP: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
