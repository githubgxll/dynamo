# SPDX-License-Identifier: Apache-2.0
"""Strict, dependency-free identity helpers for the SeedVR2 RMSNorm backport."""
from __future__ import annotations

import base64
import csv
import hashlib
import importlib.metadata as metadata
import io
import json
import os
from pathlib import Path
import tempfile

HERE = Path(__file__).resolve().parent
MODULE_RELATIVE_PATH = "vllm_omni/diffusion/models/seedvr2/nadit.py"
MODULE_NAME = "vllm_omni.diffusion.models.seedvr2.nadit"
OLD_DOC = b'    """Reference ``CustomRMSNorm``: normalise in the input dtype, optional affine."""'
NEW_DOC = b'    """RMS normalization with FP32 accumulation for low-precision inputs."""'
OLD_FORWARD = b'''    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x / torch.sqrt(variance + self.eps)
        if self.weight is not None:
            x = x * self.weight.to(x.dtype)
        return x
'''
NEW_FORWARD = b'''    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        # Squaring finite FP16 activations above ~256 overflows to infinity,
        # silently zeroing the entire token when it is divided by the RMS.
        if input_dtype in (torch.float16, torch.bfloat16):
            x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = (x / torch.sqrt(variance + self.eps)).to(input_dtype)
        if self.weight is not None:
            x = x * self.weight.to(x.dtype)
        return x
'''


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dump(path: Path, value: object) -> None:
    atomic_write(path, (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def lock_data() -> dict:
    lock = json.loads((HERE / "patch.lock.json").read_text(encoding="utf-8"))
    if lock["schema"] != "h3-seedvr2-rmsnorm-patch/1":
        raise ValueError("Unsupported RMSNorm patch lock")
    for relative, expected in lock["vendored_sha256"].items():
        path = (HERE / relative).resolve()
        if not path.is_relative_to(HERE) or sha256(path.read_bytes()) != expected:
            raise ValueError("Vendored upstream input hash mismatch: " + relative)
    return lock


def patch_bytes(original: bytes, lock: dict) -> bytes:
    if sha256(original) != lock["before_sha256"]:
        raise ValueError("Installed nadit.py is not the reviewed unpatched source")
    if original.count(OLD_DOC) != 1 or original.count(OLD_FORWARD) != 1:
        raise ValueError("RMSNorm original context is not unique")
    patched = original.replace(OLD_DOC, NEW_DOC).replace(OLD_FORWARD, NEW_FORWARD)
    if sha256(patched) != lock["after_sha256"]:
        raise ValueError("Backport did not produce the exact approved upstream source")
    compile(patched, "nadit.py", "exec")
    return patched


def wheel_record_digest(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")


def update_record(record: bytes, old: bytes, new: bytes) -> bytes:
    rows = list(csv.reader(io.StringIO(record.decode("utf-8"), newline="")))
    matches = [row for row in rows if row and row[0] == MODULE_RELATIVE_PATH]
    if len(matches) != 1 or len(matches[0]) != 3:
        raise ValueError("Expected exactly one nadit.py entry in installed RECORD")
    if matches[0][1:] != [wheel_record_digest(old), str(len(old))]:
        raise ValueError("Original RECORD does not match the reviewed nadit.py")
    matches[0][1:] = [wheel_record_digest(new), str(len(new))]
    result = io.StringIO(newline="")
    csv.writer(result, lineterminator="\n").writerows(rows)
    return result.getvalue().encode("utf-8")


def locate_installed() -> tuple[metadata.Distribution, Path, Path]:
    dist = metadata.distribution("vllm-omni")
    if dist.version != "0.31.0rc1":
        raise ValueError("Only the reviewed vllm-omni 0.31.0rc1 distribution is supported")
    entries = list(dist.files or [])
    if MODULE_RELATIVE_PATH not in {str(item) for item in entries}:
        raise ValueError("Installed distribution does not own the expected SeedVR2 source")
    record_entries = [item for item in entries if str(item).endswith(".dist-info/RECORD")]
    if len(record_entries) != 1:
        raise ValueError("Expected one installed distribution RECORD")
    root = Path(dist.locate_file("")).resolve()
    module = Path(dist.locate_file(MODULE_RELATIVE_PATH)).resolve()
    record = Path(dist.locate_file(record_entries[0])).resolve()
    if not module.is_relative_to(root) or not record.is_relative_to(root):
        raise ValueError("Installed distribution paths escape their installation root")
    return dist, module, record


def inventory() -> list[dict]:
    # A list preserves duplicate metadata rather than silently overwriting it.
    return sorted(
        ({"name": d.metadata["Name"], "version": d.version} for d in metadata.distributions()),
        key=lambda entry: (entry["name"].lower(), entry["version"]),
    )


def source_hashes(module: Path) -> dict[str, str]:
    return {name: sha256((module.parent / name).read_bytes()) for name in (
        "nadit.py", "vae.py", "pipeline_seedvr2.py", "config.py", "long_video.py")}


def check_sources(actual: dict, lock: dict, *, patched: bool) -> None:
    expected = dict(lock["source_sha256"])
    expected["nadit.py"] = lock["after_sha256"] if patched else lock["before_sha256"]
    if actual != expected:
        changed = sorted(name for name in set(expected) | set(actual) if actual.get(name) != expected.get(name))
        raise ValueError("SeedVR2 source identity mismatch: " + ", ".join(changed))
