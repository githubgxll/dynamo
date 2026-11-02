# SPDX-License-Identifier: Apache-2.0
"""Source-locked SR deltas for the official Omni 0.32.0 wheel.

This module only prepares bytes. The enclosing installer owns atomic writes,
RECORD updates and the complete installed-source manifest. Historical runtime
contracts are deliberately not imported: their package/image identities belong
to the 0.31 lineage. Keep the sibling pure generators named in ``lock()``.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
LEGACY_DIR = HERE.parent
PREFIX = "vllm_omni/diffusion/models/seedvr2/"
UPSTREAM_COMMIT = "5917730a4b02aedb44d374173cdc1b10484ccd52"
C14_FILES = (
    PREFIX + "pipeline_seedvr2.py", PREFIX + "vae.py", PREFIX + "long_video.py",
    "vllm_omni/entrypoints/openai/serving_video.py",
    "vllm_omni/entrypoints/openai/video_api_utils.py",
    "vllm_omni/diffusion/utils/media_utils.py",
)
CHANGES = (*C14_FILES, "vllm_omni/diffusion/ipc.py")
ADDITIONS = (PREFIX + "c14_capture.py", PREFIX + "sr_telemetry.py", PREFIX + "long_video_support.py")
ABSORBED = (PREFIX + "nadit.py",)
GENERATOR_FILES = (
    "seedvr2-c14/source_patches.py", "seedvr2-c14/c14_capture.py",
    "seedvr2-optimization/pipeline_patch.py", "seedvr2-optimization/long_video_patch.py",
    "seedvr2-optimization/long_video_support.py", "seedvr2-metrics-repair/transfer_patch.py",
    "seedvr2-metrics-repair/support_patch.py", "seedvr2-metrics-repair/telemetry.py",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def lock() -> dict:
    value = json.loads((HERE / "sr-patches.lock.json").read_text(encoding="utf-8"))
    if (value.get("schema") != "omni032-sr-source-patches/1"
            or value.get("upstream_commit") != UPSTREAM_COMMIT
            or value.get("upstream_version") != "0.32.0"
            or set(value.get("changes", {})) != set(CHANGES)
            or set(value.get("additions", {})) != set(ADDITIONS)
            or set(value.get("absorbed", {})) != set(ABSORBED)
            or set(value.get("generator_sha256", {})) != set(GENERATOR_FILES)):
        raise ValueError("Unexpected reviewed Omni 0.32 SR patch scope")
    return value


def _generator_bytes(relative: str) -> bytes:
    if relative not in GENERATOR_FILES:
        raise ValueError("Unknown historical pure generator: " + relative)
    path = LEGACY_DIR / relative
    if path.is_symlink() or not path.resolve().is_relative_to(LEGACY_DIR.resolve()):
        raise ValueError("Historical pure generator is a symlink or escaped its root")
    data = path.read_bytes()
    if digest(data) != lock()["generator_sha256"][relative]:
        raise ValueError("Historical pure generator differs: " + relative)
    return data


def _generator(relative: str):
    data = _generator_bytes(relative)
    # Never bind the generic 'source_patches' name used by old runtime verifiers.
    name = "_omni032_sr_generator_" + relative.replace("/", "_").replace(".", "_")
    path = LEGACY_DIR / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(data, str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def transform(relative: str, data: bytes) -> bytes:
    """Prepare one exact 0.32 source delta; reject repeat/unknown inputs."""
    if relative not in CHANGES:
        raise ValueError("Unknown Omni 0.32 SR source: " + relative)
    item = lock()["changes"][relative]
    if digest(data) != item["before_sha256"]:
        raise ValueError("Omni 0.32 SR before source differs: " + relative)
    result = data
    if relative in C14_FILES:
        result = _generator("seedvr2-c14/source_patches.py").transform(relative, result)
    if relative == PREFIX + "pipeline_seedvr2.py":
        result = _generator("seedvr2-optimization/pipeline_patch.py").transform(result)
    elif relative == PREFIX + "long_video.py":
        result = _generator("seedvr2-optimization/long_video_patch.py").transform(result)
    elif relative == "vllm_omni/diffusion/ipc.py":
        result = _generator("seedvr2-metrics-repair/transfer_patch.py").transform(result)
    compile(result, relative, "exec")
    if digest(result) != item["after_sha256"]:
        raise ValueError("Omni 0.32 SR after source differs: " + relative)
    return result


def additions() -> dict[str, bytes]:
    """Return three helper bytes; no model or package installation occurs."""
    result = {
        PREFIX + "c14_capture.py": _generator_bytes("seedvr2-c14/c14_capture.py"),
        PREFIX + "sr_telemetry.py": _generator_bytes("seedvr2-metrics-repair/telemetry.py"),
        PREFIX + "long_video_support.py": _generator("seedvr2-metrics-repair/support_patch.py").transform(
            _generator_bytes("seedvr2-optimization/long_video_support.py")),
    }
    for name, data in result.items():
        compile(data, name, "exec")
        if digest(data) != lock()["additions"][name]:
            raise ValueError("Omni 0.32 SR helper differs: " + name)
    return result


def _source_file(root: Path, relative: str) -> Path:
    root = Path(root).resolve()
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("Installed SR source is a symlink or escaped its root")
    return path


def verify_absorbed(root: Path) -> dict[str, str]:
    """Verify upstream PR 8767 is already present; never repatch RMSNorm."""
    result = {}
    for name, item in lock()["absorbed"].items():
        actual = digest(_source_file(root, name).read_bytes())
        if actual != item["sha256"]:
            raise ValueError("Upstream absorbed RMSNorm source differs: " + name)
        result[name] = actual
    return result


def locked_changes(root: Path) -> dict[str, tuple[bytes, bytes]]:
    """Validate all inputs before returning deltas; leave the tree untouched."""
    verify_absorbed(root)
    for relative in GENERATOR_FILES:
        _generator_bytes(relative)
    helpers = additions()
    for name in helpers:
        if _source_file(root, name).exists():
            raise ValueError("Cannot overwrite an existing SR helper: " + name)
    changes = {}
    for name in CHANGES:
        before = _source_file(root, name).read_bytes()
        changes[name] = (before, transform(name, before))
    return changes


def expected_sources(upstream: dict[str, str]) -> dict[str, str]:
    """Derive the post-patch manifest from the complete official wheel tree."""
    value = lock()
    result = dict(upstream)
    for name, item in value["absorbed"].items():
        if result.get(name) != item["sha256"]:
            raise ValueError("Upstream absorbed RMSNorm manifest differs: " + name)
    for name, item in value["changes"].items():
        if result.get(name) != item["before_sha256"]:
            raise ValueError("Upstream Omni 0.32 SR manifest differs: " + name)
        result[name] = item["after_sha256"]
    for name, data in additions().items():
        if name in result:
            raise ValueError("Cannot overwrite an upstream SR helper: " + name)
        result[name] = digest(data)
    return result
