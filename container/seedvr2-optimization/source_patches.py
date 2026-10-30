# SPDX-License-Identifier: Apache-2.0
"""Two exact Python deltas over the already repaired parent; no patch replay."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PREFIX = "vllm_omni/diffusion/models/seedvr2/"
CHANGES = {PREFIX + "pipeline_seedvr2.py": "pipeline_patch.py", PREFIX + "long_video.py": "long_video_patch.py"}
ADDITIONS = {PREFIX + "sr_telemetry.py": "telemetry.py", PREFIX + "long_video_support.py": "long_video_support.py"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def transform(relative, data):
    return load("_sr_opt_" + Path(CHANGES[relative]).stem, HERE / CHANGES[relative]).transform(data)


def lock():
    result = json.loads((HERE / "source-patches.lock.json").read_text(encoding="utf-8"))
    if result.get("schema") != "seedvr2-optimization-source-patches/1" or set(result["changes"]) != set(CHANGES) or set(result["additions"]) != set(ADDITIONS):
        raise ValueError("Unexpected SeedVR2 source scope")
    return result


def expected_sources(parent):
    result = dict(parent)
    patch = lock()
    for name, item in patch["changes"].items():
        if result.get(name) != item["before_sha256"]:
            raise ValueError("SeedVR2 parent differs: " + name)
        result[name] = item["after_sha256"]
    for name, value in patch["additions"].items():
        if name in result or digest((HERE / ADDITIONS[name]).read_bytes()) != value:
            raise ValueError("SeedVR2 helper collision or changed bytes: " + name)
        result[name] = value
    return result


def locked_changes(root):
    patch = lock()
    result = {}
    for name, item in patch["changes"].items():
        before = (root / name).read_bytes()
        if digest(before) != item["before_sha256"]:
            raise ValueError("Wrong installed SeedVR2 parent source: " + name)
        after = transform(name, before)
        if digest(after) != item["after_sha256"]:
            raise ValueError("SeedVR2 transform differs from reviewed bytes: " + name)
        compile(after, name, "exec")
        result[name] = (before, after)
    return result
