# SPDX-License-Identifier: Apache-2.0
"""Exact telemetry-only changes over the fixed, published 9547 image."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PREFIX = "vllm_omni/diffusion/models/seedvr2/"
REPLACEMENTS = {PREFIX + "sr_telemetry.py": "telemetry.py"}
TRANSFORMS = {"vllm_omni/diffusion/ipc.py": "transfer_patch.py", PREFIX + "long_video_support.py": "support_patch.py"}
CHANGES = {**REPLACEMENTS, **TRANSFORMS}
ADDITIONS = {}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def transform(relative, data):
    if relative in REPLACEMENTS:
        return (HERE / REPLACEMENTS[relative]).read_bytes()
    return load("_sr_metrics_" + Path(TRANSFORMS[relative]).stem, HERE / TRANSFORMS[relative]).transform(data)


def lock():
    value = json.loads((HERE / "source-patches.lock.json").read_text(encoding="utf-8"))
    if value.get("schema") != "seedvr2-metrics-repair-source-patches/1" or set(value["changes"]) != set(CHANGES) or value["additions"] != {}:
        raise ValueError("Unexpected metrics repair source scope")
    return value


def expected_sources(parent):
    result = dict(parent)
    for name, item in lock()["changes"].items():
        if result.get(name) != item["before_sha256"]:
            raise ValueError("Wrong inherited telemetry/transport source: " + name)
        result[name] = item["after_sha256"]
    return result


def locked_changes(root):
    changes = {}
    for name, item in lock()["changes"].items():
        before = (root / name).read_bytes()
        if digest(before) != item["before_sha256"]:
            raise ValueError("Unexpected installed repair parent: " + name)
        after = transform(name, before)
        if digest(after) != item["after_sha256"]:
            raise ValueError("Changed reviewed repair bytes: " + name)
        compile(after, name, "exec")
        changes[name] = (before, after)
    return changes
