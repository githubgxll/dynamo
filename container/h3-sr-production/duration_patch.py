# SPDX-License-Identifier: Apache-2.0
"""Allow one rational container tick in the c88 duration precheck only.

Container duration can round a legal 65-frame window up by one time-base
tick. The unchanged per-frame decode admission remains the hard frame cap.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHANGES = ("vllm_omni/diffusion/models/seedvr2/pipeline_seedvr2.py",)

BEFORE = '''        if stream.duration is not None and stream.time_base is not None:
            if stream.duration * stream.time_base * rate > max_frames():
                raise OmniClientError("SeedVR2 input exceeds the maximum duration")
'''
AFTER = '''        if stream.duration is not None and stream.time_base is not None:
            # Container duration may round up by one time-base tick. Keep the
            # exact rational precheck; decoded frames still face the hard cap.
            if (stream.duration - 1) * Fraction(stream.time_base) * rate > max_frames():
                raise OmniClientError("SeedVR2 input exceeds the maximum duration")
'''


def _apply(data):
    source = data.decode("utf-8")
    if source.count(BEFORE) != 1:
        raise ValueError("Pinned c88 duration precheck anchor must occur exactly once")
    result = source.replace(BEFORE, AFTER, 1).encode("utf-8")
    compile(result, CHANGES[0], "exec")
    return result


def transform(relative, data):
    if relative not in CHANGES:
        raise ValueError("Unknown duration source target")
    policy = json.loads((HERE / "duration-patches.lock.json").read_text(encoding="utf-8"))
    item = policy["changes"][relative]
    if hashlib.sha256(data).hexdigest() != item["before_sha256"]:
        raise ValueError("Duration source differs from the published c88 input")
    result = _apply(data)
    if hashlib.sha256(result).hexdigest() != item["after_sha256"]:
        raise ValueError("Duration output differs from the reviewed one-tick delta")
    return result
