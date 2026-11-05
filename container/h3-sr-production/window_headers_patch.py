"""Preserve a restore-long job's owned permit in its internal window HTTP."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHANGES = ("vllm_omni/diffusion/models/seedvr2/long_video.py",)


def _apply(data):
    source = data.decode("utf-8")
    anchor = "import requests\n"
    pool_import = "from concurrent.futures import Future, ThreadPoolExecutor\n"
    submission = "return pool.submit(_restore, frames, width, height, seed, color_correction_method, port, authorization, context, measurement, metrics)"
    original = 'headers={"Authorization": authorization} if authorization else {}, timeout=900)'
    if any(source.count(value) != 1 for value in (anchor, pool_import, submission, original)):
        raise ValueError("Pinned SR internal HTTP header/context anchor differs")
    source = source.replace(pool_import, pool_import + "from contextvars import copy_context as _h3sr_copy_context\n", 1)
    source = source.replace(anchor, anchor + "from service.operation_context import internal_window_headers as _h3sr_window_headers\n", 1)
    # ThreadPoolExecutor.submit does not inherit ContextVars. Each window must
    # capture a distinct Context so two in-flight calls can enter concurrently.
    source = source.replace(submission, submission.replace("pool.submit(_restore,", "pool.submit(_h3sr_copy_context().run, _restore,"), 1)
    source = source.replace(original, 'headers=_h3sr_window_headers(authorization), timeout=900)', 1)
    compile(source, CHANGES[0], "exec")
    return source.encode("utf-8")


def transform(relative, data):
    if relative not in CHANGES:
        raise ValueError("Unknown window header source target")
    policy = json.loads((HERE / "window-headers-patches.lock.json").read_text(encoding="utf-8"))
    item = policy["changes"][relative]
    if hashlib.sha256(data).hexdigest() != item["before_sha256"]:
        raise ValueError("SR timing output differs from reviewed window-header input")
    result = _apply(data)
    if hashlib.sha256(result).hexdigest() != item["after_sha256"]:
        raise ValueError("Window-header delta differs from reviewed output")
    return result
