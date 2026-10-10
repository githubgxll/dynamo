# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gateway protocol imports must not require an Omni worker installation."""

import subprocess
import sys

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def test_protocol_helpers_do_not_load_optional_omni_engine():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys
class NoOmni(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'vllm_omni' or fullname.startswith('vllm_omni.'):
            raise ModuleNotFoundError('optional Omni engine is unavailable', name=fullname)
sys.meta_path.insert(0, NoOmni())
from dingo.vllm.omni.detached_tasks import DetachedOmniTaskManager
from dingo.vllm.omni.minimax_h3_timings import extract_model_execution
assert DetachedOmniTaskManager is not None
assert callable(extract_model_execution)
assert 'dingo.vllm.omni.base_handler' not in sys.modules
try:
    from dingo.vllm.omni import BaseOmniHandler
except ModuleNotFoundError as exc:
    assert exc.name.startswith('vllm_omni')
else:
    raise AssertionError('requesting an engine handler must require Omni')
""",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
