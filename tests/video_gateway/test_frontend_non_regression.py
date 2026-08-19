# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys


def test_importing_video_package_has_no_optional_media_or_http_side_effects():
    script = """
import json
import sys
import dingo.video_gateway
print(json.dumps({
    'aiohttp': 'aiohttp' in sys.modules,
    'av': 'av' in sys.modules,
    'PIL': 'PIL' in sys.modules,
    'frontend': 'dingo.frontend' in sys.modules,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == (
        '{"aiohttp": false, "av": false, "PIL": false, "frontend": false}'
    )


def test_importing_dingo_does_not_activate_video_gateway():
    script = """
import json
import sys
import dingo
print(json.dumps({
    'video': any(name.startswith('dingo.video_gateway') for name in sys.modules),
    'aiohttp': 'aiohttp' in sys.modules,
    'av': 'av' in sys.modules,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == (
        '{"video": false, "aiohttp": false, "av": false}'
    )


def test_importing_existing_frontend_does_not_reach_video_package():
    script = """
import json
import sys
import dingo.frontend
print(json.dumps({
    'video': any(name.startswith('dingo.video_gateway') for name in sys.modules),
    'av': 'av' in sys.modules,
    'PIL': 'PIL' in sys.modules,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == ('{"video": false, "av": false, "PIL": false}')
