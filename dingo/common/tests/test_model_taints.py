# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from dingo.common import model_taints

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


class _Runtime:
    def __init__(self) -> None:
        self.route_name: str | None = None
        self.handler = None

    def register_engine_route(self, name, handler) -> None:
        self.route_name = name
        self.handler = handler


def test_model_taint_route_updates_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    update = AsyncMock()
    monkeypatch.setattr(model_taints, "update_model_taints", update)
    runtime = _Runtime()
    endpoint = object()

    model_taints.register_model_taint_route(runtime, endpoint)

    assert runtime.route_name == "update/model_taints"
    response = asyncio.run(
        runtime.handler({"taints": ["capacity/fast", "capacity/fast"]})
    )
    assert response == {
        "status": "ok",
        "taints": ["capacity/fast"],
    }
    update.assert_awaited_once_with(endpoint, {"capacity/fast"})


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({}, "'taints' must be a JSON array of strings"),
        ({"taints": "fast"}, "'taints' must be a JSON array of strings"),
        ({"taints": [1]}, "'taints' must be a JSON array of strings"),
        (
            {"taints": ["dynamo.topology/zone=west"]},
            "uses reserved prefix 'dynamo.topology/'",
        ),
    ],
)
def test_model_taint_route_rejects_invalid_requests(body, message) -> None:
    runtime = _Runtime()
    model_taints.register_model_taint_route(runtime, object())

    with pytest.raises(ValueError, match=message):
        asyncio.run(runtime.handler(body))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("[]", []),
        ('["lane/GL", "capacity/long", "lane/GL"]', ["capacity/long", "lane/GL"]),
    ],
)
def test_startup_taints_allow_unconfigured_empty_and_static_lists(value, expected):
    assert model_taints.resolve_startup_model_taints(value) == expected


def test_startup_taints_select_exact_stable_pod_name(monkeypatch, tmp_path):
    config_file = tmp_path / "taints.json"
    config_file.write_text(
        json.dumps({"prefill-0": ["lane/S"], "prefill-1": ["lane/GL"]}),
        encoding="utf-8",
    )
    value = f"@{config_file}"
    monkeypatch.setenv("POD_NAME", "prefill-1")

    # Resolve by stable Pod name, independently of a worker's runtime identity.
    assert model_taints.resolve_startup_model_taints(value) == ["lane/GL"]
    monkeypatch.setenv("POD_NAME", "prefill-0")
    assert model_taints.resolve_startup_model_taints(value) == ["lane/S"]


def test_startup_taints_read_updated_source_on_next_start(monkeypatch, tmp_path):
    monkeypatch.setenv("POD_NAME", "prefill-0")
    config_file = tmp_path / "taints.json"
    config_file.write_text('{"prefill-0": ["lane/GL"]}', encoding="utf-8")
    initial = model_taints.resolve_startup_model_taints(f"@{config_file}")
    config_file.write_text('{"prefill-0": ["lane/S"]}', encoding="utf-8")

    assert initial == ["lane/GL"]
    assert model_taints.resolve_startup_model_taints(f"@{config_file}") == ["lane/S"]


@pytest.mark.parametrize(
    ("value", "pod_name", "message"),
    [
        ("", None, "Invalid --model-taints"),
        ('"lane/S"', None, "JSON array of strings"),
        ("[1]", None, "JSON array of strings"),
        ('["dynamo.topology/zone=west"]', None, "reserved prefix"),
        ('{"prefill-0": ["lane/S"]}', None, "POD_NAME is required"),
        ('{"prefill-0": ["lane/S"]}', "prefill-1", "no taint mapping"),
        ('{"prefill-0": "lane/S"}', "prefill-0", "JSON array of strings"),
        ('{"": []}', "prefill-0", "Pod names must be non-empty"),
        # Validate other Pods too, so a typo does not wait for a later restart.
        (
            '{"prefill-0": [], "prefill-1": ["dynamo.topology/zone=west"]}',
            "prefill-0",
            "reserved prefix",
        ),
    ],
)
def test_startup_taints_fail_closed_on_invalid_configuration(
    monkeypatch, value, pod_name, message
):
    if pod_name is None:
        monkeypatch.delenv("POD_NAME", raising=False)
    else:
        monkeypatch.setenv("POD_NAME", pod_name)
    with pytest.raises(ValueError, match=message):
        model_taints.resolve_startup_model_taints(value)


def test_startup_taints_reject_missing_file(tmp_path: Path):
    with pytest.raises(ValueError, match="Invalid --model-taints"):
        model_taints.resolve_startup_model_taints(f"@{tmp_path / 'missing.json'}")
