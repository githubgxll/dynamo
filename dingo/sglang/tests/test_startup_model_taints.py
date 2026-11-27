# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Startup taint configuration must be validated before creating the engine."""

import argparse

import pytest

from dingo.sglang.backend_args import DynamoSGLangArgGroup, DynamoSGLangConfig

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def _parse(argv):
    parser = argparse.ArgumentParser()
    DynamoSGLangArgGroup().add_arguments(parser)
    config = DynamoSGLangConfig.from_cli_args(parser.parse_args(argv))
    config.validate()
    return config


def test_startup_taints_default_preserves_unconfigured_workers(monkeypatch):
    monkeypatch.delenv("DYN_MODEL_TAINTS", raising=False)
    assert _parse([]).startup_model_taints is None


def test_startup_taints_cli_overrides_environment(monkeypatch):
    monkeypatch.setenv("DYN_MODEL_TAINTS", '["lane/GL"]')
    assert _parse([]).startup_model_taints == ["lane/GL"]
    assert _parse(["--model-taints", '["lane/S"]']).startup_model_taints == ["lane/S"]


def test_startup_taints_environment_loads_pod_map(monkeypatch, tmp_path):
    config_file = tmp_path / "taints.json"
    config_file.write_text('{"prefill-0": ["lane/S"]}', encoding="utf-8")
    monkeypatch.setenv("DYN_MODEL_TAINTS", f"@{config_file}")
    monkeypatch.setenv("POD_NAME", "prefill-0")
    assert _parse([]).startup_model_taints == ["lane/S"]


def test_startup_taints_reject_unassigned_pod_during_argument_validation(monkeypatch):
    monkeypatch.setenv("POD_NAME", "prefill-1")
    with pytest.raises(ValueError, match="no taint mapping"):
        _parse(["--model-taints", '{"prefill-0": ["lane/S"]}'])
