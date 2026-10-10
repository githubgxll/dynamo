# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for shipped Dingo blueprints (t5 intentionally removed TRT-LLM)."""

import pytest

from dingo.profiler.utils.dgd_template import load_dgd_template

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]


@pytest.mark.parametrize(
    ("backend", "mode"),
    [
        ("vllm", "agg"),
        ("vllm", "disagg"),
        ("sglang", "agg"),
        ("sglang", "disagg"),
        ("mocker", "disagg"),
    ],
)
def test_profiler_blueprints_are_private_and_component_shaped(
    backend: str, mode: str
) -> None:
    config = load_dgd_template(backend, mode)

    assert config["apiVersion"] == "nvidia.com/v1beta1"
    components = config["spec"]["components"]
    assert isinstance(components, list)
    assert components
    assert all(component.get("name") for component in components)
    assert all(component.get("type") for component in components)


@pytest.mark.parametrize(
    ("backend", "mode"),
    [
        ("vllm", "agg"),
        ("vllm", "disagg"),
        ("sglang", "agg"),
        ("sglang", "disagg"),
        ("mocker", "disagg"),
    ],
)
def test_profiler_blueprints_materialize_frontend_cli_defaults(
    backend: str, mode: str
) -> None:
    config = load_dgd_template(backend, mode)

    frontend = _main_container(config, "Frontend")
    assert frontend["command"] == ["python3"]
    assert frontend["args"] == ["-m", "dingo.frontend"]


def _main_container(config: dict, component_name: str) -> dict:
    component = next(
        component
        for component in config["spec"]["components"]
        if component["name"] == component_name
    )
    return next(
        container
        for container in component["podTemplate"]["spec"]["containers"]
        if container["name"] == "main"
    )


def _component_args(config: dict, component_name: str) -> list[str]:
    return _main_container(config, component_name).get("args", [])


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("mode", ["agg", "disagg"])
def test_production_frontend_has_hf_token_secret(backend: str, mode: str) -> None:
    config = load_dgd_template(backend, mode)

    env_from = _main_container(config, "Frontend").get("envFrom", [])
    assert {"secretRef": {"name": "hf-token-secret"}} in env_from


def test_vllm_decode_blueprint_does_not_enable_kv_transfer() -> None:
    config = load_dgd_template("vllm", "disagg")

    assert "--kv-transfer-config" not in _component_args(config, "decode")
    assert "--kv-transfer-config" in _component_args(config, "prefill")


def test_mocker_blueprint_does_not_reference_unmounted_profile_data() -> None:
    config = load_dgd_template("mocker", "disagg")

    for component_name in ("decode", "prefill"):
        assert "--planner-profile-data" not in _component_args(config, component_name)


def test_removed_trtllm_blueprints_are_not_shipped():
    from dingo.profiler.utils.config_modifiers import CONFIG_MODIFIERS

    assert "trtllm" not in CONFIG_MODIFIERS
    for mode in ("agg", "disagg"):
        with pytest.raises(FileNotFoundError):
            load_dgd_template("trtllm", mode)
