# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy

import pytest

from dingo.video_gateway.adapters import create_adapter
from dingo.video_gateway.config import canonical_backend_target, parse_config


def _raw(tmp_path):
    return {
        "schema_version": 1,
        "deployment_id": "renamed-deployment",
        "runtime": {},
        "http": {},
        "task_store": {"kind": "memory"},
        "artifact_store": {"kind": "filesystem", "root": str(tmp_path)},
        "pools": [
            {
                "pool_id": "first-pool",
                "served_models": ["alias-a"],
                "backend_model": "internal-a",
                "backend_target": "dyn://not-a-k8s-namespace.component-a.endpoint-a",
                "adapter": {
                    "name": "minimax_h3",
                    "workflow": "fl2va",
                    "compatibility_version": "wire-v1",
                },
            },
            {
                "pool_id": "second-pool",
                "served_models": ["alias-b"],
                "backend_model": "internal-b",
                "backend_target": "other-scope.component-b.endpoint-b",
                "adapter": {
                    "name": "minimax_h3",
                    "workflow": "ref2va",
                    "compatibility_version": "wire-v1",
                },
            },
        ],
    }


def test_config_maps_arbitrary_full_targets_without_namespace_assumptions(tmp_path):
    config = parse_config(_raw(tmp_path))

    assert config.pool_for_model("alias-a").backend_target == (
        "dyn://not-a-k8s-namespace.component-a.endpoint-a"
    )
    assert config.pool_for_model("alias-b").endpoint_path == (
        "other-scope.component-b.endpoint-b"
    )
    assert config.pools[0].configuration_revision.startswith("sha256:")


@pytest.mark.parametrize(
    "target",
    ["two.parts", "four.parts.are.invalid", "dyn://scope.bad/name.endpoint", ""],
)
def test_invalid_backend_targets_fail_fast(target):
    with pytest.raises(ValueError):
        canonical_backend_target(target)


def test_duplicate_public_model_fails_fast(tmp_path):
    raw = _raw(tmp_path)
    raw["pools"][1]["served_models"] = ["alias-a"]

    with pytest.raises(ValueError, match="maps to multiple pools"):
        parse_config(raw)


def test_unknown_adapter_fails_fast(tmp_path):
    raw = deepcopy(_raw(tmp_path))
    raw["pools"][0]["adapter"]["name"] = "implicit-plugin"

    with pytest.raises(ValueError, match="unknown video adapter"):
        parse_config(raw)


def test_string_boolean_is_rejected(tmp_path):
    raw = _raw(tmp_path)
    raw["pools"][0]["scheduling"] = {"accept_without_workers": "false"}

    with pytest.raises(TypeError, match="must be a boolean"):
        parse_config(raw)


def test_unknown_model_adapter_option_fails_before_runtime_start(tmp_path):
    raw = _raw(tmp_path)
    raw["pools"][0]["adapter"]["typo_flow_shift"] = 12
    config = parse_config(raw)

    with pytest.raises(ValueError, match="unknown MiniMax-H3 adapter options"):
        create_adapter(config.pools[0])


def test_unknown_discovery_backend_fails_config_validation(tmp_path):
    raw = _raw(tmp_path)
    raw["runtime"] = {"discovery_backend": "namespace-name"}

    with pytest.raises(ValueError, match="discovery_backend"):
        parse_config(raw)


def test_vllm_omni_http_compatibility_options(tmp_path):
    raw = _raw(tmp_path)
    raw["http"] = {
        "default_model": "alias-a",
        "async_submit_status_code": 200,
    }

    config = parse_config(raw)

    assert config.http.default_model == "alias-a"
    assert config.http.async_submit_status_code == 200


def test_default_model_must_be_configured(tmp_path):
    raw = _raw(tmp_path)
    raw["http"] = {"default_model": "missing"}

    with pytest.raises(ValueError, match="is not a configured served model"):
        parse_config(raw)


@pytest.mark.parametrize("status", [199, 201, 204])
def test_async_submit_status_code_is_restricted(tmp_path, status):
    raw = _raw(tmp_path)
    raw["http"] = {"async_submit_status_code": status}

    with pytest.raises(ValueError, match="must be 200 or 202"):
        parse_config(raw)


def test_media_limits_have_safe_defaults_and_derived_budget_weight(tmp_path):
    config = parse_config(_raw(tmp_path))

    assert config.media.max_parts == 32
    assert config.media.max_single_file_bytes == 50 * 1024 * 1024
    assert config.media.max_result_encoded_bytes >= config.media.max_result_bytes
    assert (
        config.media.inflight_memory_budget_bytes
        >= config.media.max_task_memory_bytes
    )


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("http", "max_body_bytes", 513 * 1024 * 1024, "hard cap"),
        ("media", "max_single_file_bytes", 51 * 1024 * 1024, "hard cap"),
        ("media", "max_parts", 33, "hard cap"),
        ("media", "max_result_bytes", 129 * 1024 * 1024, "hard cap"),
    ],
)
def test_media_protocol_hard_caps_cannot_be_overridden(
    tmp_path, section, field, value, message
):
    raw = _raw(tmp_path)
    raw[section] = {field: value}

    with pytest.raises(ValueError, match=message):
        parse_config(raw)


def test_media_budget_must_fit_one_maximum_task(tmp_path):
    raw = _raw(tmp_path)
    raw["media"] = {"inflight_memory_budget_bytes": 128 * 1024 * 1024}

    with pytest.raises(ValueError, match="fit one maximum-size task"):
        parse_config(raw)


def test_media_limit_boolean_is_not_accepted_as_an_integer(tmp_path):
    raw = _raw(tmp_path)
    raw["media"] = {"max_parts": True}

    with pytest.raises(TypeError, match="must be an integer"):
        parse_config(raw)
