# SPDX-License-Identifier: Apache-2.0
import pytest

from dingo.worker.config import load_worker_settings


def test_existing_startup_defaults_without_config_are_preserved():
    assert (
        load_worker_settings(
            None, adapter="context_ir_compile", execution_capacity=16
        ).execution_capacity
        == 16
    )
    assert (
        load_worker_settings(
            None,
            adapter="h3_generate",
            execution_capacity=1,
            prefetch_capacity=1,
            max_prefetch_capacity=1,
        ).prefetch_capacity
        == 1
    )


def test_config_selects_bound_policy_and_can_lower_backend_capacity(tmp_path):
    p = tmp_path / "worker.yaml"
    p.write_text(
        "worker:\n  adapter: h3_generate\n  execution_capacity: 1\n  prefetch_capacity: 0\n"
    )
    cfg = load_worker_settings(
        p,
        adapter="h3_generate",
        execution_capacity=2,
        prefetch_capacity=1,
        max_execution_capacity=2,
        max_prefetch_capacity=1,
    )
    assert cfg.execution_capacity == 1 and cfg.prefetch_capacity == 0


def test_oversized_configuration_is_rejected(tmp_path):
    path = tmp_path / "worker.yaml"
    path.write_bytes(b" " * (64 * 1024 + 1))
    with pytest.raises(ValueError, match="64 KiB"):
        load_worker_settings(path, adapter="context_ir_compile", execution_capacity=16)


@pytest.mark.parametrize(
    "body",
    [
        "worker: {adapter: context_ir_compile}",
        "worker: {execution_capacity: 2}",
        "worker: {execution_capacity: true}",
        "worker: {prefetch_capacity: 1}",
        "worker: {ignore_oom: true}",
        "worker: {adapter: arbitrary.module.Class}",
        "worker: []",
        "other: {}",
    ],
)
def test_incompatible_or_unsupported_configuration_fails_before_model_start(
    tmp_path, body
):
    p = tmp_path / "worker.yaml"
    p.write_text(body)
    with pytest.raises(ValueError):
        load_worker_settings(
            p, adapter="seedvr2_sr", execution_capacity=1, max_execution_capacity=1
        )
