# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bridge the stage-config API rename in vLLM-Omni 0.31.

The public Dingo option remains ``stage_configs_path``. On Omni 0.31 it
must point to a native deploy config; removed ``stage_args`` YAML is not
silently converted to the structurally different deploy schema.
"""

from pathlib import Path
from typing import Any

import yaml
from vllm_omni.entrypoints import utils as _entrypoint_utils

_legacy_loader = getattr(_entrypoint_utils, "load_and_resolve_stage_configs", None)
if _legacy_loader is None:
    from vllm_omni.engine.omni_engine_base import (
        load_and_resolve_stage_configs as _current_loader,
    )


def stage_config_engine_kwargs(path: str | None) -> dict[str, Any]:
    """Choose the installed engine's config key without guessing YAML semantics."""
    if path is None:
        return {}
    if _legacy_loader is not None:
        return {"stage_configs_path": path}
    # Let upstream resolve named deploy profiles. Inspect explicit local files
    # only, so a removed legacy schema fails before constructing any engines.
    config_file = Path(path)
    if config_file.is_file():
        config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        if isinstance(config, dict) and "stage_args" in config:
            raise ValueError(
                "vLLM-Omni 0.31 removed the stage_args YAML schema. "
                "Use a native deploy config (stages), or migrate the legacy "
                "Dingo disaggregated stage-worker configuration before launch."
            )
    return {"deploy_config": path}


def load_and_resolve_stage_configs(
    model: str, stage_configs_path: str | None, *, kwargs: dict[str, Any]
) -> tuple[str | None, list[Any]]:
    """Preserve Dingo's two-value result across old and new upstream APIs."""
    if _legacy_loader is not None:
        return _legacy_loader(model, stage_configs_path, kwargs=kwargs)

    engine_kwargs = dict(kwargs)
    engine_kwargs.update(stage_config_engine_kwargs(stage_configs_path))
    config_inputs = _entrypoint_utils.prepare_stage_config_inputs(
        model,
        engine_kwargs,
        trust_remote_code=engine_kwargs.get("trust_remote_code"),
    )
    resolution = _current_loader(
        config_inputs.model,
        config_inputs.kwargs,
        trust_remote_code=config_inputs.trust_remote_code,
        deploy_config_path=config_inputs.deploy_config_path,
        stage_overrides=config_inputs.stage_overrides,
        strategy_config_path=config_inputs.strategy_config_path,
    )
    return resolution.config_path, list(resolution.stage_configs)
