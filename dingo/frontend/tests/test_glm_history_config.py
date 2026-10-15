# SPDX-License-Identifier: Apache-2.0
import argparse

import pytest

from dingo.frontend.frontend_args import FrontendArgGroup, FrontendConfig

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def config_for(args):
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    return FrontendConfig.from_cli_args(parser.parse_args(args))


def test_glm_history_config_default_and_backend_gate(monkeypatch):
    monkeypatch.delenv("DYN_GLM_HISTORY_ARGUMENTS_MODELS", raising=False)
    assert config_for([]).glm_history_arguments_models == []
    config = config_for(["--dyn-chat-processor", "dynamo", "--glm-history-arguments-models", "glm-5.3"])
    with pytest.raises(ValueError, match="requires --dyn-chat-processor sglang"):
        config.validate()


def test_glm_history_cli_overrides_environment(monkeypatch):
    monkeypatch.setenv("DYN_GLM_HISTORY_ARGUMENTS_MODELS", "env-model")
    assert config_for([]).glm_history_arguments_models == ["env-model"]
    config = config_for(["--dyn-chat-processor", "sglang", "--glm-history-arguments-models", "glm-5.3, second"])
    config.validate()
    assert config.glm_history_arguments_models == ["glm-5.3", "second"]
    assert config_for(["--glm-history-arguments-models", ""]).glm_history_arguments_models == []
