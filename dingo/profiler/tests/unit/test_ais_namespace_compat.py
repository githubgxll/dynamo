# SPDX-License-Identifier: Apache-2.0
"""AIS must call Dingo's config modifiers after the package rename."""

import pytest

pytestmark = [pytest.mark.gpu_0, pytest.mark.pre_merge]


def test_ais_uses_dingo_registry_and_enum():
    from dynamo.profiler.utils.config_modifiers import CONFIG_MODIFIERS as ais_registry
    from dynamo.profiler.utils.defaults import EngineType as ais_engine_type
    from dingo.profiler.utils.config_modifiers import CONFIG_MODIFIERS
    from dingo.profiler.utils.defaults import EngineType

    assert ais_registry is CONFIG_MODIFIERS
    assert ais_engine_type is EngineType
    assert set(ais_registry) == {"vllm", "sglang"}
