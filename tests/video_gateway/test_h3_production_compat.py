"""Keep production SR opt-in separate from the published native SR protocol."""
from dataclasses import replace

import pytest

from dingo.common.video_super_resolution import H3_SR_PROFILE_V3, SR_PAYLOAD_KEY
from dingo.video_gateway.adapters.h3_output_policy import H3_SR_V3_PRESETS, plan_h3_output
from dingo.video_gateway.adapters.minimax_h3 import MiniMaxH3VideoAdapter


def _adapter(make_gateway_config, options):
    pool = make_gateway_config().pools[0]
    return MiniMaxH3VideoAdapter(
        replace(pool, adapter=replace(pool.adapter, options=options))
    )


@pytest.mark.parametrize("output", list(H3_SR_V3_PRESETS))
def test_native_v3_payload_and_capabilities_preserved(make_gateway_config, tmp_path, output):
    adapter = _adapter(make_gateway_config, {
        "super_resolution": {"enabled": True, "profile": H3_SR_PROFILE_V3},
    })
    plan = plan_h3_output(*output, sr_profile=H3_SR_PROFILE_V3)
    sr = {
        "schema_version": 1,
        "profile": plan.sr_profile,
        "base_width": plan.base_width,
        "base_height": plan.base_height,
        "output_width": plan.output_width,
        "output_height": plan.output_height,
        "sr_width": plan.sr_width,
        "sr_height": plan.sr_height,
    }
    normalized = {
        "backend_model": adapter.pool.backend_model,
        "prompt": "native request",
        "width": output[0],
        "height": output[1],
        "fps": 24,
        "num_frames": 120,
        "num_inference_steps": 50,
        "super_resolution": sr,
    }
    payload = adapter.build_worker_payload(normalized, [], tmp_path)
    assert payload[SR_PAYLOAD_KEY] == sr
    assert payload["size"] == f"{plan.base_width}x{plan.base_height}"
    assert "sr_tier" not in payload
    capability = adapter.capabilities(max_result_bytes=128 * 1024 * 1024)
    assert capability["super_resolution"]["profile"] == H3_SR_PROFILE_V3
    assert capability["operations"] == ["generate"]


def test_production_and_native_sr_configurations_cannot_be_ambiguous(make_gateway_config):
    with pytest.raises(ValueError, match="cannot be combined"):
        _adapter(make_gateway_config, {
            "enable_sr": True,
            "super_resolution": {"enabled": True, "profile": H3_SR_PROFILE_V3},
        })


def test_production_sr_rejects_unimplemented_shared_reference_transport(make_gateway_config):
    with pytest.raises(ValueError, match="requires inline"):
        _adapter(make_gateway_config, {"enable_sr": True, "reference_transport": "shared_files_v1"})
