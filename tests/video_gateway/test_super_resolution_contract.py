# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import json
import time

import pytest

from dingo.common.video_super_resolution import (
    H3_SR_PROFILE,
    SR_CAPABILITY,
    SR_PAYLOAD_KEY,
    configured_sr_profile,
    negotiated_sr_profiles,
)
from dingo.common.video_task_protocol import ENVELOPE_KEY, DetachedTaskIdentity
from dingo.video_gateway.adapters.minimax_h3 import MiniMaxH3VideoAdapter
from dingo.video_gateway.errors import GatewayError
from dingo.vllm.omni.detached_tasks import DetachedOmniTaskManager
from tests.video_gateway.test_dispatcher import (
    FakeClient,
    FakeContext,
    _DetachedHandler,
    _stack,
)
from tests.video_gateway.test_minimax_h3_adapter import _png_upload


def config(make, sr=True, **changes):
    pool = {
        "pool_id": "fl-pool",
        "served_models": ["alias-without-model-family"],
        "backend_model": "worker-fl",
        "backend_target": "dyn://scope-a.backend.generate",
        "execution_mode": "detached",
        "adapter": {"name": "minimax_h3", "workflow": "fl2va"},
        "scheduling": {"worker_capacity": 1, "accept_without_workers": True},
    }
    if sr is not None:
        pool["adapter"]["super_resolution"] = {"enabled": sr, "profile": H3_SR_PROFILE}
    pool.update(changes)
    return make(pools=[pool])


def fields(size="1890x1080"):
    return {"prompt": ["test"], "seconds": ["5"], "size": [size]}


def advertisement():
    return {
        "schema_version": 1,
        "capabilities": ["execution_capacity_v1", SR_CAPABILITY],
        "execution_capacity": 1,
        "accepting": True,
        "super_resolution": {"ready": True, "profiles": [H3_SR_PROFILE]},
    }


@pytest.mark.parametrize(
    "raw",
    [
        True,
        [],
        {"enabled": "true"},
        {"enabled": 1},
        {"profile": "other.v1"},
        {"url": "http://example.invalid"},
    ],
)
def test_invalid_sr_config_rejected(raw):
    with pytest.raises(ValueError):
        configured_sr_profile("minimax_h3", {"super_resolution": raw})


def test_model_policy_is_not_global():
    with pytest.raises(ValueError):
        configured_sr_profile("other_model", {"super_resolution": {"enabled": True}})
    assert configured_sr_profile("other_model", {}) is None


def test_sr_requires_detached_mode(make_gateway_config):
    with pytest.raises(ValueError, match="detached"):
        config(make_gateway_config, execution_mode="stream")


def test_1440_normalization_and_worker_validation(make_gateway_config, tmp_path):
    from dingo.vllm.omni.seedvr2_handler import validate_sr_request

    adapter = MiniMaxH3VideoAdapter(config(make_gateway_config).pools[0])
    normalized = adapter.normalize_request(fields("2520x1440"), [], "h3-alias")
    payload = adapter.build_worker_payload(normalized, [], tmp_path)
    assert (normalized["width"], normalized["height"]) == (2520, 1440)
    assert payload["size"] == "1344x768"
    plan = validate_sr_request(payload)
    assert (plan["sr_width"], plan["sr_height"]) == (2576, 1472)
    corrupted = copy.deepcopy(payload)
    corrupted[SR_PAYLOAD_KEY]["sr_height"] = 1152
    with pytest.raises(ValueError):
        validate_sr_request(corrupted)


def test_normalization_payload_and_alias(make_gateway_config, tmp_path):
    cfg = config(make_gateway_config)
    adapter = MiniMaxH3VideoAdapter(cfg.pools[0])
    n = adapter.normalize_request(fields(), [], "alias-without-model-family")
    assert (n["width"], n["height"]) == (1890, 1080)
    assert "seed" not in n
    payload = adapter.build_worker_payload(n, [], tmp_path)
    assert payload["size"] == "1344x768"
    assert "seed" not in payload["nvext"]
    assert payload[SR_PAYLOAD_KEY]["output_width"] == 1890
    assert payload[SR_PAYLOAD_KEY]["sr_width"] == 1904
    corrupted = copy.deepcopy(n)
    corrupted["super_resolution"]["base_width"] = 2048
    with pytest.raises(ValueError, match="persisted"):
        adapter.build_worker_payload(corrupted, [], tmp_path)
    disabled = MiniMaxH3VideoAdapter(config(make_gateway_config, False).pools[0])
    with pytest.raises(GatewayError):
        disabled.normalize_request(fields(), [], "alias-without-model-family")
    assert "super_resolution" not in disabled.capabilities(max_result_bytes=100)
    assert (
        len(
            adapter.capabilities(max_result_bytes=100)["super_resolution"][
                "output_sizes"
            ]
        )
        == 4
    )


def test_base_normalization_and_revision_stable(make_gateway_config):
    old = config(make_gateway_config, None).pools[0]
    enabled = config(make_gateway_config).pools[0]
    disabled = config(make_gateway_config, False).pools[0]
    n = MiniMaxH3VideoAdapter(old).normalize_request(fields("1344x768"), [], "alias")
    for p in (enabled, disabled):
        assert (
            MiniMaxH3VideoAdapter(p).normalize_request(fields("1344x768"), [], "alias")
            == n
        )
        assert p.request_configuration_revision(n) == old.configuration_revision
    assert enabled.configuration_revision != old.configuration_revision
    changed = config(make_gateway_config, backend_model="another-model").pools[0]
    assert changed.request_configuration_revision(n) != old.configuration_revision


@pytest.mark.parametrize(
    "change",
    [
        {"accepting": False},
        {"capabilities": []},
        {"super_resolution": {"ready": False, "profiles": [H3_SR_PROFILE]}},
        {"super_resolution": {"ready": True, "profiles": H3_SR_PROFILE}},
        {"super_resolution": {"ready": True, "profiles": ["other.v1"]}},
    ],
)
def test_missing_or_malformed_capability_fails_closed(change):
    assert not negotiated_sr_profiles({**advertisement(), **change})


async def test_mixed_workers_probe_failure_and_expiry(make_gateway_config):
    cfg = config(make_gateway_config)
    responses = {7: {}, 8: advertisement()}

    class Client:
        def instance_ids(self):
            return [7, 8]

        async def direct(self, request, instance, context):
            async def stream():
                if responses[instance] is None:
                    raise RuntimeError("unreachable")
                yield responses[instance]

            return stream()

    _, _, dispatcher, _ = _stack(cfg, {"fl-pool": Client()})
    pool = dispatcher.pools["fl-pool"]
    await dispatcher._refresh_instances(pool)
    n = {"super_resolution": {"profile": H3_SR_PROFILE}}
    assert dispatcher._worker_slots(pool, n) == [(8, 0)]
    assert dispatcher._worker_slots(pool) == [(8, 0)]  # SR pool fails closed on unknown capability
    assert dispatcher.has_sr_workers("fl-pool", n)
    pool.capacity_cache[8] = (1, time.monotonic() - 1)
    assert not dispatcher.has_sr_workers("fl-pool", n)
    responses[8] = None
    await dispatcher._refresh_instances(pool)
    assert not dispatcher.has_sr_workers("fl-pool", n)
    assert 8 not in pool.sr_profiles
    assert not dispatcher._worker_slots(pool, {"super_resolution": None})


async def test_sr_refresh_keeps_valid_capability_until_response(make_gateway_config):
    entered, release = asyncio.Event(), asyncio.Event()

    class Client:
        pause = False
        calls = 0

        def instance_ids(self):
            return [8]

        async def direct(self, request, instance, context):
            self.calls += 1

            async def stream():
                if self.pause:
                    entered.set()
                    await release.wait()
                yield advertisement()

            return stream()

    client = Client()
    _, _, dispatcher, _ = _stack(config(make_gateway_config), {"fl-pool": client})
    pool = dispatcher.pools["fl-pool"]
    await dispatcher._refresh_instances(pool)
    normalized = {"super_resolution": {"profile": H3_SR_PROFILE}}
    pool.capacity_cache[8] = (1, time.monotonic() + 2.5)
    client.pause = True
    refresh = asyncio.create_task(dispatcher._refresh_instances(pool))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        assert dispatcher.has_sr_workers("fl-pool", normalized)
    finally:
        release.set()
        await refresh
    assert client.calls == 2
    assert pool.capacity_cache[8][1] > time.monotonic() + 4
    assert dispatcher.has_sr_workers("fl-pool", normalized)


async def submit(service, size, key):
    return await service.submit(
        fields=fields(size),
        uploads=[],
        upload_root=await service.artifacts.create_upload(),
        delivery_mode="async",
        idempotency_key=key,
    )


async def test_no_capable_worker_never_enqueues_sr(make_gateway_config):
    store, _, _, service = _stack(
        config(make_gateway_config), {"fl-pool": FakeClient()}
    )
    with pytest.raises(GatewayError) as exc:
        await submit(service, "1890x1080", "no-worker")
    assert exc.value.code == "super_resolution_unavailable"
    assert await store.queue_depth("fl-pool") == 0


async def test_base_replay_across_sr_enablement(make_gateway_config):
    store, _, _, old = _stack(
        config(make_gateway_config, None), {"fl-pool": FakeClient()}
    )
    original = await submit(old, "1344x768", "same-base")
    _, _, _, upgraded = _stack(config(make_gateway_config), {"fl-pool": FakeClient()})
    upgraded.store = store
    replay = await submit(upgraded, "1344x768", "same-base")
    assert not replay.created and replay.stored.task.id == original.stored.task.id
    with pytest.raises(GatewayError) as exc:
        await submit(upgraded, "1890x1080", "same-base")
    assert exc.value.code == "idempotency_conflict"


async def test_queued_base_survives_sr_config_upgrade(make_gateway_config):
    store, _, _, old = _stack(
        config(make_gateway_config, None), {"fl-pool": FakeClient()}
    )
    original = await submit(old, "1344x768", "queued-base")
    _, _, dispatcher, _ = _stack(config(make_gateway_config), {"fl-pool": FakeClient()})
    dispatcher.store = store
    await dispatcher._recover()
    assert (await store.get_task(original.stored.task.id)).task.status.value == "queued"


def test_ref2va_plan_preserves_reference_order_and_seed_zero(
    make_gateway_config, tmp_path
):
    cfg = config(
        make_gateway_config,
        adapter={
            "name": "minimax_h3",
            "workflow": "ref2va",
            "super_resolution": {"enabled": True, "profile": H3_SR_PROFILE},
        },
    )
    adapter = MiniMaxH3VideoAdapter(cfg.pools[0])
    uploads = [_png_upload(tmp_path, ordinal=i) for i in (0, 1)]
    n = adapter.normalize_request(
        {**fields("2520x1440"), "seed": ["0"]}, uploads, "ref-alias"
    )
    manifest = [u.manifest_entry(relative_to=tmp_path) for u in uploads]
    payload = adapter.build_worker_payload(n, manifest[::-1], tmp_path)
    ordinary = adapter.normalize_request(
        {**fields("1344x768"), "seed": ["0"]}, uploads, "ref-alias"
    )
    base_payload = adapter.build_worker_payload(ordinary, manifest, tmp_path)
    assert payload["size"] == "1344x768"
    assert payload["nvext"]["seed"] == 0
    assert json.loads(payload["input_reference"]) == json.loads(
        base_payload["input_reference"]
    )
    assert payload[SR_PAYLOAD_KEY]["output_width"] == 2520


async def test_sr_replay_and_changed_size_conflict(make_gateway_config):
    _, _, dispatcher, service = _stack(
        config(make_gateway_config), {"fl-pool": FakeClient()}
    )
    pool = dispatcher.pools["fl-pool"]
    pool.instance_ids = [8]
    pool.capacity_cache[8] = (1, time.monotonic() + 60)
    pool.sr_profiles[8] = frozenset({H3_SR_PROFILE})
    original = await submit(service, "1890x1080", "same-sr")
    pool.sr_profiles.clear()
    replay = await submit(service, "1890x1080", "same-sr")
    assert not replay.created and replay.stored.task.id == original.stored.task.id
    with pytest.raises(GatewayError) as exc:
        await submit(service, "2520x1440", "same-sr")
    assert exc.value.code == "idempotency_conflict"


async def test_base_worker_rejects_private_sr_before_execution(tmp_path):
    handler = _DetachedHandler()
    manager = DetachedOmniTaskManager(handler, artifact_root=tmp_path)
    identity = DetachedTaskIdentity("dep", "pool", "video-test", 1, "a" * 32)
    with pytest.raises(ValueError, match="not supported"):
        await manager._submit(identity, {SR_PAYLOAD_KEY: {}})
    with pytest.raises(ValueError, match="not supported"):
        async for _ in manager.generate({SR_PAYLOAD_KEY: {}}, FakeContext("test", {})):
            pass
    response = [
        v
        async for v in manager.generate(
            {ENVELOPE_KEY: {"schema_version": 1, "op": "capabilities"}},
            FakeContext("caps", {}),
        )
    ]
    assert not negotiated_sr_profiles(response[0])
    assert handler.calls == 0
