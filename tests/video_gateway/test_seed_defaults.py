# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import dataclasses
import json
from pathlib import Path

import pytest

from dingo.video_gateway.config import SeedDefaultsConfig, parse_config
from dingo.video_gateway.errors import GatewayError
from dingo.video_gateway.models import TaskStatus, VideoTask
from dingo.video_gateway.performance import performance_record
from tests.video_gateway.test_config import _raw
from tests.video_gateway.test_dispatcher import FakeClient, _stack
from tests.video_gateway.test_minimax_h3_adapter import _png_upload


@pytest.mark.parametrize(
    "raw",
    [
        {},
        None,
        {"strategy": "other"},
        {"strategy": "fixed"},
        {"strategy": "fixed", "value": True},
        {"strategy": "fixed", "value": 1.5},
        {"strategy": "fixed", "value": "42"},
        {"strategy": "fixed", "value": -1},
        {"strategy": "fixed", "value": 2**32},
        {"strategy": "random", "value": None},
        {"strategy": "random", "unknown": 1},
    ],
)
def test_invalid_seed_defaults(tmp_path, raw):
    config = _raw(tmp_path)
    config["seed_defaults"] = raw
    with pytest.raises((ValueError, TypeError), match="seed_defaults"):
        parse_config(config)


@pytest.mark.parametrize(
    "raw",
    [
        {"strategy": "random"},
        {"strategy": "fixed", "value": 0},
        {"strategy": "fixed", "value": 2**32 - 1},
    ],
)
def test_seed_defaults_do_not_change_pool_revision(tmp_path, raw):
    config = _raw(tmp_path)
    original = parse_config(config)
    config["seed_defaults"] = raw
    parsed = parse_config(config)
    assert parsed.seed_defaults.strategy == raw["strategy"]
    assert parsed.seed_defaults.value == raw.get("value")
    assert [p.configuration_revision for p in parsed.pools] == [
        p.configuration_revision for p in original.pools
    ]


async def submit(service, seed=None, key=None, reference=False):
    root = await service.artifacts.create_upload()
    fields = {
        "model": ["public-fl"],
        "prompt": ["seed test"],
        "seconds": ["4"],
        "size": ["1344x768"],
    }
    if seed is not None:
        fields["seed"] = [str(seed)]
    return await service.submit(
        fields=fields,
        uploads=[_png_upload(root)] if reference else [],
        upload_root=root,
        delivery_mode="async",
        idempotency_key=key,
    )


@pytest.mark.parametrize("workflow", ["fl2va", "ref2va"])
@pytest.mark.parametrize(
    "policy,explicit,expected,source",
    [
        (None, None, None, None),
        (SeedDefaultsConfig("fixed", 42), None, 42, "default_fixed"),
        (SeedDefaultsConfig("fixed", 0), None, 0, "default_fixed"),
        (SeedDefaultsConfig("random"), None, 1234, "default_random"),
        (SeedDefaultsConfig("fixed", 42), 0, 0, "request"),
        (SeedDefaultsConfig("random"), 2**32 - 1, 2**32 - 1, "request"),
    ],
)
async def test_persist_restore_and_forward_seed(
    make_gateway_config, monkeypatch, workflow, policy, explicit, expected, source
):
    config = make_gateway_config(accept_without_workers=True)
    pool = config.pools[0]
    pool = dataclasses.replace(
        pool, adapter=dataclasses.replace(pool.adapter, workflow=workflow)
    )
    config = dataclasses.replace(
        config,
        seed_defaults=policy,
        pools=(pool,),
        pools_by_id={pool.pool_id: pool},
        pools_by_model={"public-fl": pool},
    )
    store, _, _, service = _stack(config, {"fl-pool": FakeClient()})
    monkeypatch.setattr(
        "dingo.video_gateway.service.secrets.randbits", lambda bits: 1234
    )
    created = await submit(service, explicit, reference=workflow == "ref2va")
    task = VideoTask.from_dict(json.loads(json.dumps(created.stored.task.to_dict())))
    persisted = json.loads(Path(task.request_path).read_text())
    # Dispatch and persisted recovery retain the original resolved request.
    await store.transition(
        task.id, expected={TaskStatus.QUEUED}, patch={"status": TaskStatus.DISPATCHING}
    )
    restored = await store.transition(
        task.id,
        expected={TaskStatus.DISPATCHING},
        patch={"status": TaskStatus.IN_PROGRESS},
    )
    assert restored.task.normalized_request == task.normalized_request
    adapter = service.adapters["fl-pool"]
    manifest = json.loads(Path(task.input_manifest_path).read_text())
    payload = adapter.build_worker_payload(
        task.normalized_request, manifest, Path(task.request_path).parent
    )
    for value in (persisted, task.public_dict(), task.diagnostics_dict()):
        if expected is None:
            assert "seed" not in value and "seed_source" not in value
        else:
            assert value["seed"] == expected
            assert value["seed_source"] == source
            assert value["seed_generated"] == (source == "default_random")
    assert payload["nvext"].get("seed") == expected
    assert performance_record(task)["generation"]["seed_source"] == source


async def test_random_idempotency_race_and_policy_change(
    make_gateway_config, monkeypatch
):
    config = dataclasses.replace(
        make_gateway_config(accept_without_workers=True),
        seed_defaults=SeedDefaultsConfig("random"),
    )
    store, _, _, service = _stack(config, {"fl-pool": FakeClient()})
    values = iter([123, 456, 789])
    monkeypatch.setattr(
        "dingo.video_gateway.service.secrets.randbits", lambda bits: next(values)
    )
    original = store.create_task
    entered = 0
    both_entered = asyncio.Event()

    async def race(*args, **kwargs):
        nonlocal entered
        entered += 1
        if entered == 2:
            both_entered.set()
        await both_entered.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(store, "create_task", race)
    first, second = await asyncio.gather(
        submit(service, key="same"), submit(service, key="same")
    )
    assert first.stored.task.id == second.stored.task.id
    seed = first.stored.task.normalized_request["seed"]
    assert second.stored.task.normalized_request["seed"] == seed
    assert first.created != second.created
    # A later Gateway configuration change must not change an existing task or its digest.
    service.config = dataclasses.replace(
        config, seed_defaults=SeedDefaultsConfig("fixed", 42)
    )
    replay = await submit(service, key="same")
    assert not replay.created and replay.stored.task.normalized_request["seed"] == seed
    with pytest.raises(GatewayError) as error:
        await submit(service, seed=seed, key="same")
    assert error.value.code == "idempotency_conflict"
    monkeypatch.setattr(store, "create_task", original)
    service.config = config
    new = await submit(service)
    assert new.stored.task.normalized_request["seed"] == 789


@pytest.mark.parametrize("seed", ["-1", "4294967296", "1.5", ""])
async def test_invalid_request_not_replaced_by_default(make_gateway_config, seed):
    config = dataclasses.replace(
        make_gateway_config(accept_without_workers=True),
        seed_defaults=SeedDefaultsConfig("fixed", 42),
    )
    _, _, _, service = _stack(config, {"fl-pool": FakeClient()})
    with pytest.raises(GatewayError):
        await submit(service, seed=seed)


@pytest.mark.parametrize(
    "policy", [SeedDefaultsConfig("fixed", 42), SeedDefaultsConfig("random")]
)
async def test_etcd_retry_and_gateway_takeover_preserve_resolved_seed(
    make_gateway_config, policy,
):
    from dingo.video_gateway.models import now_ms
    from dingo.video_gateway.task_store import EtcdTaskStore
    from tests.video_gateway.test_etcd_task_store import FakeEtcd
    from tests.video_gateway.test_task_store import _lease

    config = dataclasses.replace(
        make_gateway_config(accept_without_workers=True), seed_defaults=policy
    )
    _, _, _, service = _stack(config, {"fl-pool": FakeClient()})
    submitted = await submit(service)
    original = submitted.stored.task
    store = EtcdTaskStore(
        FakeEtcd(), prefix="/seed-test", deployment_id=config.deployment_id
    )
    await store.register_gateway("generation", ttl_s=30)
    stored, _ = await store.create_task(
        original, principal_hash="p", idempotency_hash=None, queue_limit=8
    )
    lease = _lease(original)
    lease.execution_token = "c" * 32
    reserved = await store.reserve(
        stored, lease, deadline_at_ms=now_ms() + 60000, reserve_retry=True
    )
    retried = await store.requeue_failed_attempt(reserved, queue_limit=8)
    assert retried is not None
    assert retried.task.normalized_request == original.normalized_request
    second_lease = _lease(retried.task, instance_id=8)
    second_lease.execution_token = "d" * 32
    second = await store.reserve(retried, second_lease, deadline_at_ms=now_ms() + 60000)
    await store.register_gateway("next-generation", ttl_s=30)
    owner_lease = (await store.client.get(store._gateway_key("generation"))).lease
    await store.unregister_gateway(owner_lease)
    await store.client.lease_revoke(second.task.worker_lease_id)
    taken = await store.claim_orphaned_active(
        second, new_owner_generation="next-generation"
    )
    assert taken is not None
    assert taken.task.normalized_request == original.normalized_request
