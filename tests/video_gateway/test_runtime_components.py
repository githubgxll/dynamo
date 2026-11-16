# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import replace

import pytest

from dingo.video_gateway.task_runtime.capacity import available_candidates
from dingo.video_gateway.task_runtime.records import RecordRepository
from dingo.video_gateway.task_runtime.registry import (
    CapabilityRegistry,
    refresh_instances,
)
from dingo.video_gateway.task_runtime.repository import (
    KeySnapshot,
    RuntimeKeys,
    WorkerRef,
    WorkerSnapshot,
)
from tests.video_gateway.test_etcd_task_store import FakeEtcd
from tests.video_gateway.test_task_protocol_v2 import capabilities


async def test_discovery_invalidates_stale_cache_and_bounds_parallel_probes():
    cache = {99: (16, float('inf'))}
    active = peak = 0
    published = []

    async def probe(instance):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        return 16, float('inf')

    await refresh_instances(list_instances=lambda: list(range(20))+[1], cache=cache,
                            probe=probe, publish=lambda ids, healthy: published.append((ids, healthy)))
    assert published == [(list(range(20)), True)]
    assert len(cache) == 20 and 99 not in cache and peak == 8


async def test_discovery_failure_publishes_no_workers_and_clears_cache():
    cache = {1: (16, float('inf'))}
    def failed():
        raise ConnectionError('discovery down')
    states = []
    await refresh_instances(list_instances=failed, cache=cache, probe=None,
                            publish=lambda *state: states.append(state))
    assert states == [([], False)] and cache == {}


def test_v2_registry_expiry_generation_and_admission():
    registry = CapabilityRegistry()
    wire = capabilities()
    registry.observe('w', wire, now=10)
    opts = {'task_kind': 'context_ir', 'storage_id': 'shared',
                'model_adapter': ('minimax_h3', 1), 'protocol_adapter': ('open_h3_ir', 1)}
    assert len(registry.eligible(**opts, now=11)) == 1
    assert not registry.eligible(**opts, now=15)
    assert not registry.eligible(**{**opts, 'storage_id':'other'}, now=11)
    registry.observe('w', {**wire, 'generation':'next', 'accepting':False}, now=20)
    assert not registry.eligible(**opts, now=21)
    assert registry.eligible(**opts, queue=True, now=21)[0].generation == 'next'
    with pytest.raises(ValueError):
        registry.observe('w', {**wire, 'instance_id':'wrong'}, now=22)
    assert not registry.eligible(**opts, queue=True, now=22)
    registry.observe('w', wire, now=23)
    registry.reconcile([])
    assert not registry.records


def worker(instance='one', generation='g', capacity=16):
    return WorkerSnapshot(WorkerRef('pool', instance, generation, instance),
                          KeySnapshot('/worker/'+instance, 1, b'ready'), capacity)


def test_physical_capacity_shared_by_aliases_and_independent_instances():
    first = worker()
    alias = replace(first, ref=replace(first.ref, pool_id='alias'))
    keys = RuntimeKeys('deployment')
    assert len(available_candidates([first, alias], [], deployment_id='deployment')) == 16
    assert len(available_candidates([first, worker('two')], [], deployment_id='deployment')) == 32
    occupied = [keys.slot(first.ref, i) for i in range(16)]
    assert not available_candidates([first, alias], occupied, deployment_id='deployment')
    assert not available_candidates([first], [], deployment_id='deployment', blocked_domains=['one'])
    new_boot = replace(first, ref=replace(first.ref, generation='new'))
    assert not available_candidates([first,new_boot], [], deployment_id='deployment')
    assert len(available_candidates([first], [], deployment_id='deployment', configured_capacity=4)) == 4


def test_capacity_prefers_free_fraction_then_rotates_ties():
    first, second = worker(), worker('two')
    keys = RuntimeKeys('deployment')
    occupied = [keys.slot(first.ref, i) for i in range(10)]
    candidates = available_candidates([first,second], occupied, deployment_id='deployment')
    assert candidates[0][0] == second
    tied = available_candidates([first,second], [], deployment_id='deployment', cursor=1)
    assert tied[0][0] == second


async def test_record_adapter_preserves_bytes_and_owner_fence():
    client = FakeEtcd()
    repo = RecordRepository(client, lambda task_id:'/tasks/'+task_id, lambda v:v)
    assert await repo.load('missing') is None
    await client.txn([], [client.put('/tasks/id', b'{"schema_version":1}')])
    before = await repo.load('id')
    assert before.value == b'{"schema_version":1}'
    owner_guard = client.compare_value('/owner', 'current')
    ok, _ = await repo.update('id', before.mod_revision, b'{"schema_version":1,"status":"done"}', compares=[owner_guard])
    assert not ok and (await repo.load('id')) == before
    await client.txn([], [client.put('/owner','current')])
    ok, _ = await repo.update('id', before.mod_revision, b'new', compares=[owner_guard])
    assert ok
    ok, _ = await repo.update('id', before.mod_revision, b'stale', compares=[owner_guard])
    assert not ok and (await repo.load('id')).value == b'new'


def test_duplicate_runtime_registrations_cannot_multiply_one_qwen():
    first=worker()
    duplicate=replace(first,ref=replace(first.ref,instance_id="second-registration"))
    assert not available_candidates([first,duplicate],[],deployment_id="deployment")


@pytest.mark.parametrize("changed", [{"generation": "next"}, {"instance_id": "next"}])
def test_old_physical_slot_blocks_new_registration_until_reconciled(changed):
    original, independent = worker(), worker("two")
    replacement = replace(original, ref=replace(original.ref, **changed))
    occupied = [RuntimeKeys("deployment").slot(original.ref, 0)]
    choices = available_candidates(
        [replacement, independent], occupied, deployment_id="deployment"
    )
    assert len(choices) == 16
    assert all(candidate == independent for candidate, _ in choices)
    assert len(available_candidates([replacement], [], deployment_id="deployment")) == 16
