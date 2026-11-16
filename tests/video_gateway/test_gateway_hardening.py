# SPDX-License-Identifier: Apache-2.0
"""Regression for audited integration paths, real rejection and parent quotas."""
import asyncio
import base64

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from dingo.video_gateway.errors import EtcdLeaseNotFound, StoreUnavailable
from dingo.video_gateway.etcd_http import EtcdHttpClient
from dingo.video_gateway.models import TaskStatus, now_ms
from dingo.video_gateway.task_runtime.transactions import UnconfirmedTransaction
from dingo.video_gateway.task_store import EtcdTaskStore, MemoryTaskStore
from tests.video_gateway.test_context_ir_http import form
from tests.video_gateway.test_context_ir_worker_http import (
    wait_terminal,
)
from tests.video_gateway.test_context_ir_worker_http import (
    worker_stack as worker_stack,  # noqa: PLC0414 - fixture re-export
)
from tests.video_gateway.test_etcd_task_store import FakeEtcd
from tests.video_gateway.test_runtime_transactions_etcd import (
    runtime_etcd as runtime_etcd,  # noqa: PLC0414 - fixture re-export
)
from tests.video_gateway.test_task_store import _lease, _task


async def test_compiler_active_recovery_cannot_bypass_unknown_claim(worker_stack):
    service, manager, client, gate = worker_stack
    gate.clear()
    kv = service.store.client
    txn, get_many = kv.txn, kv.get_many
    injected = asyncio.Event()
    calls = 0
    original_execute = manager.handler.execute

    async def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return await original_execute(*args, **kwargs)

    async def lose_claim_reply(compare, writes, *args):
        result = await txn(compare, writes, *args)
        is_claim = any(
            "request_put" in op
            and "/active/context_ir/"
            in base64.b64decode(op["request_put"]["key"]).decode()
            for op in writes
        )
        if is_claim and not injected.is_set():
            injected.set()
            raise TimeoutError("claim committed but ACK lost")
        return result

    async def unavailable(*args, **kwargs):
        raise StoreUnavailable("read-back temporarily unavailable")

    manager.handler.execute = counted
    kv.txn, kv.get_many = lose_claim_reply, unavailable
    try:
        response = await client.post("/v1/context-ir", data=form())
        assert response.status == 202
        task_id = (await response.json())["id"]
        await asyncio.wait_for(injected.wait(), 2)
        await asyncio.sleep(0.6)  # Includes active-attempt recovery ticks.
        assert task_id in service.dispatcher.worker.coordinator.uncertain
        assert calls == 0 and not manager.jobs
        kv.txn, kv.get_many = txn, get_many
        gate.set()
        final = await wait_terminal(client, task_id)
        assert final["status"] == "completed" and calls == 1
    finally:
        kv.txn, kv.get_many = txn, get_many
        gate.set()


async def test_compiler_unknown_adoption_is_retained_by_common_coordinator(
    worker_stack, monkeypatch
):
    service, manager, client, gate = worker_stack
    worker = service.dispatcher.worker
    tick = worker.tick

    async def paused():
        pass

    # Persist an owned active attempt without launching it. This isolates the
    # recovery entry from normal queue selection and from the background tick.
    monkeypatch.setattr(worker, "tick", paused)
    monkeypatch.setattr(worker.coordinator, "enabled", lambda: False)
    await asyncio.sleep(0.2)
    response = await client.post("/v1/context-ir", data=form())
    task_id = (await response.json())["id"]
    await worker.refresh()
    candidate = (await worker.scheduling.candidates())[0]
    claimed = await worker.repo.claim(*candidate)
    assert claimed.task.id == task_id
    observed = False
    adoptions = 0

    async def resolve():
        if not observed:
            raise UnconfirmedTransaction(resolve)
        return claimed

    async def unknown_adoption(stored):
        nonlocal adoptions
        adoptions += 1
        raise UnconfirmedTransaction(resolve)

    monkeypatch.setattr(worker.repo, "adopt", unknown_adoption)
    monkeypatch.setattr(worker, "tick", tick)
    gate.clear()
    try:
        await asyncio.sleep(0.6)
        assert task_id in worker.coordinator.uncertain
        assert adoptions == 1 and not manager.jobs
        observed = True
        gate.set()
        assert (await wait_terminal(client, task_id))["status"] == "completed"
        assert adoptions == 1
    finally:
        gate.set()


@pytest.mark.parametrize(
    "body,typed",
    [
        ({"error": "etcdserver: requested lease not found", "code": 5}, True),
        ({"error": "etcdserver: requested lease not found", "code": 6}, False),
        ({"error": "some other not found", "code": 5}, False),
        ("proxy: etcdserver: requested lease not found", False),
    ],
)
async def test_only_structured_etcd_lease_rejection_is_definitive(body, typed):
    async def reject(request):
        if isinstance(body, dict):
            return web.json_response(body, status=404)
        return web.Response(text=body, status=404)

    app = web.Application()
    app.router.add_post("/v3/kv/txn", reject)
    server = TestServer(app)
    await server.start_server()
    client = EtcdHttpClient(str(server.make_url("")).rstrip("/"))
    try:
        with pytest.raises(EtcdLeaseNotFound if typed else StoreUnavailable) as error:
            await client.txn([], [])
        assert isinstance(error.value, EtcdLeaseNotFound) is typed
    finally:
        await client.close()
        await server.close()


async def test_real_expired_lease_before_claim_returns_to_queue_without_retry(
    runtime_etcd,
):
    client = runtime_etcd
    store = EtcdTaskStore(client, prefix="/expired-before-claim", deployment_id="test")
    original, _ = await store.create_task(
        _task("request"), principal_hash="p", idempotency_hash=None, queue_limit=8
    )
    lease = _lease(original.task)
    lease.execution_token = "a" * 32
    await store.register_gateway(lease.owner_generation, ttl_s=30)
    grant = client.lease_grant
    once = False

    async def expired_once(*args, **kwargs):
        nonlocal once
        result = await grant(*args, **kwargs)
        if not once:
            once = True
            await client.lease_revoke(result.lease_id)
        return result

    client.lease_grant = expired_once
    assert await store.reserve(original, lease, deadline_at_ms=now_ms() + 60000) is None
    unchanged = await store.get_task(original.task.id)
    assert unchanged.task.status is TaskStatus.QUEUED and unchanged.task.attempt == 0
    assert await store.queue_depth(unchanged.task.pool_id) == 1
    assert not await store.list_leases(unchanged.task.pool_id)
    claimed = await store.reserve(unchanged, lease, deadline_at_ms=now_ms() + 60000)
    assert claimed.task.attempt == 1
    assert await store.queue_depth(claimed.task.pool_id) == 0
    await store.heartbeat_lease(
        claimed.task.pool_id,
        claimed.task.worker_key,
        claimed.task.id,
        claimed.task.worker_lease_id,
    )


async def _parents(store):
    records = []
    for tid in ["parent-a", "parent-b"]:
        task = _task(tid, pool_id="parents")
        task.normalized_request = {"_pipeline": {"phase": "base", "children": {}}}
        stored, _ = await store.create_task(
            task, principal_hash="p", idempotency_hash=None, queue_limit=8
        )
        records.append(stored)
    return records


async def _check_parent_quota(first, second):
    a, b = await _parents(first)
    started = await asyncio.gather(
        first.start_pipeline_parent(a, max_active=1, deadline_at_ms=now_ms() + 60000),
        second.start_pipeline_parent(b, max_active=1, deadline_at_ms=now_ms() + 60000),
    )
    assert sum(v is not None for v in started) == 1
    counts = await first.task_counts("parents")
    assert counts[TaskStatus.DISPATCHING] == 1 and counts[TaskStatus.QUEUED] == 1
    active = next(v for v in started if v)
    await first.transition(
        active.task.id,
        expected={TaskStatus.DISPATCHING},
        patch={"status": TaskStatus.FAILED},
    )
    waiting = a if started[0] is None else b
    assert await second.start_pipeline_parent(
        waiting, max_active=1, deadline_at_ms=now_ms() + 60000
    )


@pytest.mark.parametrize("kind", ["memory", "fake-etcd"])
async def test_concurrent_parent_start_is_bounded_and_releases_on_terminal(kind):
    if kind == "memory":
        first = second = MemoryTaskStore()
    else:
        client = FakeEtcd()
        first = EtcdTaskStore(client, prefix="/quota", deployment_id="test")
        second = EtcdTaskStore(client, prefix="/quota", deployment_id="test")
    await _check_parent_quota(first, second)


async def test_real_concurrent_gateways_share_parent_quota(runtime_etcd):
    first = EtcdTaskStore(runtime_etcd, prefix="/quota", deployment_id="test")
    # Separate adapter instances share only the persisted etcd keys.
    second = EtcdTaskStore(runtime_etcd, prefix="/quota", deployment_id="test")
    await _check_parent_quota(first, second)


@pytest.mark.parametrize("kind", ["memory", "fake-etcd"])
async def test_existing_parent_is_counted_without_new_fence_bootstrap(kind):
    store = (
        MemoryTaskStore()
        if kind == "memory"
        else EtcdTaskStore(FakeEtcd(), prefix="/quota", deployment_id="test")
    )
    a, b = await _parents(store)
    # Simulate an existing parent started by the previous implementation.
    await store.transition(
        a.task.id,
        expected={TaskStatus.QUEUED},
        patch={"status": TaskStatus.DISPATCHING},
    )
    assert (
        await store.start_pipeline_parent(
            b, max_active=1, deadline_at_ms=now_ms() + 60000
        )
        is None
    )


@pytest.mark.parametrize("kind", ["memory", "fake-etcd"])
async def test_cancelled_or_stale_parent_cannot_consume_quota(kind):
    store = (
        MemoryTaskStore()
        if kind == "memory"
        else EtcdTaskStore(FakeEtcd(), prefix="/quota", deployment_id="test")
    )
    a, b = await _parents(store)
    await store.request_cancel(a.task.id)
    assert (
        await store.start_pipeline_parent(
            a, max_active=1, deadline_at_ms=now_ms() + 60000
        )
        is None
    )
    assert await store.start_pipeline_parent(
        b, max_active=1, deadline_at_ms=now_ms() + 60000
    )
