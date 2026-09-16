import asyncio
import dataclasses
import errno
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dingo.video_gateway.finalization import ResultFinalizer
from dingo.video_gateway.models import TaskStatus, now_ms
from dingo.video_gateway.result_handoff import HANDOFF_KEY
from tests.video_gateway.test_result_handoff import patch_for, setup
from tests.video_gateway.test_task_store import _lease, _task


async def ready_task(make_gateway_config, tmp_path, operation):
    store, _, stored, _ = await setup("memory")
    ready = await store.transition(
        stored.task.id,
        expected={TaskStatus.IN_PROGRESS},
        expected_revision=stored.revision,
        patch=patch_for(stored),
        release_lease=True,
        release_execution=True,
    )
    config = make_gateway_config()
    scheduling = dataclasses.replace(
        config.pools[0].scheduling, finalization_retry_delay_s=0.001
    )
    pool = SimpleNamespace(
        config=SimpleNamespace(scheduling=scheduling),
        adapter=SimpleNamespace(
            validate_artifact=None,
            prepare_artifact=None,
            artifact_requires_processing=None,
            inspect_artifact_for_publication=None,
        ),
    )
    artifacts = SimpleNamespace(finalize_worker_mp4=operation)
    finalizer = ResultFinalizer(store, artifacts, config, Mock(), "generation")
    return store, ready, pool, finalizer


async def test_slow_finalization_does_not_hold_or_release_new_worker_slot(
    make_gateway_config, tmp_path
):
    entered, finish = asyncio.Event(), asyncio.Event()
    path = tmp_path / "candidate.mp4"

    async def operation(*args, **kwargs):
        entered.set()
        await finish.wait()
        path.write_bytes(b"mp4")
        return path, 3, "a" * 64, {}

    store, ready, pool, f = await ready_task(make_gateway_config, tmp_path, operation)
    pending = asyncio.create_task(f.run(ready, pool))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        task = _task("next")
        nxt, _ = await store.create_task(
            task, principal_hash="p", idempotency_hash=None, queue_limit=8
        )
        lease = _lease(task)
        lease.execution_token = "d" * 32
        assert (
            await store.reserve(nxt, lease, deadline_at_ms=now_ms() + 60000) is not None
        )
        finish.set()
        await asyncio.wait_for(pending, 1)
        assert (await store.get_task("first")).task.status == TaskStatus.COMPLETED
        assert (await store.list_leases(task.pool_id))[0].task_id == "next"
    finally:
        finish.set()
        await asyncio.gather(pending, return_exceptions=True)


async def test_transient_error_retries_only_postprocessing(
    make_gateway_config, tmp_path
):
    count = 0
    path = tmp_path / "result.mp4"

    async def operation(*args, **kwargs):
        nonlocal count
        count += 1
        if count < 3:
            raise OSError(errno.EIO, "transient storage error")
        path.write_bytes(b"mp4")
        return path, 3, "a" * 64, {}

    store, ready, pool, f = await ready_task(make_gateway_config, tmp_path, operation)
    await f.run(ready, pool)
    task = (await store.get_task("first")).task
    assert task.status == TaskStatus.COMPLETED and task.attempt == 1
    assert count == 3 and task.normalized_request[HANDOFF_KEY]["failures"] == 2
    assert not await store.list_leases(task.pool_id)


async def test_permanent_result_failure_is_not_worker_failure(
    make_gateway_config, tmp_path
):
    async def operation(*args, **kwargs):
        raise ValueError("invalid video")

    store, ready, pool, f = await ready_task(make_gateway_config, tmp_path, operation)
    await f.run(ready, pool)
    task = (await store.get_task("first")).task
    assert task.status == TaskStatus.FAILED and task.error.code == "finalization_failed"
    assert task.attempt == 1 and not await store.list_leases(task.pool_id)


async def test_cancel_during_finalize_discards_only_own_candidate(
    make_gateway_config, tmp_path
):
    entered, finish = asyncio.Event(), asyncio.Event()
    path = tmp_path / "candidate.mp4"

    async def operation(*args, **kwargs):
        entered.set()
        await finish.wait()
        path.write_bytes(b"mp4")
        return path, 3, "a" * 64, {}

    store, ready, pool, f = await ready_task(make_gateway_config, tmp_path, operation)
    pending = asyncio.create_task(f.run(ready, pool))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        await store.request_cancel("first")
        finish.set()
        await asyncio.wait_for(pending, 1)
        assert (await store.get_task("first")).task.status == TaskStatus.CANCELLED
        assert not path.exists()
    finally:
        finish.set()
        await asyncio.gather(pending, return_exceptions=True)


async def test_lost_commit_response_does_not_delete_published_result(
    make_gateway_config, tmp_path
):
    path = tmp_path / "candidate.mp4"

    async def operation(*args, **kwargs):
        path.write_bytes(b"mp4")
        return path, 3, "a" * 64, {}

    store, ready, pool, f = await ready_task(make_gateway_config, tmp_path, operation)
    original = store.transition

    async def ambiguous(*args, **kwargs):
        result = await original(*args, **kwargs)
        if kwargs["patch"].get("status") == TaskStatus.COMPLETED:
            raise ConnectionError("lost etcd reply after commit")
        return result

    store.transition = ambiguous
    with pytest.raises(ConnectionError):
        await f.run(ready, pool)
    assert path.read_bytes() == b"mp4"
    await f.run(ready, pool)
    assert (await store.get_task("first")).task.status == TaskStatus.COMPLETED


async def test_transient_errors_exhaust_bounded_retry_budget(
    make_gateway_config, tmp_path
):
    calls = 0

    async def operation(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise OSError(errno.EIO, "persistent outage")

    store, ready, pool, f = await ready_task(make_gateway_config, tmp_path, operation)
    await f.run(ready, pool)
    assert (
        calls == 3 and (await store.get_task("first")).task.status == TaskStatus.FAILED
    )


async def test_processing_deadline_is_not_a_worker_failure(
    make_gateway_config, tmp_path
):
    calls = 0

    async def operation(*args, **kwargs):
        nonlocal calls
        calls += 1
        await asyncio.sleep(10)

    store, ready, pool, f = await ready_task(make_gateway_config, tmp_path, operation)
    ref = dict(ready.task.normalized_request[HANDOFF_KEY])
    ref["deadline_at_ms"] = now_ms() + 30
    ready = await store.transition(
        "first",
        expected={TaskStatus.FINALIZING},
        patch={
            "normalized_request": {**ready.task.normalized_request, HANDOFF_KEY: ref}
        },
    )
    await asyncio.wait_for(f.run(ready, pool), 1)
    task = (await store.get_task("first")).task
    assert calls == 1 and task.status == TaskStatus.FAILED
    assert task.error.code == "finalization_timeout" and task.attempt == 1


async def test_resuming_finalizer_uses_original_durable_result(
    make_gateway_config, tmp_path
):
    path = tmp_path / "result.mp4"
    entered = asyncio.Event()

    async def operation(*args, **kwargs):
        entered.set()
        await asyncio.sleep(10)

    store, ready, pool, f = await ready_task(make_gateway_config, tmp_path, operation)
    pending = asyncio.create_task(f.run(ready, pool))
    await asyncio.wait_for(entered.wait(), 1)
    pending.cancel()
    await asyncio.gather(pending, return_exceptions=True)
    assert (await store.get_task("first")).task.status == TaskStatus.FINALIZING

    async def recovered(*args, **kwargs):
        assert args[5] == ready.task.normalized_request[HANDOFF_KEY]["artifact"]
        path.write_bytes(b"mp4")
        return path, 3, "a" * 64, {}

    f.artifacts.finalize_worker_mp4 = recovered
    await f.run(ready, pool)
    assert (await store.get_task("first")).task.status == TaskStatus.COMPLETED
    assert not await store.list_leases(ready.task.pool_id)
