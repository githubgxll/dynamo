import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock

import pytest

from dingo.video_gateway import dispatcher as m
from dingo.video_gateway.models import TaskStatus, now_ms
from dingo.common.video_task_protocol import WAIT_TERMINAL_CAPABILITY


def fixture():
    d = object.__new__(m.VideoDispatcher)
    d._gateway_owner_healthy = d._task_watch_healthy = True
    d._worker_liveness_checks = asyncio.Semaphore(4)
    d.telemetry = MagicMock()
    task = NS(id='liveness-test', deployment_id='isolated', pool_id='pool',
              attempt=1, execution_token='a' * 32, worker_instance_id=101,
              deadline_at_ms=now_ms() + 60000)
    target = 'dyn://isolated.backend.generate'
    pool = NS(config=NS(pool_id='pool', backend_target=target), instance_ids=[],
              discovery_healthy=True, lease_watch_healthy=True)
    d.store = NS(discovery_truth_supported=True,
                 discovery_instance_snapshot=AsyncMock(return_value={target: set()}))
    d._current_owned_execution = AsyncMock(return_value=NS(task=NS(
        status=TaskStatus.IN_PROGRESS, cancel_requested_at_ms=None)))
    status = dict(schema_version=1, deployment_id=task.deployment_id,
                  pool_id=task.pool_id, task_id=task.id, attempt=task.attempt,
                  execution_token=task.execution_token, state='running',
                  updated_at_ms=now_ms() - 60000,
                  capabilities=[WAIT_TERMINAL_CAPABILITY])
    d.artifacts = NS(read_detached_status=AsyncMock(return_value=status))
    return d, pool, task, status


async def forever():
    await asyncio.Event().wait()


async def test_five_second_default_and_four_check_limit():
    assert m._WORKER_LIVENESS_CHECK_INTERVAL_S == 5.0
    assert m._WORKER_LIVENESS_CHECK_CONCURRENCY == 4


async def test_100_registered_executions_do_not_read_remote_state():
    d, pool, task, _ = fixture()
    pool.instance_ids = [101]
    assert not any(await asyncio.gather(*(d._confirm_worker_loss(pool, task) for _ in range(100))))
    d.store.discovery_instance_snapshot.assert_not_awaited()
    d.artifacts.read_detached_status.assert_not_awaited()


@pytest.mark.parametrize('flag', ['owner', 'task_watch', 'discovery', 'lease_watch'])
async def test_unhealthy_control_plane_never_means_worker_loss(flag):
    d, pool, task, _ = fixture()
    if flag == 'owner': d._gateway_owner_healthy = False
    if flag == 'task_watch': d._task_watch_healthy = False
    if flag == 'discovery': pool.discovery_healthy = False
    if flag == 'lease_watch': pool.lease_watch_healthy = False
    assert not await d._confirm_worker_loss(pool, task)
    d.store.discovery_instance_snapshot.assert_not_awaited()


async def test_local_missing_but_authoritatively_present_is_not_lost():
    d, pool, task, _ = fixture()
    d.store.discovery_instance_snapshot.return_value = {pool.config.backend_target: {101}}
    assert not await d._confirm_worker_loss(pool, task)
    d.artifacts.read_detached_status.assert_not_awaited()


@pytest.mark.parametrize('where', ['discovery', 'artifact'])
async def test_failed_evidence_reads_are_inconclusive(where):
    d, pool, task, _ = fixture()
    call = d.store.discovery_instance_snapshot if where == 'discovery' else d.artifacts.read_detached_status
    call.side_effect = RuntimeError('unavailable/corrupt test evidence')
    assert not await d._confirm_worker_loss(pool, task)


@pytest.mark.parametrize('state', ['completed', 'failed', 'cancelled', 'not_found'])
async def test_reported_terminal_or_unknown_status_is_not_worker_loss(state):
    d, pool, task, status = fixture(); status['state'] = state
    assert not await d._confirm_worker_loss(pool, task)


@pytest.mark.parametrize('heartbeat', [None, True, 'invalid', 'fresh', 'future'])
async def test_invalid_or_recent_heartbeat_does_not_trigger_retry(heartbeat):
    d, pool, task, status = fixture()
    status['updated_at_ms'] = now_ms() + (60000 if heartbeat == 'future' else 0) if heartbeat in ('fresh', 'future') else heartbeat
    assert not await d._confirm_worker_loss(pool, task)


async def test_cancel_and_owner_change_prevent_retry():
    d, pool, task, _ = fixture()
    d._current_owned_execution.return_value.task.cancel_requested_at_ms = now_ms()
    assert not await d._confirm_worker_loss(pool, task)
    d._current_owned_execution.return_value = None
    with pytest.raises(m._TaskOwnershipLost): await d._confirm_worker_loss(pool, task)


async def test_expired_execution_deadline_is_not_reclassified():
    d, pool, task, _ = fixture(); task.deadline_at_ms = now_ms() - 1
    assert not await d._confirm_worker_loss(pool, task)
    d.store.discovery_instance_snapshot.assert_not_awaited()


async def test_max_four_remote_confirmations_and_recheck_after_waiting():
    d, pool, task, _ = fixture(); entered = 0; maximum = 0; active = 0
    gate = asyncio.Event()
    async def snapshot(_):
        nonlocal entered, maximum, active
        entered += 1; active += 1; maximum = max(maximum, active)
        try: await gate.wait(); return {pool.config.backend_target: {101}}
        finally: active -= 1
    d.store.discovery_instance_snapshot.side_effect = snapshot
    tasks = [asyncio.create_task(d._confirm_worker_loss(pool, task)) for _ in range(100)]
    try:
        for _ in range(10): await asyncio.sleep(0)
        assert entered == maximum == 4
        pool.instance_ids = [101]; gate.set()
        assert not any(await asyncio.gather(*tasks))
        assert entered == 4
    finally:
        gate.set()
        for t in tasks: t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_hung_attached_wait_is_interrupted_and_children_are_cleaned(monkeypatch):
    d, pool, task, status = fixture()
    monkeypatch.setattr(m, '_WORKER_LIVENESS_CHECK_INTERVAL_S', .01)
    attached = asyncio.Event(); stream_closed = asyncio.Event()
    async def stream():
        try:
            yield {**status, 'state': 'watching'}
            attached.set()
            await forever()
        finally: stream_closed.set()
    async def direct(*_): return stream()
    pool.client = NS(direct=direct); pool.instance_ids = [101]
    d._shrink_to_result_memory_budget = AsyncMock()
    running = NS(worker_accepted=True)
    heartbeat = asyncio.create_task(forever())
    operation = asyncio.create_task(d._run_with_lease_monitor(
        d._consume_detached_worker(pool, NS(task=task), None, None, None,
                                   running, initial_worker_status=status),
        heartbeat, forever(), d._monitor_worker_liveness(pool, task, running)))
    try:
        await asyncio.wait_for(attached.wait(), 1)
        pool.instance_ids = []
        with pytest.raises(m._RetryableWorkerFailure): await asyncio.wait_for(operation, 1)
        assert stream_closed.is_set()
        assert d.store.discovery_instance_snapshot.await_count >= 1
    finally:
        heartbeat.cancel(); operation.cancel()
        await asyncio.gather(heartbeat, operation, return_exceptions=True)


async def test_registered_long_wait_and_cancel_cleanup(monkeypatch):
    d, pool, task, _ = fixture(); pool.instance_ids = [101]
    monkeypatch.setattr(m, '_WORKER_LIVENESS_CHECK_INTERVAL_S', .01)
    heartbeat = asyncio.create_task(forever())
    operation = asyncio.create_task(d._run_with_lease_monitor(
        forever(), heartbeat, forever(), d._monitor_worker_liveness(pool, task, NS(worker_accepted=True))))
    try:
        await asyncio.sleep(.055)
        assert not operation.done()
        d.store.discovery_instance_snapshot.assert_not_awaited()
        d.artifacts.read_detached_status.assert_not_awaited()
    finally:
        operation.cancel(); heartbeat.cancel()
        await asyncio.gather(operation, heartbeat, return_exceptions=True)


async def test_completed_result_wins_simultaneous_failure():
    d, _, _, _ = fixture()
    async def completed(): return 'result'
    async def lost(): raise m._RetryableWorkerFailure('lost')
    heartbeat = asyncio.create_task(forever())
    try: assert await d._run_with_lease_monitor(completed(), heartbeat, None, lost()) == 'result'
    finally:
        heartbeat.cancel(); await asyncio.gather(heartbeat, return_exceptions=True)


async def test_lease_loss_wins_simultaneous_liveness_failure():
    d, _, _, _ = fixture()
    async def lease_lost(): raise m._WorkerLeaseLost('owner lost')
    async def worker_lost(): raise m._RetryableWorkerFailure('worker lost')
    heartbeat = asyncio.create_task(lease_lost())
    with pytest.raises(m._WorkerLeaseLost):
        await d._run_with_lease_monitor(forever(), heartbeat, None, worker_lost())
