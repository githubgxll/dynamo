import asyncio

import pytest

from dingo.common.video_result_file import BINARY_RESULT_WRITER
from dingo.video_gateway.models import TaskStatus
from dingo.video_gateway.result_handoff import read_handoff
from dingo.vllm.omni.detached_tasks import DetachedOmniTaskManager
from tests.video_gateway.test_dispatcher import (
    _MINIMAL_MP4,
    _DetachedClient,
    _pool,
    _stack,
    _submit,
)


@pytest.mark.parametrize("prefetch", [0, 1])
async def test_next_task_starts_while_previous_gateway_finalization_is_blocked(
    make_gateway_config, monkeypatch, prefetch
):
    calls = []
    second_started = asyncio.Event()
    first_finalizing = asyncio.Event()
    finish = asyncio.Event()

    class Handler:
        async def generate(self, request, context):
            calls.append(context)
            if len(calls) == 2:
                second_started.set()
            desc = await BINARY_RESULT_WRITER.get().write(_MINIMAL_MP4)
            yield {
                "status": "completed",
                "data": [{"output_format": "mp4", "artifact": desc}],
            }

    pool = _pool("fl-pool", "public-fl", "dyn://scope.backend.generate")
    pool["execution_mode"] = "detached"
    pool["scheduling"]["early_release_slot"] = True
    pool["scheduling"]["worker_prefetch_capacity"] = prefetch
    config = make_gateway_config(pools=[pool])
    manager = DetachedOmniTaskManager(
        Handler(),
        config.artifact_store.root,
        binary_results=True,
        inline_results=True,
        prefetch_capacity=prefetch,
    )
    store, artifacts, dispatcher, service = _stack(
        config, {"fl-pool": _DetachedClient(manager)}
    )
    original = artifacts.finalize_worker_mp4
    count = 0

    async def slow(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 1:
            first_finalizing.set()
            await finish.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(artifacts, "finalize_worker_mp4", slow)
    await dispatcher.start()
    try:
        first = await _submit(service, "public-fl")
        await asyncio.wait_for(first_finalizing.wait(), 3)
        current = await store.get_task(first.stored.task.id)
        assert current.task.status == TaskStatus.FINALIZING and read_handoff(
            current.task
        )
        assert not await store.list_leases("fl-pool")
        second = await _submit(service, "public-fl")
        await asyncio.wait_for(second_started.wait(), 2)
        assert not finish.is_set()
        finish.set()
        terminals = [
            await dispatcher.wait_terminal(submitted.stored.task.id, 3)
            for submitted in [first, second]
        ]
        assert all(item.task.status == TaskStatus.COMPLETED for item in terminals)
        if prefetch:
            assert all(
                item.task.stage_durations is not None
                and item.task.stage_durations["worker_queue_wait"] >= 0
                and item.task.public_dict()["metrics"]["worker_queue_wait_s"]
                == item.task.stage_durations["worker_queue_wait"]
                for item in terminals
            )
        else:
            assert all(
                "worker_queue_wait" not in (item.task.stage_durations or {})
                for item in terminals
            )
        assert len(calls) == 2
    finally:
        finish.set()
        await dispatcher.stop()
        await manager.shutdown()


@pytest.mark.parametrize(
    "fault",
    [
        "lost_handoff_reply",
        "completion_store_unavailable",
        "lost_reservation",
        "store_unavailable_then_recovers",
        "deterministic_handoff_error",
        "deterministic_handoff_error_after_slot_reuse",
        "broken_terminal_writer",
    ],
)
async def test_handoff_storage_faults_do_not_rerun_model(
    make_gateway_config, monkeypatch, fault
):
    from dingo.video_gateway.errors import HandoffReservationLost, StoreUnavailable

    calls = 0

    class Handler:
        async def generate(self, request, context):
            nonlocal calls
            calls += 1
            desc = await BINARY_RESULT_WRITER.get().write(_MINIMAL_MP4)
            yield {
                "status": "completed",
                "data": [{"output_format": "mp4", "artifact": desc}],
            }

    pool = _pool("fl-pool", "public-fl", "dyn://scope.backend.generate")
    pool["execution_mode"] = "detached"
    pool["scheduling"]["early_release_slot"] = True
    config = make_gateway_config(pools=[pool])
    manager = DetachedOmniTaskManager(
        Handler(), config.artifact_store.root, binary_results=True, inline_results=True
    )
    store, artifacts, dispatcher, service = _stack(
        config, {"fl-pool": _DetachedClient(manager)}
    )
    original = store.transition
    injected = False
    handoff_attempts = 0

    async def transition(*args, **kwargs):
        nonlocal handoff_attempts, injected
        handoff = (
            kwargs.get("release_execution", False)
            and kwargs["patch"].get("status") == TaskStatus.FINALIZING
        )
        if handoff:
            handoff_attempts += 1
        if not injected and handoff and fault == "lost_reservation":
            injected = True
            raise HandoffReservationLost("injected reservation replacement")
        if handoff and fault in {
            "deterministic_handoff_error",
            "broken_terminal_writer",
        }:
            injected = True
            raise RuntimeError("injected deterministic handoff bug")
        if (
            fault == "broken_terminal_writer"
            and kwargs["patch"].get("status") == TaskStatus.FAILED
        ):
            raise RuntimeError("terminal writer is broken too")
        if handoff and fault == "deterministic_handoff_error_after_slot_reuse":
            injected = True
            key = next(iter(store._leases))
            store._leases[key].execution_token = "e" * 32
            raise RuntimeError("injected deterministic handoff bug after slot reuse")
        if (
            handoff
            and fault == "store_unavailable_then_recovers"
            and handoff_attempts <= 3
        ):
            injected = True
            raise StoreUnavailable("injected etcd outage before publication")
        if (
            not injected
            and fault == "completion_store_unavailable"
            and kwargs["patch"].get("status") == TaskStatus.COMPLETED
        ):
            injected = True
            raise ConnectionError("injected etcd outage before publication")
        result = await original(*args, **kwargs)
        if not injected and handoff and fault == "lost_handoff_reply":
            injected = True
            raise ConnectionError("injected lost handoff reply after commit")
        return result

    monkeypatch.setattr(store, "transition", transition)
    await dispatcher.start()
    try:
        submitted = await _submit(service, "public-fl")
        if fault == "broken_terminal_writer":
            await asyncio.wait_for(dispatcher._fatal_event.wait(), 3)
            assert injected and calls == handoff_attempts == 1
            assert not dispatcher.ready and dispatcher.draining
            current = await store.get_task(submitted.stored.task.id)
            assert current.task.status == TaskStatus.IN_PROGRESS
            for _ in range(100):
                if (await dispatcher.memory_budget.snapshot()).used_bytes == 0:
                    break
                await asyncio.sleep(0.01)
            assert (await dispatcher.memory_budget.snapshot()).used_bytes == 0
            return
        terminal = await dispatcher.wait_terminal(submitted.stored.task.id, 6)
        assert injected and calls == 1 and terminal.task.attempt == 1
        if fault in {
            "lost_reservation",
            "deterministic_handoff_error",
            "deterministic_handoff_error_after_slot_reuse",
        }:
            assert terminal.task.status == TaskStatus.FAILED
            assert terminal.task.error.code == (
                "result_handoff_failed"
                if fault == "deterministic_handoff_error"
                else "result_handoff_lost_reservation"
            )
        else:
            assert terminal.task.status == TaskStatus.COMPLETED
            assert not await store.list_leases("fl-pool")
        if fault in {
            "deterministic_handoff_error",
            "deterministic_handoff_error_after_slot_reuse",
        }:
            assert handoff_attempts == 1
        elif fault == "store_unavailable_then_recovers":
            assert handoff_attempts == 4
        for _ in range(100):
            if (await dispatcher.memory_budget.snapshot()).used_bytes == 0:
                break
            await asyncio.sleep(0.01)
        assert (await dispatcher.memory_budget.snapshot()).used_bytes == 0
        assert not dispatcher._finalizing
        if fault == "deterministic_handoff_error":
            assert not await store.list_leases("fl-pool")
        elif fault == "deterministic_handoff_error_after_slot_reuse":
            leases = await store.list_leases("fl-pool")
            assert len(leases) == 1 and leases[0].execution_token == "e" * 32
    finally:
        await dispatcher.stop()
        await manager.shutdown()
