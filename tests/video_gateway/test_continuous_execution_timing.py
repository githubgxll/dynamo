"""Opt-in local continuity experiment, not a GPU/DingoFS throughput benchmark."""

import asyncio
import json
import os
import time

import pytest

from dingo.common.video_result_file import BINARY_RESULT_WRITER
from dingo.video_gateway.models import TaskStatus
from dingo.vllm.omni.detached_tasks import DetachedOmniTaskManager
from tests.video_gateway.test_dispatcher import (
    _MINIMAL_MP4,
    _DetachedClient,
    _pool,
    _stack,
    _submit,
)


@pytest.mark.skipif(
    os.environ.get("DINGO_CONTINUOUS_TIMING") != "1",
    reason="opt-in wall-clock continuity experiment (~2 minutes)",
)
@pytest.mark.parametrize("compute_s", [3, 5])
@pytest.mark.parametrize("enabled", [False, True])
async def test_worker_continuity_with_half_second_postprocessing(
    make_gateway_config, monkeypatch, compute_s, enabled, record_property
):
    starts, finishes = [], []

    class Handler:
        async def generate(self, request, context):
            starts.append(time.monotonic())
            await asyncio.sleep(compute_s)
            descriptor = await BINARY_RESULT_WRITER.get().write(_MINIMAL_MP4)
            finishes.append(time.monotonic())
            yield {
                "status": "completed",
                "data": [{"output_format": "mp4", "artifact": descriptor}],
            }

    pool = _pool("fl-pool", "public-fl", "dyn://scope.backend.generate")
    pool["execution_mode"] = "detached"
    pool["scheduling"].update(
        early_release_slot=enabled,
        worker_prefetch_capacity=int(enabled),
        execution_timeout_s=60,
    )
    config = make_gateway_config(pools=[pool])
    manager = DetachedOmniTaskManager(
        Handler(),
        config.artifact_store.root,
        binary_results=True,
        inline_results=True,
        prefetch_capacity=int(enabled),
    )
    store, artifacts, dispatcher, service = _stack(
        config, {"fl-pool": _DetachedClient(manager)}
    )
    original = artifacts.finalize_worker_mp4

    async def slow(*args, **kwargs):
        await asyncio.sleep(0.5)
        return await original(*args, **kwargs)

    monkeypatch.setattr(artifacts, "finalize_worker_mp4", slow)
    await dispatcher.start()
    try:
        started = time.monotonic()
        submitted = [await _submit(service, "public-fl") for _ in range(6)]
        for item in submitted:
            terminal = await dispatcher.wait_terminal(item.stored.task.id, 60)
            assert terminal.task.status == TaskStatus.COMPLETED
        elapsed = time.monotonic() - started
        gaps = [starts[i] - finishes[i - 1] for i in range(1, len(starts))]
        measured = dict(
            compute_s=compute_s,
            early_release=enabled,
            prefetch=int(enabled),
            tasks=len(starts),
            elapsed_s=elapsed,
            requests_per_s=len(starts) / elapsed,
            mean_execution_gap_s=sum(gaps) / len(gaps),
            max_execution_gap_s=max(gaps),
            execution_gaps_s=gaps,
        )
        print("CONTINUOUS_TIMING " + json.dumps(measured), flush=True)
        record_property("continuous_timing", json.dumps(measured))
        assert len(starts) == 6 and len(finishes) == 6
        assert not await store.list_leases("fl-pool")
        if enabled:
            assert max(gaps) < 0.25, measured
        else:
            assert min(gaps) >= 0.5, measured
    finally:
        await dispatcher.stop()
        await manager.shutdown()
