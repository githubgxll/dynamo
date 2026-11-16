# SPDX-License-Identifier: Apache-2.0
"""Shared-file admission and recovery must not reserve original media copies."""
from types import SimpleNamespace

import pytest

from dingo.video_gateway.config import MediaConfig
from dingo.video_gateway.memory_budget import WeightedMemoryBudget, task_memory_bytes
from dingo.video_gateway.task_runtime.video_dispatch import (
    VideoSchedulingAdapter,
    VideoSelection,
)
from tests.video_gateway.test_dispatcher import _pool, _stack, _submit


@pytest.mark.parametrize("encoded", [0, 1024, 384 * 1024**2])
def test_shared_input_weight_is_independent_of_original_media(encoded):
    m = MediaConfig()
    n = {"shared_inputs": {"storage_id": "test"}}
    assert (
        task_memory_bytes(m, n, encoded_reference_bytes=encoded)
        == m.result_task_memory_bytes
    )
    assert (
        task_memory_bytes(
            m, n, persisted_estimate=2 * encoded + m.result_task_memory_bytes
        )
        == m.result_task_memory_bytes
    )


def test_inline_and_sr_only_keep_their_existing_reserves():
    m = MediaConfig()
    assert (
        task_memory_bytes(m, {}, encoded_reference_bytes=80 * 1024**2)
        == 160 * 1024**2 + m.result_task_memory_bytes
    )
    assert task_memory_bytes(m, {}, persisted_estimate=123) == 123
    assert task_memory_bytes(m, {}, persisted_estimate=0) == m.max_task_memory_bytes
    assert (
        task_memory_bytes(m, {"operation": "sr_only"}, encoded_reference_bytes=0)
        == m.result_task_memory_bytes
    )
    # A reverse-order FL2VA input remains inline: no persisted shared marker.
    assert (
        task_memory_bytes(m, {"frame_indices": [-1, 0]}, encoded_reference_bytes=8)
        == 16 + m.result_task_memory_bytes
    )


@pytest.mark.parametrize("shared", [False, True])
async def test_submission_persists_transport_specific_budget(
    make_gateway_config, monkeypatch, shared
):
    pool = _pool("fl-pool", "public-fl", "dyn://scope.backend.generate")
    pool["scheduling"]["accept_without_workers"] = True
    config = make_gateway_config(pools=[pool])
    _store, _artifacts, _dispatcher, service = _stack(
        config, {"fl-pool": SimpleNamespace()}
    )
    adapter = service.adapters["fl-pool"]
    normalize = adapter.normalize_request

    def normalized(*args, **kwargs):
        n = normalize(*args, **kwargs)
        if shared:
            n["shared_inputs"] = {"storage_id": "test"}
        return n

    monkeypatch.setattr(adapter, "normalize_request", normalized)
    monkeypatch.setattr(
        adapter, "estimate_encoded_reference_bytes", lambda manifest: 80 * 1024**2
    )
    submitted = await _submit(service, "public-fl")
    assert (
        submitted.stored.task.estimated_payload_bytes
        == config.media.result_task_memory_bytes + (0 if shared else 160 * 1024**2)
    )


@pytest.mark.parametrize("adopting", [False, True])
async def test_old_shared_task_uses_correct_weight_at_dispatch_and_takeover(adopting):
    m = MediaConfig()
    seen = []

    async def acquire(tid, weight):
        seen.append((tid, weight))
        return False

    pool = SimpleNamespace(budget_waiter_id=None)
    dispatcher = SimpleNamespace(
        config=SimpleNamespace(media=m),
        memory_budget=SimpleNamespace(try_acquire=acquire),
    )
    task = SimpleNamespace(
        id="old-shared",
        normalized_request={"shared_inputs": {"storage_id": "test"}},
        estimated_payload_bytes=m.max_task_memory_bytes,
    )
    candidate = VideoSelection(
        SimpleNamespace(task=task), [], False, 32, adopting=adopting
    )
    await VideoSchedulingAdapter(dispatcher, pool).claim(candidate)
    assert seen == [("old-shared", m.result_task_memory_bytes)]


async def test_two_large_shared_requests_fit_budget_and_release_without_growth():
    m = MediaConfig()
    budget = WeightedMemoryBudget(2 * m.result_task_memory_bytes)
    for tid in ["a", "b"]:
        weight = task_memory_bytes(
            m,
            {"shared_inputs": {"storage_id": "test"}},
            encoded_reference_bytes=384 * 1024**2,
        )
        assert await budget.try_acquire(tid, weight)
        assert await budget.shrink(tid, m.result_task_memory_bytes) == 0
    assert (await budget.snapshot()).used_bytes == 2 * m.result_task_memory_bytes
    for tid in ["a", "b"]:
        await budget.release(tid)
    assert (await budget.snapshot()).used_bytes == 0
