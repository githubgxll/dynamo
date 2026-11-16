# SPDX-License-Identifier: Apache-2.0
"""Three-image CPU concurrency and request-global admission invariants."""
import asyncio
import threading
import time

import pytest

from dingo.vllm.omni import h3_image_preprocess as sizing
from tests.video_gateway.test_h3_image_preprocess import adapter
from tests.video_gateway.test_h3_shared_inputs import wait_until


def test_three_threads_preserve_order_and_bound_parallelism(tmp_path, monkeypatch):
    active = maximum = 0
    lock = threading.Lock()
    barrier = threading.Barrier(3)

    def prepare(root, entry):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(active, maximum)
        try:
            barrier.wait(timeout=5)
            time.sleep((2 - entry["ordinal"] % 3) * 0.01)
            return {"ordinal": entry["ordinal"]}
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(sizing, "_prepare_one", prepare)
    receipt = sizing.prepare_images(
        tmp_path,
        [{"ordinal": i, "content_type": "image/jpeg"} for i in range(9)],
        parallelism=3,
    )
    assert maximum == 3 and receipt["parallelism"] == 3
    assert [r["ordinal"] for r in receipt["images"]] == list(range(9))


async def test_bulk_cancel_releases_partial_permits_for_next_request():
    slots = asyncio.Semaphore(3)
    lock = asyncio.Lock()
    await slots.acquire()
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def bulk():
        async with sizing.group_permits(slots, 3, lock):
            entered.set()
            await finish.wait()

    a = asyncio.create_task(bulk())
    await wait_until(lambda: slots._value == 0)
    b = asyncio.create_task(bulk())
    a.cancel()
    with pytest.raises(asyncio.CancelledError):
        await a
    slots.release()
    await asyncio.wait_for(entered.wait(), 2)
    assert slots._value == 0 and not b.done()
    finish.set()
    await b
    assert slots._value == 3 and not lock.locked()


@pytest.mark.parametrize("value", ["0", "4", "invalid"])
def test_invalid_image_concurrency_rejected(tmp_path, monkeypatch, value):
    monkeypatch.setenv("DINGO_H3_IMAGE_PREPROCESS_CONCURRENCY", value)
    with pytest.raises(ValueError):
        adapter(tmp_path, monkeypatch)


async def test_three_lane_config_shares_same_allowance(tmp_path, monkeypatch):
    monkeypatch.setenv("DINGO_H3_IMAGE_PREPROCESS_CONCURRENCY", "3")
    a = adapter(tmp_path, monkeypatch)
    assert a.image_preprocess_parallelism == 3 and a.image_prepare_slots._value == 3
    async with sizing.group_permits(
        a.image_prepare_slots, 3, a.image_prepare_group_lock
    ):
        assert a.image_prepare_slots._value == 0
    assert a.image_prepare_slots._value == 3
