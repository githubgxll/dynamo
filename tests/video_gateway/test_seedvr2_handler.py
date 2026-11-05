# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dingo.common.video_result_file import BINARY_RESULT_WRITER, BinaryResultWriter
from dingo.common.video_super_resolution import H3_SR_PROFILE, SR_PAYLOAD_KEY
from dingo.vllm.omni.seedvr2_execution import (
    SeedVR2Execution,
    UnsafeRecovery,
    write_journal,
)
from dingo.vllm.omni.seedvr2_handler import SeedVR2Handler, media, validate_sr_request
from dingo.common.video_worker_error import VideoWorkerFailure


def payload():
    return {
        "model": "h3",
        "prompt": "test",
        "size": "1344x768",
        "nvext": {"fps": 24},
        SR_PAYLOAD_KEY: {
            "schema_version": 1,
            "profile": H3_SR_PROFILE,
            "base_width": 1344,
            "base_height": 768,
            "output_width": 1890,
            "output_height": 1080,
            "sr_width": 1904,
            "sr_height": 1088,
        },
    }


class H3:
    def __init__(self):
        self.calls = []

    async def generate(self, request, context):
        self.calls.append(request)
        assert SR_PAYLOAD_KEY not in request
        assert "seed" not in request["nvext"]
        artifact = await BINARY_RESULT_WRITER.get().write(b"base")
        yield {
            "status": "completed",
            "data": [{"output_format": "mp4", "artifact": artifact}],
        }


class SR:
    def __init__(self):
        self.calls = []

    async def submit(self, source, **kwargs):
        assert source.read_bytes() == b"base"
        self.calls.append(kwargs)
        return "a" * 32

    async def status(self, job):
        return {"status": "completed", "frames": 122}

    async def download(self, job, destination):
        destination.write_bytes(b"sr")


@pytest.fixture
def pipeline(monkeypatch):
    async def probe(path):
        width, height = (1904, 1088) if path.name == "sr-raw.mp4" else (1344, 768)
        return {
            "width": width,
            "height": height,
            "frames": 122,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    async def finalize(raw, base, output, **kwargs):
        assert kwargs == {"width": 1890, "height": 1080}
        assert raw.read_bytes() == b"sr"
        output.write_bytes(b"final")
        return {
            "postprocess_mode": "encode",
            "postprocess_timings": {
                "mux_or_encode_s": 0.0,
                **{
                    f"{label}_{part}": 0.0
                    for label in ("base", "raw", "final")
                    for part in ("hash_s", "video_scan_s", "audio_scan_s")
                },
            },
        }

    monkeypatch.setattr(media, "probe", probe)
    monkeypatch.setattr(media, "finalize", finalize)
    h3, sr = H3(), SR()
    executor = SeedVR2Execution(sr, owner="owner", quarantine=lambda _: None)
    handler = SeedVR2Handler(h3, executor, engine_identity="pinned-image-and-weights")
    return handler, h3, sr


async def generate(handler, root):
    root.mkdir(parents=True)
    token = BINARY_RESULT_WRITER.set(BinaryResultWriter(root))
    try:
        return [
            chunk
            async for chunk in handler.generate(
                payload(), SimpleNamespace(is_stopped=lambda: False)
            )
        ]
    finally:
        BINARY_RESULT_WRITER.reset(token)


@pytest.mark.asyncio
async def test_only_final_result_is_published(pipeline, tmp_path):
    handler, h3, sr = pipeline
    root = tmp_path / "attempts" / "1-token"
    chunks = await generate(handler, root)
    assert len(chunks) == 1
    result = chunks[0]["data"][0]["artifact"]
    assert (root / result["filename"]).read_bytes() == b"final"
    assert len(h3.calls) == len(sr.calls) == 1
    assert sr.calls[0]["seed"] is None
    assert {
        "h3_base_total_s",
        "sr_service_s",
        "sr_finalize_s",
        "sr_finalize_encode_s",
        "sr_finalize_final_video_scan_s",
        "sr_download_s",
        "sr_result_persist_s",
        "sr_postprocess_total_s",
    } <= chunks[0]["stage_durations"].keys()
    from dingo.common.video_result_file import normalize_inline_result

    normalized = normalize_inline_result(chunks[0])
    assert len(normalized["stage_durations"]) <= 32


@pytest.mark.asyncio
async def test_copy_timing_is_distinguished_from_encoding(
    pipeline, tmp_path, monkeypatch
):
    handler, _, _ = pipeline
    original = media.finalize

    async def copied(*args, **kwargs):
        result = await original(*args, **kwargs)
        result["postprocess_mode"] = "stream_copy"
        return result

    monkeypatch.setattr(media, "finalize", copied)
    chunks = await generate(handler, tmp_path / "attempts" / "1-token")
    assert "sr_finalize_remux_s" in chunks[0]["stage_durations"]
    assert "sr_finalize_encode_s" not in chunks[0]["stage_durations"]


@pytest.mark.asyncio
async def test_new_attempt_reuses_verified_base(pipeline, tmp_path):
    handler, h3, sr = pipeline
    await generate(handler, tmp_path / "attempts" / "1-token")
    await generate(handler, tmp_path / "attempts" / "2-token")
    assert len(h3.calls) == 1
    assert len(sr.calls) == 2


@pytest.mark.asyncio
async def test_ambiguous_old_attempt_does_not_resubmit(pipeline, tmp_path):
    handler, h3, sr = pipeline
    old = tmp_path / "attempts" / "1-token"
    await generate(handler, old)
    write_journal(old / "sr-journal.json", {"phase": "submit_intent"})
    with pytest.raises(UnsafeRecovery):
        await generate(handler, tmp_path / "attempts" / "2-token")
    assert len(h3.calls) == len(sr.calls) == 1


@pytest.mark.asyncio
async def test_corrupted_base_is_never_used(pipeline, tmp_path):
    handler, h3, sr = pipeline
    old = tmp_path / "attempts" / "1-token"
    await generate(handler, old)
    next((old / "base").glob("*.mp4")).write_bytes(b"corrupt")
    with pytest.raises(UnsafeRecovery):
        await generate(handler, tmp_path / "attempts" / "2-token")
    assert len(h3.calls) == len(sr.calls) == 1


@pytest.mark.asyncio
async def test_quarantine_blocks_base_as_well_as_sr(pipeline, tmp_path):
    handler, h3, sr = pipeline
    handler.quarantine("unknown GPU state")
    assert not handler.sr_ready
    with pytest.raises(RuntimeError, match="unavailable"):
        await generate(handler, tmp_path / "attempts" / "1-token")
    ordinary = payload()
    del ordinary[SR_PAYLOAD_KEY]
    with pytest.raises(RuntimeError, match="unavailable"):
        _ = [chunk async for chunk in handler.generate(ordinary, None)]
    assert not h3.calls and not sr.calls


@pytest.mark.parametrize(
    "field,value",
    [
        ("base_width", 1056),
        ("schema_version", True),
        ("profile", "other.v1"),
        ("sr_width", 1920),
    ],
)
def test_worker_revalidates_geometry(field, value):
    request = payload()
    request[SR_PAYLOAD_KEY][field] = value
    with pytest.raises(ValueError):
        validate_sr_request(request)


def attach_engine(h3):
    client = SimpleNamespace(collective_rpc_async=AsyncMock(return_value=[None]))
    h3.engine_client = SimpleNamespace(
        check_health=AsyncMock(),
        engine=SimpleNamespace(stage_pools=[SimpleNamespace(clients=[client])]),
    )
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["resource_exhausted", "unknown"])
@pytest.mark.parametrize("sr", [True, False])
async def test_settled_failures_can_repeat_then_generate_successfully(
    pipeline, tmp_path, monkeypatch, reason, sr
):
    handler, h3, _ = pipeline
    client = attach_engine(h3)
    original = h3.generate
    failures = 0
    async def fail_twice(request, context):
        nonlocal failures
        if failures < 2:
            failures += 1
            yield {"status": "failed", "error": "request failed",
                   "_internal_error": {"schema_version": 1, "reason": reason,
                                       "stage": "base_generation"}}
        else:
            async for chunk in original(request, context):
                yield chunk
    monkeypatch.setattr(h3, "generate", fail_twice)
    for attempt in (1, 2):
        with pytest.raises(VideoWorkerFailure):
            if sr:
                await generate(handler, tmp_path / "attempts" / str(attempt))
            else:
                request = payload(); del request[SR_PAYLOAD_KEY]
                _ = [chunk async for chunk in handler.generate(request, None)]
        assert handler.sr_ready and not handler.quarantined
        assert handler.pause_started is None
    assert client.collective_rpc_async.await_count == 2
    chunks = await generate(handler, tmp_path / "attempts" / "3")
    assert chunks[0]["status"] == "completed"
    assert handler.sr_ready


@pytest.mark.asyncio
async def test_oom_settlement_beyond_old_ten_second_budget_can_continue(
    pipeline, tmp_path, monkeypatch
):
    handler, h3, _ = pipeline
    client = attach_engine(h3)
    entered = asyncio.Event()
    original = h3.generate

    async def slow_settlement(*args, **kwargs):
        entered.set()
        await asyncio.sleep(11)
        return [None]

    async def failed(request, context):
        yield {"status": "failed", "error": "CUDA out of memory"}

    client.collective_rpc_async.side_effect = slow_settlement
    monkeypatch.setattr(h3, "generate", failed)
    request = payload(); del request[SR_PAYLOAD_KEY]

    async def consume():
        return [chunk async for chunk in handler.generate(request, None)]

    task = asyncio.create_task(consume())
    await entered.wait()
    await asyncio.sleep(10.2)
    assert not task.done()
    assert not handler.sr_ready and not handler.quarantined
    with pytest.raises(VideoWorkerFailure):
        await task
    assert handler.sr_ready and not handler.quarantined
    monkeypatch.setattr(h3, "generate", original)
    chunks = await generate(handler, tmp_path / "attempts" / "after-slow-settlement")
    assert chunks[0]["status"] == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["resource_exhausted", "unknown"])
@pytest.mark.parametrize("settlement_error", [TimeoutError("rank did not settle"), RuntimeError("executor unavailable")])
async def test_rank_settlement_failure_quarantines_instead_of_readmitting(
    pipeline, tmp_path, monkeypatch, caplog, reason, settlement_error
):
    handler, h3, _ = pipeline
    client = attach_engine(h3)
    client.collective_rpc_async.side_effect = settlement_error
    async def failed(request, context):
        yield {"status": "failed", "error": "base generation failed",
               "_internal_error": {"schema_version": 1, "reason": reason,
                                   "stage": "base_generation"}}
    monkeypatch.setattr(h3, "generate", failed)
    with pytest.raises(VideoWorkerFailure) as caught:
        await generate(handler, tmp_path / "attempts" / "1")
    assert handler.quarantined and not handler.sr_ready
    assert handler.pause_started is None
    assert caught.value.__cause__ is settlement_error
    assert caught.value.internal_error == {
        "schema_version": 1, "reason": reason, "stage": "base_generation"
    }
    assert handler.last_failure == caught.value.internal_error
    assert "video_base_failure_settlement_failed" in caplog.text
    assert f"reason={reason}" in caplog.text
    assert ("settlement=timeout" if isinstance(settlement_error, TimeoutError)
            else "settlement=error") in caplog.text


@pytest.mark.asyncio
async def test_stream_without_terminal_quarantines(pipeline, tmp_path, monkeypatch):
    handler, h3, _ = pipeline
    async def incomplete(request, context):
        yield {"status": "in_progress"}
    monkeypatch.setattr(h3, "generate", incomplete)
    with pytest.raises(RuntimeError, match="without a terminal"):
        await generate(handler, tmp_path / "attempts" / "1")
    assert handler.quarantined


@pytest.mark.asyncio
async def test_cancellation_before_terminal_quarantines(pipeline, tmp_path, monkeypatch):
    handler, h3, _ = pipeline
    async def interrupted(request, context):
        raise asyncio.CancelledError
        yield
    monkeypatch.setattr(h3, "generate", interrupted)
    with pytest.raises(asyncio.CancelledError):
        await generate(handler, tmp_path / "attempts" / "1")
    assert handler.quarantined


@pytest.mark.asyncio
async def test_confirmed_h3_cancellation_can_continue(pipeline, tmp_path, monkeypatch):
    handler, h3, _ = pipeline
    client = attach_engine(h3)
    original = h3.generate
    async def interrupted(request, context):
        raise asyncio.CancelledError
        yield
    monkeypatch.setattr(h3, "generate", interrupted)
    with pytest.raises(asyncio.CancelledError):
        await generate(handler, tmp_path / "attempts" / "1")
    assert not handler.quarantined and handler.sr_ready
    client.collective_rpc_async.assert_awaited_once()
    monkeypatch.setattr(h3, "generate", original)
    chunks = await generate(handler, tmp_path / "attempts" / "2")
    assert chunks[0]["status"] == "completed"


@pytest.mark.asyncio
async def test_stopped_context_with_empty_abort_stream_can_continue(
    pipeline, monkeypatch
):
    handler, h3, _ = pipeline
    client = attach_engine(h3)
    async def aborted(request, context):
        if False:
            yield
    monkeypatch.setattr(h3, "generate", aborted)
    request = payload(); del request[SR_PAYLOAD_KEY]
    with pytest.raises(asyncio.CancelledError):
        _ = [chunk async for chunk in handler.generate(
            request, SimpleNamespace(is_stopped=lambda: True)
        )]
    assert handler.sr_ready and not handler.quarantined
    client.collective_rpc_async.assert_awaited_once()
