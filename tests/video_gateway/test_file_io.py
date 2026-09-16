"""Real-thread regressions: slow metadata must not block the event loop."""

import asyncio
import base64
import hashlib
import threading
import time

import pytest

from dingo.video_gateway.artifact_store import FileArtifactStore, TaskArtifactCandidate
from dingo.video_gateway.file_io import (
    opened_file,
    run_cancellable_file_io,
    run_file_io,
)

REAL_TO_THREAD = asyncio.to_thread


async def test_finalize_uses_one_thread_dispatch(tmp_path, monkeypatch):
    calls = []

    async def counted(function, /, *args, **kwargs):
        calls.append(function.__name__)
        return await REAL_TO_THREAD(function, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", counted)
    store = FileArtifactStore(tmp_path / "artifacts")
    main_thread = threading.get_ident()
    stages = []

    def processor(path, _normalized):
        stages.append(threading.get_ident())
        assert threading.get_ident() != main_thread
        path.write_bytes(b"processed-video")

    def validator(path, _normalized):
        stages.append(threading.get_ident())
        assert path.read_bytes() == b"processed-video"
        return {"validated": True}

    final, size, digest, media = await store.finalize_b64_mp4(
        store.task_root("deployment", "pool", "task"),
        base64.b64encode(b"source-video").decode(),
        {},
        validator,
        processor,
    )
    assert calls == ["_finalize"]
    assert len(set(stages)) == 1 and stages[0] != main_thread
    assert final.read_bytes() == b"processed-video"
    assert size == len(b"processed-video")
    assert digest == hashlib.sha256(b"processed-video").hexdigest()
    assert media == {"validated": True}


async def test_cancellable_file_io_receives_signal_and_drains(tmp_path, monkeypatch):
    monkeypatch.setattr(asyncio, "to_thread", REAL_TO_THREAD)
    entered, release = threading.Event(), threading.Event()
    events = []

    def work(cancelled):
        entered.set()
        assert release.wait(2)
        assert cancelled.is_set()
        events.append("drained")
        raise asyncio.CancelledError

    task = asyncio.create_task(run_cancellable_file_io(work))
    try:
        while not entered.is_set():
            await asyncio.sleep(0.001)
        task.cancel()
        await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert events == ["drained"]


@pytest.mark.parametrize("boundary", ["processor", "validator", "renamed"])
async def test_cancelled_pipeline_cleans_only_own_candidate(
    tmp_path, monkeypatch, boundary
):
    import dingo.video_gateway.artifact_store as module

    monkeypatch.setattr(asyncio, "to_thread", REAL_TO_THREAD)
    entered, release = threading.Event(), threading.Event()
    store = FileArtifactStore(tmp_path / "artifacts")
    task_root = store.task_root("deployment", "pool", "task")
    result_dir = task_root / "result"
    result_dir.mkdir(parents=True)
    winner = result_dir / "video-other-owner.mp4"
    winner.write_bytes(b"winner")
    stages = []
    original_replace = module.os.replace

    def pause():
        entered.set()
        assert release.wait(2)

    def processor(_path, _normalized):
        stages.append("processor")
        if boundary == "processor":
            pause()

    def validator(_path, _normalized):
        stages.append("validator")
        if boundary == "validator":
            pause()
        return {}

    def replace(source, target):
        original_replace(source, target)
        if boundary == "renamed":
            pause()

    monkeypatch.setattr(module.os, "replace", replace)
    task = asyncio.create_task(
        store.finalize_b64_mp4(
            task_root,
            base64.b64encode(b"video").decode(),
            {},
            validator,
            processor,
            publication_scope="cancelled-owner",
        )
    )
    try:
        while not entered.is_set():
            await asyncio.sleep(0.001)
        task.cancel()
        await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done(), "released file ownership before the thread drained"
        assert winner.read_bytes() == b"winner"
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    if boundary == "processor":
        assert stages == ["processor"], "continued expensive work after cancellation"
    assert list(result_dir.iterdir()) == [winner]
    assert not list((task_root / "tmp").iterdir())
    assert winner.read_bytes() == b"winner"


@pytest.mark.parametrize("boundary", ["processor", "validator"])
async def test_failed_pipeline_cleans_partial_result(tmp_path, monkeypatch, boundary):
    monkeypatch.setattr(asyncio, "to_thread", REAL_TO_THREAD)
    store = FileArtifactStore(tmp_path / "artifacts")
    task_root = store.task_root("deployment", "pool", "task")

    def processor(_path, _normalized):
        if boundary == "processor":
            raise RuntimeError("processor failed")

    def validator(_path, _normalized):
        raise RuntimeError("validator failed")

    with pytest.raises(RuntimeError, match=boundary + " failed"):
        await store.finalize_b64_mp4(
            task_root, base64.b64encode(b"video").decode(), {}, validator, processor
        )
    assert not list((task_root / "result").iterdir())
    assert not list((task_root / "tmp").iterdir())


@pytest.mark.parametrize(
    "operation",
    [
        "create",
        "task_root",
        "commit",
        "write",
        "read",
        "status",
        "cancel",
        "consume",
        "finalize",
        "discard",
        "trash",
        "open_result",
    ],
)
async def test_slow_containment_runs_off_loop(tmp_path, monkeypatch, operation):
    monkeypatch.setattr(asyncio, "to_thread", REAL_TO_THREAD)
    store = FileArtifactStore(tmp_path / "artifacts")
    upload = await store.create_upload()
    task = store.task_root("deployment", "pool", "task")
    task.mkdir(parents=True)
    (task / "x.json").write_text("{}")
    result = task / "result.mp4"
    result.write_bytes(b"video")
    token = "a" * 32
    attempt = store.detached_attempt_root("deployment", "pool", "task", 1, token)
    attempt.mkdir(parents=True)
    payload = b'{"status":"completed"}\n'
    (attempt / "worker-response.jsonl").write_bytes(payload)
    main_thread = threading.get_ident()
    original = store._contained
    calls = []

    def slow(path):
        assert threading.get_ident() != main_thread, (
            "filesystem metadata ran on event loop"
        )
        calls.append(path)
        time.sleep(0.03)
        return original(path)

    monkeypatch.setattr(store, "_contained", slow)
    ticks = []

    async def heartbeat():
        while True:
            await asyncio.sleep(0.002)
            ticks.append(time.monotonic())

    beat = asyncio.create_task(heartbeat())
    try:
        if operation == "create":
            await store.create_upload()
        elif operation == "task_root":
            await store.resolve_task_root("deployment", "pool", "task")
        elif operation == "commit":
            await store.commit_upload(
                upload, "deployment", "pool", "new-task", artifact_manifest={}
            )
        elif operation == "write":
            await store.write_json(task / "new.json", {})
        elif operation == "read":
            await store.read_json(task / "x.json")
        elif operation == "status":
            assert (
                await store.read_detached_status("deployment", "pool", "task", 1, token)
                is None
            )
        elif operation == "cancel":
            await store.request_detached_cancel("deployment", "pool", "task", 1, token)
        elif operation == "consume":

            class Consumer:
                def consume(self, value):
                    assert value["status"] == "completed"

            await store.consume_detached_response(
                "deployment",
                "pool",
                "task",
                1,
                token,
                Consumer(),
                expected_sha256=hashlib.sha256(payload).hexdigest(),
                max_response_bytes=1024,
            )
        elif operation == "finalize":
            await store.finalize_b64_mp4(
                task, base64.b64encode(b"video").decode(), {}, lambda *_: {}
            )
        elif operation == "discard":
            await store.discard(task)
        elif operation == "trash":
            await store.trash_orphan(TaskArtifactCandidate("task", task, 10, True))
        elif operation == "open_result":
            async with opened_file(
                lambda: store.result_path(result).open("rb")
            ) as stream:
                assert await run_file_io(stream.read) == b"video"
        assert calls
        assert len(ticks) >= 2, "event loop heartbeat stopped during metadata access"
    finally:
        beat.cancel()
        await asyncio.gather(beat, return_exceptions=True)


async def test_cancel_waits_for_actual_io_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(asyncio, "to_thread", REAL_TO_THREAD)
    entered, release = threading.Event(), threading.Event()
    completed = []

    def write():
        entered.set()
        assert release.wait(2)
        (tmp_path / "data").write_bytes(b"ok")
        completed.append(True)

    task = asyncio.create_task(run_file_io(write))
    try:
        while not entered.is_set():
            await asyncio.sleep(0.001)
        task.cancel()
        await asyncio.sleep(0.02)
        task.cancel()  # repeated cancellation must not abandon the worker
        await asyncio.sleep(0.02)
        assert not task.done()
        assert not completed
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert completed


async def test_cancel_during_open_closes_returned_handle(tmp_path, monkeypatch):
    monkeypatch.setattr(asyncio, "to_thread", REAL_TO_THREAD)
    entered, release = threading.Event(), threading.Event()
    streams = []

    def open_slow():
        stream = (tmp_path / "data").open("wb")
        streams.append(stream)
        entered.set()
        assert release.wait(2)
        return stream

    async def owner():
        async with opened_file(open_slow):
            pytest.fail("cancelled open must not enter the body")

    task = asyncio.create_task(owner())
    try:
        while not entered.is_set():
            await asyncio.sleep(0.001)
        task.cancel()
        await asyncio.sleep(0.02)
        assert not streams[0].closed
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert streams[0].closed


async def test_discard_rejects_traversal_and_escaped_parent(tmp_path):
    store = FileArtifactStore(tmp_path / "artifacts")
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "victim"
    victim.write_text("keep")
    link = outside / "link"
    link.symlink_to(victim)
    (store.root / "escape").symlink_to(outside, target_is_directory=True)
    for path in [
        store.root / ".." / "outside" / "link",
        store.root / "escape" / "link",
        store.root,
    ]:
        with pytest.raises(RuntimeError):
            await store.discard(path)
    assert victim.read_text() == "keep"
    assert link.is_symlink()


async def test_live_responds_while_submission_path_lookup_is_blocked(
    make_gateway_config, monkeypatch
):
    from dingo.video_gateway.api import _SERVICE_KEY
    from tests.video_gateway.test_api import _client, _form

    monkeypatch.setattr(asyncio, "to_thread", REAL_TO_THREAD)
    client = await _client(make_gateway_config)
    artifacts = client.server.app[_SERVICE_KEY].artifacts
    original = artifacts._contained
    entered, release = threading.Event(), threading.Event()
    main_thread = threading.get_ident()

    def blocked(path):
        assert threading.get_ident() != main_thread
        entered.set()
        assert release.wait(2)
        return original(path)

    monkeypatch.setattr(artifacts, "_contained", blocked)
    submission = asyncio.create_task(client.post("/v1/videos", data=_form()))
    try:

        async def started():
            while not entered.is_set():
                await asyncio.sleep(0.001)

        await asyncio.wait_for(started(), 0.5)
        response = await asyncio.wait_for(client.get("/live"), 0.5)
        assert response.status == 200
        assert not submission.done()
    finally:
        release.set()
        await asyncio.gather(submission, return_exceptions=True)
        await client.close()
