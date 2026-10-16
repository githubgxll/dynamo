import asyncio
import errno
import hashlib
import json
import os
import threading

import pytest

from dingo.common.video_result_file import (
    BINARY_RESULT_WRITER,
    BinaryResultWriter,
    validate_descriptor,
)
from dingo.video_gateway.artifact_store import FileArtifactStore
from dingo.video_gateway.errors import ResultTooLarge
from dingo.video_gateway.models import TaskStatus
from dingo.vllm.omni.detached_tasks import DetachedOmniTaskManager
from tests.video_gateway.test_dispatcher import (
    _MINIMAL_MP4,
    _DetachedClient,
    _pool,
    _stack,
    _submit,
)

REAL_THREAD = asyncio.to_thread


def test_atomic_status_existing_directory_needs_no_mkdir_or_unlink(
    tmp_path, monkeypatch
):
    from pathlib import Path

    path = tmp_path / "worker-status.json"

    def unexpected(*args, **kwargs):
        raise AssertionError("redundant metadata operation")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "mkdir", unexpected)
        patch.setattr(Path, "unlink", unexpected)
        DetachedOmniTaskManager._atomic_json(path, {"state": "accepted"})
        DetachedOmniTaskManager._atomic_json(path, {"state": "completed"})
    assert json.loads(path.read_text()) == {"state": "completed"}


def test_atomic_status_still_creates_missing_parent(tmp_path):
    path = tmp_path / "new" / "attempt" / "worker-status.json"
    DetachedOmniTaskManager._atomic_json(path, {"state": "accepted"})
    assert json.loads(path.read_text()) == {"state": "accepted"}
    assert not list(tmp_path.rglob("*.part-*"))


@pytest.mark.parametrize("operation", ["fsync", "replace"])
def test_atomic_status_failure_keeps_old_status_and_cleans_temporary(
    tmp_path, monkeypatch, operation
):
    path = tmp_path / "worker-status.json"
    DetachedOmniTaskManager._atomic_json(path, {"state": "running"})

    def fail(*args, **kwargs):
        raise OSError("injected " + operation)

    monkeypatch.setattr(os, operation, fail)
    with pytest.raises(OSError, match="injected"):
        DetachedOmniTaskManager._atomic_json(path, {"state": "completed"})
    assert json.loads(path.read_text()) == {"state": "running"}
    assert not list(tmp_path.glob("*.part-*"))


async def test_binary_success_keeps_both_fsyncs_without_missing_temp_unlink(
    tmp_path, monkeypatch
):
    from pathlib import Path

    original = os.fsync
    calls = []

    def fsync(fd):
        calls.append(fd)
        return original(fd)

    def unexpected(*args, **kwargs):
        raise AssertionError("unlink after successful rename")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", unexpected)
        patch.setattr(os, "fsync", fsync)
        desc = await BinaryResultWriter(tmp_path).write(_MINIMAL_MP4)
    assert len(calls) == 2
    assert (tmp_path / desc["filename"]).read_bytes() == _MINIMAL_MP4


async def test_binary_output_timings_are_bounded_and_descriptor_unchanged(tmp_path):
    import math

    from dingo.common.video_result_file import normalize_inline_result

    writer = BinaryResultWriter(tmp_path)
    desc = await writer.write(_MINIMAL_MP4)
    assert set(desc) == {"schema_version", "filename", "bytes", "sha256"}
    stages = writer.stage_durations
    assert set(stages) == {
        "artifact_queue_s",
        "artifact_hash_s",
        "artifact_open_s",
        "artifact_write_s",
        "artifact_file_fsync_s",
        "artifact_rename_s",
        "artifact_dir_fsync_s",
        "artifact_close_s",
        "artifact_work_s",
        "artifact_resume_s",
    }
    assert all(math.isfinite(v) and v >= 0 for v in stages.values())
    assert (
        sum(
            stages[k]
            for k in [
                "artifact_hash_s",
                "artifact_open_s",
                "artifact_write_s",
                "artifact_file_fsync_s",
                "artifact_rename_s",
                "artifact_dir_fsync_s",
                "artifact_close_s",
            ]
        )
        <= stages["artifact_work_s"]
    )
    result = normalize_inline_result(
        {
            "status": "completed",
            "stage_durations": stages,
            "data": [{"output_format": "mp4", "artifact": desc}],
        }
    )
    assert result["stage_durations"] == stages


@pytest.mark.parametrize("operation", ["fsync", "replace"])
async def test_binary_failure_cleans_unpublished_temp(tmp_path, monkeypatch, operation):
    def fail(*args, **kwargs):
        raise OSError("injected " + operation)

    monkeypatch.setattr(os, operation, fail)
    with pytest.raises(OSError, match="injected"):
        await BinaryResultWriter(tmp_path).write(_MINIMAL_MP4)
    assert not list(tmp_path.iterdir())


async def test_detached_path_checks_run_off_loop_for_all_operations(
    tmp_path, monkeypatch
):
    from dingo.common.video_task_protocol import DetachedTaskIdentity

    monkeypatch.setattr(asyncio, "to_thread", REAL_THREAD)
    started = asyncio.Event()
    release = asyncio.Event()

    class Handler:
        async def generate(self, request, context):
            started.set()
            await release.wait()
            yield {
                "status": "completed",
                "data": [
                    {
                        "output_format": "mp4",
                        "artifact": await BINARY_RESULT_WRITER.get().write(
                            _MINIMAL_MP4
                        ),
                    }
                ],
            }

    manager = DetachedOmniTaskManager(
        Handler(), tmp_path, binary_results=True, inline_results=True
    )
    identity = DetachedTaskIdentity("deployment", "pool", "task", 1, "a" * 32)
    task_root = manager._attempt_root(identity).parent.parent
    task_root.mkdir(parents=True)
    (task_root / "_artifact.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                deployment_id="deployment",
                pool_id="pool",
                task_id="task",
            )
        )
    )
    loop_thread = threading.get_ident()
    original = manager._attempt_root
    checks = []

    def checked(value):
        assert threading.get_ident() != loop_thread
        checks.append(value.key)
        return original(value)

    monkeypatch.setattr(manager, "_attempt_root", checked)
    try:
        assert (await manager._submit(identity, {}))["accepted"]
        await asyncio.wait_for(started.wait(), 2)
        assert (await manager._status(identity))["state"] == "running"
        waiter = manager._wait_terminal(identity)
        assert (await anext(waiter))["state"] == "watching"
        release.set()
        assert (await anext(waiter))["state"] == "completed"
        await waiter.aclose()
        assert (await manager._cancel(identity))["state"] == "cancel_requested"
        assert checks
    finally:
        release.set()
        await manager.shutdown()


async def test_detached_revalidates_symlinks_on_later_status_operation(tmp_path):
    from dingo.common.video_task_protocol import DetachedTaskIdentity

    manager = DetachedOmniTaskManager(object(), tmp_path / "artifacts")
    identity = DetachedTaskIdentity("deployment", "pool", "task", 1, "a" * 32)
    assert (await manager._status(identity))["state"] == "not_found"
    outside = tmp_path / "outside"
    outside.mkdir()
    (manager.root / "deployment").symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink"):
        await manager._status(identity)
    assert not list(outside.iterdir())


async def setup_result(tmp_path):
    store = FileArtifactStore(tmp_path / "artifacts")
    root = store.detached_attempt_root("deployment", "pool", "task", 1, "a" * 32)
    root.mkdir(parents=True)
    desc = await BinaryResultWriter(root).write(_MINIMAL_MP4)
    return store, root, desc


async def publish(store, desc, **kwargs):
    return await store.finalize_worker_mp4(
        "deployment",
        "pool",
        "task",
        1,
        "a" * 32,
        desc,
        {},
        kwargs.pop("validator", lambda p, n: {"container": "mp4"}),
        kwargs.pop("processor", lambda p, n: None),
        kwargs.pop("requires_processing", lambda p, n: False),
        **kwargs,
    )


async def test_binary_candidates_are_independent_and_share_no_duplicate_data(tmp_path):
    store, root, desc = await setup_result(tmp_path)
    left, size, sha, _ = await publish(store, desc)
    right, *_ = await publish(store, desc)
    assert left != right
    assert size == len(_MINIMAL_MP4) and sha == hashlib.sha256(_MINIMAL_MP4).hexdigest()
    assert left.stat().st_ino == (root / desc["filename"]).stat().st_ino
    left.unlink()
    assert right.read_bytes() == _MINIMAL_MP4
    assert (root / desc["filename"]).read_bytes() == _MINIMAL_MP4


@pytest.mark.parametrize("fault", ["size", "missing", "symlink", "other_attempt"])
async def test_binary_rejects_bad_or_cross_attempt_results(tmp_path, fault):
    store, root, desc = await setup_result(tmp_path)
    source = root / desc["filename"]
    if fault == "size":
        desc["bytes"] += 1
    if fault == "missing":
        source.unlink()
    if fault == "symlink":
        source.unlink()
        source.symlink_to(tmp_path / "outside")
    if fault == "other_attempt":
        other = root.parent / "2-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        other.mkdir()
        source.rename(other / source.name)
    with pytest.raises(RuntimeError):
        await publish(store, desc)


async def test_binary_trusts_worker_digest_without_full_read(tmp_path, monkeypatch):
    from pathlib import Path

    store, root, desc = await setup_result(tmp_path)
    desc["sha256"] = (
        "0" * 64
    )  # intentional policy: no independent content digest comparison

    def unexpected_read(*args, **kwargs):
        raise AssertionError("unexpected whole-file read")

    def unexpected_hash(*args, **kwargs):
        raise AssertionError("unexpected Gateway hash")

    with monkeypatch.context() as check:
        check.setattr(Path, "open", unexpected_read)
        check.setattr(hashlib, "sha256", unexpected_hash)
        final, size, sha, _ = await publish(store, desc)
    assert size == len(_MINIMAL_MP4) and sha == desc["sha256"]
    assert final.read_bytes() == _MINIMAL_MP4


async def test_binary_rechecks_candidate_size_after_media_validation(tmp_path):
    store, root, desc = await setup_result(tmp_path)

    def mutate(path, normalized):
        path.write_bytes(b"short")
        return {}

    with pytest.raises(RuntimeError, match="size/type changed"):
        await publish(store, desc, validator=mutate)
    assert not list(store.root.rglob("video-a*.mp4"))


@pytest.mark.parametrize("merged", [False, True])
async def test_binary_allows_dingofs_hardlink_mtime_update(
    tmp_path, monkeypatch, merged
):
    store, root, desc = await setup_result(tmp_path)
    original = os.link
    before = (root / desc["filename"]).stat().st_mtime_ns

    def dingofs_link(src, dst, **kwargs):
        original(src, dst, **kwargs)
        value = os.stat(dst)
        os.utime(dst, ns=(value.st_atime_ns, value.st_mtime_ns + 1_000_000_000))

    monkeypatch.setattr(os, "link", dingofs_link)
    final, size, sha, _ = await publish(
        store,
        desc,
        inspector=(lambda p, n: (False, {"container": "mp4"})) if merged else None,
    )
    assert final.stat().st_mtime_ns != before
    assert final.read_bytes() == _MINIMAL_MP4 and size == len(_MINIMAL_MP4)
    assert sha == desc["sha256"]


@pytest.mark.parametrize(
    "name", ["../outside.mp4", "/tmp/out.mp4", "worker-video-x.mp4"]
)
def test_descriptor_rejects_paths(name):
    with pytest.raises(ValueError):
        validate_descriptor(
            dict(schema_version=1, filename=name, bytes=1, sha256="a" * 64)
        )


async def test_binary_processing_never_changes_worker_source(tmp_path):
    store, root, desc = await setup_result(tmp_path)
    processed = b"changed-video"
    path, size, sha, _ = await publish(
        store,
        desc,
        requires_processing=lambda p, n: True,
        processor=lambda p, n: p.write_bytes(processed),
    )
    assert path.read_bytes() == processed and size == len(processed)
    assert sha == hashlib.sha256(processed).hexdigest()
    assert (root / desc["filename"]).read_bytes() == _MINIMAL_MP4


async def test_binary_validation_failure_cleans_only_candidate(tmp_path):
    store, root, desc = await setup_result(tmp_path)
    good, *_ = await publish(store, desc)

    def invalid(p, n):
        raise RuntimeError("invalid media")

    with pytest.raises(RuntimeError, match="invalid media"):
        await publish(store, desc, validator=invalid)
    assert list(good.parent.iterdir()) == [good]
    assert (root / desc["filename"]).exists()


async def test_binary_limit_and_duplicate_output(tmp_path):
    store, root, desc = await setup_result(tmp_path)
    with pytest.raises(ResultTooLarge):
        await publish(store, desc, max_result_bytes=1)
    writer = BinaryResultWriter(root, max_bytes=4)
    with pytest.raises(ValueError):
        await writer.write(b"12345")
    await writer.write(b"1234")
    with pytest.raises(RuntimeError):
        await writer.write(b"1")


async def test_binary_hardlink_unsupported_falls_back_to_copy(tmp_path, monkeypatch):
    store, root, desc = await setup_result(tmp_path)

    def unsupported(*args, **kwargs):
        raise OSError(errno.EOPNOTSUPP, "unsupported")

    monkeypatch.setattr(os, "link", unsupported)
    final, *_ = await publish(store, desc)
    assert final.read_bytes() == _MINIMAL_MP4
    assert final.stat().st_ino != (root / desc["filename"]).stat().st_ino


async def test_binary_cancel_drains_writer_before_return(tmp_path, monkeypatch):
    monkeypatch.setattr(asyncio, "to_thread", REAL_THREAD)
    writer = BinaryResultWriter(tmp_path)
    started = threading.Event()
    release = threading.Event()
    original = writer._write

    def slow(data):
        started.set()
        release.wait(3)
        return original(data)

    monkeypatch.setattr(writer, "_write", slow)
    task = asyncio.create_task(writer.write(_MINIMAL_MP4))
    try:
        assert await REAL_THREAD(started.wait, 1)
        task.cancel()
        await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("wait_supported", [False, True])
async def test_dispatcher_finishes_binary_worker_and_keeps_small_manifest(
    make_gateway_config, inline, wait_supported
):
    class Handler:
        async def generate(self, request, context):
            desc = await BINARY_RESULT_WRITER.get().write(_MINIMAL_MP4)
            yield {
                "status": "completed",
                "data": [{"output_format": "mp4", "artifact": desc}],
            }

    pool = _pool("fl-pool", "public-fl", "dyn://scope.backend.generate")
    pool["execution_mode"] = "detached"
    config = make_gateway_config(pools=[pool])

    class Manager(DetachedOmniTaskManager):
        def _base_status(self, identity, state):
            status = super()._base_status(identity, state)
            if not wait_supported:
                status.pop("capabilities", None)
            return status

    manager = Manager(
        Handler(),
        config.artifact_store.root,
        binary_results=True,
        inline_results=inline,
    )
    store, artifacts, dispatcher, service = _stack(
        config, {"fl-pool": _DetachedClient(manager)}
    )
    await dispatcher.start()
    try:
        submitted = await _submit(service, "public-fl")
        terminal = await dispatcher.wait_terminal(submitted.stored.task.id, 3)
        assert terminal.task.status == TaskStatus.COMPLETED, terminal.task.error
        assert terminal.task.result_bytes == len(_MINIMAL_MP4)
        responses = list(artifacts.root.rglob("worker-response.jsonl"))
        if inline:
            assert not responses
            status = json.loads(
                next(artifacts.root.rglob("worker-status.json")).read_text()
            )
            assert status["result_format"] == "binary_mp4_inline_v1"
            assert "artifact" in status["inline_result"]["data"][0]
            assert (
                not {"response_path", "response_bytes", "response_sha256"}
                & status.keys()
            )
        else:
            assert len(responses) == 1 and responses[0].stat().st_size < 1024
            assert "b64_json" not in responses[0].read_text()
        assert not await store.list_leases("fl-pool")
    finally:
        await dispatcher.stop()
        await manager.shutdown()
    assert BINARY_RESULT_WRITER.get() is None


@pytest.mark.parametrize("sound", [True, False])
@pytest.mark.parametrize("merged", [False, True])
async def test_binary_real_media_audio_policy(
    tmp_path, make_gateway_config, sound, merged
):
    from dingo.video_gateway.adapters import create_adapter
    from tests.video_gateway.test_minimax_h3_adapter import _write_h264_aac_mp4

    sample = tmp_path / "sample.mp4"
    _write_h264_aac_mp4(sample, frames=124, width=256, height=256)
    store = FileArtifactStore(tmp_path / "artifacts")
    root = store.detached_attempt_root("deployment", "pool", "task", 1, "a" * 32)
    root.mkdir(parents=True)
    desc = await BinaryResultWriter(root).write(sample.read_bytes())
    config = make_gateway_config()
    adapter = create_adapter(config.pools[0])
    adapter.options["validate_media"] = True
    normalized = {
        "width": 256,
        "height": 256,
        "fps": 24,
        "num_frames": 124,
        "seconds": 5,
        "generate_sound": sound,
    }
    final, _, _, media = await store.finalize_worker_mp4(
        "deployment",
        "pool",
        "task",
        1,
        "a" * 32,
        desc,
        normalized,
        adapter.validate_artifact,
        adapter.prepare_artifact,
        adapter.artifact_requires_processing,
        inspector=adapter.inspect_artifact_for_publication if merged else None,
    )
    assert media["audio_codec"] == ("aac" if sound else None)
    assert (
        hashlib.sha256((root / desc["filename"]).read_bytes()).hexdigest()
        == desc["sha256"]
    )
    assert (final.stat().st_ino == (root / desc["filename"]).stat().st_ino) == sound


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("sound", [False, True])
async def test_binary_merged_probe_opens_once_unless_copied(
    tmp_path, make_gateway_config, monkeypatch, fallback, sound
):
    import av

    from dingo.video_gateway.adapters import create_adapter
    from tests.video_gateway.test_minimax_h3_adapter import _write_h264_aac_mp4

    adapter = create_adapter(make_gateway_config().pools[0])
    adapter.options["validate_media"] = True
    normalized = {
        "width": 256,
        "height": 256,
        "num_frames": 124,
        "generate_sound": sound,
    }
    sample = tmp_path / "sample.mp4"
    _write_h264_aac_mp4(sample, frames=124, width=256, height=256)
    adapter.prepare_artifact(sample, normalized)
    expected = adapter.validate_artifact(sample, normalized)
    store = FileArtifactStore(tmp_path / "artifacts")
    root = store.detached_attempt_root("deployment", "pool", "task", 1, "a" * 32)
    root.mkdir(parents=True)
    desc = await BinaryResultWriter(root).write(sample.read_bytes())
    opened = []
    original = av.open

    def counted(path, *args, **kwargs):
        opened.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(av, "open", counted)
    if fallback:

        def unsupported(*args, **kwargs):
            raise OSError(errno.EXDEV, "cross-device")

        monkeypatch.setattr(os, "link", unsupported)
    final, _, _, media = await store.finalize_worker_mp4(
        "deployment",
        "pool",
        "task",
        1,
        "a" * 32,
        desc,
        normalized,
        adapter.validate_artifact,
        adapter.prepare_artifact,
        adapter.artifact_requires_processing,
        inspector=adapter.inspect_artifact_for_publication,
    )
    assert media == expected
    assert opened == [str(root / desc["filename"])] + ([str(final)] if fallback else [])


@pytest.mark.parametrize("fault", ["dimensions", "frames", "audio_required"])
async def test_binary_merged_probe_keeps_media_rejections(
    tmp_path, make_gateway_config, fault
):
    from dingo.video_gateway.adapters import create_adapter
    from tests.video_gateway.test_minimax_h3_adapter import _write_h264_aac_mp4

    adapter = create_adapter(make_gateway_config().pools[0])
    adapter.options["validate_media"] = True
    normalized = {
        "width": 256,
        "height": 256,
        "num_frames": 124,
        "generate_sound": False,
    }
    sample = tmp_path / "sample.mp4"
    _write_h264_aac_mp4(sample, frames=124, width=256, height=256)
    adapter.prepare_artifact(sample, normalized)
    store = FileArtifactStore(tmp_path / "artifacts")
    root = store.detached_attempt_root("deployment", "pool", "task", 1, "a" * 32)
    root.mkdir(parents=True)
    desc = await BinaryResultWriter(root).write(sample.read_bytes())
    if fault == "dimensions":
        normalized["width"] = 512
    if fault == "frames":
        normalized["num_frames"] = 200
    if fault == "audio_required":
        normalized["generate_sound"] = True
    with pytest.raises(
        RuntimeError,
        match={
            "dimensions": "dimensions",
            "frames": "frame count",
            "audio_required": "AAC",
        }[fault],
    ):
        await store.finalize_worker_mp4(
            "deployment",
            "pool",
            "task",
            1,
            "a" * 32,
            desc,
            normalized,
            adapter.validate_artifact,
            adapter.prepare_artifact,
            adapter.artifact_requires_processing,
            inspector=adapter.inspect_artifact_for_publication,
        )
    assert not list(store.root.rglob("video-a*.mp4"))
    assert (root / desc["filename"]).exists()


async def test_binary_inspected_source_change_is_rejected(tmp_path):
    store, root, desc = await setup_result(tmp_path)

    def inspect(path, normalized):
        path.write_bytes(b"changed")
        return False, {"container": "mp4"}

    with pytest.raises(RuntimeError, match="changed during validation"):
        await publish(store, desc, inspector=inspect)
    assert not list(store.root.rglob("video-a*.mp4"))


@pytest.mark.parametrize("disk_failure", [False, True])
@pytest.mark.parametrize("inline", [False, True])
async def test_binary_worker_restart_is_idempotent_and_write_failure_is_not_completed(
    tmp_path, monkeypatch, disk_failure, inline
):
    from dingo.common.video_task_protocol import detached_envelope

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

    if disk_failure:

        def fail(*args):
            raise OSError("controlled disk failure")

        monkeypatch.setattr(BinaryResultWriter, "_write", fail)
    store = FileArtifactStore(tmp_path / "artifacts")
    task_root = store.task_root("deployment", "pool", "task")
    task_root.mkdir(parents=True)
    (task_root / "_artifact.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                deployment_id="deployment",
                pool_id="pool",
                task_id="task",
            )
        )
    )
    identity = dict(
        deployment_id="deployment",
        pool_id="pool",
        task_id="task",
        attempt=1,
        execution_token="a" * 32,
    )
    manager = DetachedOmniTaskManager(
        Handler(), store.root, binary_results=True, inline_results=inline
    )

    async def operation(m, op):
        return [
            x
            async for x in m.generate(
                detached_envelope(
                    op=op, **identity, payload={} if op == "submit" else None
                ),
                None,
            )
        ]

    await operation(manager, "submit")
    terminal = (await operation(manager, "wait"))[-1]
    assert terminal["state"] == ("failed" if disk_failure else "completed")
    await manager.shutdown()
    replacement = DetachedOmniTaskManager(
        Handler(), store.root, binary_results=True, inline_results=inline
    )
    try:
        reply = (await operation(replacement, "submit"))[0]
        assert reply["accepted"] is False and reply["state"] == terminal["state"]
        assert calls == 1
        if not disk_failure:
            attempt_root = store.detached_attempt_root(**identity)
            response = (
                reply["inline_result"]
                if inline
                else json.loads((attempt_root / "worker-response.jsonl").read_text())
            )
            desc = response["data"][0]["artifact"]
            final, *_ = await publish(store, desc)
            assert final.read_bytes() == _MINIMAL_MP4
    finally:
        await replacement.shutdown()


@pytest.mark.parametrize(
    "error",
    [
        "Executor shut down",
        {"code": "invalid_request", "message": "bad parameter"},
        {"code": "worker_failed", "message": "Executor shut down"},
    ],
)
def test_inline_errors_preserve_retry_classification(error):
    from dingo.common.video_result_file import normalize_inline_result
    from dingo.video_gateway.errors import worker_execution_error

    new = normalize_inline_result({"status": "failed", "error": error})["error"]
    assert type(worker_execution_error(new)) is type(worker_execution_error(error))
    assert str(worker_execution_error(new)) == str(worker_execution_error(error))


@pytest.mark.parametrize(
    "data",
    [
        None,
        [],
        [{"output_format": "mp4", "b64_json": "AAAA"}],
        [{"output_format": "mp4", "artifact": {"filename": "../other"}}],
    ],
)
def test_inline_descriptor_rejects_missing_binary_or_legacy_payload(data):
    from dingo.common.video_result_file import normalize_inline_result

    with pytest.raises(ValueError):
        normalize_inline_result({"status": "completed", "data": data})


async def test_inline_cancel_after_binary_write_never_publishes_completed(tmp_path):
    from dingo.common.video_task_protocol import detached_envelope

    store = FileArtifactStore(tmp_path / "artifacts")
    root = store.task_root("deployment", "pool", "task")
    root.mkdir(parents=True)
    (root / "_artifact.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                deployment_id="deployment",
                pool_id="pool",
                task_id="task",
            )
        )
    )
    ready = asyncio.Event()

    class Handler:
        async def generate(self, request, context):
            descriptor = await BINARY_RESULT_WRITER.get().write(_MINIMAL_MP4)
            ready.set()
            yield {
                "status": "completed",
                "data": [{"output_format": "mp4", "artifact": descriptor}],
            }
            await context.wait_stopped()

    manager = DetachedOmniTaskManager(
        Handler(),
        store.root,
        binary_results=True,
        inline_results=True,
        cancel_poll_interval_s=0.01,
    )

    def request(op):
        return detached_envelope(
            op=op,
            deployment_id="deployment",
            pool_id="pool",
            task_id="task",
            attempt=1,
            execution_token="a" * 32,
            payload={} if op == "submit" else None,
        )

    async def call(op):
        return [x async for x in manager.generate(request(op), None)]

    await call("submit")
    try:
        await asyncio.wait_for(ready.wait(), 1)
        assert (await call("status"))[0]["state"] == "running"
        await call("cancel")
        terminal = (await asyncio.wait_for(call("wait"), 1))[-1]
        assert terminal["state"] == "cancelled" and "inline_result" not in terminal
        assert not list(store.root.rglob("worker-response.jsonl"))
    finally:
        await manager.shutdown()
