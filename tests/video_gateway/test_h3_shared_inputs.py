"""Real file/subprocess tests for shared input staging and prefetch ordering."""

import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from dingo.common.video_task_protocol import DetachedTaskIdentity
from dingo.vllm.omni.detached_tasks import DetachedOmniTaskManager
from dingo.vllm.omni.h3_shared_inputs import (
    CURRENT,
    TYPE,
    SharedInputSession,
    stage_files,
    validate,
)
from dingo.vllm.omni.request_adapters.minimax_h3 import MiniMaxH3RequestAdapter


def fixture(root, task="video-A", contents=None):
    identity = DetachedTaskIdentity("deployment", "pool", task, 1, "a" * 32)
    task_root = root / "deployment/v1/pools/pool/tasks" / task
    (task_root / "inputs").mkdir(parents=True)
    inputs = contents or [b"raw media bytes"]
    entries = []
    for i, content in enumerate(inputs):
        (task_root / "inputs" / f"{i}.jpg").write_bytes(content)
        entries.append(
            dict(
                ordinal=i,
                path=f"inputs/{i}.jpg",
                content_type="image/jpeg",
                bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
            )
        )
    raw = json.dumps(entries).encode()
    (task_root / "input-manifest.json").write_bytes(raw)
    (task_root / "_artifact.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                deployment_id="deployment",
                pool_id="pool",
                task_id=task,
            )
        )
    )
    value = dict(
        type=TYPE,
        schema_version=1,
        storage_id="storage",
        workflow="fl2va",
        deployment_id="deployment",
        pool_id="pool",
        task_id=task,
        attempt=1,
        execution_token="a" * 32,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        frame_indices=[0] if len(entries) == 1 else [0, -1],
        entries=entries,
    )
    return identity, value, task_root


def adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("DINGO_H3_SHARED_INPUTS_ENABLED", "1")
    monkeypatch.setenv("DINGO_H3_SHARED_INPUTS_STORAGE_ID", "storage")
    return MiniMaxH3RequestAdapter("fl2va", str(tmp_path / "media"), 100 * 1024**2)


async def wait_until(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_real_subprocess_stages_more_than_tcp_limit(tmp_path, monkeypatch):
    identity, value, root = fixture(
        tmp_path / "pvc", contents=[b"a" * (17 * 1024**2), b"b" * (17 * 1024**2)]
    )
    a = adapter(tmp_path, monkeypatch)
    session = SharedInputSession(value, a, tmp_path / "pvc", identity)
    await session.prepare()
    assert session.prepared
    assert sum(p.stat().st_size for p in session.root.iterdir()) > 32 * 1024**2
    assert len(json.dumps(value)) < 2048
    for entry in value["entries"]:
        assert (
            hashlib.sha256(session.path(entry["ordinal"]).read_bytes()).hexdigest()
            == entry["sha256"]
        )
    session.cleanup()
    assert a._media_capacity._used == 0


@pytest.mark.parametrize(
    "fault", ["symlink", "modified", "missing", "manifest", "fifo"]
)
def test_bad_inputs_do_not_escape_or_silently_change(tmp_path, fault):
    identity, value, root = fixture(tmp_path / "pvc")
    path = root / value["entries"][0]["path"]
    if fault == "symlink":
        path.unlink()
        path.symlink_to("/etc/passwd")
    elif fault == "modified":
        path.write_bytes(b"changed input!!")
    elif fault == "missing":
        path.unlink()
    elif fault == "manifest":
        (root / "input-manifest.json").write_text("[]")
    elif fault == "fifo":
        import os

        path.unlink()
        os.mkfifo(path)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises((ValueError, OSError)):
        stage_files(tmp_path / "pvc", out, value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("task_id", "other"),
        ("attempt", 2),
        ("execution_token", "b" * 32),
        ("storage_id", "wrong"),
        ("workflow", "ref2va"),
    ],
)
def test_identity_is_bound_to_detached_envelope(tmp_path, field, value):
    identity, refs, _ = fixture(tmp_path / "pvc")
    refs[field] = value
    with pytest.raises(ValueError):
        validate(refs, storage_id="storage", workflow="fl2va", identity=identity)


@pytest.mark.asyncio
@pytest.mark.parametrize("isolate", [False, True])
async def test_B_stages_while_A_runs_but_only_executes_when_usable(
    tmp_path, monkeypatch, isolate
):
    root = tmp_path / "pvc"
    a = adapter(tmp_path, monkeypatch)

    class Handler:
        sr_ready = True
        request_adapter = a

        def __init__(self):
            self.release = asyncio.Event()
            self.calls = []
            self.staged = []

        async def generate(self, payload, context):
            session = CURRENT.get()
            assert session.prepared
            self.calls.append(session.value["task_id"])
            if len(self.calls) == 1:
                await self.release.wait()
                self.sr_ready = not isolate
            yield {"phase": "complete"}

    h = Handler()
    manager = DetachedOmniTaskManager(
        h, root, prefetch_capacity=1, binary_results=False, inline_results=False
    )
    for task in ("video-A", "video-B"):
        identity, refs, _ = fixture(root, task)
        payload = {"input_reference": json.dumps(refs)}
        result = await manager._submit(identity, payload)
        assert result["accepted"]
        if task == "video-A":
            await wait_until(lambda: h.calls == ["video-A"])
    # Both raw files have been staged while only A has entered the handler.
    await wait_until(
        lambda: len(list((a.media_root / "h3-shared-inputs").glob("request-*/0"))) == 2
    )
    assert h.calls == ["video-A"]
    h.release.set()
    await wait_until(lambda: not manager._running)
    b_identity, _, _ = fixture(tmp_path / "unused", "video-B")
    state = await manager._status(b_identity)
    if isolate:
        assert state["error"]["code"] == "worker_unavailable_before_execution"
        assert h.calls == ["video-A"]
    else:
        assert state["state"] == "completed"
        assert h.calls == ["video-A", "video-B"]
    assert a._media_capacity._used == 0
    await manager.shutdown()


@pytest.mark.asyncio
async def test_adapter_decodes_only_a_bound_prepared_session(tmp_path, monkeypatch):
    import io

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(buf, format="JPEG")
    identity, refs, _ = fixture(tmp_path / "pvc", contents=[buf.getvalue()])
    a = adapter(tmp_path, monkeypatch)
    req = SimpleNamespace(input_reference=json.dumps(refs), input_references=None)
    with pytest.raises(ValueError):
        await a.load_reference(req, None)
    session = SharedInputSession(refs, a, tmp_path / "pvc", identity)
    await session.prepare()

    class Loader:
        @staticmethod
        def _open_image_sync(stream):
            with Image.open(stream) as image:
                return image.convert("RGB")

    token = CURRENT.set(session)
    try:
        result = await a.load_reference(req, Loader())
        assert result.frame_indices == (0,)
        assert result.images[0].getpixel((0, 0))[0] > 250
    finally:
        CURRENT.reset(token)
        session.cleanup()


def test_gateway_builds_only_a_small_descriptor_without_reading_media(
    tmp_path, make_gateway_config, monkeypatch
):
    from dataclasses import replace
    from dingo.video_gateway.adapters.minimax_h3 import MiniMaxH3VideoAdapter
    import dingo.video_gateway.adapters.minimax_h3 as module

    config = make_gateway_config()
    pool = config.pools[0]
    pool = replace(
        pool,
        execution_mode="detached",
        adapter=replace(
            pool.adapter,
            options={
                **pool.adapter.options,
                "reference_transport": "shared_files_v1",
                "shared_storage_id": "storage",
            },
        ),
    )
    gateway = MiniMaxH3VideoAdapter(pool)
    identity, refs, root = fixture(tmp_path / "pvc")
    normalized = gateway.normalize_request(
        {
            "prompt": ["test"],
            "seconds": ["4"],
            "size": ["1344x768"],
            "num_inference_steps": ["2"],
        },
        [],
        "public-fl",
    )
    normalized.update(
        shared_inputs={"storage_id": "storage", "protocol": "shared_files_v1"},
        frame_indices=[0],
    )
    # A descriptor can still be built without opening any original media.
    (root / refs["entries"][0]["path"]).unlink()
    monkeypatch.setattr(
        module, "_data_url", lambda *a, **kw: pytest.fail("Base64 builder was called")
    )
    payload = gateway.build_worker_payload(normalized, refs["entries"], root)
    value = json.loads(payload["input_reference"])
    assert value["entries"] == refs["entries"]
    assert value["manifest_sha256"] == refs["manifest_sha256"]
    assert "base64" not in json.dumps(payload)
    assert "seed" not in payload["nvext"]


def test_shared_tasks_only_select_fresh_matching_worker_capabilities(
    tmp_path, make_gateway_config
):
    import time
    from dingo.video_gateway.dispatcher import VideoDispatcher

    config = make_gateway_config()
    pool = SimpleNamespace(
        config=config.pools[0],
        instance_ids=[1, 2, 3],
        shared_inputs={
            1: {"storage_id": "storage", "workflow": "fl2va"},
            2: {"storage_id": "wrong", "workflow": "fl2va"},
            3: {"storage_id": "storage", "workflow": "fl2va"},
        },
        capacity_cache={
            1: (1, time.monotonic() + 10),
            2: (1, time.monotonic() + 10),
            3: (1, 0),
        },
    )
    assert VideoDispatcher._worker_slots(
        pool, {"shared_inputs": {"storage_id": "storage"}}
    ) == [(1, 0)]


@pytest.mark.asyncio
async def test_preparation_timeout_kills_child_and_releases_capacity(
    tmp_path, monkeypatch
):
    identity, refs, _ = fixture(tmp_path / "pvc")
    a = adapter(tmp_path, monkeypatch)
    session = SharedInputSession(refs, a, tmp_path / "pvc", identity)
    # The short deadline exercises actual process termination/reaping.
    with pytest.raises(TimeoutError):
        await session.prepare(timeout=0.000001)
    session.cleanup()
    assert a._media_capacity._used == 0
    assert not list((a.media_root / "h3-shared-inputs").iterdir())


@pytest.mark.asyncio
async def test_cancel_staging_reaps_its_process_before_cleanup(tmp_path, monkeypatch):
    import sys

    identity, refs, _ = fixture(tmp_path / "pvc")
    a = adapter(tmp_path, monkeypatch)
    session = SharedInputSession(refs, a, tmp_path / "pvc", identity)
    spawn = asyncio.create_subprocess_exec
    children = []
    started = asyncio.Event()

    async def slow_spawn(*args, **kwargs):
        child = await spawn(
            sys.executable, "-I", "-c", "import time; time.sleep(30)", **kwargs
        )
        children.append(child)
        started.set()
        return child

    monkeypatch.setattr(asyncio, "create_subprocess_exec", slow_spawn)
    preparing = asyncio.create_task(session.prepare())
    await started.wait()
    preparing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await preparing
    assert children[0].returncode is not None
    session.cleanup()
    assert a._media_capacity._used == 0


@pytest.mark.asyncio
async def test_prepare_failure_never_enters_model_and_is_not_engine_failure(
    tmp_path, monkeypatch
):
    root = tmp_path / "pvc"
    identity, refs, task_root = fixture(root)
    (task_root / refs["entries"][0]["path"]).unlink()
    a = adapter(tmp_path, monkeypatch)

    class Handler:
        request_adapter = a
        sr_ready = True

        async def generate(self, payload, context):
            pytest.fail("input failure entered the model")
            yield {}

    manager = DetachedOmniTaskManager(
        Handler(), root, prefetch_capacity=1, binary_results=False, inline_results=False
    )
    await manager._submit(identity, {"input_reference": json.dumps(refs)})
    await wait_until(lambda: not manager._running)
    state = await manager._status(identity)
    assert state["error"]["code"] == "input_processing_failed"
    assert state["error"]["internal_error"]["reason"] == "input_processing_failed"
    assert manager.accepting
    assert a._media_capacity._used == 0
    await manager.shutdown()


def test_transport_changes_do_not_change_model_configuration_revision(
    make_gateway_config,
):
    import copy

    pool = {
        "pool_id": "fl-pool",
        "served_models": ["public-fl"],
        "backend_model": "worker-fl",
        "backend_target": "dyn://scope-a.backend.generate",
        "execution_mode": "detached",
        "adapter": {"name": "minimax_h3", "workflow": "fl2va"},
    }
    before = make_gateway_config(pools=[pool]).pools[0]
    updated = copy.deepcopy(pool)
    updated["adapter"].update(
        reference_transport="shared_files_v1", shared_storage_id="storage"
    )
    after = make_gateway_config(pools=[updated]).pools[0]
    assert before.configuration_revision == after.configuration_revision
    assert before.base_configuration_revision == after.base_configuration_revision


@pytest.mark.asyncio
async def test_shared_upload_rejects_an_inline_only_worker_before_enqueue(
    make_gateway_config,
):
    from dingo.video_gateway.adapters.minimax_h3 import MiniMaxH3VideoAdapter
    from dingo.video_gateway.artifact_store import FileArtifactStore
    from dingo.video_gateway.dispatcher import VideoDispatcher
    from dingo.video_gateway.errors import GatewayError
    from dingo.video_gateway.service import VideoGatewayService
    from dingo.video_gateway.task_store import MemoryTaskStore
    from tests.video_gateway.test_dispatcher import FakeClient, FakeContext
    from tests.video_gateway.test_minimax_h3_adapter import _base_fields, _png_upload

    pool = {
        "pool_id": "fl-pool",
        "served_models": ["public-fl"],
        "backend_model": "worker-fl",
        "backend_target": "dyn://scope-a.backend.generate",
        "execution_mode": "detached",
        "adapter": {
            "name": "minimax_h3",
            "workflow": "fl2va",
            "reference_transport": "shared_files_v1",
            "shared_storage_id": "storage",
        },
    }
    config = make_gateway_config(pools=[pool])
    artifacts = FileArtifactStore(config.artifact_store.root)
    store = MemoryTaskStore()
    adapters = {"fl-pool": MiniMaxH3VideoAdapter(config.pools[0])}
    dispatcher = VideoDispatcher(
        config,
        store,
        artifacts,
        {"fl-pool": FakeClient()},
        adapters,
        context_factory=FakeContext,
    )
    await dispatcher._refresh_instances(dispatcher.pools["fl-pool"])
    assert dispatcher.pool_instances("fl-pool") == [7]
    assert not dispatcher.has_workers("fl-pool")
    service = VideoGatewayService(config, store, artifacts, dispatcher, adapters)
    upload_root = await artifacts.create_upload()
    upload = _png_upload(upload_root)
    with pytest.raises(GatewayError) as caught:
        await service.submit(
            fields=_base_fields(),
            uploads=[upload],
            upload_root=upload_root,
            delivery_mode="async",
            idempotency_key=None,
        )
    assert caught.value.status == 503
    assert await store.queue_depth("fl-pool") == 0
    assert dispatcher.media_runtime_snapshot().legacy_input_encoded_bytes == 0
