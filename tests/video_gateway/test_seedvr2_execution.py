# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from dingo.vllm.omni.seedvr2_client import SeedVR2Client, SubmissionUnknown
from dingo.vllm.omni.seedvr2_execution import (
    SeedVR2Execution,
    UnsafeRecovery,
    read_journal,
    write_journal,
)

JOB = "a" * 32


class Client:
    job_route = staticmethod(SeedVR2Client.job_route)

    def __init__(self):
        self.posts = 0
        self.deletes = 0
        self.states = [{"status": "completed", "frames": 122}]
        self.error = None
        self.submitted = asyncio.Event()
        self.block = False

    async def submit(self, *args, **kwargs):
        self.posts += 1
        self.submitted.set()
        if self.block:
            await asyncio.Event().wait()
        if self.error:
            raise self.error
        return JOB

    async def status(self, job):
        assert job == JOB
        if len(self.states) > 1:
            return self.states.pop(0)
        return self.states[0]

    async def cancel_and_wait(self, job, **kwargs):
        self.deletes += 1
        return {"status": "cancelled", "frames": 0}


def setup(tmp_path, client=None, **settings):
    client = client or Client()
    quarantine = []
    engine = SeedVR2Execution(
        client,
        owner="process-incarnation-1",
        quarantine=quarantine.append,
        poll_s=0.001,
        **settings,
    )
    path = tmp_path / "sr-journal.json"
    kwargs = dict(
        fingerprint="request-digest",
        width=2048,
        height=1152,
        frames=122,
        stopped=lambda: False,
    )
    return client, quarantine, engine, path, kwargs


@pytest.mark.asyncio
async def test_success_and_reattach_do_not_repeat_post(tmp_path):
    client, q, engine, path, kwargs = setup(tmp_path)
    source = tmp_path / "base.mp4"
    assert await engine.run(source, path, **kwargs) == JOB
    record = read_journal(path)
    assert record["phase"] == "completed"
    assert record["job_id"] == JOB
    assert record["seed"] is None
    assert await engine.run(source, path, **kwargs) == JOB
    assert client.posts == 1
    assert not q


@pytest.mark.asyncio
async def test_ambiguous_submit_persists_intent_and_cannot_repost(tmp_path):
    client, q, engine, path, kwargs = setup(tmp_path)
    client.error = SubmissionUnknown("lost response")
    with pytest.raises(SubmissionUnknown):
        await engine.run(tmp_path / "base.mp4", path, **kwargs)
    assert read_journal(path)["phase"] == "submit_intent"
    assert q and client.posts == 1 and client.deletes == 0
    with pytest.raises(UnsafeRecovery):
        await engine.run(tmp_path / "base.mp4", path, **kwargs)
    assert client.posts == 1


@pytest.mark.asyncio
async def test_cancellation_during_post_quarantines(tmp_path):
    client, q, engine, path, kwargs = setup(tmp_path)
    client.block = True
    task = asyncio.create_task(engine.run(tmp_path / "base.mp4", path, **kwargs))
    await client.submitted.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert q and client.posts == 1
    assert read_journal(path)["phase"] == "submit_intent"


@pytest.mark.asyncio
async def test_known_job_timeout_with_confirmed_cancellation_can_continue(tmp_path):
    client, q, engine, path, kwargs = setup(tmp_path, timeout_s=0.01)
    client.states = [{"status": "running", "frames": 1}]
    with pytest.raises(TimeoutError):
        await engine.run(tmp_path / "base.mp4", path, **kwargs)
    assert not q and client.deletes == 1
    assert read_journal(path)["phase"] == "cancelled"
    client.states = [{"status": "completed", "frames": 122}]
    assert await engine.run(tmp_path / "base.mp4", tmp_path / "next.json", **kwargs) == JOB


@pytest.mark.asyncio
async def test_failed_sr_job_does_not_poison_next_request(tmp_path):
    client, q, engine, path, kwargs = setup(tmp_path)
    client.states = [{"status": "failed", "frames": 0}]
    with pytest.raises(RuntimeError):
        await engine.run(tmp_path / "base.mp4", path, **kwargs)
    assert not q and read_journal(path)["phase"] == "failed"
    client.states = [{"status": "completed", "frames": 122}]
    assert await engine.run(tmp_path / "base.mp4", tmp_path / "next.json", **kwargs) == JOB
    assert client.posts == 2


@pytest.mark.asyncio
async def test_unconfirmed_cancellation_keeps_worker_quarantined(tmp_path):
    client, q, engine, path, kwargs = setup(tmp_path, timeout_s=0.01)
    client.states = [{"status": "running", "frames": 1}]
    pauses = []
    engine.pause = lambda: pauses.append("pause")
    engine.resume = lambda: pauses.append("resume")
    async def never_settles(*args, **kwargs):
        return {"status": "running", "frames": 1}
    client.cancel_and_wait = never_settles
    with pytest.raises(TimeoutError):
        await engine.run(tmp_path / "base.mp4", path, **kwargs)
    assert q and pauses == ["pause"]


@pytest.mark.asyncio
async def test_admission_stays_paused_until_cancel_is_confirmed(tmp_path):
    client, q, engine, path, kwargs = setup(tmp_path, timeout_s=0.01)
    client.states = [{"status": "running", "frames": 1}]
    cancelling, settled = asyncio.Event(), asyncio.Event()
    transitions = []
    engine.pause = lambda: transitions.append("pause")
    engine.resume = lambda: transitions.append("resume")
    async def cancel(*args, **kwargs):
        cancelling.set()
        await settled.wait()
        return {"status": "cancelled", "frames": 1}
    client.cancel_and_wait = cancel
    task = asyncio.create_task(engine.run(tmp_path / "base.mp4", path, **kwargs))
    await asyncio.wait_for(cancelling.wait(), .5)
    assert transitions == ["pause"] and not q and not task.done()
    settled.set()
    with pytest.raises(TimeoutError):
        await task
    assert transitions == ["pause", "resume"] and not q


@pytest.mark.asyncio
async def test_second_cancellation_does_not_interrupt_native_settlement(tmp_path):
    client, q, engine, path, kwargs = setup(tmp_path, cancel_timeout_s=.5)
    client.states = [{"status": "running", "frames": 1}]
    cancelling, settled = asyncio.Event(), asyncio.Event()
    transitions = []
    engine.pause = lambda: transitions.append("pause")
    engine.resume = lambda: transitions.append("resume")
    async def cancel(*args, **kwargs):
        cancelling.set()
        await settled.wait()
        return {"status": "cancelled", "frames": 1}
    client.cancel_and_wait = cancel
    task = asyncio.create_task(engine.run(tmp_path / "base.mp4", path, **kwargs))
    await client.submitted.wait()
    task.cancel()
    await asyncio.wait_for(cancelling.wait(), .5)
    task.cancel()  # Mirrors the detached manager's five-second abort enforcer.
    await asyncio.sleep(.01)
    assert transitions == ["pause"] and not q and not task.done()
    settled.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert transitions == ["pause", "resume"] and not q
    assert read_journal(path)["phase"] == "cancelled"


@pytest.mark.asyncio
async def test_repeated_cancellation_does_not_extend_settlement_deadline(tmp_path):
    client, q, engine, path, kwargs = setup(tmp_path, cancel_timeout_s=.05)
    client.states = [{"status": "running", "frames": 1}]
    cancelling = asyncio.Event()
    async def never_settles(*args, **kwargs):
        cancelling.set()
        await asyncio.Event().wait()
    client.cancel_and_wait = never_settles
    task = asyncio.create_task(engine.run(tmp_path / "base.mp4", path, **kwargs))
    await client.submitted.wait(); task.cancel()
    await asyncio.wait_for(cancelling.wait(), .5)
    task.cancel(); await asyncio.sleep(.01)
    assert not q and not task.done()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, .5)
    assert q  # Deadline elapsed without a terminal native state.


@pytest.mark.asyncio
async def test_changed_owner_never_adopts_existing_job(tmp_path):
    client, q, engine, path, kwargs = setup(tmp_path)
    await engine.run(tmp_path / "base.mp4", path, **kwargs)
    engine.owner = "process-incarnation-2"
    with pytest.raises(UnsafeRecovery):
        await engine.run(tmp_path / "base.mp4", path, **kwargs)
    assert q and client.posts == 1


@pytest.mark.asyncio
async def test_changed_fingerprint_rejected(tmp_path):
    client, q, engine, path, kwargs = setup(tmp_path)
    await engine.run(tmp_path / "base.mp4", path, **kwargs)
    kwargs["fingerprint"] = "different"
    with pytest.raises(UnsafeRecovery):
        await engine.run(tmp_path / "base.mp4", path, **kwargs)
    assert client.posts == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state",
    [
        {"status": "completed", "frames": 121},
        {"status": "completed", "frames": 123},
        {"status": "failed", "frames": 3},
    ],
)
async def test_bad_completion_is_not_success(tmp_path, state):
    client, q, engine, path, kwargs = setup(tmp_path)
    client.states = [state]
    with pytest.raises(RuntimeError):
        await engine.run(tmp_path / "base.mp4", path, **kwargs)


@pytest.mark.asyncio
async def test_already_cancelled_never_submits(tmp_path):
    client, q, engine, path, kwargs = setup(tmp_path)
    kwargs["stopped"] = lambda: True
    with pytest.raises(asyncio.CancelledError):
        await engine.run(tmp_path / "base.mp4", path, **kwargs)
    assert client.posts == 0 and not path.exists() and not q


def test_journal_rejects_symlink_and_oversized_metadata(tmp_path):
    target = tmp_path / "target"
    target.write_text("{}")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(ValueError):
        read_journal(link)
    with pytest.raises(ValueError):
        write_journal(link, {})
    target.write_text("x" * (65536 + 1))
    with pytest.raises(ValueError):
        read_journal(target)


@pytest.mark.parametrize("job", ["../x", "", "a" * 31, "a" * 33, "g" * 32, None])
def test_job_id_cannot_escape_route(job):
    with pytest.raises(ValueError):
        SeedVR2Client.job_route(job)

@pytest.mark.asyncio
async def test_native_oom_classification_survives_status_and_journal(tmp_path):
    from dingo.common.video_worker_error import VideoWorkerFailure
    client,q,engine,path,kwargs=setup(tmp_path)
    detail={'schema_version':1,'reason':'resource_exhausted','stage':'super_resolution'}
    client.states=[{'status':'failed','frames':0,'internal_error':detail}]
    with pytest.raises(VideoWorkerFailure) as caught:
        await engine.run(tmp_path/'base.mp4',path,**kwargs)
    assert caught.value.internal_error==detail
    assert read_journal(path)['phase']=='failed'
    assert client.deletes==0
