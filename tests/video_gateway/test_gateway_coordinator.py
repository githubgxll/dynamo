# SPDX-License-Identifier: Apache-2.0
"""Common dispatch contracts, including an unrelated finite CPU workload."""
import asyncio

import pytest

from dingo.video_gateway.task_runtime.coordinator import (
    DispatchClaim,
    DispatchCoordinator,
    ExecutionSupervisor,
    monitor_execution,
)
from dingo.video_gateway.task_runtime.transactions import UnconfirmedTransaction


class Policy:
    def __init__(self, kind):
        self.kind = kind
        self.queue = ["one"]
        self.claims = self.commits = self.calls = self.releases = 0
        self.started, self.finish = asyncio.Event(), asyncio.Event()
        self.decision = DispatchClaim("record")

    async def candidates(self):
        return list(self.queue)

    def key(self, candidate):
        return f"{self.kind}-{candidate}"

    async def claim(self, candidate):
        self.claims += 1
        return self.decision

    def committed(self, candidate, record):
        self.commits += 1

    async def release_preparation(self, candidate):
        self.releases += 1

    async def execute(self, record):
        assert record == "record"
        self.calls += 1
        self.started.set()
        await self.finish.wait()


def runtime():
    errors = []
    supervisor = ExecutionSupervisor(lambda key, error: errors.append((key, error)))
    return DispatchCoordinator(supervisor), errors


@pytest.mark.parametrize("kind", ["h3", "sr", "compiler", "cpu-thumbnail"])
async def test_common_claim_execute_dedup_and_connection_independence(kind):
    coordinator, errors = runtime()
    policy = Policy(kind)
    assert await coordinator.dispatch(policy)
    await asyncio.wait_for(policy.started.wait(), 1)
    assert not await coordinator.dispatch(policy)
    assert policy.claims == policy.calls == 1

    # A caller can stop waiting while the controller and model remain alive.
    async def caller():
        await asyncio.shield(coordinator.supervisor.tasks[policy.key("one")])

    waiter = asyncio.create_task(caller())
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert coordinator.supervisor.contains(policy.key("one"))
    policy.finish.set()
    await asyncio.gather(*coordinator.supervisor.executions)
    await asyncio.sleep(0)
    assert not coordinator.supervisor.tasks and not errors
    assert policy.commits == 1 and policy.releases == 0


@pytest.mark.parametrize("retry", [True, False])
async def test_conflict_and_resource_wait_release_only_preparation(retry):
    coordinator, _ = runtime()
    policy = Policy("cpu")
    policy.decision = DispatchClaim(retry_now=retry)
    assert await coordinator.dispatch(policy) is retry
    assert policy.calls == policy.commits == 0 and policy.releases == 1


async def test_local_launch_failure_retains_claim_and_resumes_while_draining(
    monkeypatch,
):
    coordinator, _ = runtime()
    policy = Policy("cpu")
    original = coordinator.supervisor.spawn
    monkeypatch.setattr(
        coordinator.supervisor,
        "spawn",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("local launch failed")),
    )
    with pytest.raises(RuntimeError, match="local launch failed"):
        await coordinator.dispatch(policy)
    assert policy.key("one") in coordinator.pending and policy.releases == 0
    monkeypatch.setattr(coordinator.supervisor, "spawn", original)
    coordinator.enabled = lambda: False
    await coordinator.dispatch(policy)
    await asyncio.wait_for(policy.started.wait(), 1)
    assert policy.claims == policy.calls == 1 and not coordinator.pending
    policy.finish.set()
    await asyncio.gather(*coordinator.supervisor.executions)


async def test_unknown_claim_never_executes_or_releases_then_resumes_same_claim():
    coordinator, _ = runtime()
    policy = Policy("cpu")
    observed = False

    async def resolve():
        if not observed:
            raise UnconfirmedTransaction(resolve)
        return "record"

    async def unknown(candidate):
        policy.claims += 1
        raise UnconfirmedTransaction(resolve)

    policy.claim = unknown
    assert not await coordinator.dispatch(policy)
    assert not await coordinator.dispatch(policy)
    assert policy.calls == policy.releases == 0 and policy.claims == 1
    observed = True
    policy.queue.clear()
    await coordinator.dispatch(policy)
    await asyncio.wait_for(policy.started.wait(), 1)
    assert policy.calls == policy.claims == 1 and not coordinator.uncertain
    policy.finish.set()
    await asyncio.gather(*coordinator.supervisor.executions)


async def test_unknown_claim_with_proven_conflict_releases_without_execution():
    coordinator, _ = runtime()
    policy = Policy("cpu")

    async def resolve():
        return None

    async def unknown(candidate):
        raise UnconfirmedTransaction(resolve)

    policy.claim = unknown
    await coordinator.dispatch(policy)
    policy.queue.clear()
    await coordinator.dispatch(policy)
    assert policy.releases == 1 and policy.calls == 0 and not coordinator.uncertain


async def test_concurrent_claimers_cannot_release_each_others_local_budget():
    coordinator, _ = runtime()
    policy = Policy("cpu")
    entered, release = asyncio.Event(), asyncio.Event()
    original = policy.claim

    async def slow(candidate):
        entered.set()
        await release.wait()
        return await original(candidate)

    policy.claim = slow
    first = asyncio.create_task(coordinator.claim_one(policy, "one"))
    await entered.wait()
    assert not await coordinator.claim_one(policy, "one")
    assert policy.releases == 0
    release.set()
    assert await first
    await policy.started.wait()
    policy.finish.set()
    await asyncio.gather(*coordinator.supervisor.executions)
    assert policy.claims == 1


async def test_old_controller_callback_cannot_remove_new_controller():
    coordinator, errors = runtime()
    supervisor = coordinator.supervisor
    gate = asyncio.Event()
    supervisor.spawn("task", lambda: asyncio.sleep(0))
    old = supervisor.tasks["task"]
    # Complete the old task without running its scheduled done callback yet.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert old.done()
    supervisor.spawn("task", gate.wait)
    await asyncio.sleep(0)
    assert supervisor.contains("task")
    gate.set()
    await asyncio.gather(*supervisor.executions)
    assert not errors


async def test_failed_execution_is_reported_and_next_task_can_run():
    coordinator, errors = runtime()
    policy = Policy("cpu")

    async def failed(record):
        raise ValueError("backend failed")

    policy.execute = failed
    await coordinator.dispatch(policy)
    await asyncio.gather(*coordinator.supervisor.executions, return_exceptions=True)
    await asyncio.sleep(0)
    assert len(errors) == 1 and not coordinator.supervisor.tasks


async def test_operation_completion_wins_and_external_monitor_is_not_cancelled():
    gate = asyncio.Event()
    external = asyncio.create_task(gate.wait())

    async def finished(name, task):
        pytest.fail("operation must win")

    assert (
        await monitor_execution(
            asyncio.sleep(0, result="done"),
            monitors=[("owner", external)],
            on_monitor_finished=finished,
        )
        == "done"
    )
    assert not external.done()
    gate.set()
    await external


async def test_monitor_priority_and_owned_children_are_joined():
    stopped = asyncio.Event()

    async def work():
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    async def finished(name, task):
        raise RuntimeError(name)

    with pytest.raises(RuntimeError, match="owner"):
        await monitor_execution(
            work(),
            monitors=[("owner", asyncio.sleep(0)), ("worker", asyncio.sleep(0))],
            on_monitor_finished=finished,
        )
    assert stopped.is_set()
