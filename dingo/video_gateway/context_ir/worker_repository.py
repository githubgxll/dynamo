# SPDX-License-Identifier: Apache-2.0
"""Compiler business records adapted to shared physical-slot transactions."""
import json
import secrets
from dataclasses import asdict, replace

from dingo.video_gateway.models import now_ms
from dingo.video_gateway.task_runtime.repository import (
    AttemptFence,
    AttemptSnapshot,
    KeySnapshot,
    OwnerSnapshot,
    RuntimeKeys,
    TaskRef,
    TaskSnapshot,
    ValidatedOutcome,
    WorkerRef,
)
from dingo.video_gateway.task_runtime.transactions import (
    claim_transaction,
    compare_snapshot,
    encode,
    execute_fenced_transaction,
    confirm_record_write,
    settle_transaction,
)
from .models import StoredContextIRTask


def observed(value):
    return KeySnapshot(value.key, value.mod_revision, value.value)


def fence_from(value):
    return AttemptFence(
        TaskRef(**value["task"]),
        WorkerRef(**value["worker"]),
        value["attempt"],
        value["execution_token"],
    )


class WorkerTaskRepository:
    def __init__(self, store, owner, generation):
        self.store, self.client = store, store.client
        self.owner, self.generation = owner, generation

    def ref(self, task):
        return TaskRef(task.deployment_id, "context_ir", task.id)

    async def owner_snapshot(self):
        value = await self.client.get(self.store.key(f"owners/{self.owner}"))
        if value is None or value.value.decode() != self.generation:
            raise RuntimeError("compiler owner lease lost")
        return OwnerSnapshot(self.generation, observed(value))

    async def claim(self, stored, worker, slot_id):
        old = stored.task
        if old.status != "queued" or old.queue_deadline_ms <= now_ms():
            return None
        backend = self.store.config.backends[old.backend_id]
        if old.backend_epoch != backend.epoch:
            return None
        route = await self.client.get(
            self.store.key(f"backends/{old.backend_id}/config")
        )
        if route is None or json.loads(route.value)["revision"] != backend.revision:
            return None
        queue = await self.client.get(self.store.queue_key(old))
        count = await self.client.get(
            self.store.key(f"counters/{old.backend_id}/queued")
        )
        record = await self.client.get(self.store.task_key(old.id))
        if (
            not queue
            or not count
            or not record
            or record.mod_revision != stored.revision
        ):
            return None
        fence = AttemptFence(
            self.ref(old), worker.ref, old.attempt + 1, secrets.token_hex(16)
        )
        task = replace(
            old,
            status="in_progress",
            phase="preparing",
            attempt=fence.attempt,
            slot_id=slot_id,
            owner_id=self.owner,
            owner_generation=self.generation,
            started_at_ms=old.started_at_ms or now_ms(),
            execution_deadline_ms=old.execution_deadline_ms
            or now_ms() + int(backend.execution_timeout_s * 1000),
        )
        owner = await self.owner_snapshot()
        compare, writes = claim_transaction(
            self.client,
            task=TaskSnapshot(self.ref(old), observed(record)),
            worker=worker,
            owner=owner,
            route=observed(route),
            queue=observed(queue),
            queued=observed(count),
            slot_id=slot_id,
            fence=fence,
            task_json=task.to_dict(),
        )
        keys = RuntimeKeys(old.deployment_id)

        async def confirm():
            values, _ = await self.client.get_many(
                [
                    self.store.task_key(old.id),
                    keys.active(fence.task),
                    keys.attempt(fence),
                    keys.slot(fence.worker, slot_id),
                    owner.record.key,
                ]
            )
            record, active, execution, slot, current_owner = values
            if any(v is None for v in values):
                return None
            current = self.store.decode(record)
            if (
                current.task.attempt != fence.attempt
                or current.task.owner_id != self.owner
                or current.task.owner_generation != self.generation
                or active.value != encode(asdict(fence)).encode()
                or slot.value != active.value
                or current_owner.value != owner.record.value
            ):
                return None
            held = json.loads(execution.value)
            if (
                held["fence"] != asdict(fence)
                or held["owner_key"] != owner.record.key
                or held["owner_generation"] != owner.generation
            ):
                return None
            return current

        return await execute_fenced_transaction(
            self.client,
            compare,
            writes,
            committed=lambda rev: StoredContextIRTask(task, rev),
            confirm=confirm,
        )

    async def commit(self, task, compare, writes, *, witnesses=(), before=None):
        """Preserve the compiler codec while sharing mutation reconciliation."""
        notified = False
        def notify(record):
            nonlocal notified
            if record is not None and before is not None and not notified:
                self.store.record_transition(before, record.task)
                notified = True
            return record

        async def confirm():
            return notify(await confirm_record_write(
                self.client, self.store.task_key(task.id), encode(task.to_dict()),
                self.store.decode, witnesses=witnesses))

        return await execute_fenced_transaction(
            self.client, compare, writes,
            committed=lambda rev: notify(StoredContextIRTask(task, rev)),
            confirm=confirm,
        )

    async def attempt(self, stored):
        task = stored.task
        keys = RuntimeKeys(task.deployment_id)
        active = await self.client.get(keys.active(self.ref(task)))
        if active is None:
            return None
        fence = fence_from(json.loads(active.value))
        value = await self.client.get(keys.attempt(fence))
        if value is None:
            raise RuntimeError("active compiler attempt missing")
        slot = await self.client.get(
            keys.slot(fence.worker, json.loads(value.value)["slot_id"])
        )
        if slot is None:
            raise RuntimeError("active compiler slot missing")
        record = await self.client.get(self.store.task_key(task.id))
        return AttemptSnapshot(
            fence,
            observed(value),
            TaskSnapshot(self.ref(task), observed(record)),
            observed(slot),
            observed(active),
            await self.owner_snapshot(),
        )

    async def adopt(self, stored):
        old = stored.task
        if old.owner_id == self.owner and old.owner_generation == self.generation:
            return stored
        if not old.owner_id:
            return None
        old_key = self.store.key(f"owners/{old.owner_id}")
        if await self.client.get(old_key) is not None:
            return None
        keys = RuntimeKeys(old.deployment_id)
        active = await self.client.get(keys.active(self.ref(old)))
        if active is None:
            return None
        fence = fence_from(json.loads(active.value))
        execution = await self.client.get(keys.attempt(fence))
        if execution is None:
            return None
        data = json.loads(execution.value)
        task = replace(old, owner_id=self.owner, owner_generation=self.generation)
        owner = await self.owner_snapshot()
        data.update(owner_key=owner.record.key, owner_generation=self.generation)
        compare = [
            self.client.compare_mod(self.store.task_key(old.id), stored.revision),
            self.client.compare_version(old_key, 0),
            *compare_snapshot(self.client, execution.key, execution),
            *compare_snapshot(self.client, active.key, active),
            self.client.compare_value(owner.record.key, self.generation),
        ]
        writes = [
            self.client.put(self.store.task_key(old.id), encode(task.to_dict())),
            self.client.put(execution.key, encode(data)),
        ]
        return await self.commit(
            task,
            compare,
            writes,
            witnesses=(
                (execution.key, encode(data)),
                (active.key, active.value),
                (owner.record.key, owner.record.value),
            ),
        )

    async def settle(
        self,
        stored,
        status,
        result=None,
        usage=None,
        error=None,
        metrics=None,
        descriptor=None,
    ):
        attempt = await self.attempt(stored)
        if attempt is None:
            return stored
        if attempt.task.record.revision != stored.revision:
            return None
        # Never publish an old execution into a newer business record.
        if stored.task.attempt != attempt.fence.attempt:
            return None
        if (
            stored.task.owner_id != self.owner
            or stored.task.owner_generation != self.generation
        ):
            return None
        state = (
            "succeeded"
            if status == "completed"
            else ("cancelled" if status == "cancelled" else "failed")
        )
        task = replace(
            stored.task,
            status=stored.task.status if stored.task.status == "cancelled" else status,
            phase="finished",
            terminal_at_ms=stored.task.terminal_at_ms or now_ms(),
            expires_at_ms=stored.task.expires_at_ms
            or now_ms() + int(self.store.config.result_retention_s * 1000),
            usage=usage,
            error=error,
            metrics=metrics or stored.task.metrics,
            **(result or {}),
        )
        outcome = ValidatedOutcome(
            state, descriptor if state == "succeeded" else None, encode(usage).encode()
        )
        compare, writes = settle_transaction(
            self.client, attempt=attempt, outcome=outcome, task_json=task.to_dict()
        )
        return await self.commit(
            task,
            compare,
            writes,
            witnesses=((attempt.active.key, None), (attempt.slot.key, None)),
            before=stored.task,
        )

    async def mark_dispatched(self, stored):
        if stored.task.status == "cancelled":
            return stored
        attempt = await self.attempt(stored)
        if attempt is None or attempt.task.record.revision != stored.revision:
            return None
        changed = replace(stored.task, phase="compiling")
        compares = [
            self.client.compare_mod(self.store.task_key(changed.id), stored.revision),
            self.client.compare_value(attempt.owner.record.key, self.generation),
            self.client.compare_value(attempt.active.key, attempt.active.value),
        ]
        writes = [
            self.client.put(self.store.task_key(changed.id), encode(changed.to_dict()))
        ]
        return await self.commit(
            changed,
            compares,
            writes,
            witnesses=(
                (attempt.active.key, attempt.active.value),
                (attempt.owner.record.key, attempt.owner.record.value),
            ),
        )

    async def requeue_not_accepted(self, stored):
        """Only a definite Worker negative ACK permits this transition."""
        attempt = await self.attempt(stored)
        if attempt is None or attempt.task.record.revision != stored.revision:
            return None
        if stored.task.status == "cancelled":
            return await self.settle(stored, "cancelled")
        task = replace(
            stored.task,
            status="queued",
            phase="queued",
            slot_id=None,
            owner_id=None,
            owner_generation=None,
            started_at_ms=None,
            execution_deadline_ms=None,
        )
        qkey = self.store.key(f"counters/{task.backend_id}/queued")
        counter = await self.client.get(qkey)
        count = int(counter.value) if counter else 0
        if count < 0:
            raise RuntimeError("negative compiler queue count")
        queue = self.store.queue_key(task)
        compare, writes = settle_transaction(
            self.client,
            attempt=attempt,
            outcome=ValidatedOutcome("failed", None, b"null"),
            task_json=task.to_dict(),
        )
        compare += [
            *compare_snapshot(self.client, qkey, counter),
            self.client.compare_version(queue, 0),
        ]
        writes += [
            self.client.put(queue, task.id),
            self.client.put(qkey, str(count + 1)),
        ]
        return await self.commit(
            task,
            compare,
            writes,
            witnesses=(
                (attempt.active.key, None),
                (attempt.slot.key, None),
                (queue, task.id),
            ),
        )
