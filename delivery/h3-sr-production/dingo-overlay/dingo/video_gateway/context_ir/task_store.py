"""Compiler resource policy on the shared etcd CAS/admission primitives."""
import asyncio
from dataclasses import replace, asdict
import json
import time
from dingo.video_gateway.errors import GatewayError, StoreConflict
from dingo.video_gateway.models import now_ms
from dingo.video_gateway.task_runtime.transactions import (
    admission_transaction, compare_snapshot, encode, get_idempotent,
    get_record, read_counter, sequence_token,
)
from .models import ContextIRTask, StoredContextIRTask, TERMINAL, failure


class ContextIRStore:
    def __init__(self, client, deployment_id, config):
        self.client, self.config = client, config
        self.root = f"/dingo/context-ir/v1/{deployment_id}"

    def key(self, suffix):
        return f"{self.root}/{suffix}"

    def task_key(self, task_id):
        return self.key(f"tasks/{task_id}")

    def queue_key(self, task):
        return self.key(f"backends/{task.backend_id}/queue/{sequence_token(task.created_seq)}/{task.id}")

    def slot_key(self, task):
        return self.key(f"backends/{task.backend_id}/slots/{task.slot_id}")

    async def records(self, suffix):
        values, _ = await self.client.range_all(self.key(suffix), prefix=True)
        return values

    @staticmethod
    def decode(value):
        return StoredContextIRTask(ContextIRTask.from_dict(json.loads(value.value)), value.mod_revision)

    async def get_task(self, task_id):
        return await get_record(self.client, self.task_key(task_id), self.decode)

    async def idempotent(self, principal, key_hash, digest):
        if key_hash is None:
            return None
        return await get_idempotent(self.client, self.key(f"idempotency/{principal}/{key_hash}"), digest, self.get_task)

    async def ensure_backend(self, name, backend):
        key = self.key(f"backends/{name}/config")
        existing = await self.client.get(key)
        data = {"revision": backend.revision, "epoch": backend.epoch,
                "max_inflight": backend.max_inflight, "adapter": backend.adapter}
        if existing is not None:
            if json.loads(existing.value) != data:
                return False
            return True
        ok, _ = await self.client.txn([self.client.compare_version(key, 0)],
                                     [self.client.put(key, encode(data))])
        return ok or await self.ensure_backend(name, backend)

    async def backend_compares(self, backend_id):
        b = self.config.backends[backend_id]
        key = self.key(f"backends/{backend_id}/config")
        value = await self.client.get(key)
        if value is None or json.loads(value.value)["revision"] != b.revision:
            raise GatewayError(503, "context_ir_configuration_mismatch", "compiler backend configuration is inconsistent")
        state_key = self.key(f"backends/{backend_id}/state")
        state = await self.client.get(state_key)
        if state is not None and json.loads(state.value).get("state") == "quarantined":
            raise GatewayError(503, "context_ir_backend_quarantined", "compiler backend execution requires operator reconciliation")
        return [*compare_snapshot(self.client, key, value), *compare_snapshot(self.client, state_key, state)]

    async def reserve_upload(self, upload_id, task_id, owner, amount):
        key = self.key(f"reservations/{upload_id}")
        counter_key = self.key("counters/reserved_bytes")
        for _ in range(40):
            count, counter = await read_counter(self.client, counter_key)
            if count + amount > self.config.managed_storage_budget_bytes:
                raise GatewayError(503, "context_ir_global_storage_capacity_exceeded", "compiler storage budget is exhausted")
            record = {"task_id": task_id, "owner": owner, "created_at_ms": now_ms(),
                      "bytes": amount, "backend_id": None, "admitted": False}
            ok, _ = await self.client.txn(
                [self.client.compare_version(key, 0), *compare_snapshot(self.client, counter_key, counter)],
                [self.client.put(key, encode(record)), self.client.put(counter_key, str(count + amount))])
            if ok:
                return
            current = await self.client.get(key)
            if current is not None:
                if json.loads(current.value) != record:
                    raise StoreConflict("upload reservation already exists")
                return
            await asyncio.sleep(.01)
        raise StoreConflict("compiler reservation contention")

    async def release_reservation(self, reservation_id):
        key = self.key(f"reservations/{reservation_id}")
        for _ in range(40):
            record = await self.client.get(key)
            if record is None:
                return
            data = json.loads(record.value)
            compares = compare_snapshot(self.client, key, record)
            writes = [self.client.delete(key)]
            counters = [self.key("counters/reserved_bytes")]
            if data["backend_id"]:
                counters.append(self.key(f"counters/{data['backend_id']}/reserved_bytes"))
            for ck in counters:
                count, cv = await read_counter(self.client, ck)
                if count < data["bytes"]:
                    raise RuntimeError("compiler storage ledger underflow")
                compares += compare_snapshot(self.client, ck, cv)
                writes.append(self.client.put(ck, str(count - data["bytes"])))
            ok, _ = await self.client.txn(compares, writes)
            if ok:
                return
        raise StoreConflict("compiler reservation release contention")

    async def shrink_reservation(self, reservation_id, remaining_bytes):
        """Release only a proved reduction; repeated cleanup cannot double-debit."""
        key = self.key(f"reservations/{reservation_id}")
        for _ in range(40):
            record = await self.client.get(key)
            if record is None:
                return
            data = json.loads(record.value)
            if remaining_bytes >= data["bytes"]:
                return
            released = data["bytes"] - remaining_bytes
            compares = compare_snapshot(self.client, key, record)
            writes = []
            keys = [self.key("counters/reserved_bytes")]
            if data["backend_id"]:
                keys.append(self.key(f"counters/{data['backend_id']}/reserved_bytes"))
            for counter_key in keys:
                count, value = await read_counter(self.client, counter_key)
                if count < released:
                    raise RuntimeError("compiler storage ledger underflow")
                compares += compare_snapshot(self.client, counter_key, value)
                writes.append(self.client.put(counter_key, str(count-released)))
            data["bytes"] = remaining_bytes
            writes.append(self.client.put(key, encode(data)))
            ok, _ = await self.client.txn(compares, writes)
            if ok:
                return
        raise StoreConflict("compiler quota reconciliation contention")

    async def create(self, task):
        b = self.config.backends[task.backend_id]
        idem = self.key(f"idempotency/{task.principal_hash}/{task.idempotency_hash}") if task.idempotency_hash else None
        for _ in range(60):
            existing = await self.idempotent(task.principal_hash, task.idempotency_hash, task.request_digest)
            if existing:
                return existing, False
            bc = await self.backend_compares(task.backend_id)
            qk = self.key(f"counters/{task.backend_id}/queued")
            sk = self.key("counters/sequence")
            bk = self.key(f"counters/{task.backend_id}/reserved_bytes")
            q, qv = await read_counter(self.client, qk)
            seq, sv = await read_counter(self.client, sk, maximum=10**20 - 1)
            used, bv = await read_counter(self.client, bk)
            rk = self.key(f"reservations/{task.reservation_id}")
            rv = await self.client.get(rk)
            if rv is None:
                raise StoreConflict("compiler upload reservation lost")
            reservation = json.loads(rv.value)
            if reservation["task_id"] != task.id or reservation["admitted"]:
                raise StoreConflict("compiler reservation identity mismatch")
            if q >= b.queue_limit:
                raise GatewayError(429, "context_ir_queue_full", "compiler queue is full")
            if used + reservation["bytes"] > b.storage_budget_bytes:
                raise GatewayError(503, "context_ir_backend_storage_capacity_exceeded", "compiler backend storage budget is exhausted")
            assigned = replace(task, created_seq=seq + 1)
            reservation.update(backend_id=task.backend_id, admitted=True)
            compares, writes = admission_transaction(
                self.client, task_key=self.task_key(task.id), task_json=assigned.to_dict(),
                sequence_key=sk, sequence=seq, sequence_value=sv,
                queue_counter_key=qk, queue_count=q, queue_counter_value=qv,
                queue_writes=[self.client.put(self.queue_key(assigned), task.id)],
                idempotency_key=idem, task_id=task.id, request_digest=task.request_digest,
                compares=[*bc, *compare_snapshot(self.client, bk, bv), *compare_snapshot(self.client, rk, rv)],
                writes=[self.client.put(bk, str(used + reservation["bytes"])), self.client.put(rk, encode(reservation))])
            ok, revision = await self.client.txn(compares, writes)
            if ok:
                return StoredContextIRTask(assigned, revision), True
            await asyncio.sleep(.01)
        raise StoreConflict("compiler admission contention")

    async def claim(self, backend_id, owner, generation):
        b = self.config.backends[backend_id]
        bc = await self.backend_compares(backend_id)
        queued = await self.client.range(self.key(f"backends/{backend_id}/queue/"), prefix=True, limit=1)
        if not queued:
            return None
        stored = await self.get_task(queued[0].value.decode())
        if stored is None or stored.task.status != "queued":
            raise RuntimeError("compiler queue index is inconsistent")
        task = stored.task
        if task.model_adapter_version != 1 or task.protocol_adapter_version != 1:
            return None
        if task.queue_deadline_ms <= now_ms():
            await self.finish(stored, "failed", error=failure("context_ir_queue_timeout", "compiler queue deadline exceeded"))
            return None
        if task.backend_epoch != b.epoch:
            await self.finish(stored, "failed", error=failure("backend_epoch_changed", "compiler backend epoch changed; resubmit the task"))
            return None
        slots = await self.records(f"backends/{backend_id}/slots/")
        ik = self.key(f"counters/{backend_id}/inflight")
        count, iv = await read_counter(self.client, ik)
        if len(slots) != count:
            raise RuntimeError("compiler slots and inflight ledger disagree")
        if count >= b.max_inflight:
            return None
        occupied = {int(s.key.rsplit("/", 1)[1]) for s in slots}
        slot = next(i for i in range(b.max_inflight) if i not in occupied)
        qk = self.key(f"counters/{backend_id}/queued")
        q, qv = await read_counter(self.client, qk)
        if q <= 0:
            raise RuntimeError("compiler queue ledger underflow")
        claimed = replace(task, status="in_progress", phase="preparing", attempt=task.attempt+1,
                          owner_id=owner, owner_generation=generation, slot_id=slot,
                          started_at_ms=task.started_at_ms or now_ms(),
                          execution_deadline_ms=task.execution_deadline_ms or now_ms()+int(b.execution_timeout_s*1000))
        slotkey = self.slot_key(claimed)
        ownerkey = self.key(f"owners/{owner}")
        ok, rev = await self.client.txn(
            [self.client.compare_mod(self.task_key(task.id), stored.revision),
             self.client.compare_version(slotkey, 0), self.client.compare_value(ownerkey, generation),
             *bc, *compare_snapshot(self.client, ik, iv), *compare_snapshot(self.client, qk, qv)],
            [self.client.put(self.task_key(task.id), encode(claimed.to_dict())),
             self.client.delete(self.queue_key(task)), self.client.put(qk, str(q-1)),
             self.client.put(ik, str(count+1)), self.client.put(slotkey, encode(self.identity(claimed)))])
        return StoredContextIRTask(claimed, rev) if ok else None

    @staticmethod
    def identity(task):
        return {k: getattr(task, k) for k in ("id", "attempt", "owner_id", "owner_generation", "backend_epoch")}

    async def update(self, stored, **patch):
        task = replace(stored.task, **patch)
        compares = [self.client.compare_mod(self.task_key(task.id), stored.revision)]
        if task.owner_id:
            compares.append(self.client.compare_value(self.key(f"owners/{task.owner_id}"), task.owner_generation))
            compares.append(self.client.compare_value(self.slot_key(task), encode(self.identity(task))))
        ok, rev = await self.client.txn(compares, [self.client.put(self.task_key(task.id), encode(task.to_dict()))])
        return StoredContextIRTask(task, rev) if ok else None

    async def finish(self, stored, status, *, error=None, result=None):
        if status not in {"completed", "failed", "cancelled"}:
            raise ValueError("invalid compiler terminal transition")
        old = stored.task
        if old.status in TERMINAL:
            return stored
        stamp = now_ms()
        task = replace(old, status=status, phase="finished", terminal_at_ms=stamp,
                       expires_at_ms=stamp+int(self.config.result_retention_s*1000), error=error,
                       **(result or {}))
        compares = [self.client.compare_mod(self.task_key(old.id), stored.revision)]
        if status == "completed":
            compares += [self.client.compare_value(self.key(f"owners/{old.owner_id}"), old.owner_generation),
                         self.client.compare_value(self.slot_key(old), encode(self.identity(old)))]
            current_config = await self.client.get(self.key(f"backends/{old.backend_id}/config"))
            if current_config is None or json.loads(current_config.value)["epoch"] != old.backend_epoch:
                return None
            compares += compare_snapshot(self.client, current_config.key, current_config)
        writes = [self.client.put(self.task_key(task.id), encode(task.to_dict()))]
        if old.status == "queued":
            qk = self.key(f"counters/{old.backend_id}/queued")
            q, qv = await read_counter(self.client, qk)
            if q <= 0:
                raise RuntimeError("compiler queued counter underflow")
            compares += compare_snapshot(self.client, qk, qv)
            writes += [self.client.delete(self.queue_key(old)), self.client.put(qk, str(q-1))]
        for kind, stamp_ms in (("expiry", task.expires_at_ms), ("input_cleanup", stamp+int(self.config.input_terminal_retention_s*1000))):
            writes.append(self.client.put(self.key(f"indexes/{kind}/{stamp_ms:020d}/{task.id}"), task.id))
        ok, rev = await self.client.txn(compares, writes)
        return StoredContextIRTask(task, rev) if ok else None

    async def release_slot(self, task):
        key = self.slot_key(task)
        for _ in range(30):
            slot = await self.client.get(key)
            if slot is None or json.loads(slot.value) != self.identity(task):
                return False
            ik = self.key(f"counters/{task.backend_id}/inflight")
            count, iv = await read_counter(self.client, ik)
            if count <= 0:
                raise RuntimeError("compiler inflight counter underflow")
            ok, _ = await self.client.txn(
                [*compare_snapshot(self.client, key, slot), *compare_snapshot(self.client, ik, iv)],
                [self.client.delete(key), self.client.put(ik, str(count-1))])
            if ok:
                return True
        raise StoreConflict("compiler slot release contention")

    async def quarantine(self, task):
        key = self.key(f"backends/{task.backend_id}/state")
        slotkey = self.slot_key(task)
        ok, _ = await self.client.txn(
            [self.client.compare_value(slotkey, encode(self.identity(task)))],
            [self.client.put(key, encode({"state": "quarantined", "epoch": task.backend_epoch,
                                         "reason": "context_ir_execution_unknown", "at_ms": now_ms()}))])
        return ok

    async def adopt(self, stored, owner, generation):
        old = stored.task
        if not old.owner_id or await self.client.get(self.key(f"owners/{old.owner_id}")) is not None:
            return None
        task = replace(old, owner_id=owner, owner_generation=generation,
                       attempt=old.attempt + (1 if old.upstream_call_intent_at_ms is None and old.status not in TERMINAL else 0))
        ok, revision = await self.client.txn(
            [self.client.compare_mod(self.task_key(old.id), stored.revision),
             self.client.compare_version(self.key(f"owners/{old.owner_id}"), 0),
             self.client.compare_value(self.key(f"owners/{owner}"), generation),
             self.client.compare_value(self.slot_key(old), encode(self.identity(old)))],
            [self.client.put(self.task_key(task.id), encode(task.to_dict())),
             self.client.put(self.slot_key(task), encode(self.identity(task)))])
        return StoredContextIRTask(task, revision) if ok else None
