"""Compiler admission, durable result access and conservative lifecycle policy."""
import asyncio
from dataclasses import asdict, replace
import hashlib
import json
import logging
import time
from pathlib import Path
from dingo.video_gateway.errors import GatewayError, StoreUnavailable, StoreConflict
from dingo.video_gateway.file_io import run_file_io
from dingo.video_gateway.ids import new_video_id
from dingo.video_gateway.models import now_ms
from dingo.video_gateway.etcd_http import EtcdHttpClient
from dingo.video_gateway.telemetry import GatewayTelemetry
from dingo.video_gateway.task_runtime.transactions import encode, compare_snapshot
from .artifact_store import ContextIRArtifacts
from .task_store import ContextIRStore
from .models import ContextIRTask, TERMINAL, failure
from .dispatcher import ContextIRDispatcher
from .validation import normalize, request_digest

log = logging.getLogger(__name__)


class ContextIRService:
    def __init__(self, video, *, client=None):
        self.video, self.config = video, video.config.context_ir
        self.telemetry = GatewayTelemetry()
        self.uploading = set()
        cfg = video.config
        if client is None:
            client = EtcdHttpClient(cfg.task_store.endpoints or cfg.task_store.url,
                                    timeout_s=cfg.task_store.request_timeout_s)
        self.store = ContextIRStore(client, cfg.deployment_id, self.config)
        self.artifacts = ContextIRArtifacts(cfg.artifact_store.root, cfg.deployment_id, self.config.result_max_bytes)
        self.dispatcher = ContextIRDispatcher(self)

    async def start(self):
        await self.dispatcher.start()

    async def stop(self):
        await self.dispatcher.stop()
        await self.store.client.close()

    def check_accepting(self):
        if not self.config.enabled or not self.dispatcher.accepting or self.video.dispatcher.draining:
            raise GatewayError(503, "context_ir_unavailable", "compiler task admission is unavailable")

    async def begin_upload(self):
        self.check_accepting()
        required = self.video.config.media.max_total_file_bytes + self.config.derived_bytes_per_task + self.config.response_max_bytes + self.config.result_max_bytes + 1024**2
        await self.video.ensure_submission_capacity(required)
        root = await self.artifacts.create_upload()
        self.uploading.add(root.name)
        task_id = new_video_id().replace("video-", "context-ir-", 1)
        try:
            await self.store.reserve_upload(root.name, task_id, self.dispatcher.owner, required)
        except StoreUnavailable:
            # The reservation transaction may have committed: retain evidence.
            self.uploading.discard(root.name)
            raise
        except BaseException:
            self.uploading.discard(root.name)
            await self.artifacts.discard(root)
            raise
        return root, task_id, required

    async def abandon_upload(self, root):
        # Only a positively unadmitted reservation may be released. This also
        # handles a response lost after the task admission transaction committed.
        reservation = await self.store.client.get(self.store.key(f"reservations/{root.name}"))
        if reservation is None:
            await self.artifacts.discard(root)
            return
        data = json.loads(reservation.value)
        if data["admitted"] or await self.store.get_task(data["task_id"]) is not None:
            return
        task_root = self.artifacts.task_root(self.video.config.deployment_id, "", data["task_id"])
        await self.artifacts.discard(task_root)
        await self.artifacts.discard(root)
        await self.store.release_reservation(root.name)

    async def idempotent_request(self, principal_hash, key_hash, normalized):
        digest = request_digest(normalized)
        try:
            return await self.store.idempotent(principal_hash, key_hash, digest)
        except GatewayError as conflict:
            if conflict.code != "idempotency_conflict":
                raise
            legacy = dict(normalized)
            legacy["references"] = [{k: v for k, v in entry.items() if k != "role_stated"}
                                    for entry in normalized["references"]]
            legacy_digest = request_digest(legacy)
            if legacy_digest == digest:
                raise
            # Only the original version lacked role_stated. New tasks must continue to
            # distinguish omitted roles from explicit roles, even when defaults match.
            existing = await self.store.idempotent(principal_hash, key_hash, legacy_digest)
            if existing is None:
                raise conflict
            original = await self.artifacts.read_bounded(existing.task.request_path, self.config.result_max_bytes)
            if any("role_stated" in entry for entry in original.get("references", [])):
                raise conflict
            if request_digest(original) != legacy_digest:
                raise conflict
            return existing

    async def submit(self, parsed, task_id, reserved_bytes, idempotency_key, principal="validation"):
        self.check_accepting()
        if idempotency_key is not None and (not idempotency_key or len(idempotency_key.encode()) > 256):
            raise GatewayError(400, "invalid_idempotency_key", "Idempotency-Key must be between 1 and 256 bytes")
        async with self.dispatcher.cpu_slots:
            normalized = await run_file_io(normalize, self.video, parsed.fields, parsed.uploads)
        principal_hash = hashlib.sha256(principal.encode()).hexdigest()
        key_hash = hashlib.sha256(idempotency_key.encode()).hexdigest() if idempotency_key else None
        digest = request_digest(normalized)
        existing = await self.idempotent_request(principal_hash, key_hash, normalized)
        if existing:
            return existing, False
        reserved_bytes = parsed.total_bytes + self.config.derived_bytes_per_task + self.config.response_max_bytes + self.config.result_max_bytes + 1024**2
        await self.store.shrink_reservation(parsed.upload_root.name, reserved_bytes)
        backend_id = self.config.model_bindings[normalized["model"]]
        backend = self.config.backends[backend_id]
        if backend_id not in self.dispatcher.backend_ready:
            raise GatewayError(503, "context_ir_configuration_mismatch", "compiler backend configuration is inconsistent")
        await self.store.backend_compares(backend_id)
        route = {"backend_id": backend_id, "backend": asdict(backend), "model": normalized["model"],
                 "adapter": "minimax_h3", "adapter_version": 1, "protocol": "open_h3_ir", "protocol_version": 1,
                 "policy": {k:v for k,v in asdict(self.config).items() if k not in {"backends", "model_bindings"}}}
        revision = hashlib.sha256(encode(route).encode()).hexdigest()
        upload = parsed.upload_root
        await self.artifacts.write_json(upload / "request.json", normalized)
        await self.artifacts.write_json(upload / "input-manifest.json", normalized["references"])
        await self.artifacts.write_json(upload / "route-snapshot.json", route)
        await self.artifacts.write_json(upload / "_artifact.json", {"kind": "context_ir", "schema_version": 1,
                                        "deployment_id": self.video.config.deployment_id, "task_id": task_id})
        task_root = await self.artifacts.commit_upload(upload, self.video.config.deployment_id, "", task_id)
        stamp = now_ms()
        task = ContextIRTask(id=task_id, deployment_id=self.video.config.deployment_id, principal_hash=principal_hash,
            model=normalized["model"], target_pool_id=normalized["target_pool_id"], backend_id=backend_id,
            backend_epoch=backend.epoch, route_revision=revision, request_digest=digest, idempotency_hash=key_hash,
            request_path=self.artifacts.relative(task_root / "request.json"),
            input_manifest_path=self.artifacts.relative(task_root / "input-manifest.json"),
            route_snapshot_path=self.artifacts.relative(task_root / "route-snapshot.json"), reservation_id=upload.name,
            created_at_ms=stamp, queue_deadline_ms=stamp+int(backend.queue_timeout_s*1000),
            input_bytes=parsed.total_bytes, reserved_bytes=reserved_bytes)
        try:
            stored, created = await self.store.create(task)
        except GatewayError as conflict:
            if conflict.code != "idempotency_conflict":
                raise
            stored = await self.idempotent_request(principal_hash, key_hash, normalized)
            if stored is None:
                raise
            return stored, False
        except StoreUnavailable:
            # Never clean the renamed directory on an ambiguous transaction.
            stored = await self.store.get_task(task.id)
            if stored is None:
                stored = await self.idempotent_request(principal_hash, key_hash, normalized)
            if stored is None:
                raise
            return stored, stored.task.id == task.id
        if created:
            self.telemetry.increment("dingo_context_ir_submitted_total", labels={"model": task.model, "backend": task.backend_id})
        return stored, created

    async def metrics(self):
        for name, backend in self.config.backends.items():
            labels = {"backend": name}
            for kind in ("queued", "inflight", "reserved_bytes"):
                value = await self.store.client.get(self.store.key(f"counters/{name}/{kind}"))
                self.telemetry.set_gauge(f"dingo_context_ir_{kind}", int(value.value) if value else 0, labels=labels)
            value = await self.store.client.get(self.store.key(f"backends/{name}/state"))
            self.telemetry.set_gauge("dingo_context_ir_quarantined", int(value is not None and json.loads(value.value).get("state")=="quarantined"), labels=labels)
            self.telemetry.set_gauge("dingo_context_ir_max_inflight", backend.max_inflight, labels=labels)
        return self.telemetry.render_prometheus()

    async def get(self, task_id, principal="validation"):
        stored = await self.store.get_task(task_id)
        if stored is None or stored.task.principal_hash != hashlib.sha256(principal.encode()).hexdigest():
            raise GatewayError(404, "context_ir_not_found", "compiler task was not found")
        if stored.task.status == "expired":
            raise GatewayError(410, "context_ir_expired", "compiler task has expired")
        return stored

    async def public(self, stored):
        result = None
        if stored.task.status == "completed":
            try:
                result = await self.artifacts.read_bounded(stored.task.result_path,
                    self.config.result_max_bytes, stored.task.result_sha256)
            except FileNotFoundError:
                latest = await self.store.get_task(stored.task.id)
                if latest is None or latest.task.status == "expired":
                    raise GatewayError(410, "context_ir_expired", "compiler task has expired")
                raise
        return stored.task.public_dict(result)

    async def delete(self, task_id, principal="validation"):
        for _ in range(30):
            stored = await self.store.get_task(task_id)
            if stored is None or stored.task.principal_hash != hashlib.sha256(principal.encode()).hexdigest():
                raise GatewayError(404, "context_ir_not_found", "compiler task was not found")
            if stored.task.status == "expired":
                return None
            if stored.task.status not in TERMINAL:
                updated = await self.store.finish(stored, "cancelled",
                    result={"cancel_requested_at_ms": now_ms()})
                if updated:
                    return updated
            elif await self.expire(stored):
                return None
        raise StoreConflict("compiler cancellation contention")

    async def expire(self, stored):
        task = replace(stored.task, status="expired", phase="finished", expired_at_ms=now_ms(), cleanup_pending=True)
        ok, _ = await self.store.client.txn(
            [self.store.client.compare_mod(self.store.task_key(task.id), stored.revision)],
            [self.store.client.put(self.store.task_key(task.id), encode(task.to_dict()))])
        return ok

    async def sweep(self):
        # Scanning is bounded by the managed namespace; each destructive action
        # has a leased CAS lock and rechecks the persistent execution slot.
        client = self.store.client
        for value in await self.store.records("tasks/"):
            stored = self.store.decode(value)
            task = stored.task
            if task.status == "queued" and task.queue_deadline_ms <= now_ms():
                await self.store.finish(stored, "failed", error=failure("context_ir_queue_timeout", "compiler queue deadline exceeded"))
                continue
            if task.status in TERMINAL and task.status != "expired" and task.expires_at_ms <= now_ms():
                await self.expire(stored)
                continue
            if task.status not in TERMINAL:
                continue
            if task.slot_id is not None and await client.get(self.store.slot_key(task)) is not None:
                continue
            cleanup_inputs = (task.inputs_cleaned_at_ms is None and
                              now_ms() >= task.terminal_at_ms+int(self.config.input_terminal_retention_s*1000))
            if task.status != "expired" and not cleanup_inputs:
                continue
            lock = self.store.key(f"gc/{task.id}")
            ok, _ = await client.txn([client.compare_version(lock, 0), client.compare_mod(self.store.task_key(task.id), stored.revision)],
                                    [client.put(lock, self.dispatcher.generation, lease_id=self.dispatcher.lease_id)])
            if not ok:
                continue
            try:
                root = self.artifacts.task_root(task.deployment_id, "", task.id)
                if task.status == "expired":
                    await self.artifacts.discard(root)
                    await self.store.release_reservation(task.reservation_id)
                    changed = replace(task, cleanup_pending=False)
                else:
                    await self.artifacts.discard(root / "inputs")
                    for path in root.glob("attempts/*/derived"):
                        await self.artifacts.discard(path)
                    remaining = await run_file_io(self.artifacts._tree_size, root)
                    await self.store.shrink_reservation(task.reservation_id, remaining)
                    changed = replace(task, inputs_cleaned_at_ms=now_ms(), reserved_bytes=remaining)
                if task.status == "expired" and now_ms() >= task.expired_at_ms+int(self.config.tombstone_retention_s*1000):
                    writes = [client.delete(self.store.task_key(task.id))]
                    if task.idempotency_hash:
                        writes.append(client.delete(self.store.key(f"idempotency/{task.principal_hash}/{task.idempotency_hash}")))
                else:
                    writes = [client.put(self.store.task_key(task.id), encode(changed.to_dict()))]
                await client.txn([client.compare_mod(self.store.task_key(task.id), stored.revision), client.compare_value(lock, self.dispatcher.generation)], writes)
            except Exception:
                log.exception("context_ir_cleanup_pending task_id=%s", task.id)
            finally:
                await client.txn([client.compare_value(lock, self.dispatcher.generation)], [client.delete(lock)])
        for value in await self.store.records("reservations/"):
            reservation = json.loads(value.value)
            if reservation["admitted"] or now_ms()-reservation["created_at_ms"] < (self.config.upload_timeout_s+60)*1000:
                continue
            reservation_id = value.key.rsplit("/", 1)[1]
            if reservation["owner"] == self.dispatcher.owner:
                if reservation_id in self.uploading:
                    continue
            elif await client.get(self.store.key(f"owners/{reservation['owner']}")) is not None:
                continue
            await self.abandon_upload(self.artifacts.upload_root / reservation_id)
