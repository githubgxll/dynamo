# SPDX-License-Identifier: Apache-2.0
"""Worker transport coordination using shared registry, capacity and CAS slots."""
import asyncio
import hashlib
import json
import logging
import time
from dataclasses import asdict

from dingo.common.task_protocol import (
    ControlRequest,
    InputDescriptor,
    TaskResponse,
)
from dingo.video_gateway.file_io import run_file_io
from dingo.video_gateway.task_runtime.capacity import available_candidates
from dingo.video_gateway.task_runtime.coordinator import (
    DispatchClaim,
    DispatchCoordinator,
)
from dingo.video_gateway.task_runtime.registry import CapabilityRegistry
from dingo.video_gateway.task_runtime.repository import (
    RuntimeKeys,
    WorkerRef,
    WorkerSnapshot,
)
from dingo.video_gateway.task_runtime.transactions import compare_snapshot, encode
from dingo.video_gateway.task_runtime.transport import control_request

from .models import TERMINAL, failure
from .worker_repository import WorkerTaskRepository, observed


class WorkerCoordinator:
    def __init__(self, dispatcher):
        self.dispatcher, self.service = dispatcher, dispatcher.service
        self.store, self.artifacts = self.service.store, self.service.artifacts
        self.repo = WorkerTaskRepository(
            self.store, dispatcher.owner, dispatcher.generation
        )
        self.clients = getattr(self.service.video, "context_ir_clients", {})
        self.registries = {}
        self.workers = {}
        self.next_refresh = 0
        self.context_factory = self.service.video.dispatcher.context_factory
        self.coordinator = DispatchCoordinator(
            dispatcher.supervisor,
            enabled=lambda: dispatcher.accepting
            and not dispatcher.stopping
            and not self.service.video.dispatcher.draining,
        )
        self.scheduling = CompilerSchedulingAdapter(self)

    async def rpc(self, client, instance, request, timeout=5):
        context = self.context_factory(
            "context-ir-control", {"task_kind": "context_ir"}
        )
        try:
            return await control_request(
                client, instance, request.to_dict(), context, timeout_s=timeout
            )
        finally:
            context.stop_generating()

    async def refresh(self):
        self.workers = {}
        for name, backend in self.service.config.backends.items():
            if backend.transport != "worker" or name not in self.clients:
                continue
            registry = self.registries.setdefault(name, CapabilityRegistry())
            client = self.clients[name]
            try:
                ids = client.instance_ids()
            except Exception:
                logging.getLogger(__name__).warning(
                    "compiler discovery unavailable", exc_info=True
                )
                ids = []
            registry.reconcile([str(i) for i in ids])
            for instance in ids:
                try:
                    caps = await self.rpc(
                        client,
                        instance,
                        ControlRequest(
                            "capabilities",
                            self.service.video.config.deployment_id,
                            backend.worker_pool,
                            str(instance),
                        ),
                    )
                    registry.observe(str(instance), caps)
                except Exception:
                    logging.getLogger(__name__).warning(
                        "compiler capabilities unavailable", exc_info=True
                    )
                    registry.records.pop(str(instance), None)
            available = registry.eligible(
                task_kind="context_ir",
                storage_id=backend.storage_id,
                model_adapter=("minimax_h3", 1),
                protocol_adapter=("open_h3_ir", 1),
            )
            for caps in available:
                if ("intent_resolution", 1) not in caps.protocol_adapters:
                    continue
                key = (
                    f"/dingo/task-runtime/v2/{self.service.video.config.deployment_id}/workers/"
                    f"{caps.capacity_domain}/{caps.instance_id}/current"
                )
                value = encode(asdict(caps))
                current = await self.store.client.get(key)
                if current is None or current.value.decode() != value:
                    await self.store.client.txn(
                        compare_snapshot(self.store.client, key, current),
                        [self.store.client.put(key, value)],
                    )
                    current = await self.store.client.get(key)
                if current is None or current.value.decode() != value:
                    continue
                ref = WorkerRef(
                    backend.worker_pool,
                    caps.instance_id,
                    caps.generation,
                    caps.capacity_domain,
                )
                self.workers.setdefault(name, []).append(
                    WorkerSnapshot(
                        ref,
                        observed(current),
                        min(caps.execution_capacity, backend.worker_capacity),
                    )
                )

    async def tick(self):
        if time.monotonic() >= self.next_refresh:
            await self.refresh()
            self.next_refresh = time.monotonic() + 2
        await self.coordinator.dispatch(self.scheduling)
        deployment = self.service.video.config.deployment_id
        runtime = RuntimeKeys(deployment)
        # Resume owned and orphaned attempts, including publicly cancelled work
        # that still occupies a physical slot.
        active, _ = await self.store.client.range_all(
            runtime.root + "/active/context_ir/", prefix=True
        )
        for value in active:
            task_id = value.key.rsplit("/", 1)[1]
            if self.coordinator.manages(task_id):
                continue
            stored = await self.store.get_task(task_id)
            if stored is None:
                continue
            # Adoption has the same confirmation/unknown state as a fresh
            # claim. It must not start a second, independent recovery path.
            await self.coordinator.claim_one(self.scheduling, (stored, None, None))

    def worker_root(self, fence):
        t = fence.task
        return (
            self.artifacts.root
            / t.deployment_id
            / "v2"
            / "context_ir"
            / t.task_id
            / "attempts"
            / f"{fence.attempt}-{fence.execution_token}"
        )

    async def execute(self, initial):
        task = initial.task
        backend = self.service.config.backends[task.backend_id]
        attempt = await self.repo.attempt(initial)
        if attempt is None:
            return
        fence = attempt.fence
        identity = fence.identity()
        root = self.worker_root(fence)
        self.artifacts.path(task.request_path)
        manifest = self.artifacts.path(task.input_manifest_path)
        route = self.artifacts.path(task.route_snapshot_path)
        payload = InputDescriptor(
            backend.storage_id,
            task.request_path,
            task.input_manifest_path,
            hashlib.sha256(await run_file_io(manifest.read_bytes)).hexdigest(),
            task.route_snapshot_path,
            hashlib.sha256(await run_file_io(route.read_bytes)).hexdigest(),
            task.request_digest,
        )
        request = ControlRequest(
            "submit",
            task.deployment_id,
            backend.worker_pool,
            identity.instance_id,
            identity.generation,
            identity,
            payload,
            task.execution_deadline_ms,
        )
        client = self.clients[task.backend_id]
        # If a response is lost, the durable worker record remains authoritative.
        # A duplicate submit to the SAME attempt is safe while its registration lives.
        submitted = (root / "accepted.json").exists()
        if not submitted:
            try:
                response = TaskResponse.parse(
                    await self.rpc(client, identity.instance_id, request)
                )
                if response.identity != identity:
                    raise ValueError("compiler ACK identity mismatch")
                if response.state == "not_accepted":
                    await self.repo.requeue_not_accepted(
                        await self.dispatcher.owned(task.id)
                    )
                    self.workers[task.backend_id] = [
                        w
                        for w in self.workers.get(task.backend_id, [])
                        if w.ref.instance_id != identity.instance_id
                    ]
                    return
            except (TimeoutError, ConnectionError, RuntimeError):
                pass
        latest = await self.dispatcher.owned(task.id)
        if latest.task.status not in TERMINAL:
            await self.repo.mark_dispatched(latest)
        while not self.dispatcher.stopping:
            current = await self.dispatcher.owned(task.id)
            if (
                current.task.status == "cancelled"
                or time.time() * 1000 >= task.execution_deadline_ms
            ):
                if root.exists():
                    await self.artifacts.write_json(
                        root / "cancel.json", {"cancelled": True}
                    )
                try:
                    cancel = ControlRequest(
                        "cancel",
                        task.deployment_id,
                        backend.worker_pool,
                        identity.instance_id,
                        identity.generation,
                        identity,
                    )
                    await self.rpc(client, identity.instance_id, cancel, timeout=2)
                except Exception:
                    logging.getLogger(__name__).debug(
                        "compiler cancel delivery pending", exc_info=True
                    )
            status_path = root / "status.json"
            status = (
                await run_file_io(
                    lambda path=status_path: json.loads(path.read_bytes())
                )
                if status_path.exists()
                else None
            )
            if status is not None:
                response = TaskResponse.parse(status)
                if response.identity != identity:
                    raise ValueError("compiler status identity mismatch")
                if response.state in {"succeeded", "failed", "cancelled"}:
                    break
            await asyncio.sleep(0.5)
        else:
            return
        current = await self.dispatcher.owned(task.id)
        outcome_path = root / "outcome.json"
        outcome = (
            await self.artifacts.read_bounded(
                self.artifacts.relative(outcome_path), 2 * 1024**2
            )
            if outcome_path.exists()
            else {}
        )
        if response.state == "succeeded":
            proof = response.result
            if proof.storage_id != backend.storage_id:
                raise ValueError("compiler result storage mismatch")
            outcome = await self.artifacts.read_bounded(
                proof.path, 512 * 1024, proof.sha256
            )
            result = outcome["result"]
            adapter = self.service.video.adapters[task.target_pool_id]
            if (
                len(result["content"]["prompt"].encode())
                > adapter.limits.max_prompt_bytes
            ):
                raise ValueError("compiler prompt exceeds configured limit")
            canonical = self.artifacts.path(
                f"tasks/{task.id}/attempts/{task.attempt:04d}/result.json"
            )
            await self.artifacts.write_json(canonical, result)
            raw = await run_file_io(canonical.read_bytes)
            settled = await self.repo.settle(
                current,
                "completed",
                result={
                    "result_path": self.artifacts.relative(canonical),
                    "result_sha256": hashlib.sha256(raw).hexdigest(),
                    "result_bytes": len(raw),
                },
                usage=outcome.get("usage"),
                metrics={
                    **outcome.get("metrics", {}),
                    "execution_time_s": max(0, time.time() - task.started_at_ms / 1000),
                    "queue_wait_s": (task.started_at_ms - task.created_at_ms) / 1000,
                },
                descriptor=proof,
            )
        else:
            settled = await self.repo.settle(
                current,
                "cancelled" if response.state == "cancelled" else "failed",
                usage=outcome.get("usage"),
                error=None
                if response.state == "cancelled"
                else failure(
                    response.error_code or "context_ir_worker_failed",
                    "compiler Worker execution failed",
                ),
            )

        if settled is not None:
            await self.service.reconcile_worker_storage(settled)


class CompilerSchedulingAdapter:
    """Compiler capability/slot policy; lifecycle is owned by the coordinator."""

    def __init__(self, worker):
        self.worker = worker

    async def candidates(self):
        worker = self.worker
        deployment = worker.service.video.config.deployment_id
        runtime = RuntimeKeys(deployment)
        occupied, _ = await worker.store.client.range_all(
            runtime.root + "/slots/", prefix=True
        )
        occupied_keys = {v.key for v in occupied}
        selected = []
        for name, workers in worker.workers.items():
            backend = worker.service.config.backends[name]
            available = available_candidates(
                workers,
                occupied_keys,
                deployment_id=deployment,
                configured_capacity=backend.worker_capacity,
            )
            queued = await worker.store.client.range(
                runtime.root + f"/queues/context_ir/{backend.worker_pool}/",
                prefix=True,
                limit=backend.queue_limit,
            )
            for entry, (instance, slot) in zip(queued, available):
                stored = await worker.store.get_task(entry.value.decode())
                if stored and stored.task.backend_id == name:
                    selected.append((stored, instance, slot))
        return selected

    def key(self, candidate):
        return candidate[0].task.id

    async def claim(self, candidate):
        stored, instance, slot = candidate
        if instance is None:
            return DispatchClaim(await self.worker.repo.adopt(stored))
        return DispatchClaim(await self.worker.repo.claim(stored, instance, slot))

    def committed(self, candidate, record):
        pass

    async def release_preparation(self, candidate):
        pass

    async def execute(self, record):
        await self.worker.dispatcher.execute(record)
