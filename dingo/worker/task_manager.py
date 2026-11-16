# SPDX-License-Identifier: Apache-2.0
"""Durable detached control loop, independent of video/compiler semantics."""
import asyncio
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from dingo.common.task_protocol import (
    TERMINAL_STATES,
    ResultDescriptor,
    identifier,
    parse_request,
)

from .contracts import RecoveryPolicy, WorkerProfile
from .execution import ExecutionContext, ExecutionKernel
from .runtime_io import atomic_json, read_json


class DetachedTaskManager:
    def __init__(
        self,
        handler,
        artifact_root,
        *,
        deployment_id,
        pool_id,
        instance_id,
        capacity_domain,
        storage_id,
        profile: WorkerProfile,
        recovery=None,
        capacity=1,
        engine_idle=None,
    ):
        for value in (
            deployment_id,
            pool_id,
            str(instance_id),
            capacity_domain,
            storage_id,
        ):
            identifier(value, "worker identity")
        if type(capacity) is not int or capacity < 1:
            raise ValueError("invalid capacity")
        self.handler, self.root = handler, Path(artifact_root).resolve()
        self.deployment_id, self.pool_id = deployment_id, pool_id
        self.instance_id, self.domain = str(instance_id), capacity_domain
        self.storage_id, self.capacity = storage_id, capacity
        if not isinstance(profile, WorkerProfile):
            raise TypeError("WorkerProfile is required")
        self.profile = profile
        self.recovery = recovery if recovery is not None else RecoveryPolicy()
        self.generation = uuid.uuid4().hex
        self.engine_idle = engine_idle
        self.executions = ExecutionKernel(capacity)
        self.lock = self.executions.lock
        self.draining = False
        self.recovering = False
        self.components_ready = True
        self.pending = []
        self.recovery_task = None

    @property
    def jobs(self):
        """Compatibility view for health reporting and existing adapters."""
        return {
            key: (r.execution, r.context.cancelled)
            for key, r in self.executions.records.items()
        }

    @property
    def draining(self):
        return self.executions.admission.draining

    @draining.setter
    def draining(self, value):
        self.executions.admission.draining = value

    @property
    def recovering(self):
        return self.executions.admission.recovering

    @recovering.setter
    def recovering(self, value):
        self.executions.admission.recovering = value

    @property
    def components_ready(self):
        return self.executions.admission.components_ready

    @components_ready.setter
    def components_ready(self, value):
        self.executions.admission.components_ready = value

    async def start(self):
        await self.recovery.acquire(self)
        # A process generation is not evidence that its backend has restarted.
        directory = self.root / self.deployment_id / "v2" / self.profile.task_kind
        for accepted in directory.glob("*/attempts/*/accepted.json"):
            record = read_json(accepted)
            if record.get("capacity_domain") != self.domain:
                continue
            status_path = accepted.parent / "status.json"
            if (
                not status_path.exists()
                or read_json(status_path)["state"] not in TERMINAL_STATES
            ):
                self.pending.append(accepted.parent)
        self.recovering = bool(self.pending)
        if self.pending:
            self.recovery_task = asyncio.create_task(self._recover())

    async def _recover(self):
        while self.pending:
            if self.engine_idle is not None and await self.engine_idle():
                for root in self.pending:
                    record = read_json(root / "accepted.json")
                    outcome = self.recovery.outcome(self, root)
                    atomic_json(root / "outcome.json", outcome)
                    atomic_json(
                        root / "status.json",
                        {
                            "schema_version": 2,
                            "identity": record["identity"],
                            "state": "failed",
                            "error_code": self.profile.recovery_error_code,
                        },
                    )
                self.pending.clear()
                self.recovering = False
                return
            await asyncio.sleep(2)

    def attempt_root(self, identity):
        return (
            self.root
            / identity.deployment_id
            / "v2"
            / identity.task_kind
            / identity.task_id
            / "attempts"
            / f"{identity.attempt}-{identity.execution_token}"
        )

    @property
    def accepting(self):
        return self.executions.admission.can_execute()

    def capabilities(self):
        return {
            "instance_id": self.instance_id,
            "generation": self.generation,
            "capacity_domain": self.domain,
            "storage_id": self.storage_id,
            "protocol_versions": [2],
            "task_kinds": [self.profile.task_kind],
            "execution_capacity": self.capacity,
            "prefetch_capacity": 0,
            "model_adapters": dict(self.profile.model_adapters),
            "protocol_adapters": dict(self.profile.protocol_adapters),
            "queue_allowed": self.executions.admission.can_queue(),
            "accepting": self.accepting,
            "draining": self.draining,
        }

    def response(self, identity, state, **extra):
        return dict(schema_version=2, identity=asdict(identity), state=state, **extra)

    def attempt_status(self, root, identity):
        try:
            return read_json(root / "status.json")
        except FileNotFoundError:
            # Durable acceptance can precede the first status publication.
            # Recovery owns deciding what happened; readers must not resubmit.
            return self.response(identity, "accepted")

    async def generate(self, message, context=None):
        request = parse_request(message)
        if (
            request.deployment_id != self.deployment_id
            or request.pool_id != self.pool_id
        ):
            raise ValueError("worker deployment/pool mismatch")
        if request.op == "capabilities":
            # Runtime connection IDs change on CPU restart, but callers may query
            # old task identities through the new registration during recovery.
            if request.instance_id != self.instance_id:
                raise ValueError("worker registration mismatch")
            yield self.capabilities()
            return
        if request.op == "drain":
            if (
                request.instance_id != self.instance_id
                or request.generation != self.generation
            ):
                raise ValueError("drain generation mismatch")
            self.draining = True
            yield self.capabilities()
            return
        identity = request.identity
        if identity.task_kind != self.profile.task_kind:
            raise ValueError("task kind mismatch")
        root = self.attempt_root(identity)
        if request.op == "submit":
            async with self.lock:
                if (root / "accepted.json").exists():
                    prior = read_json(root / "accepted.json")
                    if prior["identity"] != asdict(identity) or prior[
                        "payload"
                    ] != asdict(request.payload):
                        raise ValueError("duplicate task identity with different input")
                    yield self.attempt_status(root, identity)
                    return
                if (
                    identity.instance_id != self.instance_id
                    or identity.generation != self.generation
                    or not self.accepting
                    or len(self.jobs) >= self.capacity
                    or request.deadline_at_ms <= time.time() * 1000
                ):
                    yield self.response(
                        identity, "not_accepted", error_code="worker_unavailable"
                    )
                    return
                atomic_json(
                    root / "accepted.json",
                    {
                        "identity": asdict(identity),
                        "payload": asdict(request.payload),
                        "capacity_domain": self.domain,
                        "deadline_at_ms": request.deadline_at_ms,
                    },
                )
                atomic_json(root / "status.json", self.response(identity, "accepted"))
                context = ExecutionContext(identity.task_id)
                self.executions.start(
                    str(root),
                    identity,
                    context,
                    self._run(request, root, context.cancelled),
                )
                yield read_json(root / "status.json")
                return
        if not (root / "accepted.json").exists() or read_json(root / "accepted.json")[
            "identity"
        ] != asdict(identity):
            yield self.response(identity, "not_found")
            return
        if request.op == "cancel":
            atomic_json(root / "cancel.json", {"cancelled": True})
            self.executions.cancel(str(root))
        if request.op == "wait":
            end = time.monotonic() + request.wait_timeout_ms / 1000
            record = self.executions.records.get(str(root))
            if record is not None:
                await self.executions.wait(record, request.wait_timeout_ms / 1000)
            else:
                while time.monotonic() < end:
                    if self.attempt_status(root, identity)["state"] in TERMINAL_STATES:
                        break
                    await asyncio.sleep(0.1)
        yield self.attempt_status(root, identity)

    async def _run(self, request, root, cancelled):
        identity = request.identity
        try:
            atomic_json(root / "status.json", self.response(identity, "running"))
            outcome = await self.handler.execute(request, root, cancelled)
            if outcome.get("settled") is False:
                # Keep the physical execution permit and stop new admissions
                # while native backend work may still exist.
                self.recovering = True
                while self.engine_idle is None or not await self.engine_idle():
                    await asyncio.sleep(2)
            if cancelled.is_set() or outcome.get("cancelled"):
                state, extra = "cancelled", {}
            elif outcome.get("failed"):
                state, extra = "failed", {"error_code": outcome["error_code"]}
            else:
                size, digest = atomic_json(root / "result.json", outcome)
                descriptor = ResultDescriptor(
                    self.storage_id,
                    str((root / "result.json").relative_to(self.root)),
                    digest,
                    size,
                )
                state, extra = "succeeded", {"result": asdict(descriptor)}
            # Even cancelled/failed attempts retain accounting for reconciliation.
            atomic_json(root / "outcome.json", outcome)
            terminal = self.response(identity, state, **extra)
            await self.executions.publish_terminal(
                str(root),
                terminal,
                lambda: asyncio.to_thread(atomic_json, root / "status.json", terminal),
            )
            if not self.pending and not any(
                not j.done() and t != str(root) for t, (j, e) in self.jobs.items()
            ):
                self.recovering = False
        except Exception:
            atomic_json(
                root / "status.json",
                self.response(
                    identity, "failed", error_code=self.profile.preparation_error_code
                ),
            )
            raise

    async def shutdown(self, timeout_s=960):
        self.draining = True
        if not await self.executions.drain(timeout_s):
            return False
        await self.recovery.close()
        return True
