"""HTTP execution with durable intent, owner fencing and physical-slot retention."""
import asyncio
from dataclasses import asdict, replace
import hashlib
import json
import logging
import time
import uuid
import aiohttp
from dingo.video_gateway.errors import StoreUnavailable, GatewayError
from dingo.video_gateway.file_io import run_file_io
from dingo.video_gateway.models import now_ms
from dingo.video_gateway.task_runtime.transactions import encode
from .client import CompilerHTTP, UpstreamUnknown, UpstreamRejected
from .models import TERMINAL, failure
from .protocol import prepare, validate_result

log = logging.getLogger(__name__)


class ContextIRDispatcher:
    def __init__(self, service):
        self.service = service
        self.store, self.artifacts, self.config = service.store, service.artifacts, service.config
        self.owner = uuid.uuid4().hex
        self.generation = uuid.uuid4().hex
        self.lease_id = None
        self.running = {}
        self.accepting = False
        self.stopping = False
        self.backend_ready = set()
        self.cpu_slots = asyncio.Semaphore(2)
        self.session = None
        self.loop_task = self.heartbeat_task = None

    async def start(self):
        lease = await self.store.client.lease_grant(self.config.owner_lease_ttl_s)
        self.lease_id = lease.lease_id
        key = self.store.key(f"owners/{self.owner}")
        ok, _ = await self.store.client.txn([self.store.client.compare_version(key, 0)],
            [self.store.client.put(key, self.generation, lease_id=self.lease_id)])
        if not ok:
            raise RuntimeError("ContextIR owner registration conflict")
        self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=64, limit_per_host=32))
        for name, backend in self.config.backends.items():
            if await self.store.ensure_backend(name, backend):
                self.backend_ready.add(name)
        self.accepting = True
        self.heartbeat_task = asyncio.create_task(self.heartbeat(), name="context-ir-owner")
        self.loop_task = asyncio.create_task(self.run(), name="context-ir-dispatch")

    async def heartbeat(self):
        try:
            while not self.stopping:
                await asyncio.sleep(self.config.owner_heartbeat_s)
                lease = await self.store.client.lease_keepalive(self.lease_id)
                if lease.ttl <= 0:
                    raise StoreUnavailable("compiler owner lease expired")
        except asyncio.CancelledError:
            raise
        except Exception:
            self.accepting = False
            log.exception("context_ir_owner_lost")

    def launch(self, stored):
        task_id = stored.task.id
        if task_id in self.running:
            return
        execution = asyncio.create_task(self.execute(stored), name=f"context-ir-{task_id}")
        self.running[task_id] = execution
        def done(future):
            self.running.pop(task_id, None)
            if not future.cancelled() and future.exception():
                log.error("context_ir_supervisor_failed task_id=%s", task_id, exc_info=future.exception())
        execution.add_done_callback(done)

    async def run(self):
        next_sweep = 0
        while not self.stopping:
            try:
                if self.accepting and not self.service.video.dispatcher.draining:
                    for backend in tuple(self.backend_ready):
                        try:
                            stored = await self.store.claim(backend, self.owner, self.generation)
                            if stored:
                                self.launch(stored)
                        except GatewayError:
                            pass  # Per-backend quarantine does not stop another backend.
                if time.monotonic() >= next_sweep:
                    await self.recover()
                    await self.service.sweep()
                    next_sweep = time.monotonic() + self.config.sweeper_interval_s
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("context_ir_dispatch_iteration_failed")
            await asyncio.sleep(.25)

    async def recover(self):
        for backend in self.backend_ready:
            for slot in await self.store.records(f"backends/{backend}/slots/"):
                identity = json.loads(slot.value)
                if await self.store.client.get(self.store.key(f"owners/{identity['owner_id']}")) is not None:
                    continue
                stored = await self.store.get_task(identity["id"])
                if stored is None:
                    raise RuntimeError("orphan compiler slot without task metadata")
                adopted = await self.store.adopt(stored, self.owner, self.generation)
                if adopted:
                    self.launch(adopted)

    async def owned(self, task_id):
        current = await self.store.get_task(task_id)
        if current is None or current.task.owner_id != self.owner or current.task.owner_generation != self.generation:
            raise UpstreamUnknown("compiler execution ownership changed")
        return current

    async def execute(self, initial):
        task = initial.task
        backend = self.config.backends[task.backend_id]
        http = CompilerHTTP(self.session, backend, self.config.response_max_bytes)
        root = self.artifacts.path(f"tasks/{task.id}")
        attempt_root = root / "attempts" / f"{task.attempt:04d}"
        proof_path = attempt_root / "response-received.json"
        lease_token = f"{task.id}-{task.attempt}"
        called = task.upstream_call_intent_at_ms is not None
        confirmed = False
        asset_lease = False
        started = time.monotonic()
        timings = {}
        try:
            route = await self.artifacts.read_bounded(task.route_snapshot_path, 65536)
            if (hashlib.sha256(encode(route).encode()).hexdigest() != task.route_revision
                    or route["backend"]["url"] != backend.url or route["backend"]["epoch"] != task.backend_epoch):
                raise ValueError("compiler route snapshot mismatch")
            normalized = await self.artifacts.read_bounded(task.request_path, self.config.result_max_bytes)
            current = await self.owned(task.id)
            if current is None:
                return
            if called:
                if not proof_path.exists():
                    raise UpstreamUnknown("previous compiler execution has no durable response")
                proof = await self.artifacts.read_bounded(self.artifacts.relative(proof_path), 4096)
                response = await self.artifacts.read_bounded(self.artifacts.relative(attempt_root / "upstream-response.json"),
                                                             self.config.response_max_bytes, proof["sha256"])
                expected = await self.artifacts.read_bounded(self.artifacts.relative(attempt_root / "reference-map.json"), 65536)
                status = proof["status"]
                timings = proof.get("metrics", {})
                confirmed = status in {200, 201, 400, 404, 413, 422}
            else:
                if current.task.status in TERMINAL:
                    confirmed = True
                    return
                if task.attempt > self.config.max_pre_call_attempts:
                    raise UpstreamRejected("context_ir_prepare_attempts_exhausted")
                remaining = (task.execution_deadline_ms-now_ms())/1000
                if remaining <= 0:
                    raise TimeoutError("compiler execution deadline expired before preparation")
                async with asyncio.timeout(min(backend.prepare_timeout_s, remaining)):
                    async with self.cpu_slots:
                        payload, files, expected = await run_file_io(
                            prepare, normalized, root, attempt_root, self.config.derived_bytes_per_task)
                    timings["prepare_time_s"] = time.monotonic()-started
                    await self.artifacts.write_json(attempt_root / "upstream-request.json", payload)
                    await self.artifacts.write_json(attempt_root / "reference-map.json", expected)
                    current = await self.owned(task.id)
                    if current.task.status in TERMINAL:
                        confirmed = True
                        return
                    current = await self.store.update(current, phase="uploading")
                    if current is None:
                        raise UpstreamUnknown("compiler execution ownership changed")
                    code, caps = await http.json("GET", "/v1/capabilities")
                    if code != 200 or not caps.get("gateway_contract", {}).get("asset_leases"):
                        raise UpstreamRejected("compiler lacks required in-flight asset protection")
                    code, _ = await http.json("PUT", f"/v1/asset-leases/{lease_token}", json={"digests": list(files)})
                    if code not in {200, 201}:
                        raise UpstreamRejected("compiler asset lease could not be reserved")
                    asset_lease = True
                    upload_started = time.monotonic()
                    for digest, path in files.items():
                        await http.upload(digest, path, lease_token)
                    timings["asset_upload_time_s"] = time.monotonic()-upload_started
                current = await self.owned(task.id)
                if current.task.status in TERMINAL:
                    confirmed = True
                    return
                remaining = (task.execution_deadline_ms-now_ms())/1000
                if remaining <= 0:
                    raise TimeoutError("compiler execution deadline expired before call")
                current = await self.store.update(current, phase="compiling", upstream_call_intent_at_ms=now_ms())
                if current is None:
                    # No POST was made, so this attempt is physically settled.
                    confirmed = True
                    return
                task = current.task
                called = True
                compile_started = time.monotonic()
                status, response = await http.compile_once(payload, lease_token, min(backend.compile_timeout_s, remaining))
                timings["compile_time_s"] = time.monotonic()-compile_started
                response_path = attempt_root / "upstream-response.json"
                await self.artifacts.write_json(response_path, response)
                response_hash = await run_file_io(lambda: hashlib.sha256(response_path.read_bytes()).hexdigest())
                await self.artifacts.write_json(proof_path, {"status": status, "sha256": response_hash, "received_at_ms": now_ms(), "metrics": timings})
                confirmed = status in {200, 201, 400, 404, 413, 422}
            from .protocol import response_usage
            usage = response_usage(response)
            if usage is not None:
                current = await self.owned(task.id)
                if current is not None:
                    await self.store.update(current, usage=usage)
            if not confirmed:
                raise UpstreamUnknown("compiler response does not prove execution terminated")
            current = await self.owned(task.id)
            if current.task.status in TERMINAL:
                return
            result = validate_result(normalized, expected, status, response)
            adapter = self.service.video.adapters[task.target_pool_id]
            if len(result["content"]["prompt"].encode()) > adapter.limits.max_prompt_bytes:
                raise UpstreamRejected("context_ir_prompt_too_large")
            if len(encode(result).encode()) > self.config.result_max_bytes:
                raise UpstreamRejected("context_ir_result_too_large")
            current = await self.store.update(current, phase="finalizing", upstream_response_received_at_ms=now_ms())
            if current is None:
                return
            result_path = attempt_root / "result.json"
            finalize_started = time.monotonic()
            await self.artifacts.write_json(result_path, result)
            raw = await run_file_io(result_path.read_bytes)
            timings["finalize_time_s"] = time.monotonic()-finalize_started
            finished = await self.store.finish(current, "completed", result={
                "result_path": self.artifacts.relative(result_path), "result_sha256": hashlib.sha256(raw).hexdigest(),
                "result_bytes": len(raw), "metrics": {**timings, "queue_wait_s": (task.started_at_ms-task.created_at_ms)/1000,
                                                        "execution_time_s": time.monotonic()-started}})
            if finished:
                labels={"backend":task.backend_id,"model":task.model,"state":"completed"}
                self.service.telemetry.increment("dingo_context_ir_terminal_total", labels=labels)
                for phase,seconds in timings.items():
                    self.service.telemetry.observe("dingo_context_ir_stage_duration_seconds",seconds,
                        labels={"backend":task.backend_id,"model":task.model,"stage":phase},
                        buckets=(.01,.1,1,5,10,30,60,120,300,600,900))
        except asyncio.CancelledError:
            # Process shutdown after call intent is recovered conservatively;
            # never release its slot merely because this coroutine was cancelled.
            if called and not confirmed:
                try:
                    await self.store.quarantine(task)
                except Exception:
                    log.exception("context_ir_shutdown_quarantine_unconfirmed")
            raise
        except Exception as exc:
            unknown = called and not confirmed
            if unknown:
                await self.store.quarantine(task)
            current = await self.owned(task.id)
            if current is not None and current.task.status not in TERMINAL:
                code = "context_ir_execution_unknown" if unknown else (
                    str(exc) if isinstance(exc, UpstreamRejected) and str(exc).startswith("context_ir_")
                    else "context_ir_preparation_failed")
                finished = await self.store.finish(current, "failed", error=failure(code,
                    "compiler execution outcome is unknown" if unknown else "compiler input preparation or result validation failed"))
                if finished:
                    self.service.telemetry.increment("dingo_context_ir_terminal_total",labels={"backend":task.backend_id,"model":task.model,"state":"failed"})
            log.warning("context_ir_task_failed task_id=%s attempt=%s unknown=%s error_type=%s", task.id, task.attempt, unknown, type(exc).__name__)
            confirmed = not unknown
        finally:
            if confirmed:
                if asset_lease or called:
                    try:
                        await http.json("DELETE", f"/v1/asset-leases/{lease_token}", timeout=10)
                    except Exception:
                        log.warning("context_ir_asset_lease_release_pending task_id=%s", task.id)
                # CAS identity prevents a stale owner from freeing a new attempt.
                latest = await self.store.get_task(task.id)
                if latest is not None and latest.task.status in TERMINAL:
                    await self.store.release_slot(task)

    async def stop(self):
        self.accepting = False
        self.stopping = True
        if self.loop_task:
            self.loop_task.cancel()
            await asyncio.gather(self.loop_task, return_exceptions=True)
        if self.running:
            _, pending = await asyncio.wait(list(self.running.values()), timeout=30)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            await asyncio.gather(self.heartbeat_task, return_exceptions=True)
        if self.session:
            await self.session.close()
        if self.lease_id:
            await self.store.client.lease_revoke(self.lease_id)
