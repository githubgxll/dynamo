"""SR-only internal control, with independent lightweight rank identity."""
from __future__ import annotations

import asyncio
import hmac
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import urllib.request

from . import worker_extension as worker
from .state import FailureLatch, atomic_json, identifier, process_birth, read_json
from .operation_context import bind_operation, reset_operation

IDENTITY_PATH = "/internal/h3sr/identity"
RELEASE_PATH = "/internal/h3sr/release-idle-cache"
BUSINESS_PATHS = {"/v1/videos/sync", "/v1/seedvr2/restore-long"}
CANCEL = re.compile(r"/v1/(?:videos|seedvr2/restore-long)/[^/]+/cancel\Z")
DELETE_JOB = re.compile(r"/v1/(?:videos|seedvr2/restore-long)/[^/]+\Z")


def handoff_identifier(epoch, action_id):
    identifier(epoch)
    identifier(action_id)
    return "handoff-" + hashlib.sha256((epoch + "\0" + action_id).encode()).hexdigest()[:48]


def singleton(value):
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return value


class Rejected(RuntimeError):
    def __init__(self, message, status=409):
        self.status = status
        super().__init__(message)


class ControlGate:
    def __init__(self, *, root, epoch, token, long_video, engine_provider, extension_sha256,
                 birth_reader=process_birth):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.epoch, self.token = identifier(epoch), token
        self.long_video, self.engine_provider = long_video, engine_provider
        self.extension_sha256 = extension_sha256
        self.birth_reader = birth_reader
        self.latch = FailureLatch(self.root)
        self.active_business_requests = 0
        if not isinstance(token, str) or len(token) < 32:
            raise ValueError("Private control credential is required")

    def authenticate(self, header, host):
        try:
            local = ipaddress.ip_address(host).is_loopback
        except (ValueError, TypeError):
            local = False
        if not local or not isinstance(header, str) or not hmac.compare_digest(header, self.token):
            raise Rejected("Internal control requires authenticated loopback", 403)

    def _idle(self):
        task = self.long_video.active_task
        if task is not None:
            if not task.done():
                raise Rejected("SR job remains active after HTTP202; no handoff")
            if task.cancelled() or task.exception() is not None:
                raise Rejected("SR job failed or was cancelled; reconcile existing task")
        if self.active_business_requests:
            raise Rejected("SR business HTTP request is active")

    def _engine(self):
        engine = self.engine_provider()
        if not callable(getattr(engine, "collective_rpc", None)):
            raise Rejected("SR engine control RPC is unavailable")
        return engine

    async def identities(self):
        async with self.long_video.job_lock:
            try:
                self.latch.assert_open()
                self._idle()
                value = singleton(await self._engine().collective_rpc(worker.IDENTITY_METHOD, timeout=60,
                    kwargs={"service_epoch": self.epoch, "extension_sha256": self.extension_sha256}, stage_ids=[0]))
                worker.validate_identity_report(value, self.epoch, self.extension_sha256)
                path = self.root / "state.json"
                frozen = read_json(path).get("sr_worker_identities") if path.exists() else None
                if frozen is not None and (worker.validate_identities(value["ranks"], self.epoch, self.extension_sha256)
                        != worker.validate_identities(frozen, self.epoch, self.extension_sha256)):
                    raise Rejected("Ready worker identities changed; reconcile before any new epoch")
                return value
            except BaseException as error:
                self.latch.trip(epoch=self.epoch, error=error)
                raise

    def require_lease(self, payload):
        action_id = identifier(payload.get("action_id"))
        lease = read_json(self.root / "active.json")
        if (lease.get("schema") != "h3sr-production-operation/1" or lease.get("action_id") != action_id
                or lease.get("service_epoch") != self.epoch
                or not isinstance(payload.get("operation_token"), str)
                or not hmac.compare_digest(lease.get("operation_token", ""), payload["operation_token"])):
            raise Rejected("GPU operation lease identity differs", 403)
        birth = lease.get("controller_birth", {})
        if self.birth_reader(birth.get("pid")) != birth:
            raise Rejected("GPU operation owner process birth changed")
        return lease

    async def release(self, payload):
        action_id = payload.get("action_id")
        self.require_lease(payload)
        async with self.long_video.job_lock:
            try:
                self.latch.assert_open()
                lease = self.require_lease(payload)
                self._idle()
                reason = payload.get("reason")
                if reason not in worker.REASONS:
                    raise Rejected("Invalid production handoff history")
                if payload.get("service_epoch") != self.epoch:
                    raise Rejected("Service epoch differs")
                expected = payload.get("expected_rank_identities")
                worker.validate_identities(expected, self.epoch, self.extension_sha256)
                state_path = self.root / "state.json"
                frozen = read_json(state_path).get("sr_worker_identities") if state_path.exists() else None
                if frozen is not None and (worker.validate_identities(expected, self.epoch, self.extension_sha256)
                        != worker.validate_identities(frozen, self.epoch, self.extension_sha256)):
                    raise Rejected("Handoff worker identities differ from Ready epoch")
                memory, timing = payload.get("collect_memory", False), payload.get("collect_timing", False)
                if type(memory) is not bool or type(timing) is not bool:
                    raise Rejected("Explicit metric controls must be boolean", 400)
                handoff_id = identifier(payload.get("handoff_id"))
                if handoff_id != handoff_identifier(self.epoch, action_id):
                    raise Rejected("Handoff ID is not bound to this operation")
                destination = self.root / "handoffs" / (handoff_id + ".json")
                intent = self.root / "handoffs" / (handoff_id + ".intent.json")
                if destination.exists() or intent.exists():
                    raise Rejected("Prior handoff result/intent exists; observe it, never replay RPC")
                atomic_json(intent, {"schema": "h3sr-production-handoff-intent/1", "status": "STARTED",
                    "service_epoch": self.epoch, "handoff_id": handoff_id, "action_id": action_id,
                    "reason": reason, "extension_sha256": self.extension_sha256,
                    "metrics": {"memory": memory, "timing": timing}})
                engine = self._engine()
                # This standard wrapper RPC drains pending async D2H/SHM
                # outputs; an extension synchronize alone would not do that.
                synced = singleton(await engine.collective_rpc("synchronize_device", timeout=60,
                    kwargs={"timeout": 60}, stage_ids=[0]))
                if synced is not None:
                    raise Rejected("SR synchronize/drain did not return successful void")
                if self.require_lease(payload) != lease:
                    raise Rejected("GPU operation lease changed while draining SR")
                self._idle()
                raw = await engine.collective_rpc(worker.METHOD, timeout=60, stage_ids=[0], kwargs={
                    "handoff_id": handoff_id, "reason": reason, "service_epoch": self.epoch,
                    "extension_sha256": self.extension_sha256, "expected_rank_identities": expected,
                    "collect_memory": memory, "collect_timing": timing})
                result = worker.validate_report(singleton(raw), handoff_id, reason, expected, self.epoch,
                                                self.extension_sha256, memory, timing)
                if self.require_lease(payload) != lease:
                    raise Rejected("GPU operation lease changed during cache release")
                self.latch.assert_open()
                atomic_json(destination, result)
                return result
            except BaseException as error:
                self.latch.trip(epoch=self.epoch, action_id=action_id, error=error)
                raise

    def allow_business(self, headers):
        self.latch.assert_open()
        lease = read_json(self.root / "active.json")
        header = headers.get("x-h3sr-operation-token", "")
        action = headers.get("x-h3sr-action-id", "")
        if (lease.get("service_epoch") != self.epoch or lease.get("action_id") != action
                or not header or not hmac.compare_digest(header, lease.get("operation_token", ""))):
            raise Rejected("SR business request must hold shared GPU operation permit", 403)


class ControlledIngress:
    def __init__(self, app, gate):
        self.app, self.gate = app, gate

    def __getattr__(self, name):
        return getattr(self.app, name)

    async def __call__(self, scope, receive, send):
        # Only known GPU-generating ingress is gated. Cancellation, status,
        # management and other Gateway routes remain usable while blocked.
        if scope["type"] != "http" or scope.get("method") not in ("POST", "PUT", "PATCH", "DELETE"):
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        if (path in (IDENTITY_PATH, RELEASE_PATH) or CANCEL.fullmatch(path)
                or (scope.get("method") == "DELETE" and DELETE_JOB.fullmatch(path))):
            return await self.app(scope, receive, send)
        try:
            if scope.get("method") != "POST" or path not in BUSINESS_PATHS:
                raise Rejected("Unsupported native GPU ingress; use the controlled Gateway", 409)
            headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
            self.gate.allow_business(headers)
        except Exception as error:
            body = json.dumps({"error": str(error)}).encode()
            await send({"type": "http.response.start", "status": getattr(error, "status", 409),
                        "headers": [(b"content-type", b"application/json")]})
            return await send({"type": "http.response.body", "body": body})
        self.gate.active_business_requests += 1
        context_token = bind_operation(headers["x-h3sr-action-id"], headers["x-h3sr-operation-token"])
        try:
            await self.app(scope, receive, send)
        except BaseException as error:
            self.gate.latch.trip(epoch=self.gate.epoch, action_id=headers.get("x-h3sr-action-id"), error=error)
            raise
        finally:
            reset_operation(context_token)
            self.gate.active_business_requests -= 1


def install(app, gate):
    from fastapi import HTTPException, Request

    async def dispatch(request, identity):
        try:
            gate.authenticate(request.headers.get("x-h3sr-control-token"), request.client.host if request.client else None)
            return await gate.identities() if identity else await gate.release(await request.json())
        except Rejected as error:
            raise HTTPException(error.status, str(error)) from error

    async def identities(request):
        return await dispatch(request, True)

    async def release(request):
        return await dispatch(request, False)

    identities.__annotations__["request"] = Request
    release.__annotations__["request"] = Request
    app.add_api_route(IDENTITY_PATH, identities, methods=["POST"], include_in_schema=False)
    app.add_api_route(RELEASE_PATH, release, methods=["POST"], include_in_schema=False)
    app.state.h3sr_production_control = gate
    # Preserve an already-built FastAPI/Starlette middleware stack.
    return ControlledIngress(app, gate)


class HTTPControlClient:
    """Private loopback client; transport timeout is UNKNOWN, never auto-retried."""
    def __init__(self, root, *, epoch, port=8124):
        self.root, self.epoch, self.port = Path(root), epoch, port

    async def _request(self, path, payload):
        def request():
            token = (self.root / "control-token").read_text(encoding="ascii").strip()
            req = urllib.request.Request(f"http://127.0.0.1:{self.port}" + path,
                data=json.dumps(payload).encode(), headers={"content-type": "application/json",
                "x-h3sr-control-token": token}, method="POST")
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(req, timeout=150) as response:
                if response.status != 200:
                    raise RuntimeError("Internal SR control HTTP failure")
                return json.load(response)
        return await asyncio.to_thread(request)

    async def identities(self):
        result = await self._request(IDENTITY_PATH, {})
        return worker.validate_identity_report(result, self.epoch, worker.source_sha256())

    async def release(self, reason, lease):
        ready = read_json(self.root / "state.json")
        expected = ready.get("sr_worker_identities")
        result = await self._request(RELEASE_PATH, {"service_epoch": self.epoch,
            "action_id": lease["action_id"], "operation_token": lease["operation_token"],
            "handoff_id": handoff_identifier(self.epoch, lease["action_id"]), "reason": reason,
            "expected_rank_identities": expected, "collect_memory": False, "collect_timing": False})
        return worker.validate_report(result, handoff_identifier(self.epoch, lease["action_id"]), reason,
                                      expected, self.epoch, worker.source_sha256())
