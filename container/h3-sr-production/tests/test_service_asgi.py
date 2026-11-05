"""Real FastAPI/Starlette CPU stack compatibility, without Omni/CUDA imports."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from service import control, state, worker_extension as worker


class ActualASGI(unittest.TestCase):
    def test_existing_built_stack_and_lifespan_are_preserved(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        lifecycle = []
        @asynccontextmanager
        async def lifespan(app):
            lifecycle.append("start"); yield; lifecycle.append("stop")
        app = FastAPI(lifespan=lifespan)
        @app.get("/health")
        async def health(): return {"ok": True}
        app.middleware_stack = app.build_middleware_stack()
        original_stack = app.middleware_stack
        with tempfile.TemporaryDirectory() as temporary:
            gate = control.ControlGate(root=temporary, epoch="test", token="t" * 64,
                long_video=SimpleNamespace(job_lock=asyncio.Lock(), active_task=None), engine_provider=lambda: None,
                extension_sha256=worker.source_sha256())
            wrapped = control.install(app, gate)
            self.assertIs(app.middleware_stack, original_stack)
            with TestClient(wrapped) as client:
                self.assertEqual(client.get("/health").status_code, 200)
                # Public client cannot use a trusted loopback control endpoint.
                self.assertEqual(client.post(control.IDENTITY_PATH).status_code, 403)
            self.assertEqual(lifecycle, ["start", "stop"])

    def test_unknown_generation_blocked_while_management_cancel_works(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        app = FastAPI()
        calls = []
        @app.post("/v1/videos")
        async def generate(): calls.append("gpu"); return {"generated": True}
        @app.post("/v1/seedvr2/restore-long/job/cancel")
        async def cancel(): calls.append("cancel"); return {"cancelled": True}
        with tempfile.TemporaryDirectory() as temporary:
            gate = control.ControlGate(root=temporary, epoch="test", token="t" * 64,
                long_video=SimpleNamespace(job_lock=asyncio.Lock(), active_task=None), engine_provider=lambda: None,
                extension_sha256=worker.source_sha256())
            gate.latch.trip(epoch="test", error=RuntimeError("unknown GPU action"))
            with TestClient(control.install(app, gate)) as client:
                self.assertEqual(client.post("/v1/videos").status_code, 409)
                self.assertEqual(client.post("/v1/seedvr2/restore-long/job/cancel").status_code, 200)
            self.assertEqual(calls, ["cancel"])


if __name__ == "__main__": unittest.main()
