"""Run actual patched long-video submit/restore code on a real CPU thread pool."""
from __future__ import annotations

import ast
import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from fractions import Fraction
import importlib.util
import io
import itertools
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from service import control, operation_context, state

RELATIVE = "vllm_omni/diffusion/models/seedvr2/long_video.py"


def load_patch(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reviewed_sources():
    parent = (ROOT / "tests/fixtures/service/long_video.parent.py.txt").read_bytes()
    timing = load_patch("sr_timing_patch").transform(RELATIVE, parent)
    patched = load_patch("window_headers_patch").transform(RELATIVE, timing)
    return timing, patched


class EndAfterHeaders(Exception):
    """Stop after real HTTP-call construction, before any media/model work."""


class RealWindowChain:
    def __init__(self, root, *, gate=None):
        self.root, self.gate = Path(root), gate
        self.observed, self.threads, self.metrics = [], [], []
        self.barrier = threading.Barrier(2)
        self.caller_thread = threading.get_ident()
        owner = self

        class Metrics:
            def __init__(self, *_args):
                self.windows = []
                owner.metrics.append(self)
            def stage(self, *_args, **_kwargs):
                return nullcontext()
            def window(self, index, start, frames):
                self.windows.append({"index": index, "start": start, "frames": frames})
                return {"window_index": index, "measurement_mode": "off"}
            def finish(self, *args):
                self.finished = args

        class Container:
            def __enter__(self): return self
            def __exit__(self, *_args): pass
            def add_stream(self, *_args, **_kwargs):
                return SimpleNamespace(codec_context=SimpleNamespace())

        class Av:
            library_versions = {}
            def open(self, path, *_args, **_kwargs):
                if isinstance(path, io.BytesIO):
                    raise EndAfterHeaders()
                return Container()

        class Response:
            content = b"not-decoded-in-this-transport-test"
            def raise_for_status(self): pass
            def close(self): pass

        def post(_url, **kwargs):
            headers = kwargs["headers"]
            if owner.gate is not None:
                owner.gate.allow_business(headers)  # Missing permit would be 403.
            owner.observed.append(headers)
            owner.threads.append(threading.get_ident())
            # Both actual native windows must be in flight concurrently. A
            # shared/re-entered Context cannot pass this barrier.
            owner.barrier.wait(timeout=3)
            return Response()

        _, output = reviewed_sources()
        tree = ast.parse(output)
        actual = [node for node in tree.body if
            isinstance(node, ast.FunctionDef) and node.name in {"_restore_to", "_restore"} or
            isinstance(node, ast.ImportFrom) and node.module in {"contextvars", "service.operation_context"}]
        # Postpone original media annotations; retain both function bodies,
        # nested restore(), pool.submit(), and actual context-related imports.
        body = ast.parse("from __future__ import annotations\n").body + actual
        self.scope = {"ThreadPoolExecutor": ThreadPoolExecutor, "deque": deque, "itertools": itertools,
            "FPS": 24, "OVERLAP": 4, "Fraction": Fraction, "json": json, "io": io,
            "LOSSLESS": {}, "JobError": RuntimeError, "JobCancelledError": type("Cancelled", (Exception,), {}),
            "requests": SimpleNamespace(post=post, HTTPError=RuntimeError), "av": Av(),
            "_frames": lambda *_args: (index for index in range(23)), "_segment": lambda _frames: b"segment",
            "_support": SimpleNamespace(ServerMetrics=Metrics, NullMetrics=Metrics,
                writer_spec=lambda *_args: {"width": 1280, "height": 768, "options": {}}),
            "_c14": SimpleNamespace(window_context=lambda *_args: {}, long_encoder=lambda *_args: None,
                media=lambda *_args: None, child_options=lambda _context, options: options)}
        exec(compile(ast.Module(body=body, type_ignores=[]), "real-patched-long-video-chain", "exec"), self.scope)

    def run(self):
        self.scope["_restore_to"](self.root / "job", self.root / "scratch", 17, 1280, 768,
            23, False, 666, "lab", 8124, "Bearer original", {"encoding_mode": "legacy"})


class NativeWindowContext(unittest.IsolatedAsyncioTestCase):
    async def test_owned_202_background_to_thread_to_real_native_pool_preserves_both_window_permits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state.atomic_json(root / "active.json", {"service_epoch": "epoch", "action_id": "job", "operation_token": "owned"})
            gate = control.ControlGate(root=root, epoch="epoch", token="t" * 64,
                long_video=SimpleNamespace(job_lock=asyncio.Lock(), active_task=None),
                engine_provider=lambda: None, extension_sha256="a" * 64)
            chain = RealWindowChain(root, gate=gate)
            tasks, messages = [], []
            proceed = asyncio.Event()
            async def background():
                await proceed.wait()
                await asyncio.to_thread(chain.run)
            async def app(_scope, _receive, send):
                tasks.append(asyncio.create_task(background()))
                await send({"type": "http.response.start", "status": 202, "headers": []})
            async def receive(): return {"type": "http.request", "body": b""}
            async def send(message): messages.append(message)
            ingress = control.ControlledIngress(app, gate)
            await ingress({"type": "http", "method": "POST", "path": "/v1/seedvr2/restore-long",
                "headers": [(b"x-h3sr-action-id", b"job"), (b"x-h3sr-operation-token", b"owned")]}, receive, send)
            self.assertEqual(messages[0]["status"], 202)
            self.assertEqual(operation_context.internal_window_headers("A"), {"Authorization": "A"})
            proceed.set()
            with self.assertRaises(EndAfterHeaders):
                await tasks[0]
            expected = {"Authorization": "Bearer original", "x-h3sr-action-id": "job", "x-h3sr-operation-token": "owned"}
            self.assertEqual(chain.observed, [expected, expected])
            self.assertEqual(len(set(chain.threads)), 2)
            self.assertNotIn(chain.caller_thread, chain.threads)
            self.assertEqual([row["frames"] for row in chain.metrics[0].windows], [17, 10])

    async def test_no_operation_context_leaves_original_authorization_only_in_real_pool(self):
        with tempfile.TemporaryDirectory() as temporary:
            chain = RealWindowChain(temporary)
            with self.assertRaises(EndAfterHeaders):
                await asyncio.to_thread(chain.run)
            self.assertEqual(chain.observed, [{"Authorization": "Bearer original"}] * 2)


class WindowSourceLock(unittest.TestCase):
    def test_real_submit_anchor_is_required_exactly_once(self):
        timing, _ = reviewed_sources()
        patch = load_patch("window_headers_patch")
        original = b"return pool.submit(_restore, frames, width, height, seed, color_correction_method, port, authorization, context, measurement, metrics)"
        for data in (timing.replace(original, b"return unreviewed_submit()"), timing + b"\n# " + original):
            with self.subTest(data_sha=__import__("hashlib").sha256(data).hexdigest()):
                with self.assertRaisesRegex(ValueError, "header/context anchor"):
                    patch._apply(data)


if __name__ == "__main__":
    unittest.main()
