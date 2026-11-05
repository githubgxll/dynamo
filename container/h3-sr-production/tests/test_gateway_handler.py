"""Actual handler methods with CPU contexts/transports, no model construction."""
from __future__ import annotations

import asyncio
import ast
import base64
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[3]
if not (REPO / "dingo/vllm/omni/production_handler.py").is_file():
    REPO = Path(importlib.metadata.distribution("ai-dingo").locate_file(""))
if not (REPO / "dingo/vllm/omni/production_handler.py").is_file():
    raise RuntimeError("Gateway checks require the actual installed ai-dingo handler")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
spec = importlib.util.spec_from_file_location("production_handler_under_test", REPO / "dingo/vllm/omni/production_handler.py")
handler_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handler_module)


class Context:
    def __init__(self): self.event = asyncio.Event()
    def id(self): return "job"
    def is_stopped(self): return self.event.is_set()
    def async_killed_or_stopped(self): return asyncio.create_task(self.event.wait())


class HandlerContracts(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "video.mp4"
        self.path.write_bytes(b"actual-local-bytes")
        self.config = SimpleNamespace(request_adapter_workflow="fl2va", model="base", served_model_name="public-fl")
        self.bridge = SimpleNamespace(workflow="fl2va", stages=False, check_ready=mock.AsyncMock(), close=mock.AsyncMock())
        async def execute(request, action, output):
            self.assertEqual(action, "job")
            return await output(self.path, {})
        self.bridge.execute = execute
        self.handler = handler_module.ProductionOmniHandler(None, self.config, bridge=self.bridge)
    async def asyncTearDown(self): self.tmp.cleanup()

    async def test_b64_result_preserves_dingo_video_envelope(self):
        chunks = [chunk async for chunk in self.handler.generate({"response_format": "b64_json"}, Context())]
        self.assertEqual(chunks[0]["status"], "completed")
        self.assertEqual(chunks[0]["model"], "public-fl")
        self.assertEqual(base64.b64decode(chunks[0]["data"][0]["b64_json"]), b"actual-local-bytes")
        self.assertNotIn("stage_durations", chunks[0])

    async def test_binary_writer_context_is_used_instead_of_base64(self):
        writer = SimpleNamespace(write=mock.AsyncMock(return_value={"schema_version": 1, "filename": "worker-video-" + "a" * 32 + ".mp4", "bytes": 18, "sha256": "b" * 64}))
        token = handler_module.BINARY_RESULT_WRITER.set(writer)
        try:
            chunks = [chunk async for chunk in self.handler.generate({"response_format": "b64_json"}, Context())]
        finally:
            handler_module.BINARY_RESULT_WRITER.reset(token)
        self.assertIn("artifact", chunks[0]["data"][0])
        self.assertNotIn("b64_json", chunks[0]["data"][0])
        writer.write.assert_awaited_once_with(b"actual-local-bytes")

    async def test_context_disconnect_waits_for_bridge_cancellation(self):
        entered, cancelled = asyncio.Event(), asyncio.Event()
        async def execute(*args):
            entered.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()
        self.bridge.execute = execute
        context = Context()
        async def run():
            return [chunk async for chunk in self.handler.generate({}, context)]
        task = asyncio.create_task(run())
        await entered.wait()
        context.event.set()
        with self.assertRaises(asyncio.CancelledError): await task
        self.assertTrue(cancelled.is_set())

    async def test_health_never_submits_a_generation(self):
        self.bridge.execute = mock.AsyncMock(side_effect=AssertionError("no generation"))
        chunks = [chunk async for chunk in self.handler.generate(self.handler.health_check_payload(), Context())]
        self.assertEqual(chunks, [{"status": "ready", "native_backends_ready": True}])
        self.bridge.check_ready.assert_awaited_once()
        self.bridge.execute.assert_not_called()

    async def test_workflow_mismatch_rejected(self):
        self.bridge.workflow = "ref2va"
        with self.assertRaises(ValueError):
            handler_module.ProductionOmniHandler(None, self.config, bridge=self.bridge)

    async def test_registration_receipt_has_actual_profile_epoch_pid_and_no_token(self):
        from service import state
        profile = Path(self.tmp.name) / "profile.json"
        profile.write_text('{"production":"fixture"}', encoding="utf-8")
        profile_hash = hashlib.sha256(profile.read_bytes()).hexdigest()
        root = Path(self.tmp.name) / "control"
        root.mkdir()
        state.atomic_json(root / "state.json", {"service_epoch": "svc-test", "profile_sha256": profile_hash})
        self.config.h3_sr_profile = str(profile)
        self.bridge.executor = SimpleNamespace(root=root, epoch="svc-test")
        birth = {"pid": os.getpid(), "process_start_ticks": 1, "pid_namespace": "pid:[test]", "boot_id": "cpu-only"}
        with mock.patch.object(state, "process_birth", return_value=birth):
            receipt = await self.handler.publish_registration()
        self.assertEqual(json.loads((root / "gateway-ready.json").read_text(encoding="utf-8")), receipt)
        self.assertEqual(receipt["profile_sha256"], profile_hash)
        self.assertEqual(receipt["pid"], os.getpid())
        self.assertEqual(receipt["birth"], birth)
        self.assertNotIn("token", json.dumps(receipt))
        with self.assertRaisesRegex(RuntimeError, "must not be replaced"):
            with mock.patch.object(state, "process_birth", return_value=birth):
                await self.handler.publish_registration()

    @unittest.skipUnless(importlib.util.find_spec("prometheus_client"), "Requires actual monitoring dependency")
    async def test_actual_detached_busy_slot_does_not_block_exact_internal_health(self):
        name = "gateway_actual_detached_cpu"
        if name not in sys.modules:
            module_spec = importlib.util.spec_from_file_location(name, REPO / "dingo/vllm/omni/detached_tasks.py")
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[name] = module
            module_spec.loader.exec_module(module)
        module = sys.modules[name]
        manager = module.DetachedOmniTaskManager(self.handler, Path(self.tmp.name) / "tasks", execution_capacity=1)
        entered, release = asyncio.Event(), asyncio.Event()
        async def execute(request, action, output):
            entered.set()
            await release.wait()
            return await output(self.path, {})
        self.bridge.execute = execute
        async def business():
            return [chunk async for chunk in manager.generate({"response_format": "b64_json"}, Context())]
        task = asyncio.create_task(business())
        await entered.wait()
        self.assertEqual(manager._direct_running, 1)
        health = [chunk async for chunk in manager.generate(self.handler.health_check_payload(), Context())]
        self.assertEqual(health, [{"status": "ready", "native_backends_ready": True}])
        self.assertEqual(manager._direct_running, 1)
        with self.assertRaisesRegex(RuntimeError, "capacity exhausted"):
            [chunk async for chunk in manager.generate({**self.handler.health_check_payload(), "prompt": "not-a-health-probe"}, Context())]
        release.set()
        await task
        self.assertEqual(manager._direct_running, 0)


class MainSelection(unittest.IsolatedAsyncioTestCase):
    async def test_opt_in_retains_registration_detached_wrapper_and_cleanup(self):
        source = ast.parse((REPO / "dingo/vllm/omni/main.py").read_text(encoding="utf-8"))
        function = next(n for n in source.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "init_omni")
        namespace = {"__package__": "dingo.vllm.omni", "asyncio": asyncio,
            "DistributedRuntime": object, "OmniConfig": object,
            "shutdown_endpoints": [], "get_fs": lambda value: None,
            "logger": mock.Mock(), "setup_metrics_collection": mock.Mock(),
            "get_output_modalities": lambda *args: "video", "ModelType": SimpleNamespace(Images="image"),
            "ModelInput": SimpleNamespace(Text="text"), "WorkerType": SimpleNamespace(Aggregated="aggregated"),
            "register_model": mock.AsyncMock(),
            "prometheus_names": SimpleNamespace(labels=SimpleNamespace(MODEL="model", MODEL_NAME="model_name")),
            "OmniEngineMonitor": mock.Mock(side_effect=AssertionError("No AsyncOmni monitor for production")),
            "VllmOmniHealthCheckPayload": mock.Mock()}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "actual-main.py", "exec"), namespace)
        handler = SimpleNamespace(check_ready=mock.AsyncMock(), start_monitor=mock.Mock(),
            generate=mock.Mock(), health_check_payload=lambda: {"_h3sr_native_health_v1": True},
            output_formatter=SimpleNamespace(close=mock.Mock()), aclose=mock.AsyncMock(), cleanup=mock.Mock())
        order = []
        namespace["register_model"].side_effect = lambda *args, **kwargs: order.append("register")
        handler.publish_registration = mock.AsyncMock(side_effect=lambda: order.append("receipt"))
        cls = mock.Mock(return_value=handler)
        detached = SimpleNamespace(generate=mock.Mock(), shutdown=mock.AsyncMock())
        manager = mock.Mock(return_value=detached)
        endpoint = SimpleNamespace(serve_endpoint=mock.AsyncMock())
        endpoint.serve_endpoint.side_effect = lambda *args, **kwargs: order.append("serve")
        runtime = SimpleNamespace(endpoint=lambda name: endpoint)
        config = SimpleNamespace(namespace="scope", component="backend", endpoint="generate", media_output_fs_url=None,
            h3_sr_profile="/etc/h3-sr/profile.json", detached_video_task_root="/tasks", media_output_http_url=None,
            detached_video_drain_timeout=1800, detached_video_prefetch_capacity=0,
            diffusion=SimpleNamespace(max_num_seqs=1), engine_args=SimpleNamespace(data_parallel_rank=0, block_size=16),
            model="/models/h3/FL2VA", served_model_name="public-fl", output_modalities=["video"])
        with mock.patch.dict(sys.modules, {
            "dingo.vllm.omni.production_handler": SimpleNamespace(ProductionOmniHandler=cls),
            "dingo.vllm.omni.detached_tasks": SimpleNamespace(DetachedOmniTaskManager=manager)}):
            await namespace["init_omni"](runtime, config, asyncio.Event())
        cls.assert_called_once()
        handler.check_ready.assert_awaited_once()
        manager.assert_called_once()
        self.assertIs(endpoint.serve_endpoint.call_args.args[0], detached.generate)
        self.assertEqual(endpoint.serve_endpoint.call_args.kwargs["health_check_payload"], {"_h3sr_native_health_v1": True})
        namespace["register_model"].assert_awaited_once()
        detached.shutdown.assert_awaited_once()
        handler.aclose.assert_awaited_once()
        handler.cleanup.assert_called_once()
        self.assertEqual(order, ["register", "receipt", "serve"])


if __name__ == "__main__": unittest.main()
