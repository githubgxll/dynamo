"""CPU control contracts with simulated ranks/backends; never GPU acceptance."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from service import control, operation_context, ownership, profile, state, worker_extension as worker

EPOCH = "service-test-epoch"
SHA = worker.source_sha256()
BIRTH = {"pid": 1000, "process_start_ticks": 99, "pid_namespace": "pid:[100]", "boot_id": "boot-test"}


def identities(epoch=EPOCH, sha=SHA):
    return [{"rank": rank, "local_cuda_index": rank, "pid": 1000 + rank,
        "process_start_ticks": 99 + rank, "pid_namespace": "pid:[100]",
        "gpu_uuid": f"GPU-{rank + 1:08x}-0000-0000-0000-000000000000", "world_size": 8,
        "service_epoch": epoch, "extension_sha256": sha, "ok": True} for rank in range(8)]


def identity_report():
    return {"schema": worker.IDENTITY_SCHEMA, "service_epoch": EPOCH, "extension_sha256": SHA, "ranks": identities()}


def release_report(action="job", reason="after-sr-load-before-first-h3"):
    handoff = control.handoff_identifier(EPOCH, action)
    return {"schema": worker.SCHEMA, "service_epoch": EPOCH, "extension_sha256": SHA,
        "handoff_id": handoff, "reason": reason, "metrics": {"memory": False, "timing": False},
        "ranks": [{**row, "handoff_id": handoff, "reason": reason} for row in identities()]}


def ready(root):
    state.atomic_json(Path(root) / "state.json", {"service_ready": True, "profile": "ref2va-sr",
        "service_epoch": EPOCH, "sr_worker_identities": identities()})


class WorkerContracts(unittest.TestCase):
    def test_default_v2_has_no_memory_or_elapsed_and_requires_all_identities(self):
        report = release_report()
        self.assertIs(worker.validate_report(report, report["handoff_id"], report["reason"], identities(), EPOCH, SHA), report)
        self.assertNotIn("before", report["ranks"][0])
        self.assertNotIn("empty_cache_wall_seconds", report["ranks"][0])

    def test_missing_duplicate_rank_or_uuid_rejected(self):
        for mutation in (lambda rows: rows.pop(), lambda rows: rows.__setitem__(1, dict(rows[0])),
                         lambda rows: rows[1].update(gpu_uuid=rows[0]["gpu_uuid"])):
            rows = identities()
            mutation(rows)
            with self.assertRaises(RuntimeError):
                worker.validate_identities(rows, EPOCH, SHA)

    def test_birth_epoch_source_change_rejected(self):
        for field, value in (("service_epoch", "other"), ("extension_sha256", "0" * 64),
                             ("process_start_ticks", 10000)):
            report = release_report()
            report["ranks"][0][field] = value
            with self.assertRaises(RuntimeError):
                worker.validate_report(report, report["handoff_id"], report["reason"], identities(), EPOCH, SHA)

    def test_legacy_schema_is_never_accepted_by_online_v2(self):
        report = release_report()
        report["schema"] = worker.legacy.SCHEMA
        with self.assertRaises(RuntimeError):
            worker.validate_report(report, report["handoff_id"], report["reason"], identities(), EPOCH, SHA)
        self.assertIs(worker.validate_legacy_report, worker.legacy.validate_report)

    def test_disabled_metric_return_is_rejected(self):
        for field, value in (("before", {}), ("empty_cache_wall_seconds", 0.1)):
            report = release_report()
            report["ranks"][0][field] = value
            with self.assertRaises(RuntimeError):
                worker.validate_report(report, report["handoff_id"], report["reason"], identities(), EPOCH, SHA)

    def test_local_failure_still_participates_in_eight_rank_gather(self):
        calls = []
        distributed = SimpleNamespace(is_initialized=lambda: True, get_rank=lambda: 0, get_world_size=lambda: 8)
        def gather(rows, local, group):
            calls.append((local, group))
            rows[:] = identities()
            rows[0] = local
        distributed.all_gather_object = gather
        fake_torch = ModuleType("torch")
        fake_torch.distributed = distributed
        group_module = ModuleType("vllm_omni.diffusion.distributed.parallel_state")
        group_module.get_world_group = lambda: SimpleNamespace(cpu_group="real-selected-group-fixture")
        with mock.patch.dict(sys.modules, {"torch": fake_torch,
                "vllm_omni.diffusion.distributed.parallel_state": group_module}):
            ext = worker.IdleCacheWorkerExtension()
            ext.model_runner = SimpleNamespace(pipeline=object())
            with self.assertRaises(RuntimeError):
                ext.h3sr_worker_identities(service_epoch=EPOCH, extension_sha256=SHA)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0]["ok"], False)

    def test_release_default_never_samples_allocator_or_elapsed(self):
        class Cuda:
            def device(self, device):
                from contextlib import nullcontext
                return nullcontext()
            def current_device(self): return 0
            def get_device_properties(self, device): return SimpleNamespace(uuid=identities()[0]["gpu_uuid"])
            def empty_cache(self): calls.append("empty")
            def __getattr__(self, name): raise AssertionError("Forbidden CUDA sampler: " + name)
        calls = []
        fake_torch = ModuleType("torch")
        def gather(rows, local, group):
            rows[:] = [{**r, "handoff_id": "handoff", "reason": "before-consecutive-h3"} for r in identities()]
            rows[0] = local
        fake_torch.distributed = SimpleNamespace(is_initialized=lambda: True, get_rank=lambda: 0,
            get_world_size=lambda: 8, all_gather_object=gather)
        fake_torch.cuda = Cuda()
        group_module = ModuleType("vllm_omni.diffusion.distributed.parallel_state")
        group_module.get_world_group = lambda: SimpleNamespace(cpu_group="group")
        ext = worker.IdleCacheWorkerExtension()
        ext.model_runner = SimpleNamespace(pipeline=type("SeedVR2Pipeline", (), {})())
        ext.device = SimpleNamespace(type="cuda", index=0)
        with mock.patch.dict(sys.modules, {"torch": fake_torch,
                "vllm_omni.diffusion.distributed.parallel_state": group_module}), \
                mock.patch.object(worker.os, "getpid", return_value=1000), \
                mock.patch.object(worker.legacy, "_birth_identity", return_value={k: BIRTH[k] for k in ("process_start_ticks", "pid_namespace")}), \
                mock.patch.object(worker.time, "perf_counter", side_effect=AssertionError("timing disabled")):
            report = ext.h3sr_release_idle_cache_v2(handoff_id="handoff", reason="before-consecutive-h3",
                service_epoch=EPOCH, extension_sha256=SHA, expected_rank_identities=identities())
        self.assertEqual(calls, ["empty"])
        self.assertNotIn("empty_cache_wall_seconds", report["ranks"][0])


class OwnershipContracts(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        ready(self.root)
    async def asyncTearDown(self): self.temp.cleanup()
    def executor(self, ref=False, handoff=None):
        return ownership.GatewayExecutor(self.root, profile="ref2va-sr" if ref else "fl2va-sr",
            service_epoch=EPOCH, handoff=handoff, ready_check=lambda: True, birth_reader=lambda: BIRTH)
    def terminal(self, action="job"):
        return ownership.TerminalResult(action, True, True, True, payload="video")

    async def test_raw_202_is_unknown_and_keeps_lease_and_blocked(self):
        executor = self.executor()
        with self.assertRaisesRegex(RuntimeError, "TerminalResult"):
            await executor.execute_owned("job", sr_callable=lambda: {"status_code": 202, "id": "sr-job"})
        self.assertTrue((self.root / "active.json").exists())
        self.assertTrue((self.root / "blocked.json").exists())

    async def test_completed_job_without_output_is_blocked(self):
        with self.assertRaises(RuntimeError):
            await self.executor().execute_owned("job", sr_callable=lambda: ownership.TerminalResult("job", True, False, True))
        self.assertTrue((self.root / "blocked.json").exists())

    async def test_lock_remains_held_through_async_sr_and_remux(self):
        executor = self.executor()
        entered, release = asyncio.Event(), asyncio.Event()
        async def sr():
            entered.set()
            await release.wait()
            return self.terminal()
        task = asyncio.create_task(executor.execute_owned("job", h3_callable=self.terminal,
            sr_callable=sr, remux_callable=self.terminal))
        await entered.wait()
        self.assertTrue((self.root / "active.json").exists())
        acquired = asyncio.Event()
        async def second():
            async with ownership.AsyncFileLock(self.root / "operation.lock"): acquired.set()
        waiter = asyncio.create_task(second())
        await asyncio.sleep(.05)
        self.assertFalse(acquired.is_set())
        release.set()
        await task
        await waiter
        self.assertTrue(acquired.is_set())
        self.assertFalse((self.root / "active.json").exists())

    async def test_cancel_persists_block_before_unlock_management_still_available(self):
        executor = self.executor()
        entered = asyncio.Event()
        async def sr():
            entered.set()
            await asyncio.Event().wait()
        task = asyncio.create_task(executor.execute_owned("job", sr_callable=sr))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task
        self.assertTrue((self.root / "blocked.json").exists())
        self.assertTrue((self.root / "active.json").exists())
        self.assertEqual(await executor.management_cancel(lambda: "cancel-observed"), "cancel-observed")
        with self.assertRaises(RuntimeError):
            await executor.execute_owned("new", h3_callable=lambda: self.terminal("new"))

    async def test_consecutive_h3_and_h3_sr_h3_have_real_history_reasons(self):
        reasons = []
        async def handoff(reason, lease):
            reasons.append(reason)
            return release_report(lease["action_id"], reason)
        executor = self.executor(True, handoff)
        await executor.execute_owned("a", h3_callable=lambda: self.terminal("a"))
        await executor.execute_owned("b", h3_callable=lambda: self.terminal("b"), sr_callable=lambda: self.terminal("b"))
        await executor.execute_owned("c", h3_callable=lambda: self.terminal("c"))
        self.assertEqual(reasons, ["after-sr-load-before-first-h3", "before-consecutive-h3", "after-sr-before-next-h3"])

    async def test_fl_never_calls_ref_handoff(self):
        with self.assertRaises(ValueError): self.executor(False, lambda *_: None)
        await self.executor().execute_owned("job", h3_callable=self.terminal)

    async def test_mismatched_terminal_action_id_is_blocked(self):
        with self.assertRaises(RuntimeError):
            await self.executor().execute_owned("job", h3_callable=lambda: self.terminal("other"))

    async def test_success_action_is_not_replayed(self):
        executor = self.executor()
        await executor.execute_owned("job", h3_callable=self.terminal)
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            await executor.execute_owned("job", h3_callable=self.terminal)

    async def test_lost_ready_after_handoff_does_not_start_h3(self):
        availability = [True]
        async def handoff(reason, lease):
            availability[0] = False
            return release_report(lease["action_id"], reason)
        executor = self.executor(True, handoff)
        executor.ready_check = lambda: availability[0]
        h3 = mock.Mock(return_value=self.terminal())
        with self.assertRaisesRegex(RuntimeError, "left Ready"):
            await executor.execute_owned("job", h3_callable=h3)
        h3.assert_not_called()
        self.assertTrue((self.root / "active.json").exists())

    async def test_persistence_error_keeps_memory_latch_and_unresolved_lease(self):
        executor = self.executor()
        with mock.patch.object(state, "atomic_json", side_effect=OSError("disk unavailable")):
            with self.assertRaises(OSError):
                await executor.execute_owned("job", sr_callable=lambda: {"status_code": 202})
        self.assertIsNotNone(executor.latch.blocked)
        self.assertTrue((self.root / "active.json").exists())
        with self.assertRaises(RuntimeError):
            await executor.execute_owned("other", h3_callable=lambda: self.terminal("other"))


class ControlContracts(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        ready(self.root)
        self.calls = []
        self.long_video = SimpleNamespace(job_lock=asyncio.Lock(), active_task=None)
        self.payload = {"service_epoch": EPOCH, "action_id": "job", "operation_token": "owned",
            "handoff_id": control.handoff_identifier(EPOCH, "job"), "reason": "after-sr-load-before-first-h3",
            "expected_rank_identities": identities()}
        state.atomic_json(self.root / "active.json", {"schema": "h3sr-production-operation/1",
            "service_epoch": EPOCH, "action_id": "job", "operation_token": "owned", "controller_birth": BIRTH})
        owner = self
        class Engine:
            async def collective_rpc(self, method, **kwargs):
                owner.calls.append(method)
                if method == "synchronize_device": return [None]
                if method == worker.IDENTITY_METHOD: return [[identity_report()]]
                return [[release_report()]]
        self.engine = Engine()
        self.gate = control.ControlGate(root=self.root, epoch=EPOCH, token="t" * 64, long_video=self.long_video,
            engine_provider=lambda: self.engine, extension_sha256=SHA, birth_reader=lambda pid: BIRTH)
    async def asyncTearDown(self): self.temp.cleanup()

    async def test_http202_active_task_blocks_despite_zero_http_count(self):
        self.long_video.active_task = SimpleNamespace(done=lambda: False)
        with self.assertRaisesRegex(control.Rejected, "HTTP202"):
            await self.gate.release(self.payload)
        self.assertEqual(self.calls, [])
        self.assertTrue((self.root / "blocked.json").exists())

    async def test_release_uses_standard_drain_then_v2_and_keeps_intent(self):
        result = await self.gate.release(self.payload)
        self.assertEqual(self.calls, ["synchronize_device", worker.METHOD])
        self.assertEqual(result["schema"], worker.SCHEMA)
        self.assertEqual(len(list((self.root / "handoffs").glob("*.intent.json"))), 1)

    async def test_duplicate_result_cannot_repeat_empty_cache(self):
        await self.gate.release(self.payload)
        with self.assertRaisesRegex(control.Rejected, "never replay"):
            await self.gate.release(self.payload)
        self.assertEqual(self.calls, ["synchronize_device", worker.METHOD])

    async def test_unknown_intent_never_retries_rpc(self):
        intent = self.root / "handoffs" / (self.payload["handoff_id"] + ".intent.json")
        state.atomic_json(intent, {"status": "STARTED"})
        with self.assertRaises(control.Rejected): await self.gate.release(self.payload)
        self.assertEqual(self.calls, [])

    async def test_lease_change_during_drain_is_blocked_before_empty_cache(self):
        async def rpc(method, **kwargs):
            self.calls.append(method)
            path = self.root / "active.json"
            current = state.read_json(path); current["operation_token"] = "changed"; state.atomic_json(path, current)
            return [None]
        self.engine.collective_rpc = rpc
        with self.assertRaises(control.Rejected): await self.gate.release(self.payload)
        self.assertEqual(self.calls, ["synchronize_device"])
        self.assertTrue((self.root / "blocked.json").exists())

    async def test_identity_birth_change_keeps_old_ready_and_blocks(self):
        async def rpc(*a, **kw):
            report = identity_report(); report["ranks"][0]["process_start_ticks"] += 1; return report
        self.engine.collective_rpc = rpc
        with self.assertRaises(control.Rejected): await self.gate.identities()
        self.assertEqual(state.read_json(self.root / "state.json")["sr_worker_identities"], identities())
        self.assertTrue((self.root / "blocked.json").exists())

    async def test_cancel_during_rpc_latches_before_job_lock_release(self):
        async def rpc(*a, **kw): raise asyncio.CancelledError()
        self.engine.collective_rpc = rpc
        recorded = []
        original = self.gate.latch.trip
        def trip(**kwargs):
            recorded.append(self.long_video.job_lock.locked())
            return original(**kwargs)
        self.gate.latch.trip = trip
        with self.assertRaises(asyncio.CancelledError): await self.gate.release(self.payload)
        self.assertEqual(recorded, [True])
        self.assertTrue((self.root / "blocked.json").exists())

    async def test_unsupported_gpu_routes_rejected_but_cancel_and_status_pass(self):
        seen = []
        async def app(scope, receive, send): seen.append(scope["path"])
        wrapper = control.ControlledIngress(app, self.gate)
        async def receive(): return {"type": "http.request", "body": b""}
        messages = []
        async def send(message): messages.append(message)
        for path in ("/v1/videos", "/v1/images/generations", "/v1/chat/completions"):
            await wrapper({"type": "http", "method": "POST", "path": path, "headers": []}, receive, send)
        self.assertEqual(seen, [])
        self.assertTrue(all(m["status"] == 409 for m in messages if m["type"] == "http.response.start"))
        for method, path in (("POST", "/v1/seedvr2/restore-long/job/cancel"), ("DELETE", "/v1/seedvr2/restore-long/job"), ("GET", "/health")):
            await wrapper({"type": "http", "method": method, "path": path, "headers": []}, receive, send)
        self.assertEqual(len(seen), 3)

    async def test_job_context_survives_202_and_to_thread_only_for_owned_job(self):
        seen = []
        done = asyncio.Event()
        async def background():
            await done.wait()
            seen.append(await asyncio.to_thread(operation_context.internal_window_headers, "Bearer original"))
        tasks = []
        async def app(scope, receive, send):
            tasks.append(asyncio.create_task(background()))
            await send({"type": "http.response.start", "status": 202, "headers": []})
        wrapper = control.ControlledIngress(app, self.gate)
        async def recv(): return {}
        async def send(message): pass
        await wrapper({"type": "http", "method": "POST", "path": "/v1/seedvr2/restore-long",
            "headers": [(b"x-h3sr-action-id", b"job"), (b"x-h3sr-operation-token", b"owned")]}, recv, send)
        self.assertEqual(operation_context.internal_window_headers("A"), {"Authorization": "A"})
        done.set(); await tasks[0]
        self.assertEqual(seen, [{"Authorization": "Bearer original", "x-h3sr-action-id": "job", "x-h3sr-operation-token": "owned"}])

    def test_long_ids_map_to_bounded_deterministic_handoff(self):
        value = control.handoff_identifier("e" * 128, "a" * 128)
        self.assertLess(len(value), 128)
        self.assertEqual(value, control.handoff_identifier("e" * 128, "a" * 128))
        self.assertNotEqual(value, control.handoff_identifier("e" * 128, "b" * 128))


class HeaderSourceChain(unittest.TestCase):
    def test_actual_restore_parent_function_preserves_authorization_and_owned_headers(self):
        """Execute the real patched _restore AST against CPU media/HTTP doubles."""
        import ast
        from contextlib import nullcontext
        def load(name):
            spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
        fixture = ROOT / "tests/fixtures/service/long_video.parent.py.txt"
        before = fixture.read_bytes()
        relative = "vllm_omni/diffusion/models/seedvr2/long_video.py"
        timing = load("sr_timing_patch").transform(relative, before)
        output = load("window_headers_patch").transform(relative, timing)
        tree = ast.parse(output)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_restore")
        fn.returns = None
        for arg in fn.args.args: arg.annotation = None
        observed = []
        class Response:
            content = b"output"
            def raise_for_status(self): pass
            def close(self): pass
        class EndAfterHeaders(Exception): pass
        class Av:
            def open(self, *a, **kw): raise EndAfterHeaders()
        fake_requests = SimpleNamespace(post=lambda *a, **kw: (observed.append(kw["headers"]) or Response()), HTTPError=RuntimeError)
        scope = {"_support": SimpleNamespace(NullMetrics=lambda: SimpleNamespace(stage=lambda *a, **kw: nullcontext())),
            "_segment": lambda frames: b"segment", "_c14": SimpleNamespace(media=lambda *a: None, child_options=lambda c, o: o),
            "LOSSLESS": {}, "json": json, "requests": fake_requests, "io": __import__("io"), "av": Av(),
            "_h3sr_window_headers": operation_context.internal_window_headers, "JobError": RuntimeError}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "actual-parent-restore", "exec"), scope)
        token = operation_context.bind_operation("job", "owned")
        try:
            with self.assertRaises(EndAfterHeaders): scope["_restore"]([], 1280, 768, 1, "lab", 8124, "Bearer kept")
        finally: operation_context.reset_operation(token)
        with self.assertRaises(EndAfterHeaders): scope["_restore"]([], 1280, 768, 1, "lab", 8124, "Bearer kept")
        self.assertEqual(observed, [{"Authorization": "Bearer kept", "x-h3sr-action-id": "job", "x-h3sr-operation-token": "owned"}, {"Authorization": "Bearer kept"}])


if __name__ == "__main__": unittest.main()
