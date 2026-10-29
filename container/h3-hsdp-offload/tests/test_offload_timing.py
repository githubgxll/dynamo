"""Pure telemetry/control-flow checks; these deliberately do not emulate CUDA."""
import ast
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "hsdp_migration.py"
API = {"_offload_rank", "_publish_offload_profile", "offload_profile_scope", "timed_offload", "count_offload"}


def telemetry_module(extra=()):
    """Load actual stdlib-only telemetry AST, with no fake Tensor execution."""
    body = []
    for node in ast.parse(SOURCE.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Import):
            if all(not x.name.startswith("torch") for x in node.names):
                body.append(node)
        elif isinstance(node, ast.ImportFrom):
            if not (node.module or "").startswith(("torch", "vllm")):
                body.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(x, ast.Name) and x.id in {"_CONVERSION_LOCK", "_OFFLOAD_PROFILE", "_TIMING_LOGGER"} for x in targets):
                body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in API | set(extra):
            body.append(node)
    namespace = {"__name__": "offload_timing_test", "init_logger": logging.getLogger, "torch": SimpleNamespace(
        distributed=SimpleNamespace(is_available=lambda: True, is_initialized=lambda: False))}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(SOURCE), "exec"), namespace)
    return SimpleNamespace(**namespace), namespace


class OffloadTimingTests(unittest.TestCase):
    def setUp(self):
        self.m, self.ns = telemetry_module()
        self.env = patch.dict(os.environ, {"H3_OFFLOAD_TIMING": "1", "H3_OFFLOAD_FASTPATH": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_disabled_never_reads_clock_generates_uuid_or_publishes(self):
        with patch.dict(os.environ, {"H3_OFFLOAD_TIMING": "0"}), \
                patch.object(self.m.time, "perf_counter", side_effect=AssertionError("clock")), \
                patch.object(self.m.uuid, "uuid4", side_effect=AssertionError("uuid")):
            with self.m.offload_profile_scope("disabled") as value:
                self.assertIsNone(value)
                with self.m.timed_offload("copy"):
                    self.m.count_offload("copies")
        self.assertIsNone(self.m._OFFLOAD_PROFILE.get())

    def test_inclusive_nested_spans_accumulate_and_do_not_add_to_wall(self):
        records = []
        self.ns["_publish_offload_profile"] = records.append
        with patch.object(self.m.time, "perf_counter", side_effect=[1, 2, 3, 5, 8, 10]):
            with self.m.offload_profile_scope("dit"):
                with self.m.timed_offload("outer"):
                    with self.m.timed_offload("inner"):
                        self.m.count_offload("copies", 2)
                        self.m.count_offload("copies")
        record = records[0]
        self.assertEqual(record["wall_seconds"], 9)
        self.assertEqual(record["seconds"], {"inner": 2, "outer": 6})
        self.assertEqual(record["counts"], {"copies": 3, "inner_calls": 1, "outer_calls": 1})
        self.assertIn("not additive", record["timing_scope"])
        self.assertEqual(record["timing_includes"]["native_apply"], ["tensor_to", "pin_memory"])

    def test_nested_profiles_share_outer_record_and_publish_once(self):
        records = []
        self.ns["_publish_offload_profile"] = records.append
        with self.m.offload_profile_scope("outer") as outer:
            with self.m.offload_profile_scope("inner") as inner:
                self.assertIs(outer, inner)
                self.m.count_offload("fastpath_hits", 50)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["label"], "outer")
        self.assertEqual(records[0]["counts"]["fastpath_hits"], 50)

    def test_failure_keeps_original_exception_records_failed_interval_and_resets(self):
        records = []
        self.ns["_publish_offload_profile"] = records.append
        failure = RuntimeError("model failed")
        with self.assertRaises(RuntimeError) as caught:
            with self.m.offload_profile_scope("failed"):
                with self.m.timed_offload("copy"):
                    raise failure
        self.assertIs(caught.exception, failure)
        self.assertEqual(records[0]["status"], "failed")
        self.assertEqual(records[0]["error_type"], "RuntimeError")
        self.assertEqual(records[0]["counts"]["copy_calls"], 1)
        self.assertIsNone(self.m._OFFLOAD_PROFILE.get())

    def test_publish_failure_never_masks_original_error_or_success(self):
        def fail(record):
            raise OSError("disk full")
        self.ns["_publish_offload_profile"] = fail
        with patch.object(self.m._TIMING_LOGGER, "warning") as warning:
            with self.m.offload_profile_scope("succeeded"):
                pass
            self.assertEqual(warning.call_count, 1)
            with self.assertRaisesRegex(ValueError, "original"):
                with self.m.offload_profile_scope("failed"):
                    raise ValueError("original")
            self.assertEqual(warning.call_count, 2)
        self.assertIsNone(self.m._OFFLOAD_PROFILE.get())

    def test_atomic_per_rank_pid_files_and_unique_scopes(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "H3_OFFLOAD_TIMING_DIR": directory, "RANK": "4", "H3_OFFLOAD_FASTPATH": "0"
        }):
            for index in range(2):
                with self.m.offload_profile_scope("request"):
                    self.m.count_offload("index", index)
            files = list(Path(directory).rglob("*.json"))
            self.assertEqual(len(files), 2)
            self.assertFalse(list(Path(directory).rglob("*.tmp")))
            records = [json.loads(p.read_text(encoding="utf-8")) for p in files]
            self.assertEqual(len({r["scope_id"] for r in records}), 2)
            self.assertTrue(all(r["schema"] == "h3-offload-timing/1" and r["rank"] == 4 for r in records))
            self.assertTrue(all(r["pid"] == os.getpid() and r["fastpath_env"] == "0" for r in records))
            self.assertTrue(all(p.parent.name == f"rank-4-pid-{os.getpid()}" for p in files))

    def test_no_directory_logs_one_structured_event(self):
        with patch.dict(os.environ, {"H3_OFFLOAD_TIMING_DIR": ""}), \
                patch.object(self.m._TIMING_LOGGER, "info") as info:
            with self.m.offload_profile_scope("dit"):
                for _ in range(50):
                    self.m.count_offload("forward")
            info.assert_called_once()
            self.assertEqual(json.loads(info.call_args.args[1])["counts"]["forward"], 50)

    def test_thread_contexts_are_isolated(self):
        records = []
        self.ns["_publish_offload_profile"] = records.append
        barrier = threading.Barrier(2)
        def worker(label):
            with self.m.offload_profile_scope(label):
                barrier.wait(timeout=3)
                self.m.count_offload(label)
            self.assertIsNone(self.m._OFFLOAD_PROFILE.get())
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(worker, ["a", "b"]))
        self.assertEqual({r["label"] for r in records}, {"a", "b"})
        self.assertTrue(all(r["counts"] == {r["label"]: 1} for r in records))

    def test_initialized_rank_precedes_environment_without_collectives(self):
        self.ns["torch"] = SimpleNamespace(distributed=SimpleNamespace(
            is_available=lambda: True, is_initialized=lambda: True, get_rank=lambda: 6))
        with patch.dict(os.environ, {"RANK": "2"}):
            self.assertEqual(self.m._offload_rank(), 6)

    def test_local_rank_fallback_and_missing_rank_do_not_claim_rank_zero(self):
        with patch.dict(os.environ, {"RANK": "invalid", "LOCAL_RANK": "3"}):
            self.assertEqual(self.m._offload_rank(), 3)
        with patch.dict(os.environ, {"RANK": "invalid", "LOCAL_RANK": "invalid"}):
            self.assertEqual(self.m._offload_rank(), -1)

    def test_unavailable_telemetry_and_logging_preserve_model_exception(self):
        def fail(record):
            raise OSError("disk unavailable")
        self.ns["_publish_offload_profile"] = fail
        with patch.object(self.m._TIMING_LOGGER, "warning", side_effect=OSError("log unavailable")):
            with self.assertRaisesRegex(ValueError, "model exception"):
                with self.m.offload_profile_scope("failed"):
                    raise ValueError("model exception")

    def test_public_migration_without_outer_scope_still_records_one_event(self):
        m, ns = telemetry_module(extra={"move_component"})
        records = []
        ns["_publish_offload_profile"] = records.append
        ns["torch"].device = lambda target: SimpleNamespace(type=target)
        def move(module, target, **kwargs):
            self.assertFalse(kwargs["allow_mixed"])
            m.count_offload("migration_noops")
            return False
        ns["_move_component"] = move
        self.assertFalse(m.move_component(object(), "cpu"))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["label"], "migration:object")
        self.assertIn("migration_total", records[0]["seconds"])
        self.assertIn("migration_to_cpu", records[0]["seconds"])
        self.assertEqual(records[0]["counts"]["migration_noops"], 1)

    def test_source_instrumentation_does_not_add_cuda_synchronizations(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "synchronize"]
        self.assertEqual(len(calls), 1)
        migration = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_move_component")
        text = ast.unparse(migration)
        for name in ("graph_preflight", "fsdp_preflight", "source_sync", "reshard", "metadata_prepare",
                     "native_apply", "tensor_to", "pin_memory", "post_validate"):
            self.assertIn(f"timed_offload('{name}')", text)
        self.assertIn("non_blocking = False", text)
        self.assertIn("child._apply(convert, recurse=False)", text)


if __name__ == "__main__":
    unittest.main()
