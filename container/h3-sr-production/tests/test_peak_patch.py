"""CPU control-flow tests on exact parent source; no fake GPU acceptance."""
from __future__ import annotations

import argparse
import ast
import asyncio
from contextlib import nullcontext
import copy
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
import time
import unittest
from unittest.mock import Mock
from typing import cast
import __future__

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("production_peak_patch", ROOT / "peak_patch.py")
PATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCH)
FIXTURES = ROOT / "tests/fixtures/peak"


def text(relative, changed=True):
    data = (FIXTURES / relative).read_bytes()
    return (PATCH.transform(relative, data) if changed else data).decode("utf-8")


def execute_nodes(nodes, namespace):
    tree = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]))
    exec(compile(tree, "exact-source-control-fixture", "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace


def functions(relative, names, namespace=None):
    nodes = [node for node in ast.parse(text(relative)).body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError("Actual patched function not found")
    return execute_nodes(nodes, {} if namespace is None else namespace)


RUNNER = "vllm_omni/diffusion/worker/diffusion_model_runner.py"
API = "vllm_omni/entrypoints/openai/api_server.py"
HELPERS = "vllm_omni/entrypoints/openai/video/generation/helpers.py"


def runner_fixture(enabled=True, primary=True):
    platform = NS(reset_peak_memory_stats=Mock(), synchronize=Mock(), is_available=lambda: True)
    distributed = NS(is_initialized=lambda: not primary, get_rank=lambda: 1)
    ns = {
        "torch": NS(no_grad=nullcontext, inference_mode=nullcontext, distributed=distributed),
        "current_omni_platform": platform, "nullcontext": nullcontext,
        "OffloadStrategy": NS(DISTRIBUTED_LAYER_WISE="distributed"),
        "resolve_offload_strategy": lambda config: "none",
        "DiffusionRequestBatch": lambda **kwargs: NS(**kwargs),
        "record_function": lambda *a, **kw: nullcontext(),
        "set_forward_context": lambda **kw: nullcontext(),
        "request_cancellation_scope": lambda *a, **kw: nullcontext(),
        "check_request_cancellation": lambda: None,
        "_normalize_pipeline_outputs": lambda outputs, **kw: outputs,
        "DiffusionRequestAbortedError": type("DiffusionRequestAbortedError", (Exception,), {}),
        "OmniClientError": type("OmniClientError", (Exception,), {}),
        "DiffusionOutput": lambda **kwargs: NS(peak_memory_mb=0.0, **kwargs),
        "RunnerOutput": lambda **kwargs: NS(**kwargs),
        "BatchRunnerOutput": NS(from_list=lambda values: NS(runner_outputs=values)),
        "logger": NS(debug=Mock(), info=Mock(), error=Mock()),
        "validate_new_request_data_identity": lambda value: None,
        "clear_pipeline_stage_durations": lambda pipeline: None,
        "consume_pipeline_stage_durations": lambda pipeline: {},
        "merge_stage_durations": lambda *a: None,
        "DiffusionKVCacheMode": NS(DENSE_LEGACY="dense", PAGED_SCHEDULER="paged"),
        "DiffusionSchedulerOutput": lambda **kw: NS(**kw),
        "NewRequestData": lambda **kw: NS(**kw),
        "CachedRequestData": NS(make_empty=lambda: NS()),
        "gc": NS(collect=Mock()),
    }
    wanted = {"_execute_request_list", "execute_model", "execute_model_batch", "execute_stepwise",
              "_execute_stepwise_core", "profile_run"}
    tree = ast.parse(text(RUNNER))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DiffusionModelRunner")
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_should_record_output_peak_memory")
    fixture_cls = ast.ClassDef(name="RunnerFixture", bases=[], keywords=[], body=methods, decorator_list=[])
    execute_nodes([helper, fixture_cls], ns)
    runner = ns["RunnerFixture"]()
    config = NS(parallel_config=NS(use_hsdp=False), cache_backend="none", enable_cache_dit_summary=False,
                streaming_output=False, record_output_peak_memory=enabled)
    runner.od_config = config
    runner.pipeline = NS(supports_request_batch=True, forward=lambda batch: [NS(peak_memory_mb=0.0) for _ in batch.requests])
    runner.vllm_config = object()
    runner.cache_backend = None
    runner._prepare_request_for_forward = lambda *a, **kw: None
    runner._refresh_cache_for_requests = lambda *a, **kw: None
    runner._prepare_output_for_transport = lambda output, sampling: output
    runner._maybe_send_stage_payload = lambda *a: None
    runner._runner_output_from_outputs = lambda reqs, outputs: NS(runner_outputs=[NS(result=o) for o in outputs])
    runner._sample_peak_memory_mb = Mock(return_value=123.5)
    runner._validate_diffusion_kv_metadata = lambda **kw: None
    runner.remove_diffusion_kv_requests = lambda *a: None
    runner.state_cache = {}
    runner.input_batch = None
    req = NS(prompt="fixture", request_id="request-1", sampling_params=NS())
    return runner, config, platform, req, ns


class ExactSourceLock(unittest.TestCase):
    def test_all_targets_match_hash_lock_and_compile(self):
        lock = PATCH.policy()
        self.assertEqual(len(PATCH.CHANGES), 9)
        for relative in PATCH.CHANGES:
            with self.subTest(relative=relative):
                before = (FIXTURES / relative).read_bytes()
                after = PATCH.transform(relative, before)
                self.assertEqual(hashlib.sha256(before).hexdigest(), lock["changes"][relative]["before_sha256"])
                self.assertEqual(hashlib.sha256(after).hexdigest(), lock["changes"][relative]["after_sha256"])
                compile(after, relative, "exec")

    def test_unreviewed_and_repeated_inputs_rejected(self):
        before = (FIXTURES / RUNNER).read_bytes()
        for relative, data in [("unknown.py", before), (RUNNER, before + b"\n"), (RUNNER, PATCH.transform(RUNNER, before))]:
            with self.subTest(relative=relative):
                with self.assertRaises(ValueError):
                    PATCH.transform(relative, data)

    def test_capacity_profile_body_unchanged(self):
        def body(source):
            cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "DiffusionModelRunner")
            return ast.dump(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "profile_run"), include_attributes=False)
        self.assertEqual(body(text(RUNNER, False)), body(text(RUNNER)))


class RequestObservation(unittest.TestCase):
    def test_disabled_single_request_has_no_memory_queries(self):
        runner, config, platform, req, ns = runner_fixture(False)
        output = runner.execute_model(req)
        self.assertEqual(output.peak_memory_mb, -1.0)
        platform.reset_peak_memory_stats.assert_not_called()
        runner._sample_peak_memory_mb.assert_not_called()

    def test_disabled_batch_has_no_queries_for_all_results(self):
        runner, config, platform, req, ns = runner_fixture(False)
        schedule = NS(scheduled_new_reqs=[NS(req=req, request_id=req.request_id, diffusion_kv_metadata=None),
                                          NS(req=req, request_id=req.request_id, diffusion_kv_metadata=None)])
        output = runner.execute_model_batch(schedule, config)
        self.assertEqual([o.result.peak_memory_mb for o in output.runner_outputs], [-1.0, -1.0])
        platform.reset_peak_memory_stats.assert_not_called()
        runner._sample_peak_memory_mb.assert_not_called()

    def test_compatibility_default_enabled_records_one_batch_peak(self):
        runner, config, platform, req, ns = runner_fixture(True)
        del config.record_output_peak_memory
        output = runner.execute_model(req)
        self.assertEqual(output.peak_memory_mb, 123.5)
        platform.reset_peak_memory_stats.assert_called_once_with()
        runner._sample_peak_memory_mb.assert_called_once_with()

    def test_enabled_secondary_rank_never_samples_primary_peak(self):
        runner, config, platform, req, ns = runner_fixture(True, primary=False)
        runner.execute_model(req)
        platform.reset_peak_memory_stats.assert_not_called()
        runner._sample_peak_memory_mb.assert_not_called()

    def test_explicit_profile_false_overrides_enabled_serving_policy(self):
        runner, config, platform, req, ns = runner_fixture(True)
        runner._execute_request_list([req], od_config=config, allow_single_output=True,
            require_request_batch_support=False, record_name="capacity-fixture", record_output_peak_memory=False)
        platform.reset_peak_memory_stats.assert_not_called()
        runner._sample_peak_memory_mb.assert_not_called()

    def test_public_step_entry_resolves_configuration(self):
        for enabled in (False, True):
            runner, config, platform, req, ns = runner_fixture(enabled)
            runner._execute_stepwise = Mock(return_value="forwarded")
            self.assertEqual(runner.execute_stepwise(NS()), "forwarded")
            self.assertIs(runner._execute_stepwise.call_args.kwargs["record_output_peak_memory"], enabled)

    def test_actual_step_core_disables_both_reset_and_read(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                runner, config, platform, req, ns = runner_fixture(enabled)
                state = NS(request_id="step-1", step_index=0, peak_memory_mb=0.0)
                runner.pipeline = NS(denoise_step=lambda batch, states: None, interrupt=True)
                runner._update_states = lambda schedule: ([state], [state.request_id])
                runner._prepare_batch_inputs = lambda states, ids: (states, object(), [])
                runner._update_states_after = lambda *a: None
                result = runner._execute_stepwise_core(NS(), record_output_peak_memory=True)
                self.assertEqual(result.runner_outputs[0].result.peak_memory_mb, 123.5 if enabled else -1.0)
                self.assertEqual(platform.reset_peak_memory_stats.call_count, int(enabled))
                self.assertEqual(runner._sample_peak_memory_mb.call_count, int(enabled))

    def test_profile_run_keeps_own_capacity_context_in_both_modes(self):
        for enabled in (True, False):
            for step in (True, False):
                with self.subTest(enabled=enabled, step=step):
                    runner, config, platform, req, ns = runner_fixture(enabled)
                    config.step_execution = step
                    runner._execute_stepwise = Mock(return_value=NS())
                    runner._execute_request_list = Mock(return_value=NS())
                    runner.profile_run([req])
                    called = runner._execute_stepwise if step else runner._execute_request_list
                    self.assertIs(called.call_args.kwargs["record_output_peak_memory"], False)
                    self.assertIs(called.call_args.kwargs["in_diffusion_kv_memory_profile"], True)
                    platform.reset_peak_memory_stats.assert_not_called()
                    platform.synchronize.assert_called_once_with()

    def test_boolean_policy_is_not_truthy_string_or_integer(self):
        runner, config, platform, req, ns = runner_fixture(False)
        for value in ("false", 0, 1, None):
            config.record_output_peak_memory = value
            with self.assertRaises(TypeError):
                runner.execute_model(req)


class ConfigProjection(unittest.TestCase):
    def test_native_cli_boolean_optional_default_and_disable(self):
        relative = "vllm_omni/entrypoints/cli/serve.py"
        call = next(n for n in ast.walk(ast.parse(text(relative))) if isinstance(n, ast.Call)
                    and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "--record-output-peak-memory")
        parser = argparse.ArgumentParser()
        execute_nodes([ast.Expr(value=call)], {"omni_config_group": parser, "argparse": argparse})
        self.assertIs(parser.parse_args([]).record_output_peak_memory, True)
        self.assertIs(parser.parse_args(["--no-record-output-peak-memory"]).record_output_peak_memory, False)
        self.assertIs(parser.parse_args(["--record-output-peak-memory"]).record_output_peak_memory, True)

    def test_field_survives_all_dataclass_projection_surfaces(self):
        expected = {PATCH.CHANGES[0]: [True], PATCH.CHANGES[1]: [True], PATCH.CHANGES[2]: [None],
                    PATCH.CHANGES[3]: [None, True]}
        for relative, defaults in expected.items():
            tree = ast.parse(text(relative))
            values = [n.value.value for n in ast.walk(tree) if isinstance(n, ast.AnnAssign)
                      and isinstance(n.target, ast.Name) and n.target.id == "record_output_peak_memory"]
            self.assertEqual(values, defaults, relative)
        self.assertIn('"record_output_peak_memory",', text(PATCH.CHANGES[3]))


class PublicAvailability(unittest.TestCase):
    def setUp(self):
        self.api = functions(API, ["_peak_memory_observation", "_output_peak_memory_headers", "_build_image_response_metrics"], {"math": math})
        self.helpers = functions(HELPERS, ["_video_peak_memory_observation", "_unpack_video_generation_result"],
                                 {"math": math, "cast": cast, "VideoAction": object})
        self.protocol = functions(PATCH.CHANGES[7], ["_protocol_peak_memory_observation"], {"math": math})

    def test_sentinel_and_legacy_zero_do_not_become_measurements(self):
        values = [(-1.0, None, "not_collected"), (None, None, "not_collected"),
                  (0.0, None, "unavailable"), (123.5, 123.5, "collected"),
                  (float("nan"), None, "unavailable"), (float("inf"), None, "unavailable"),
                  (True, None, "unavailable"), ("bad", None, "unavailable")]
        for value, peak, status in values:
            with self.subTest(value=value):
                expected = {"peak_memory_mb": peak, "peak_memory_status": status}
                for fn in (self.api["_peak_memory_observation"], self.helpers["_video_peak_memory_observation"],
                           self.protocol["_protocol_peak_memory_observation"]):
                    self.assertEqual(fn(value), expected)
                headers = self.api["_output_peak_memory_headers"](value)
                self.assertEqual(headers["X-Peak-Memory-Status"], status)
                self.assertEqual("X-Peak-Memory-MB" in headers, peak is not None)

    def test_image_metrics_null_and_status_keep_other_fields(self):
        result = self.api["_build_image_response_metrics"](response_metrics={"request_id": "keep"},
            stage_durations={}, peak_memory_mb=-1.0)
        self.assertEqual(result, {"request_id": "keep", "stage_durations": None,
                                 "peak_memory_mb": None, "peak_memory_status": "not_collected"})

    def test_internal_float_survives_video_tuple_unpack(self):
        result = self.helpers["_unpack_video_generation_result"]((b"video", {}, -1.0, None, {}))
        self.assertEqual(result[2], -1.0)
        self.assertNotIn("X-Peak-Memory-MB", self.api["_output_peak_memory_headers"](result[2]))

    def test_video_models_accept_null_and_normalize_internal_float(self):
        tree = ast.parse(text(PATCH.CHANGES[7]))
        mixin = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_PeakMemoryAvailability")
        method = copy.deepcopy(next(n for n in mixin.body if isinstance(n, ast.FunctionDef)))
        method.decorator_list = []
        ns = execute_nodes([method], self.protocol)
        result = ns["_normalize_peak_memory"](None, {"model": "m", "peak_memory_mb": -1.0})
        self.assertIsNone(result["peak_memory_mb"])
        self.assertEqual(result["peak_memory_status"], "not_collected")
        for name in ("VideoGenerationResponse", "VideoResponse"):
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
            self.assertEqual(cls.bases[0].id, "_PeakMemoryAvailability")
            peak = next(n for n in cls.body if isinstance(n, ast.AnnAssign) and n.target.id == "peak_memory_mb")
            self.assertEqual(ast.unparse(peak.annotation), "float | None")

    def test_actual_async_job_persists_null_availability(self):
        records = []
        async def get(video_id):
            return NS(file_extension="mp4")
        async def update(video_id, value):
            records.append(value)
        async def save_video(content, video_id):
            return NS(key="video-key", created_at=1, expires_at=None)
        async def generate(*a, **kw):
            return b"video", {}, -1.0, None, {}
        ns = {"time": time, "asyncio": asyncio, "math": math, "cast": cast, "VideoAction": object,
              "VIDEO_STORE": NS(get=get, update_fields=update), "STORAGE_MANAGER": NS(save=save_video),
              "logger": NS(info=Mock(), warning=Mock(), exception=Mock()),
              "VideoGenerationStatus": NS(COMPLETED="completed", IN_PROGRESS="running", FAILED="failed"),
              "EngineGenerateError": type("EngineGenerateError", (Exception,), {}),
              "EngineDeadError": type("EngineDeadError", (Exception,), {}),
              "_cleanup_video_references": lambda *a: None, "_cleanup_timeline_guides": lambda *a: None}
        functions(HELPERS, ["_video_peak_memory_observation", "_unpack_video_generation_result", "_run_video_generation_job"], ns)
        asyncio.run(ns["_run_video_generation_job"](NS(generate_video_bytes=generate), NS(), "job-1"))
        self.assertEqual(records[-1]["status"], "completed")
        self.assertIsNone(records[-1]["peak_memory_mb"])
        self.assertEqual(records[-1]["peak_memory_status"], "not_collected")

    def test_stored_null_availability_survives_get_reconstruction(self):
        tree = ast.parse(text(PATCH.CHANGES[7]))
        mixin = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_PeakMemoryAvailability")
        method = copy.deepcopy(next(n for n in mixin.body if isinstance(n, ast.FunctionDef)))
        method.decorator_list = []
        normalize = execute_nodes([method], self.protocol)["_normalize_peak_memory"]
        for status in ("unavailable", "not_collected"):
            stored = {"model": "m", "peak_memory_mb": None, "peak_memory_status": status}
            first = normalize(None, stored)
            rebuilt = normalize(None, json.loads(json.dumps(first)))
            self.assertEqual(rebuilt, stored)
        invalid = normalize(None, {"model": "m", "peak_memory_mb": None, "peak_memory_status": "collected"})
        self.assertNotEqual(invalid["peak_memory_status"], "collected")


if __name__ == "__main__":
    unittest.main()
