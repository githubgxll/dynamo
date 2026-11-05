"""CPU-only source/contract tests. Fake CUDA events do not establish GPU time."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
spec = importlib.util.spec_from_file_location("_sr_timing_patch_test", ROOT / "sr_timing_patch.py")
patcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patcher)


def source(name):
    return (HERE / "fixtures/sr-timing" / name).read_bytes()


def module(name):
    value = ModuleType("_cpu_sr_" + name.replace(".", "_"))
    value.__file__ = str(HERE / "fixtures/sr-timing" / name)
    relative = patcher.PREFIX + name
    exec(compile(patcher.transform(relative, source(name)), value.__file__, "exec"), value.__dict__)
    return value


class Event:
    def __init__(self, cuda):
        self.cuda = cuda
    def record(self, stream):
        self.cuda.event_records += 1
    def synchronize(self):
        self.cuda.waits += 1
    def query(self):
        return True
    def elapsed_time(self, other):
        return 1.25


class FakeCuda:
    def __init__(self, rank=0, allow_memory=False):
        self.rank, self.allow_memory = rank, allow_memory
        self.memory_calls = []
        self.event_records, self.waits = 0, 0
    def _memory(self, name, value=17):
        self.memory_calls.append(name)
        if not self.allow_memory:
            raise AssertionError("Memory sampling is forbidden in off/timing: " + name)
        return value
    def memory_allocated(self, device):
        return self._memory("allocated", 8)
    def memory_reserved(self, device):
        return self._memory("reserved", 17)
    def max_memory_allocated(self, device):
        return self._memory("max_allocated", 12)
    def max_memory_reserved(self, device):
        return self._memory("max_reserved", 17)
    def reset_peak_memory_stats(self, device):
        self._memory("reset_peak")
    def Event(self, **kwargs):
        return Event(self)
    def current_stream(self, device):
        return "fake-current-stream"
    def get_device_properties(self, device):
        return SimpleNamespace(uuid="GPU-CPU-FIXTURE-" + str(self.rank))


def fake_torch(rank=0, allow_memory=False):
    return SimpleNamespace(cuda=FakeCuda(rank, allow_memory), distributed=SimpleNamespace(
        is_available=lambda: True, is_initialized=lambda: True, get_rank=lambda: rank, get_world_size=lambda: 8))


def process_identity(rank):
    return {"pid": 1000 + rank, "process_start_time": {"clock_domain": "linux_boot_ticks", "value": 2000 + rank}, "nspid": [1000 + rank]}


HASHES = {"sr_telemetry.py": "a" * 64, "pipeline_seedvr2.py": "b" * 64, "ipc.py": "c" * 64}
PHASES = ("prepare_video_gpu", "vae_encode", "latent_prepare", "dit", "solver", "vae_decode", "color_uint8")


class Tensor:
    def __init__(self, device_type="cuda"):
        class Device:
            type = device_type
            def __str__(self):
                return "cuda:0" if self.type == "cuda" else "cpu"
        self.device = Device()
        self.shape = (1, 17, 16, 16, 3)
        self.dtype = "torch.uint8"
    def stride(self):
        return (13056, 768, 48, 3, 1)
    def data_ptr(self):
        return 12345
    def storage_offset(self):
        return 0
    def numel(self):
        return 13056
    def element_size(self):
        return 1
    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda: 12345)


def context(mode="timing"):
    return {"schema": patcher.TIMING_SCHEMA if mode == "timing" else "sr-timing-memory/2",
        "sr_request_id": "1" * 32, "model_request_id": "2" * 32, "measurement_mode": mode,
        "window_index": 0, "window_start": 0, "input_frame_count": 17, "padded_frame_count": 17,
        "new_frame_count": 17, "overlap_frames": 0, "configured_window": 17, "selected_window": 17,
        "model_width": 16, "model_height": 16, "delivery_width": 16, "delivery_height": 16}


class TimingSourceTests(unittest.TestCase):
    def test_all_exact_parent_targets_compile_and_lock_rejects_unknown_bytes(self):
        for relative in patcher.CHANGES:
            original = source(Path(relative).name)
            changed = patcher.transform(relative, original)
            self.assertNotEqual(changed, original)
            compile(changed, relative, "exec")
            with self.assertRaises(ValueError):
                patcher.transform(relative, original + b"\n")
            with self.assertRaises(ValueError):
                patcher.transform(relative, changed)

    def test_default_off_needs_no_context_directory_or_torch(self):
        telemetry, support = module("sr_telemetry.py"), module("long_video_support.py")
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.dict(sys.modules, {"torch": None}):
            config = support.options()
            self.assertEqual(config["measurement_mode"], "off")
            request = SimpleNamespace(sampling_params=SimpleNamespace(extra_args={}))
            self.assertFalse(telemetry.window(request, "cuda:0").enabled)
            self.assertFalse(telemetry.post_process({}, Tensor("cpu")).enabled)
            with telemetry.transfer_scope({}):
                self.assertFalse(telemetry.tensor_transfer(Tensor(), None).enabled)
            metrics = support.ServerMetrics(Path("a" * 32), config, 16, 16, 17, 17)
            self.assertFalse(metrics.enabled)
            self.assertIsNone(metrics.root)

    def test_timing_context_has_own_schema_and_rejects_legacy_schema(self):
        telemetry = module("sr_telemetry.py")
        self.assertEqual(telemetry.validate_context(context())["schema"], patcher.TIMING_SCHEMA)
        with self.assertRaises(ValueError):
            telemetry.validate_context({**context(), "schema": "sr-timing-memory/2"})
        self.assertEqual(telemetry.validate_context(context("basic"))["schema"], "sr-timing-memory/2")

    def test_timing_requires_explicit_metrics_directory(self):
        support = module("long_video_support.py")
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                support.options(measurement_mode="timing")

    def test_timing_window_calls_events_but_never_allocator_or_peak(self):
        telemetry = module("sr_telemetry.py")
        torch = fake_torch()
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, {"SEEDVR2_METRICS_DIR": root}):
            with mock.patch.object(telemetry, "_proc_identity", return_value=process_identity(0)):
                observation = telemetry.Window(context(), "cuda:0", torch_module=torch)
                observation.actual_model_request_id = "actual-window"
                with observation:
                    for phase in PHASES:
                        with observation.stage(phase):
                            pass
            record = json.loads((Path(root) / ("1" * 32) / "windows/0/rank-0.json").read_text())
            self.assertEqual(record["status"], "PASS")
            self.assertEqual(record["schema"], patcher.TIMING_SCHEMA)
            self.assertEqual(record["memory_observation"], "not_collected")
            self.assertEqual(record["explicit_stage_waits"], 0)
            self.assertEqual(record["explicit_window_boundary_waits"], 1)
            self.assertEqual(torch.cuda.memory_calls, [])
            self.assertEqual(torch.cuda.waits, 1)
            self.assertFalse(any(key.startswith("allocator") for key in record))
            self.assertTrue(all(not any(k.startswith("allocator") for k in stage) for stage in record["stages"]))

    def test_historical_basic_and_diagnostic_keep_memory_and_wait_policy(self):
        for mode, waits in (("basic", 1), ("diagnostic", 2)):
            telemetry = module("sr_telemetry.py")
            torch = fake_torch(allow_memory=True)
            with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, {"SEEDVR2_METRICS_DIR": root}):
                with telemetry.Window(context(mode), "cuda:0", torch_module=torch) as obs:
                    with obs.stage("dit"):
                        pass
                record = json.loads((Path(root) / ("1" * 32) / "windows/0/rank-0.json").read_text())
                self.assertEqual(record["schema"], "sr-timing-memory/2")
                self.assertEqual(record["measurement_source"], "torch_allocator")
                self.assertIn("reset_peak", torch.cuda.memory_calls)
                self.assertIn("max_allocated", torch.cuda.memory_calls)
                self.assertEqual(torch.cuda.waits, waits)

    def test_timing_transport_no_memory_and_no_extra_cuda_wait(self):
        telemetry = module("sr_telemetry.py")
        torch, tensor = fake_torch(), Tensor()
        output = {"metadata": {telemetry.KEY: {**context(), "actual_model_request_id": "actual-window"}},
                  "payload": {"video": tensor}}
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, {"SEEDVR2_METRICS_DIR": root}), \
                mock.patch.dict(sys.modules, {"torch": torch}), mock.patch.object(telemetry, "_proc_identity", return_value=process_identity(0)):
            with telemetry.transfer_scope(output):
                with telemetry.tensor_transfer(tensor, None):
                    with telemetry.copy_stage("synchronous_cpu"):
                        pass
            record = json.loads((Path(root) / ("1" * 32) / "windows/0/transfer-rank-0.json").read_text())
            self.assertEqual(record["status"], "PASS")
            self.assertEqual(record["memory_observation"], "not_collected")
            self.assertEqual(record["observed_copy_count"], 1)
            self.assertEqual(torch.cuda.memory_calls, [])
            self.assertEqual(torch.cuda.waits, 0)

    def test_long_video_timing_metadata_uses_versioned_schema_and_default_off(self):
        changed = patcher.transform(patcher.PREFIX + "long_video.py", source("long_video.py")).decode()
        tree = ast.parse(changed)
        # Public endpoint still defaults to off; timing includes native request
        # preprocessing and keeps the original background/job lifecycle.
        self.assertIn('measurement_mode: str = Form("off")', changed)
        self.assertIn('_support.measurement_schema(measurement_mode)', changed)
        self.assertIn('optimization["api_preprocessing_end_ns"]', changed)
        self.assertIn('active_task = asyncio.create_task', changed)
        self.assertTrue(tree.body)


class TimingBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.environment = mock.patch.dict(os.environ, {"SEEDVR2_METRICS_DIR": self.temp.name})
        self.environment.start(); self.addCleanup(self.environment.stop)
        self.telemetry, self.support = module("sr_telemetry.py"), module("long_video_support.py")
        self.job_id = "a" * 32
        self.mode = getattr(self, "_test_mode", "timing")
        config = self.support.options(window_frames=17, measurement_mode=self.mode,
                                      measurement_context=json.dumps({"source_hashes": HASHES}))
        start = time.perf_counter_ns()
        config.update(api_preprocessing_start_ns=start, api_preprocessing_end_ns=start + 1000,
                      server_started_ns=start)
        metrics = self.support.ServerMetrics(Path(self.job_id), config, 16, 16, 17, 17)
        self.item = metrics.window(0, 0, 17)
        actual_id = "actual-cpu-simulation-no-gpu"
        for rank in range(8):
            torch = fake_torch(rank, allow_memory=self.mode != "timing")
            with mock.patch.object(self.telemetry, "_proc_identity", return_value=process_identity(rank)), \
                    mock.patch.object(self.telemetry, "_source_hashes", return_value=HASHES):
                observation = self.telemetry.Window(self.item, "cuda:" + str(rank), torch_module=torch)
                observation.actual_model_request_id = actual_id
                with observation:
                    for phase in PHASES:
                        with observation.stage(phase):
                            pass
            if self.mode == "timing":
                self.assertEqual(torch.cuda.memory_calls, [])
        output = {"metadata": {self.telemetry.KEY: {**self.item, "actual_model_request_id": actual_id}},
                  "payload": {"video": Tensor()}}
        with mock.patch.dict(sys.modules, {"torch": fake_torch(allow_memory=self.mode != "timing")}), \
                mock.patch.object(self.telemetry, "_proc_identity", return_value=process_identity(0)), \
                mock.patch.object(self.telemetry, "_source_hashes", return_value=HASHES):
            with self.telemetry.transfer_scope(output):
                with self.telemetry.tensor_transfer(output["payload"]["video"], None):
                    with self.telemetry.copy_stage("synchronous_cpu"):
                        pass
            with self.telemetry.post_process(output, Tensor("cpu")) as observation:
                with observation.stage("device_to_host"):
                    pass
        for phase in ("source_decode", "window_input_encode", "window_model_http", "window_output_decode",
                      "final_video_encode", "media_validate", "artifact_publish"):
            with metrics.stage(phase, **({"window_index": 0} if phase.startswith("window_") else {})):
                pass
        metrics.finish("completed")
        self.bundle = self.support.metrics_bundle(self.job_id)

    def test_real_source_producers_to_collector_accept_timing_without_memory(self):
        self.assertEqual(self.bundle["metrics_status"], "PASS_COMPLETE", self.bundle["inconsistent_rank_records"])
        self.assertEqual(self.bundle["schema"], patcher.TIMING_SCHEMA)
        self.assertEqual(self.bundle["memory_observation"], "not_collected")
        self.assertEqual(len(self.bundle["files"]["window-map.jsonl"]), 1)

    def check_mutation(self, mutate):
        files = copy.deepcopy(self.bundle["files"])
        mutate(files)
        result = self.support.validate_bundle(self.job_id, files)
        self.assertEqual(result["metrics_status"], "METRICS_INCOMPLETE")

    def test_missing_rank_never_upgrades_to_complete(self):
        self.check_mutation(lambda files: files.pop("windows/0/rank-3.json"))

    def test_allocator_data_is_rejected_in_timing_only_schema(self):
        self.check_mutation(lambda files: files["windows/0/rank-0.json"].update(allocator_before={"allocated_bytes": 0}))

    def test_nonfinite_stage_time_is_rejected(self):
        self.check_mutation(lambda files: files["windows/0/rank-0.json"]["stages"][0].update(cuda_stream_elapsed_ms=float("nan")))

    def test_wrong_window_or_actual_request_is_rejected(self):
        self.check_mutation(lambda files: files["windows/0/rank-4.json"].update(actual_model_request_id="other"))
        self.check_mutation(lambda files: files["window-map.jsonl"][0].update(new_frame_count=16))

    def test_duplicate_worker_identity_and_gpu_is_rejected(self):
        def mutate(files):
            record, other = files["windows/0/rank-4.json"], files["windows/0/rank-0.json"]
            record.update(pid=other["pid"], process_start_time=other["process_start_time"], gpu_uuid=other["gpu_uuid"])
        self.check_mutation(mutate)

    def test_transfer_is_distinct_from_cpu_postprocess(self):
        self.check_mutation(lambda files: files.pop("windows/0/transfer-rank-0.json"))

    def test_source_mismatch_or_extra_receipt_is_rejected(self):
        self.check_mutation(lambda files: files["windows/0/rank-5.json"].update(actual_source_hashes={**HASHES, "ipc.py": "d" * 64}))
        self.check_mutation(lambda files: files.update({"windows/0/rank-0-duplicate.json": {}}))

    def test_historical_validator_is_not_replaced_by_timing_result(self):
        result = self.support.validate_bundle(self.job_id, {"request.json": {"schema": "sr-timing-memory/2", "measurement_mode": "basic"}})
        self.assertEqual(result["schema"], "sr-timing-memory/2")
        self.assertEqual(result["metrics_status"], "METRICS_INCOMPLETE")

    def test_complete_legacy_basic_and_diagnostic_producers_still_validate(self):
        for mode in ("basic", "diagnostic"):
            fixture = TimingBundleTests("test_historical_validator_is_not_replaced_by_timing_result")
            fixture._test_mode = mode
            try:
                fixture.setUp()
                self.assertEqual(fixture.bundle["schema"], "sr-timing-memory/2")
                self.assertEqual(fixture.bundle["metrics_status"], "PASS_COMPLETE",
                                 fixture.bundle["inconsistent_rank_records"])
            finally:
                fixture.doCleanups()

    def test_transfer_and_cpu_sources_cannot_disagree_with_rank_sources(self):
        self.check_mutation(lambda files: files["windows/0/transfer-rank-0.json"].update(actual_source_hashes={**HASHES, "ipc.py": "d" * 64}))
        cpu = next(name for name in self.bundle["files"] if "/postprocess-" in name)
        self.check_mutation(lambda files: files[cpu].update(actual_source_hashes={**HASHES, "ipc.py": "d" * 64}))


if __name__ == "__main__":
    unittest.main()
