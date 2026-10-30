# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import telemetry
import transfer_patch
from test_telemetry_repair import context, Device
from test_transfer_patch import original


class Stream:
    def __init__(self, name, trace, *, records_ready=False):
        self.name, self.trace, self.records_ready = name, trace, records_ready
        self.events = []

    def synchronize(self):
        self.trace.append(("original_stream_sync", self.name))
        for event in self.events:
            event.ready = True

    def __str__(self):
        return "stream:" + self.name


class Event:
    def __init__(self, cuda):
        self.cuda = cuda
        self.ready = False

    def record(self, stream):
        self.stream = stream
        stream.events.append(self)
        self.ready = stream.records_ready
        self.cuda.trace.append(("event", stream.name))

    def synchronize(self):
        raise AssertionError("Transport inserted an extra CUDA wait")

    def query(self):
        return self.ready

    def elapsed_time(self, other):
        if not self.ready or not other.ready:
            raise AssertionError("Events resolved before original synchronization")
        return 2.5


class Cuda:
    def __init__(self, trace):
        self.trace = trace
        self.torch = None
        self.event_count = 0

    def check(self, device):
        if device.type != "cuda":
            raise AssertionError("Fake CUDA strictly rejects a CPU device")

    def get_device_properties(self, device):
        self.check(device)
        return SimpleNamespace(uuid="GPU-real-device-fixture")

    def memory_allocated(self, device):
        self.check(device)
        return 123

    def memory_reserved(self, device):
        self.check(device)
        return 456

    def max_memory_allocated(self, device):
        self.check(device)
        return 789

    def max_memory_reserved(self, device):
        self.check(device)
        return 987

    def reset_peak_memory_stats(self, device):
        raise AssertionError("Transfer must not reset model peaks")

    def current_stream(self, device):
        self.check(device)
        return self.torch.stream

    def Event(self, enable_timing):
        assert enable_timing
        self.event_count += 1
        return Event(self)


class Tensor:
    dtype = "torch.uint8"
    shape = (2, 3)

    def __init__(self, torch, values=None, device="cuda:0"):
        self.torch, self.values, self.device = torch, values or list(range(6)), Device(device)

    def detach(self):
        return self

    def cpu(self):
        self.torch.trace.append(("original_cpu",))
        return Host(self.torch, self.values[:])

    def contiguous(self):
        return self

    def numpy(self):
        self.torch.trace.append(("numpy",))
        return self.values[:]

    def stride(self):
        return (3, 1)

    def data_ptr(self):
        return 10000

    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda: 10000)

    def storage_offset(self):
        return 0

    def numel(self):
        return 6

    def element_size(self):
        return 1


class Host(Tensor):
    def __init__(self, torch, values=None):
        super().__init__(torch, values, device="cpu")

    def copy_(self, source, non_blocking):
        self.torch.trace.append(("original_copy", self.torch.stream.name))
        if self.torch.fail_copy:
            raise RuntimeError("actual copy failure")
        assert non_blocking
        self.values = source.values[:]
        return self


class FakeTorch:
    bfloat16 = "torch.bfloat16"
    version = SimpleNamespace(cuda="13.0")

    def __init__(self, *, sync_ready=False, fail_copy=False):
        self.trace = []
        self.cuda = Cuda(self.trace)
        self.cuda.torch = self
        self.stream = Stream("default", self.trace, records_ready=sync_ready)
        self.fail_copy = fail_copy
        self.distributed = SimpleNamespace(is_available=lambda: True, is_initialized=lambda: True, get_rank=lambda: 0, get_world_size=lambda: 8)
        self.accelerator = SimpleNamespace(current_stream=lambda: self.stream, set_stream=self.set_stream)

    def set_stream(self, stream):
        self.trace.append(("original_set_stream", stream.name))
        self.stream = stream

    def empty(self, shape, dtype, pin_memory):
        assert pin_memory
        self.trace.append(("original_pinned_alloc",))
        return Host(self)


def tensor_pack(torch):
    node = next(node for node in ast.parse(transfer_patch.transform(original())).body
        if isinstance(node, ast.FunctionDef) and node.name == "_tensor_to_shm")
    scope = {"torch": torch, "_sr_metrics": telemetry, "_array_to_shm": lambda values: {"values": list(values)},
             "_cuda_registered_shm_copy": lambda tensor, stream: {"registered": True}}
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "actual-patched-ipc-tensor-function", "exec"), scope)
    return scope["_tensor_to_shm"]


class TransferRuntimeTests(unittest.TestCase):
    def run_pack(self, torch, *, measured=True, stream=True, identity_tensor=None):
        video = Tensor(torch)
        output = {"payload": {"video": identity_tensor or video}, "metadata": {telemetry.KEY: context()} if measured else {}}
        selected_stream = Stream("D2H", torch.trace) if stream else None
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"SEEDVR2_METRICS_DIR": root}), patch.dict(sys.modules, {"torch": torch}):
            with telemetry.transfer_scope(output):
                created = []
                result = tensor_pack(torch)(video, d2h_stream=selected_stream, created=created)
            records = [json.loads(path.read_text()) for path in Path(root).glob("request-abc/windows/1/transfer*.json")]
        return result, records, created

    def test_actual_pinned_copy_call_order_values_and_existing_wait_unchanged(self):
        torch = FakeTorch()
        result, records, created = self.run_pack(torch)
        self.assertEqual(result["values"], list(range(6)))
        self.assertEqual(created, [result])
        self.assertEqual([item[0] for item in torch.trace if not item[0] == "event"],
            ["original_set_stream", "original_pinned_alloc", "original_copy", "original_set_stream", "original_stream_sync", "numpy"])
        self.assertEqual([item for item in torch.trace if item[0] == "event"], [("event", "D2H"), ("event", "D2H")])
        record = records[0]
        self.assertEqual(record["status"], "PASS")
        self.assertEqual(record["observed_copy_count"], 1)
        self.assertTrue(record["matched_payload_video"])
        self.assertEqual(record["video_payload_path"], "payload.video")
        self.assertEqual(record["tensor_identity"]["shape"], [2, 3])
        self.assertEqual(record["transport_mode"], "async_side_stream")
        self.assertEqual(record["stages"][0]["phase"], "device_to_host_transport")
        self.assertEqual(record["stages"][0]["cuda_stream_elapsed_ms"], 2.5)
        self.assertEqual(record["stages"][0]["event_scope"], "passed_d2h_stream")
        self.assertEqual(record["framework_output_pack"]["phase"], "framework_output_pack")
        self.assertEqual(record["explicit_stage_waits"], 0)
        self.assertEqual((record["rank"], record["world_size"]), (0, 8))
        self.assertIsNone(record["allocator_window_peak"])

    def test_non_seed_output_has_no_cuda_observations(self):
        torch = FakeTorch()
        result, records, _ = self.run_pack(torch, measured=False)
        self.assertEqual(result["values"], list(range(6)))
        self.assertEqual(torch.cuda.event_count, 0)
        self.assertEqual(records, [])

    def test_other_tensor_not_matching_video_is_not_observed(self):
        torch = FakeTorch()
        result, records, _ = self.run_pack(torch, identity_tensor=Tensor(torch))
        self.assertEqual(result["values"], list(range(6)))
        self.assertEqual(records, [])

    def test_sync_copy_uses_original_expression_without_extra_wait(self):
        torch = FakeTorch(sync_ready=True)
        result, records, _ = self.run_pack(torch, stream=False)
        self.assertEqual(result["values"], list(range(6)))
        self.assertEqual(records[0]["status"], "PASS")
        self.assertEqual(records[0]["transport_mode"], "sync_current_stream")
        self.assertEqual([item for item in torch.trace if item[0] == "original_cpu"], [("original_cpu",)])
        self.assertFalse(any(item[0] == "original_stream_sync" for item in torch.trace))

    def test_unresolved_sync_event_is_missing_not_hidden_wait_or_zero(self):
        torch = FakeTorch()
        result, records, _ = self.run_pack(torch, stream=False)
        self.assertEqual(result["values"], list(range(6)))
        self.assertEqual(records[0]["status"], "METRICS_INCOMPLETE")
        self.assertIsNone(records[0]["stages"][0]["cuda_stream_elapsed_ms"])

    def test_original_copy_error_propagates_and_context_resets(self):
        torch = FakeTorch(fail_copy=True)
        video = Tensor(torch)
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"SEEDVR2_METRICS_DIR": root}), patch.dict(sys.modules, {"torch": torch}):
            with self.assertRaisesRegex(RuntimeError, "actual copy failure"):
                with telemetry.transfer_scope({"payload": {"video": video}, "metadata": {telemetry.KEY: context()}}):
                    tensor_pack(torch)(video, d2h_stream=Stream("D2H", torch.trace))
            record = json.loads(next(Path(root).glob("request-abc/windows/1/transfer*.json")).read_text())
        self.assertEqual(record["status"], "FAIL")
        self.assertEqual(torch.stream.name, "default")
        self.assertIsNone(telemetry._TRANSFER_SCOPE.get())
        self.assertIsNone(telemetry._ACTIVE_TRANSFER.get())


if __name__ == "__main__":
    unittest.main()
