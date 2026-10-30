# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import importlib.util
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


def context(mode="basic"):
    return {"schema": telemetry.SCHEMA, "measurement_mode": mode, "sr_request_id": "request-abc", "model_request_id": "model-def",
        "window_index": 1, "window_start": 13, "input_frame_count": 17, "padded_frame_count": 17,
        "new_frame_count": 13, "overlap_frames": 4, "configured_window": 17, "selected_window": 17,
        "model_width": 1840, "model_height": 1104, "delivery_width": 1800, "delivery_height": 1080,
        "run_id": "run1", "batch_id": "batch1", "case_id": "C08", "action_id": "action1", "phase": "measurement"}


class FakeEvent:
    def __init__(self, cuda):
        self.cuda = cuda

    def record(self, stream):
        self.cuda.streams.append(stream)

    def synchronize(self):
        self.cuda.waits += 1

    def elapsed_time(self, _other):
        return 1.25


class FakeCuda:
    def __init__(self):
        self.waits = 0
        self.resets = 0
        self.streams = []

    def Event(self, enable_timing):
        assert enable_timing
        return FakeEvent(self)

    def current_stream(self, device):
        return "actual-stream-" + str(device)

    def memory_allocated(self, _device):
        return 123

    def memory_reserved(self, _device):
        return 456

    def max_memory_allocated(self, _device):
        return 789

    def max_memory_reserved(self, _device):
        return 987

    def reset_peak_memory_stats(self, _device):
        self.resets += 1

    def get_device_properties(self, _device):
        return SimpleNamespace(uuid="GPU-uuid")


def torch_fake():
    return SimpleNamespace(cuda=FakeCuda(), distributed=SimpleNamespace(is_available=lambda: True, is_initialized=lambda: True,
        get_rank=lambda: 3, get_world_size=lambda: 8))


class TelemetryTests(unittest.TestCase):
    def test_off_does_not_import_torch_or_touch_cuda(self):
        request = SimpleNamespace(sampling_params=SimpleNamespace(extra_args={telemetry.KEY: context("off")}), request_id="actual")
        before = "torch" in sys.modules
        with telemetry.window(request, "cuda:0") as observation:
            with observation.stage("dit"):
                pass
            self.assertIsNone(observation.metadata())
        self.assertEqual("torch" in sys.modules, before)

    def test_basic_one_window_wait_no_stage_sync_and_rank_provenance(self):
        fake = torch_fake()
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"SEEDVR2_METRICS_DIR": root}):
            with telemetry.Window(context(), "cuda:3", fake) as observation:
                self.assertEqual(fake.cuda.waits, 0)
                for name in ("vae_encode", "dit", "vae_decode"):
                    with observation.stage(name):
                        pass
                    self.assertEqual(fake.cuda.waits, 0)
            self.assertEqual(fake.cuda.waits, 1)
            self.assertEqual(fake.cuda.resets, 1)
            record = json.loads((Path(root) / "request-abc/windows/1/rank-3.json").read_text())
        self.assertEqual(record["status"], "PASS")
        self.assertEqual((record["rank"], record["world_size"], record["gpu_uuid"]), (3, 8, "GPU-uuid"))
        self.assertEqual(record["allocator_window_peak"]["allocated_bytes"], 789)
        self.assertEqual(record["explicit_stage_waits"], 0)
        self.assertEqual(record["stages"][1]["cuda_stream_elapsed_ms"], 1.25)
        self.assertTrue(all(item == "actual-stream-cuda:3" for item in fake.cuda.streams))

    def test_diagnostic_explicitly_accounts_stage_wait(self):
        fake = torch_fake()
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"SEEDVR2_METRICS_DIR": root}):
            with telemetry.Window(context("diagnostic"), 0, fake) as observation:
                with observation.stage("dit"):
                    pass
            record = json.loads((Path(root) / "request-abc/windows/1/rank-3.json").read_text())
        self.assertEqual(fake.cuda.waits, 2)
        self.assertEqual(record["explicit_stage_waits"], 1)
        self.assertIsNotNone(record["window_boundary_wait_ms"])

    def test_model_failure_retains_identity_and_does_not_wait_failed_cuda(self):
        fake = torch_fake()
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"SEEDVR2_METRICS_DIR": root}):
            with self.assertRaisesRegex(RuntimeError, "model boom"):
                with telemetry.Window(context(), 0, fake) as observation:
                    with observation.stage("dit"):
                        raise RuntimeError("model boom")
            record = json.loads((Path(root) / "request-abc/windows/1/rank-3.json").read_text())
        self.assertEqual(record["status"], "FAIL")
        self.assertEqual(record["sr_request_id"], "request-abc")
        self.assertEqual(fake.cuda.waits, 0)
        self.assertIsNone(record["stages"][0]["cuda_stream_elapsed_ms"])

    def test_bad_metrics_path_never_turns_generation_into_failure(self):
        fake = torch_fake()
        with patch.dict(os.environ, {"SEEDVR2_METRICS_DIR": "relative-unsafe"}):
            with telemetry.Window(context(), 0, fake) as observation:
                with observation.stage("dit"):
                    result = "generated"
        self.assertEqual(result, "generated")

    def test_event_failure_has_null_missing_not_zero(self):
        fake = torch_fake()
        original = fake.cuda.Event
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"SEEDVR2_METRICS_DIR": root}):
            with telemetry.Window(context(), 0, fake) as observation:
                with patch.object(fake.cuda, "Event", side_effect=RuntimeError("event unavailable")):
                    with observation.stage("dit"):
                        pass
            record = json.loads((Path(root) / "request-abc/windows/1/rank-3.json").read_text())
        self.assertEqual(record["status"], "METRICS_INCOMPLETE")
        self.assertIsNone(record["stages"][0]["cuda_stream_elapsed_ms"])
        self.assertTrue(record["errors"])

    def test_context_validation_and_path_traversal(self):
        for field, value in (("sr_request_id", "../escape"), ("model_request_id", "/escape"), ("window_index", True), ("padded_frame_count", 18), ("input_frame_count", 19)):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                telemetry.validate_context({**context(), field: value})
        self.assertEqual(telemetry.validate_context({**context(), "arbitrary_path": "/etc"}).get("arbitrary_path"), None)

    def test_invalid_context_has_no_receipt_and_cannot_appear_complete(self):
        value = {**context(), "sr_request_id": "../invalid"}
        request = SimpleNamespace(sampling_params=SimpleNamespace(extra_args={telemetry.KEY: value}), request_id="actual")
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"SEEDVR2_METRICS_DIR": root}):
            with telemetry.window(request, "cuda:0") as observation:
                self.assertFalse(observation.enabled)
                with observation.stage("dit"):
                    pass
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_symlink_paths_rejected_when_supported(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "actual"
            target.mkdir()
            link = Path(root) / "link"
            try:
                link.symlink_to(target, target_is_directory=True)
            except OSError:
                self.skipTest("Symlink creation unavailable on this host")
            with self.assertRaises(ValueError):
                telemetry.safe_path(link / "new.json")

    def test_postprocess_filename_distinct_from_rank(self):
        fake = torch_fake()
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"SEEDVR2_METRICS_DIR": root}):
            with telemetry.Window(context(), 0, fake, execution_scope="postprocess_process", filename="postprocess-100.json", reset_allocator_peak=False) as observation:
                with observation.stage("device_to_host"):
                    pass
            record = json.loads((Path(root) / "request-abc/windows/1/postprocess-100.json").read_text())
        self.assertEqual(record["execution_scope"], "postprocess_process")
        self.assertEqual(record["device_to_host"], "executed_here")
        self.assertEqual(fake.cuda.resets, 0)
        self.assertIsNone(record["allocator_window_peak"])
        self.assertEqual(record["allocator_process_peak_observed"]["allocated_bytes"], 789)

    def test_atomic_json_write_no_temporary_residue(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "receipt.json"
            telemetry.atomic_json(path, {"value": 1})
            telemetry.atomic_json(path, {"value": 2})
            self.assertEqual(json.loads(path.read_text())["value"], 2)
            self.assertEqual(list(Path(root).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
