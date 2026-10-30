# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
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
        "actual_model_request_id": "actual-worker-request", "window_index": 1, "window_start": 13, "input_frame_count": 17, "padded_frame_count": 17,
        "new_frame_count": 13, "overlap_frames": 4, "configured_window": 17, "selected_window": 17,
        "model_width": 1840, "model_height": 1104, "delivery_width": 1800, "delivery_height": 1080,
        "run_id": "run1", "batch_id": "batch1", "case_id": "C08", "action_id": "action1", "phase": "measurement"}


class Device:
    def __init__(self, name):
        self.type = name.split(":")[0]
        self.name = name

    def __str__(self):
        return self.name


class CudaForbidden:
    def __getattr__(self, name):
        raise AssertionError("CPU path touched CUDA API: " + name)


class CpuTensor:
    device = Device("cpu")

    def __init__(self, values):
        self.values = values
        self.copy_calls = 0
        self.cpu_calls = 0
        self.numpy_calls = 0

    def cpu(self):
        self.cpu_calls += 1
        return self

    def numpy(self):
        self.numpy_calls += 1
        return self.values


class CpuPostprocessTests(unittest.TestCase):
    def test_cpu_does_not_touch_cuda_and_preserves_actual_expression(self):
        video = CpuTensor([1, 4, 8])
        output = {"payload": {"video": video}, "metadata": {telemetry.KEY: context()}}
        forbidden = SimpleNamespace(cuda=CudaForbidden())
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"SEEDVR2_METRICS_DIR": root}), patch.dict(sys.modules, {"torch": forbidden}):
            with telemetry.post_process(output, video) as observation:
                with observation.stage("device_to_host"):
                    actual = {"payload": {**output["payload"], "video": video.cpu().numpy()}, "metadata": output["metadata"]}
            files = list((Path(root) / "request-abc/windows/1").glob("postprocess-*.json"))
            record = json.loads(files[0].read_text())
        self.assertIs(actual["payload"]["video"], video.values)
        self.assertEqual((video.cpu_calls, video.numpy_calls), (1, 1))
        self.assertEqual(record["schema"], "sr-timing-memory/2")
        self.assertEqual(record["status"], "PASS")
        self.assertEqual(record["execution_scope"], "api_cpu_postprocess")
        self.assertEqual(record["stages"][0]["phase"], "cpu_video_numpy")
        self.assertEqual(record["local_d2h"], "not_executed_here")
        self.assertEqual(record["upstream_transfer_receipt"], "transfer-rank-0.json")
        for key in ("rank", "world_size", "gpu_uuid", "allocator_before", "allocator_after", "allocator_window_peak", "cuda_stream_elapsed_ms"):
            self.assertIsNone(record[key], key)
        self.assertEqual(record["cuda_status"], "not_applicable")
        self.assertEqual(record["allocator_status"], "not_applicable")

    def test_cpu_postprocess_failure_retains_original_error(self):
        video = CpuTensor([2])
        output = {"metadata": {telemetry.KEY: context()}}
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"SEEDVR2_METRICS_DIR": root}):
            with self.assertRaisesRegex(RuntimeError, "numpy failed"):
                with telemetry.post_process(output, video) as observation:
                    with observation.stage("device_to_host"):
                        raise RuntimeError("numpy failed")
            record = json.loads(next((Path(root) / "request-abc/windows/1").glob("postprocess-*.json")).read_text())
        self.assertEqual(record["status"], "FAIL")
        self.assertEqual(record["stages"][0]["failure"], "RuntimeError")

    def test_repeated_cpu_producer_is_visible_not_overwritten(self):
        video = CpuTensor([2])
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"SEEDVR2_METRICS_DIR": root}):
            for _ in range(2):
                with telemetry.post_process({"metadata": {telemetry.KEY: context()}}, video) as observation:
                    with observation.stage("device_to_host"):
                        video.cpu().numpy()
            records = [json.loads(path.read_text()) for path in (Path(root) / "request-abc/windows/1").glob("postprocess-*.json")]
        self.assertEqual(len(records), 2)
        self.assertEqual({item["status"] for item in records}, {"PASS", "METRICS_INCOMPLETE"})

    def test_old_schema_and_invalid_context_do_not_generate_false_receipt(self):
        video = CpuTensor([2])
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"SEEDVR2_METRICS_DIR": root}):
            for bad in ({**context(), "schema": "sr-timing-memory/1"}, {**context(), "sr_request_id": "../bad"}):
                with telemetry.post_process({"metadata": {telemetry.KEY: bad}}, video) as observation:
                    self.assertFalse(observation.enabled)
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_off_postprocess_is_disabled_and_does_not_write(self):
        with telemetry.post_process({"metadata": {telemetry.KEY: context("off")}}, CpuTensor([1])) as observation:
            self.assertFalse(observation.enabled)

    def test_concurrent_receipt_publish_retains_every_duplicate(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "postprocess-1.json"
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda _: telemetry.unique_json(path, {"status": "PASS", "errors": []}), range(8)))
            records = [json.loads(item.read_text()) for item in Path(root).glob("postprocess-*.json")]
            self.assertFalse(list(Path(root).glob("*.publish-lock")))
        self.assertEqual(len(records), 8)
        self.assertEqual(sum(item["status"] == "PASS" for item in records), 1)
        self.assertEqual(sum(item["status"] == "METRICS_INCOMPLETE" for item in records), 7)

    def test_metrics_publish_failure_does_not_change_cpu_result(self):
        video = CpuTensor([2, 3])
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"SEEDVR2_METRICS_DIR": root}), patch.object(telemetry, "unique_json", side_effect=OSError("metrics disk unavailable")):
            with telemetry.post_process({"metadata": {telemetry.KEY: context()}}, video) as observation:
                with observation.stage("device_to_host"):
                    actual = video.cpu().numpy()
        self.assertEqual(actual, [2, 3])

    def test_real_cpu_tensor_numpy_is_required_in_linux_image(self):
        try:
            import torch
        except ImportError:
            if os.environ.get("SR_METRICS_REQUIRE_RUNTIME") == "1":
                raise
            self.skipTest("Actual Torch CPU check runs in the Linux parent runtime")
        from unittest import mock
        video = torch.arange(72, dtype=torch.uint8).reshape(1, 2, 3, 4, 3)
        original = video.clone()
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"SEEDVR2_METRICS_DIR": root}), mock.patch.object(torch.cuda, "get_device_properties", side_effect=AssertionError("CPU path used CUDA")), mock.patch.object(torch.cuda, "Event", side_effect=AssertionError("CPU path created event")):
            with telemetry.post_process({"metadata": {telemetry.KEY: context()}}, video) as observation:
                with observation.stage("device_to_host"):
                    actual = video.cpu().numpy()
            record = json.loads(next((Path(root) / "request-abc/windows/1").glob("postprocess-*.json")).read_text())
        self.assertTrue(torch.equal(torch.from_numpy(actual), original))
        self.assertEqual(actual.__array_interface__["data"][0], video.data_ptr())
        self.assertEqual(record["status"], "PASS")
        self.assertIsNone(record["stages"][0]["cuda_stream_elapsed_ms"])


if __name__ == "__main__":
    unittest.main()
