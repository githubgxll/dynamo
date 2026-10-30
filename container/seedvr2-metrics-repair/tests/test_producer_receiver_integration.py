# SPDX-License-Identifier: Apache-2.0
"""Actual /2 producer and receiver code; strict fake CUDA, no GPU inference claim."""
from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_completeness import receiver
from test_transfer_patch import original
from test_transfer_runtime import FakeTorch, Stream, Tensor
from test_telemetry_repair import CpuTensor

HERE = Path(__file__).resolve().parents[1]


def module_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def recipe_parent(name):
    candidate = HERE.parent / name
    return candidate if candidate.is_dir() else Path("/opt") / name


def pipeline_bytes():
    relative = "vllm_omni/diffusion/models/seedvr2/pipeline_seedvr2.py"
    c14 = recipe_parent("seedvr2-c14")
    optimization = recipe_parent("seedvr2-optimization")
    first = module_file("_integration_c14_source_patch", c14 / "source_patches.py")
    second = module_file("_integration_pipeline_source_patch", optimization / "pipeline_patch.py")
    return second.transform(first.transform(relative, (c14 / "vendor/original" / relative).read_bytes()))


class ProducerReceiverIntegration(unittest.TestCase):
    def test_actual_finish_video_tensor_contract_is_valid_frame_bthwc_uint8(self):
        tree = ast.parse(pipeline_bytes())
        finish = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "finish_video")
        empty = next(node for node in ast.walk(finish) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == "empty")
        self.assertEqual(ast.unparse(empty.args[0]), "(batch_size, frame_count, height, width, 3)")
        self.assertTrue(any(option.arg == "dtype" and ast.unparse(option.value) == "torch.uint8" for option in empty.keywords))
        call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == "finish_video")
        self.assertEqual(ast.unparse(call.args[2]), "prepared.frame_count")

    def test_actual_producers_feed_strict_receiver_in_a_new_synthetic_namespace(self):
        # This does not rewrite any observed /1 receipt. The rank records below
        # explicitly belong to a new simulated process/GPU namespace. Only the
        # actual new CPU/transfer producers create those two receipt types.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            delta = module_file("_integration_transfer_source_patch", HERE / "transfer_patch.py")
            patched_ipc = delta.transform(original())
            if os.environ.get("SR_METRICS_REQUIRE_RUNTIME") == "1":
                telemetry = importlib.import_module("vllm_omni.diffusion.models.seedvr2.sr_telemetry")
            else:
                # Materialize the real three source files in their package
                # layout, so source hashes are computed normally, without
                # mocking the observer's source-hash function.
                diffusion = root / "site/vllm_omni/diffusion"
                seed = diffusion / "models/seedvr2"
                seed.mkdir(parents=True)
                (seed / "sr_telemetry.py").write_bytes((HERE / "telemetry.py").read_bytes())
                (seed / "pipeline_seedvr2.py").write_bytes(pipeline_bytes())
                (diffusion / "ipc.py").write_bytes(patched_ipc)
                telemetry = module_file("_integration_actual_telemetry", seed / "sr_telemetry.py")
            source_hashes = telemetry._source_hashes()
            self.assertEqual(set(source_hashes), {"sr_telemetry.py", "pipeline_seedvr2.py", "ipc.py"})
            self.assertTrue(all(isinstance(value, str) and len(value) == 64 for value in source_hashes.values()))
            support = receiver()
            job_id = "7" * 32
            torch = FakeTorch()
            shape = (1, 17, 1104, 1840, 3)
            class Video(Tensor):
                def stride(self):
                    return (17 * 1104 * 1840 * 3, 1104 * 1840 * 3, 1840 * 3, 3, 1)
                def numel(self):
                    return math.prod(self.shape)
            # Values remain six bounded control values; only the virtual GPU
            # tensor identity has the real model's shape. No giant allocation.
            Video.shape = shape
            video = Video(torch)
            sender_process = {"pid": os.getpid() + 10000,
                "process_start_time": {"clock_domain": "linux_boot_ticks", "value": 5000},
                "nspid": [os.getpid() + 10000]}
            cpu_process = {"pid": os.getpid(),
                "process_start_time": {"clock_domain": "linux_boot_ticks", "value": 6000}, "nspid": [os.getpid()]}
            with patch.dict(os.environ, {"SEEDVR2_METRICS_DIR": str(root / "metrics")}):
                config = support.options("single_encode", "1800x1080", 17, "basic", json.dumps({
                    "run_id": "producer-integration", "batch_id": "synthetic", "case_id": "C08",
                    "action_id": "integration-action", "phase": "measurement", "source_hashes": source_hashes,
                    "image_digest": "example.invalid/testing@sha256:" + "a" * 64,
                    "model_manifest_hash": "b" * 64, "expected_world_size": 8}))
                server = support.ServerMetrics(root / "jobs" / job_id, config, 1840, 1104, 17, 17)
                item = server.window(0, 0, 17)
                actual_request = "actual-producer-integration-request"
                marker = {**item, "actual_model_request_id": actual_request}
                output = {"payload": {"video": video}, "metadata": {telemetry.KEY: marker}}
                # Compile the real patched IPC function. Fake CUDA rejects CPU
                # devices and extra Event.synchronize; the original stream
                # synchronization is the only operation that resolves events.
                function = next(node for node in ast.parse(patched_ipc).body
                    if isinstance(node, ast.FunctionDef) and node.name == "_tensor_to_shm")
                scope = {"torch": torch, "_sr_metrics": telemetry,
                    "_array_to_shm": lambda values: {"values": list(values)},
                    "_cuda_registered_shm_copy": lambda tensor, stream: {"registered": True}}
                syntax = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
                exec(compile(ast.fix_missing_locations(syntax), "actual-patched-ipc-integration", "exec"), scope)
                with patch.dict(sys.modules, {"torch": torch}), patch.object(telemetry, "_proc_identity", return_value=sender_process):
                    with telemetry.transfer_scope(output):
                        packed = scope["_tensor_to_shm"](video, d2h_stream=Stream("D2H", torch.trace), created=[])
                self.assertEqual(packed["values"], list(range(6)))
                before_events = torch.cuda.event_count
                api_video = CpuTensor(packed["values"])
                with patch.dict(sys.modules, {"torch": torch}), patch.object(telemetry, "_proc_identity", return_value=cpu_process):
                    with telemetry.post_process({"payload": {"video": api_video}, "metadata": {telemetry.KEY: marker}}, api_video) as observation:
                        with observation.stage("device_to_host"):
                            received_values = api_video.cpu().numpy()
                self.assertEqual(received_values, list(range(6)))
                self.assertEqual(torch.cuda.event_count, before_events)
                for rank in range(8):
                    process = sender_process if rank == 0 else {"pid": sender_process["pid"] + rank,
                        "process_start_time": {"clock_domain": "linux_boot_ticks", "value": 5000 + rank},
                        "nspid": [sender_process["pid"] + rank]}
                    allocator = {"allocated_bytes": 123, "reserved_bytes": 456}
                    names = ("prepare_video_gpu", "vae_encode", "latent_prepare", "dit", "solver", "vae_decode", "color_uint8")
                    stages = [{"phase": name, "cpu_start_ns": 1000000 * (i + 1), "cpu_end_ns": 1000000 * (i + 2),
                        "cpu_wall_ms": 1.0, "cuda_stream_elapsed_ms": 2.5, "failure": None,
                        "execution_scope": "sr_rank", "event_scope": "current_cuda_stream",
                        "allocator_before": allocator, "allocator_after": allocator} for i, name in enumerate(names)]
                    receipt = {**item, **process, "actual_model_request_id": actual_request, "actual_source_hashes": source_hashes,
                        "rank": rank, "world_size": 8, "video_payload_path": "payload.video",
                        "gpu_uuid": "GPU-real-device-fixture" if rank == 0 else f"GPU-synthetic-distinct-{rank}",
                        "clock_domain": "process_perf_counter_ns", "execution_scope": "sr_rank", "stages": stages,
                        "allocator_before": allocator, "allocator_after": allocator, "allocator_window_peak": allocator,
                        "allocator_peak_scope": "window_reset", "pipeline_gpu_window_stream_elapsed_ms": 17.5,
                        "explicit_stage_waits": 0, "status": "PASS", "errors": []}
                    support.atomic_json(server.root / "windows/0" / f"rank-{rank}.json", receipt)
                for name in ("window_input_encode", "window_model_http", "window_output_decode"):
                    with server.stage(name, window_index=0):
                        pass
                for name in ("source_decode", "delivery_resize", "final_video_encode", "media_validate", "artifact_publish"):
                    with server.stage(name):
                        pass
                server.finish("completed")
                bundle = support.metrics_bundle(job_id)
                self.assertEqual(bundle["inconsistent_rank_records"], [])
                self.assertEqual(bundle["metrics_status"], "PASS_COMPLETE")
                self.assertTrue(all(value == "PASS_COMPLETE" for value in bundle["scope_completeness"].values()))
                transfer = bundle["files"]["windows/0/transfer-rank-0.json"]
                self.assertEqual(transfer["tensor_identity"]["shape"], list(shape))
                self.assertEqual(transfer["source_device_type"], "cuda")
                cpu_key = next(name for name in bundle["files"] if "/postprocess-" in name)
                self.assertEqual(bundle["files"][cpu_key]["source_device_type"], "cpu")
                self.assertIsNone(bundle["files"][cpu_key]["cuda_stream_elapsed_ms"])


if __name__ == "__main__":
    unittest.main()
