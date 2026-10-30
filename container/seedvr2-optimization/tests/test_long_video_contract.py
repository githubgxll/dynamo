# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


support = load("_sr_long_support_contract", HERE / "long_video_support.py")


class LongVideoContracts(unittest.TestCase):
    def test_legacy_defaults_preserved(self):
        config = support.options()
        self.assertEqual(config["encoding_mode"], "legacy")
        self.assertEqual(config["measurement_mode"], "off")
        self.assertIsNone(config["window_frames"])
        self.assertEqual(support.writer_spec(config, 1840, 1104)["options"]["preset"], "veryfast")

    def test_single_writer_uses_delivery_preset(self):
        config = support.options("single_encode", "1800x1080", 17)
        support.validate_source(config, 1280, 768, 1840, 1104)
        spec = support.writer_spec(config, 1840, 1104)
        self.assertEqual((spec["width"], spec["height"]), (1800, 1080))
        self.assertEqual(spec["options"], {"preset": "medium", "crf": "18"})

    def test_supported_aspects(self):
        for size, source, model in (("1800x1080", (1280, 768), (1840, 1104)),
                                    ("1890x1080", (1344, 768), (1904, 1088)),
                                    ("1080x1800", (768, 1280), (1104, 1840)),
                                    ("2400x1440", (1280, 768), (2400, 1440))):
            with self.subTest(size=size):
                support.validate_source(support.options("single_encode", size), *source, *model)

    def test_invalid_delivery_rejected(self):
        for encoding, size in (("bad", None), ("single_encode", None), ("legacy", "1800x1080"),
                               ("single_encode", "1801x1080"), ("single_encode", "1920x1081"),
                               ("single_encode", "0x1080"), ("single_encode", "1800x720")):
            with self.subTest(encoding=encoding, size=size), self.assertRaises(ValueError):
                support.options(encoding, size)
        with self.assertRaisesRegex(ValueError, "aspect ratio"):
            support.validate_source(support.options("single_encode", "1920x1080"), 1280, 768, 1840, 1104)
        with self.assertRaisesRegex(ValueError, "model size"):
            support.validate_source(support.options("single_encode", "1800x1080"), 1280, 768, 1920, 1088)

    def test_explicit_window_does_not_fallback(self):
        for selected in (17, 33, 65):
            self.assertEqual(support.select_window(65, selected), selected)
        self.assertEqual(support.select_window(33, None), 33)
        for capacity, requested in ((17, 33), (33, 65), (0, None)):
            with self.subTest(capacity=capacity, requested=requested), self.assertRaises(ValueError):
                support.select_window(capacity, requested)
        for value in (5, 9, 32, 64, True):
            with self.assertRaises(ValueError):
                support.options(window_frames=value)

    def test_measured_request_requires_explicit_shared_path(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "SEEDVR2_METRICS_DIR"):
            support.options(measurement_mode="basic")

    def test_context_validation(self):
        for value in ('[]', '{"not_a_field": 1}', '{"run_id":"../../escape"}',
                      '{"expected_world_size": 1}', '{"source_hashes":{"a":"oops"}}'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                support.options(measurement_context=value)
        identity = {"run_id": "test-run", "case_id": "C14", "expected_world_size": 8,
                    "source_hashes": {"base.mp4": "a" * 64}}
        self.assertEqual(support.options(measurement_context=json.dumps(identity))["measurement_context"], identity)

    def test_mapping_and_rank_completeness(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"SEEDVR2_METRICS_DIR": folder}):
            config = support.options(window_frames=17, measurement_mode="basic")
            metrics = support.ServerMetrics(Path(folder) / ("a" * 32), config, 1840, 1104, 20, 17)
            first, last = metrics.window(0, 0, 17), metrics.window(1, 13, 7)
            self.assertEqual((last["input_frame_count"], last["padded_frame_count"], last["new_frame_count"]), (7, 9, 3))
            self.assertEqual(first["overlap_frames"], 0)
            self.assertNotEqual(first["model_request_id"], last["model_request_id"])
            metrics.finish("completed")
            bundle = support.metrics_bundle("a" * 32)
            self.assertEqual(bundle["metrics_status"], "METRICS_INCOMPLETE")
            self.assertEqual(len(bundle["missing_rank_records"]), 16)
            for item in (first, last):
                stage_names = ("prepare_video_gpu", "vae_encode", "latent_prepare", "dit", "solver", "vae_decode", "color_uint8")
                measured = {"allocator_before": {"allocated_bytes": 1, "reserved_bytes": 2},
                    "allocator_after": {"allocated_bytes": 1, "reserved_bytes": 2},
                    "allocator_window_peak": {"allocated_bytes": 1, "reserved_bytes": 2},
                    "pipeline_gpu_window_stream_elapsed_ms": 1.0, "actual_model_request_id": "actual-123",
                    "stages": [{"phase": name, "cpu_wall_ms": 1.0, "cuda_stream_elapsed_ms": 1.0} for name in stage_names]}
                for rank in range(8):
                    support.atomic_json(metrics.root / "windows" / str(item["window_index"]) / f"rank-{rank}.json",
                        {**item, **measured, "rank": rank, "world_size": 8, "status": "PASS"})
                support.atomic_json(metrics.root / "windows" / str(item["window_index"]) / "postprocess-123.json",
                    {**item, **measured, "status": "PASS",
                     "allocator_window_peak": None,
                     "allocator_process_peak_observed": {"allocated_bytes": 1, "reserved_bytes": 2},
                     "allocator_peak_scope": "process_since_last_external_reset_not_d2h_peak",
                     "stages": [{"phase": "device_to_host", "cpu_wall_ms": 1.0, "cuda_stream_elapsed_ms": 1.0}]})
            self.assertEqual(support.metrics_bundle("a" * 32)["metrics_status"], "PASS_COMPLETE")
            rankpath = metrics.root / "windows/0/rank-3.json"
            rankdata = json.loads(rankpath.read_text())
            rankdata["model_request_id"] = "wrong-request"
            support.atomic_json(rankpath, rankdata)
            self.assertEqual(len(support.metrics_bundle("a" * 32)["inconsistent_rank_records"]), 1)

    def test_metrics_failure_is_not_generation_failure(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"SEEDVR2_METRICS_DIR": folder}):
            config = support.options(measurement_mode="basic")
            with patch.object(support, "atomic_json", side_effect=OSError("no writes")):
                metrics = support.ServerMetrics(Path(folder) / ("b" * 32), config, 80, 48, 5, 17)
                with metrics.stage("source_decode"):
                    pass
                metrics.window(0, 0, 5)
                metrics.finish("completed")
            self.assertTrue(metrics.failed)

    def test_metrics_ids_are_not_paths(self):
        for value in ("../../escape", "f" * 31, "f" * 32 + "/other"):
            with self.assertRaises(ValueError):
                support.checked_root(value)


if __name__ == "__main__":
    unittest.main()
