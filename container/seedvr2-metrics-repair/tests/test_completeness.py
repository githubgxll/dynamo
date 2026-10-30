# SPDX-License-Identifier: Apache-2.0
"""Replay real /1 failure; /2 repaired receipts below are synthetic test data."""
from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
FIXTURES = HERE / "tests/fixtures"


def receiver():
    if os.environ.get("SR_METRICS_REQUIRE_RUNTIME") == "1":
        return importlib.import_module("vllm_omni.diffusion.models.seedvr2.long_video_support")
    spec = importlib.util.spec_from_file_location("_repair_support_patch", HERE / "support_patch.py")
    delta = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(delta)
    parent = HERE.parent / "seedvr2-optimization/long_video_support.py"
    if not parent.is_file():
        parent = Path("/opt/seedvr2-optimization/long_video_support.py")
    result = types.ModuleType("_repair_support_v2")
    result.__file__ = str(HERE / "long_video_support.py")
    exec(compile(delta.transform(parent.read_bytes()), result.__file__, "exec"), result.__dict__)
    return result


def field_bundle():
    return json.loads((FIXTURES / "field_bundle_v1.json").read_text(encoding="utf-8"))


def schema_two(value):
    if isinstance(value, dict):
        return {name: "sr-timing-memory/2" if name == "schema" and item == "sr-timing-memory/1"
                else schema_two(item) for name, item in value.items()}
    if isinstance(value, list):
        return [schema_two(item) for item in value]
    return value


def synthetic_repaired_bundle():
    """Keep the eighty observed GPU records; construct explicit test-only new receipts.

    These invented CPU/D2H times and tensor addresses are fixtures, not corrected
    field evidence or claimed GPU acceptance of the repair image. The required
    third critical IPC source hash is explicitly synthesized for /2 test data;
    the original /1 fixture remains byte-for-byte unchanged.
    """
    result = schema_two(field_bundle())
    files = result["files"]
    def test_only_ipc_identity(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "source_hashes":
                    item["diffusion/ipc.py"] = "c" * 64
                elif key == "actual_source_hashes":
                    item["ipc.py"] = "c" * 64
                else:
                    test_only_ipc_identity(item)
        elif isinstance(value, list):
            for item in value:
                test_only_ipc_identity(item)
    test_only_ipc_identity(files)
    for name, record in files.items():
        if name.startswith("windows/") and "/rank-" in name:
            record["video_payload_path"] = "payload.video"
    for item in files["window-map.jsonl"]:
        label = f"windows/{item['window_index']}"
        sender = files[label + "/rank-0.json"]
        old_cpu_name = next(name for name in files if name.startswith(label + "/postprocess-"))
        old_cpu = files[old_cpu_name]
        start = sender["cpu_end_ns"] + 1000000
        end = start + 2000000
        before, after = sender["allocator_before"], sender["allocator_after"]
        stage = {"phase": "device_to_host_transport", "cpu_start_ns": start + 1000,
            "cpu_end_ns": start + 101000, "cpu_wall_ms": .1, "cuda_stream_elapsed_ms": .5,
            "execution_scope": "worker_transport", "event_scope": "passed_d2h_stream",
            "cuda_status": "applicable", "allocator_before": before, "allocator_after": after,
            "failure": None, "includes_collectives": False}
        count, height, width = item["input_frame_count"], item["model_height"], item["model_width"]
        transfer = {**item, "actual_model_request_id": sender["actual_model_request_id"],
            "actual_source_hashes": sender["actual_source_hashes"],
            **{key: sender[key] for key in ("pid", "process_start_time", "nspid", "rank", "world_size", "gpu_uuid")},
            "clock_domain": "process_perf_counter_ns", "execution_scope": "worker_transport",
            "source_device_type": "cuda", "source_device": "cuda:0", "video_payload_path": "payload.video",
            "transport_mode": "async_side_stream", "copy_kind": "pinned_cpu",
            "observed_copy_count": 1, "matched_payload_video": True,
            "tensor_identity": {"python_id": 10000 + item["window_index"], "source_device": "cuda:0",
                "shape": [1, count, height, width, 3], "stride": [count * height * width * 3, height * width * 3, width * 3, 3, 1],
                "dtype": "torch.uint8", "data_ptr": 4096, "storage_data_ptr": 4096, "storage_offset": 0,
                "numel": count * height * width * 3, "element_size": 1},
            "transport_stream_identity": "synthetic_stream_fixture", "cpu_start_ns": start, "cpu_end_ns": end,
            "video_tensor_pack_wall_ms": 2.0, "framework_output_pack_wall_ms": 2.002,
            "framework_output_pack": {"phase": "framework_output_pack", "cpu_start_ns": start - 1000,
                "cpu_end_ns": end + 1000, "cpu_wall_ms": 2.002, "execution_scope": "worker_transport",
                "event_scope": "cpu_wall_envelope", "scope": "synthetic original framework pack interval"},
            "cuda_status": "applicable", "allocator_status": "applicable", "allocator_before": before,
            "allocator_after": after, "allocator_window_peak": None,
            "allocator_process_peak_observed": sender["allocator_window_peak"],
            "allocator_peak_scope": "process_since_last_external_reset_not_D2H_peak", "stages": [stage],
            "explicit_stage_waits": 0, "explicit_window_boundary_waits": 0, "status": "PASS", "errors": []}
        files[label + "/transfer-rank-0.json"] = transfer
        cpu_start = end + 1000000
        files[old_cpu_name] = {**item,
            **{key: old_cpu[key] for key in ("pid", "process_start_time", "nspid", "actual_model_request_id", "actual_source_hashes")},
            "clock_domain": "process_perf_counter_ns", "execution_scope": "api_cpu_postprocess",
            "source_device_type": "cpu", "source_device": "cpu", "video_payload_path": "payload.video",
            "local_d2h": "not_executed_here", "upstream_transfer_receipt": "transfer-rank-0.json",
            "rank": None, "world_size": None, "gpu_uuid": None,
            "cuda_status": "not_applicable", "allocator_status": "not_applicable", "allocator_before": None,
            "allocator_after": None, "allocator_window_peak": None, "allocator_process_peak_observed": None,
            "cuda_stream_elapsed_ms": None, "event_scope": "not_applicable_cpu",
            "cpu_start_ns": cpu_start, "cpu_end_ns": cpu_start + 200000, "cpu_postprocess_wall_ms": .2,
            "stages": [{"phase": "cpu_video_numpy", "cpu_start_ns": cpu_start + 1000,
                "cpu_end_ns": cpu_start + 51000, "cpu_wall_ms": .05, "cuda_stream_elapsed_ms": None,
                "event_scope": "not_applicable_cpu", "cuda_status": "not_applicable", "allocator_status": "not_applicable",
                "allocator_before": None, "allocator_after": None, "execution_scope": "api_cpu_postprocess", "failure": None}],
            "status": "PASS", "errors": []}
    return result


class Completeness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.support = receiver()

    def validate(self, bundle):
        return self.support.validate_bundle(bundle["sr_request_id"], bundle["files"])

    def test_fixture_provenance_and_real_failure_shape(self):
        proof = json.loads((FIXTURES / "field_bundle_provenance.json").read_text())
        self.assertEqual(hashlib.sha256((FIXTURES / "field_bundle_v1.json").read_bytes()).hexdigest(), proof["embedded_member_sha256"])
        self.assertEqual(proof["source_metadata_sha256"], "77a91fb200b8cb6a83ec7398b879d32fbe4341e87a553201f5be942665349a44")
        bundle = field_bundle()
        ranks = [row for name, row in bundle["files"].items() if name.startswith("windows/") and "/rank-" in name]
        cpu = [row for name, row in bundle["files"].items() if "/postprocess-" in name]
        self.assertEqual(len(ranks), 80)
        self.assertTrue(all(row["status"] == "PASS" for row in ranks))
        self.assertEqual(len(cpu), 10)
        self.assertTrue(all(row["status"] == "METRICS_INCOMPLETE" and row["stages"] == [] for row in cpu))
        self.assertTrue(all(any("got: cpu" in error["error"] for error in row["errors"]) for row in cpu))

    def test_original_v1_is_rejected_without_relabeling_history(self):
        bundle = field_bundle()
        before = json.dumps(bundle, sort_keys=True)
        result = self.validate(bundle)
        self.assertEqual(result["metrics_status"], "METRICS_INCOMPLETE")
        self.assertEqual(bundle["schema"], "sr-timing-memory/1")
        self.assertEqual(before, json.dumps(bundle, sort_keys=True))

    def test_schema_string_change_cannot_pass(self):
        result = self.validate(schema_two(field_bundle()))
        self.assertEqual(result["metrics_status"], "METRICS_INCOMPLETE")
        self.assertEqual(result["scope_completeness"]["actual_d2h"], "METRICS_INCOMPLETE")
        self.assertEqual(result["scope_completeness"]["api_cpu_postprocess"], "METRICS_INCOMPLETE")

    def test_separate_cpu_and_actual_worker_transfer_pass(self):
        result = self.validate(synthetic_repaired_bundle())
        self.assertEqual(result["inconsistent_rank_records"], [])
        self.assertEqual(result["missing_rank_records"], [])
        self.assertEqual(result["metrics_status"], "PASS_COMPLETE")
        self.assertTrue(all(value == "PASS_COMPLETE" for value in result["scope_completeness"].values()))

    def test_api_cpu_receipt_alone_never_claims_d2h(self):
        bundle = synthetic_repaired_bundle()
        del bundle["files"]["windows/0/transfer-rank-0.json"]
        result = self.validate(bundle)
        self.assertEqual(result["metrics_status"], "METRICS_INCOMPLETE")
        self.assertEqual(result["scope_completeness"]["rank_pipeline"], "PASS_COMPLETE")
        self.assertEqual(result["scope_completeness"]["actual_d2h"], "METRICS_INCOMPLETE")
        self.assertEqual(result["scope_completeness"]["framework_output_pack"], "METRICS_INCOMPLETE")
        self.assertEqual(result["scope_completeness"]["api_cpu_postprocess"], "METRICS_INCOMPLETE")

    def test_fake_cpu_cuda_zero_is_rejected(self):
        for field in ("gpu_uuid", "rank", "world_size", "allocator_before", "allocator_after", "allocator_window_peak",
                      "allocator_process_peak_observed", "pipeline_gpu_window_stream_elapsed_ms"):
            bundle = synthetic_repaired_bundle()
            key = next(name for name in bundle["files"] if name.startswith("windows/0/postprocess-"))
            bundle["files"][key][field] = 0
            with self.subTest(field=field):
                self.assertEqual(self.validate(bundle)["metrics_status"], "METRICS_INCOMPLETE")
        bundle = synthetic_repaired_bundle()
        key = next(name for name in bundle["files"] if name.startswith("windows/0/postprocess-"))
        bundle["files"][key]["stages"][0]["cuda_stream_elapsed_ms"] = 0
        self.assertEqual(self.validate(bundle)["metrics_status"], "METRICS_INCOMPLETE")

    def test_missing_or_duplicate_cpu_receipt_fails(self):
        for duplicate in (False, True):
            bundle = synthetic_repaired_bundle()
            key = next(name for name in bundle["files"] if name.startswith("windows/0/postprocess-"))
            if duplicate:
                bundle["files"]["windows/0/postprocess-9999-duplicate.json"] = copy.deepcopy(bundle["files"][key])
            else:
                del bundle["files"][key]
            with self.subTest(duplicate=duplicate):
                self.assertEqual(self.validate(bundle)["metrics_status"], "METRICS_INCOMPLETE")

    def test_duplicate_transfer_is_not_merged_or_overwritten(self):
        bundle = synthetic_repaired_bundle()
        bundle["files"]["windows/0/transfer-rank-0-duplicate-abc.json"] = copy.deepcopy(bundle["files"]["windows/0/transfer-rank-0.json"])
        self.assertEqual(self.validate(bundle)["metrics_status"], "METRICS_INCOMPLETE")

    def test_transfer_payload_device_count_and_identity_are_strict(self):
        changes = (("source_device_type", "cpu"), ("source_device", "cpu"), ("observed_copy_count", 0),
            ("observed_copy_count", 2), ("matched_payload_video", False), ("video_payload_path", "payload.audio"),
            ("actual_model_request_id", "another-request"), ("world_size", 1), ("rank", 1), ("pid", 1),
            ("gpu_uuid", "wrong-GPU"), ("status", "FAIL"), ("explicit_window_boundary_waits", 1))
        for key, value in changes:
            bundle = synthetic_repaired_bundle()
            bundle["files"]["windows/0/transfer-rank-0.json"][key] = value
            with self.subTest(key=key, value=value):
                self.assertEqual(self.validate(bundle)["metrics_status"], "METRICS_INCOMPLETE")
        for key, value in (("shape", [17, 1104, 1840, 3]), ("dtype", "torch.float16"), ("data_ptr", 0), ("numel", 1)):
            bundle = synthetic_repaired_bundle()
            bundle["files"]["windows/0/transfer-rank-0.json"]["tensor_identity"][key] = value
            with self.subTest(tensor_field=key):
                self.assertEqual(self.validate(bundle)["metrics_status"], "METRICS_INCOMPLETE")

    def test_pack_envelope_is_separate_and_cannot_be_missing(self):
        bundle = synthetic_repaired_bundle()
        del bundle["files"]["windows/0/transfer-rank-0.json"]["framework_output_pack"]
        result = self.validate(bundle)
        self.assertEqual(result["metrics_status"], "METRICS_INCOMPLETE")
        self.assertEqual(result["scope_completeness"]["actual_d2h"], "PASS_COMPLETE")
        self.assertEqual(result["scope_completeness"]["framework_output_pack"], "METRICS_INCOMPLETE")

    def test_gpu_stage_missing_wrong_scope_or_nan_fails(self):
        for change in ("missing", "scope", "time"):
            bundle = synthetic_repaired_bundle()
            stages = bundle["files"]["windows/0/rank-4.json"]["stages"]
            if change == "missing":
                stages.pop()
            elif change == "scope":
                stages[0]["execution_scope"] = "api_cpu_postprocess"
            else:
                stages[0]["cuda_stream_elapsed_ms"] = float("nan")
            with self.subTest(change=change):
                self.assertEqual(self.validate(bundle)["metrics_status"], "METRICS_INCOMPLETE")

    def test_cpu_upstream_reference_and_identity_are_required(self):
        for key, value in (("upstream_transfer_receipt", None), ("local_d2h", "executed_here"),
                           ("actual_model_request_id", "wrong"), ("source_device_type", "cuda")):
            bundle = synthetic_repaired_bundle()
            name = next(name for name in bundle["files"] if name.startswith("windows/0/postprocess-"))
            bundle["files"][name][key] = value
            with self.subTest(key=key):
                self.assertEqual(self.validate(bundle)["metrics_status"], "METRICS_INCOMPLETE")

    def test_rank_request_window_bindings_and_exact_service_windows_required(self):
        for field, value in (("sr_request_id", "wrong-parent"), ("model_request_id", "wrong-window"),
                             ("window_start", 1), ("model_width", 1), ("image_digest", "wrong-image"),
                             ("execution_scope", "api_cpu_postprocess"), ("status", "METRICS_INCOMPLETE")):
            bundle = synthetic_repaired_bundle()
            bundle["files"]["windows/0/rank-7.json"][field] = value
            with self.subTest(field=field):
                self.assertEqual(self.validate(bundle)["metrics_status"], "METRICS_INCOMPLETE")
        bundle = synthetic_repaired_bundle()
        del bundle["files"]["windows/0/rank-7.json"]
        self.assertEqual(len(self.validate(bundle)["missing_rank_records"]), 1)
        bundle = synthetic_repaired_bundle()
        bundle["files"]["window-map.jsonl"][-1]["window_start"] += 1
        self.assertEqual(self.validate(bundle)["metrics_status"], "METRICS_INCOMPLETE")
        bundle = synthetic_repaired_bundle()
        rows = bundle["files"]["server-stages.jsonl"]
        rows[:] = [row for row in rows if not (row["stage"] == "window_input_encode" and row.get("window_index") == 9)]
        result = self.validate(bundle)
        self.assertEqual(result["metrics_status"], "METRICS_INCOMPLETE")
        self.assertEqual(result["scope_completeness"]["server_stages"], "METRICS_INCOMPLETE")

    def test_source_provenance_mismatch_is_not_treated_as_same_receipt(self):
        bundle = synthetic_repaired_bundle()
        bundle["files"]["windows/0/transfer-rank-0.json"]["actual_source_hashes"] = {"sr_telemetry.py": "a" * 64}
        self.assertEqual(self.validate(bundle)["metrics_status"], "METRICS_INCOMPLETE")

    def test_three_critical_actual_sources_need_unique_exact_frozen_matches(self):
        actual = {"sr_telemetry.py": "a" * 64, "pipeline_seedvr2.py": "b" * 64, "ipc.py": "c" * 64}
        frozen = {"diffusion/models/seedvr2/sr_telemetry.py": "a" * 64,
                  "diffusion/models/seedvr2/pipeline_seedvr2.py": "b" * 64, "diffusion/ipc.py": "c" * 64}
        self.assertTrue(self.support._v2_sources({"actual_source_hashes": actual}, {"source_hashes": frozen}))
        cases = [({}, frozen), ({"unrelated.py": "a" * 64}, frozen),
                 ({key: value for key, value in actual.items() if key != "ipc.py"}, frozen),
                 ({**actual, "unrelated.py": "d" * 64}, frozen), (actual, {}),
                 (actual, {"unrelated.py": "a" * 64}),
                 (actual, {key: value for key, value in frozen.items() if not key.endswith("ipc.py")}),
                 (actual, {**frozen, "ipc.py": "c" * 64}),
                 (actual, {**frozen, "other/ipc.py": "c" * 64}),
                 (actual, {**frozen, "diffusion/ipc.py": "bad-hash"}),
                 ({**actual, "ipc.py": int("1" * 64)}, {**frozen, "diffusion/ipc.py": int("1" * 64)})]
        for observed, expected in cases:
            with self.subTest(observed=observed, expected=expected):
                self.assertFalse(self.support._v2_sources({"actual_source_hashes": observed}, {"source_hashes": expected}))
        bundle = synthetic_repaired_bundle()
        del bundle["files"]["windows/0/rank-7.json"]["actual_source_hashes"]["ipc.py"]
        self.assertEqual(self.validate(bundle)["metrics_status"], "METRICS_INCOMPLETE")

    def test_rank_pid_start_time_gpu_binding_must_stay_fixed_across_windows(self):
        for field in ("pid", "start_time", "gpu_uuid"):
            bundle = synthetic_repaired_bundle()
            rank = bundle["files"]["windows/1/rank-3.json"]
            if field == "pid":
                rank["pid"] += 100000
            elif field == "start_time":
                rank["process_start_time"]["value"] += 1
            else:
                rank["gpu_uuid"] = "GPU-replaced-during-request"
            result = self.validate(bundle)
            with self.subTest(field=field):
                self.assertEqual(result["metrics_status"], "METRICS_INCOMPLETE")
                self.assertTrue(any("changed across windows" in error["reason"] for error in result["inconsistent_rank_records"]))

    def test_exact_parent_source_guard_preserves_all_other_helper_defaults(self):
        spec = importlib.util.spec_from_file_location("_support_delta_guard_test", HERE / "support_patch.py")
        delta = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(delta)
        parent = HERE.parent / "seedvr2-optimization/long_video_support.py"
        if not parent.is_file():
            parent = Path("/opt/seedvr2-optimization/long_video_support.py")
        before = parent.read_bytes()
        after = delta.transform(before)
        before_head = before.split(b"def metrics_bundle(job_id):", 1)[0]
        after_head = after.split(b"def metrics_bundle(job_id):", 1)[0]
        self.assertEqual(after_head.replace(b"sr-timing-memory/2", b"sr-timing-memory/1"), before_head)
        with self.assertRaises(ValueError):
            delta.transform(before + b"\n")
        with self.assertRaises(ValueError):
            delta.transform(after)

    def test_api_reads_new_files_without_touching_inference_or_filling_defaults(self):
        bundle = synthetic_repaired_bundle()
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"SEEDVR2_METRICS_DIR": folder}):
            root = Path(folder) / bundle["sr_request_id"]
            for name, value in bundle["files"].items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                if name.endswith(".jsonl"):
                    path.write_text("".join(json.dumps(row) + "\n" for row in value))
                else:
                    path.write_text(json.dumps(value))
            result = self.support.metrics_bundle(bundle["sr_request_id"])
            self.assertEqual(result["metrics_status"], "PASS_COMPLETE")


if __name__ == "__main__":
    unittest.main()
