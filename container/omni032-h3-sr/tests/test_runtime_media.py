# SPDX-License-Identifier: Apache-2.0
"""Local adapter tests and an explicit actual-installed CPU suite factory.

The factory is invoked by cpu_smoke.py in the final Linux environment. Ordinary
local discovery runs only the synthetic adapter tests below; it never labels an
unavailable target framework as a passed or skipped installed-runtime test.
"""
from __future__ import annotations

import ast
import base64
from contextlib import ExitStack
import hashlib
import importlib
import importlib.metadata as md
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


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class GPUAdapterIdentity(unittest.TestCase):
    """CPU-only dispatch fixtures: deliberately no CUDA or GPU success path."""
    def fixture(self, temporary):
        root = Path(temporary)
        child, parent = root / "new", root / "hsdp"
        child.mkdir()
        parent.mkdir()
        (child / "runtime.py").write_text("def verify_installed():\n return {'status':'PASS','scope':'synthetic identity fixture'}\n")
        (parent / "gpu_roundtrip.py").write_text("def main():\n import runtime\n return runtime.verify_installed()\n")
        (parent / "torch_contract.py").write_text("def verify():\n return {'scope':'synthetic contract fixture'}\n")
        (parent / "torch-contract.json").write_text("{}\n")
        adapter = load("_new_gpu_adapter_fixture", HERE / "gpu_roundtrip.py")
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(adapter, "HERE", child))
        stack.enter_context(patch.object(adapter, "PARENT", parent))
        for name, path in (("GATE_SHA256", parent / "gpu_roundtrip.py"),
                           ("TORCH_CONTRACT_SHA256", parent / "torch_contract.py"),
                           ("TORCH_POLICY_SHA256", parent / "torch-contract.json")):
            stack.enter_context(patch.object(adapter, name, hashlib.sha256(path.read_bytes()).hexdigest()))
        stack.enter_context(patch.dict(sys.modules))
        for name in ("runtime", "torch_contract", "_omni032_inherited_gpu_gate"):
            sys.modules.pop(name, None)
        return adapter, child, parent

    def test_gate_provenance_uses_new_runtime_not_old_lineage(self):
        with tempfile.TemporaryDirectory() as temporary:
            adapter, child, parent = self.fixture(temporary)
            gate = adapter.resolve_gate()
            self.assertEqual(gate.main()["scope"], "synthetic identity fixture")
            self.assertEqual(Path(sys.modules["runtime"].__file__).resolve(), child / "runtime.py")
            self.assertEqual(Path(gate.main.__code__.co_filename).resolve(), parent / "gpu_roundtrip.py")

    def test_changed_gpu_program_is_rejected_before_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            adapter, _, parent = self.fixture(temporary)
            (parent / "gpu_roundtrip.py").write_text("raise AssertionError('must not execute')\n")
            with self.assertRaisesRegex(ValueError, "implementation differs"):
                adapter.resolve_gate()

    def test_changed_torch_policy_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            adapter, _, parent = self.fixture(temporary)
            (parent / "torch-contract.json").write_text('{"version":"unreviewed"}\n')
            with self.assertRaisesRegex(ValueError, "implementation differs"):
                adapter.resolve_gate()

    def test_cached_old_runtime_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            adapter, _, parent = self.fixture(temporary)
            sys.modules["runtime"] = types.SimpleNamespace(__file__=str(parent / "runtime.py"))
            with self.assertRaisesRegex(ValueError, "different runtime"):
                adapter.resolve_gate()


class RuntimeRecipeGuards(unittest.TestCase):
    """Exercise cross-module patch ownership and fail-closed installer inputs."""
    def test_combined_h3_sr_preparation_preserves_the_official_input_tree(self):
        runtime = load("_omni032_combined_prepare_fixture", HERE / "runtime.py")
        sr = load("_omni032_combined_sr_fixture", HERE / "sr_patches.py")
        h3 = load("_omni032_combined_h3_fixture", HERE / "h3_patches.py")
        with tempfile.TemporaryDirectory() as temporary:
            import shutil
            root = Path(temporary) / "upstream"
            shutil.copytree(HERE / "tests/fixtures/upstream", root)
            pristine = {path.relative_to(root).as_posix(): path.read_bytes()
                        for path in root.rglob("*") if path.is_file()}
            changes, helpers, expected = runtime.prepare_changes(root, pristine)
            self.assertEqual(set(changes), set(sr.CHANGES) | set(h3.CHANGES))
            self.assertEqual(set(helpers), set(sr.ADDITIONS) | {h3.HELPER})
            self.assertTrue(set(changes).isdisjoint(helpers))
            self.assertEqual(set(expected), set(pristine) | set(helpers))
            after = {path.relative_to(root).as_posix(): path.read_bytes()
                     for path in root.rglob("*") if path.is_file()}
            self.assertEqual(after, pristine)

    def test_record_update_rejects_duplicate_patch_ownership(self):
        runtime = load("_omni032_duplicate_record_fixture", HERE / "runtime.py")
        before, after = b"before", b"reviewed after"
        signature = base64.urlsafe_b64encode(hashlib.sha256(before).digest()).rstrip(b"=").decode()
        row = f"vllm_omni/test.py,sha256={signature},{len(before)}\n".encode()
        with self.assertRaisesRegex(ValueError, "RECORD source ownership"):
            runtime.record_update(row + row, {"vllm_omni/test.py": (before, after)}, {})

    def test_record_update_rejects_unreviewed_before_hash(self):
        runtime = load("_omni032_bad_record_fixture", HERE / "runtime.py")
        row = b"vllm_omni/test.py,sha256=incorrect,6\n"
        with self.assertRaisesRegex(ValueError, "RECORD hash differs"):
            runtime.record_update(row, {"vllm_omni/test.py": (b"before", b"after")}, {})

    def test_unknown_stale_helper_is_rejected_without_deleting_it(self):
        runtime = load("_omni032_stale_guard_fixture", HERE / "runtime.py")
        source, stale = "vllm_omni/__init__.py", "vllm_omni/unknown_helper.py"
        commit = "a" * 40
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "site"
            (root / "vllm_omni").mkdir(parents=True)
            (root / source).write_bytes(b"released\n")
            (root / stale).write_bytes(b"unknown helper\n")
            info = Path(temporary) / "build-info"
            def read(path):
                if Path(path).name == "parent-verified.json":
                    return {"source_commit": commit}
                if Path(path).name == "parent-source-manifest.json":
                    return {"source_sha256": {stale: "0" * 64}}
                raise AssertionError("Unexpected receipt read: " + str(path))
            with patch.object(runtime, "INFO", info), patch.object(runtime, "policy", return_value={}), \
                 patch.object(runtime, "read", side_effect=read), \
                 patch.object(runtime, "installed", return_value=(types.SimpleNamespace(version="0.32.0"), root, root / "RECORD")), \
                 patch.object(runtime, "wheel_sources", return_value={source: b"released\n"}), \
                 patch.object(runtime.md, "version", return_value="0.32.0"), \
                 patch.object(runtime.os, "geteuid", return_value=0, create=True), \
                 patch.dict(os.environ, {runtime.SOURCE_ENV: commit}):
                with self.assertRaisesRegex(ValueError, "Unknown file survived"):
                    runtime.apply(info)
            self.assertEqual((root / stale).read_bytes(), b"unknown helper\n")


def installed_suite():
    """Mandatory final-image cases, explicitly separate from local fixtures."""
    import torch
    sr = load("_installed_omni032_sr_patch_contract", HERE / "sr_patches.py")

    class InstalledRuntime(unittest.TestCase):
        def test_imports_and_full_new_runtime_identity(self):
            self.assertEqual(md.version("vllm-omni"), "0.32.0")
            self.assertEqual(md.version("vllm"), "0.32.0")
            root = Path(md.distribution("vllm-omni").locate_file("")).resolve()
            contract = sr.lock()
            expected = {name: item["after_sha256"] for name, item in contract["changes"].items()}
            expected.update(contract["additions"])
            expected.update({name: item["sha256"] for name, item in contract["absorbed"].items()})
            for name, expected_hash in expected.items():
                with self.subTest(source=name):
                    module = importlib.import_module(name[:-3].replace("/", "."))
                    self.assertEqual(Path(module.__file__).resolve(), (root / name).resolve())
                    self.assertEqual(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(), expected_hash)
            runtime = load("_cpu_actual_omni032_runtime", HERE / "runtime.py")
            self.assertEqual(runtime.verify_installed()["status"], "PASS")

        def test_default_observations_remain_disabled(self):
            telemetry = importlib.import_module("vllm_omni.diffusion.models.seedvr2.sr_telemetry")
            capture = importlib.import_module("vllm_omni.diffusion.models.seedvr2.c14_capture")
            self.assertNotIn("SEEDVR2_METRICS_DIR", os.environ)
            self.assertNotIn("SEEDVR2_C14_PROFILE", os.environ)
            self.assertIsNone(capture.profile())
            request = types.SimpleNamespace(sampling_params=types.SimpleNamespace(extra_args={}))
            self.assertFalse(telemetry.window(request, "cpu").enabled)

        def test_actual_cpu_postprocess_never_creates_cuda_work(self):
            pipeline = importlib.import_module("vllm_omni.diffusion.models.seedvr2.pipeline_seedvr2")
            telemetry = importlib.import_module("vllm_omni.diffusion.models.seedvr2.sr_telemetry")
            video = torch.arange(5 * 48 * 80 * 3, dtype=torch.int32).to(torch.uint8).reshape(1, 5, 48, 80, 3)
            marker = {"schema": telemetry.SCHEMA, "measurement_mode": "basic", "sr_request_id": "installed-032-cpu",
                "model_request_id": "installed-window", "actual_model_request_id": "actual-032-request", "window_index": 0,
                "window_start": 0, "input_frame_count": 5, "padded_frame_count": 5, "new_frame_count": 5, "overlap_frames": 0,
                "configured_window": 17, "selected_window": 17, "model_width": 80, "model_height": 48,
                "delivery_width": 80, "delivery_height": 48}
            with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, SEEDVR2_METRICS_DIR=temporary), \
                 patch.object(torch.cuda, "Event", side_effect=AssertionError("CPU created CUDA event")), \
                 patch.object(torch.cuda, "get_device_properties", side_effect=AssertionError("CPU queried CUDA")):
                output = pipeline._seedvr2_post_process({"payload": {"video": video}, "metadata": {telemetry.KEY: marker}})
                record = json.loads(next(Path(temporary).rglob("postprocess-*.json")).read_text())
            self.assertTrue(torch.equal(torch.from_numpy(output["payload"]["video"]), video))
            self.assertEqual(output["payload"]["video"].__array_interface__["data"][0], video.data_ptr())
            self.assertEqual(record["status"], "PASS")
            self.assertEqual(record["execution_scope"], "api_cpu_postprocess")
            self.assertEqual(record["stages"][0]["phase"], "cpu_video_numpy")
            self.assertIsNone(record["stages"][0]["cuda_stream_elapsed_ms"])
            self.assertEqual(record["local_d2h"], "not_executed_here")

        def test_actual_unmarked_cpu_ipc_roundtrip_preserved(self):
            from vllm_omni.diffusion.data import DiffusionOutput
            from vllm_omni.diffusion import ipc
            video = torch.arange(1_200_000, dtype=torch.int32).reshape(100, 100, 120)
            expected = video.clone()
            output = DiffusionOutput(output={"payload": {"video": video}, "metadata": {"unmarked_model": "control"}})
            with patch.object(torch.cuda, "Event", side_effect=AssertionError("Unmarked CPU IPC created CUDA event")):
                restored = ipc.unpack_diffusion_output_shm(ipc.pack_diffusion_output_shm(output))
            self.assertTrue(torch.equal(restored.output["payload"]["video"], expected))
            self.assertEqual(restored.output["metadata"], {"unmarked_model": "control"})

        def test_actual_finish_video_equals_official_032_with_hooks_disabled(self):
            pipeline = importlib.import_module("vllm_omni.diffusion.models.seedvr2.pipeline_seedvr2")
            official = HERE / "tests/fixtures/upstream/vllm_omni/diffusion/models/seedvr2/pipeline_seedvr2.py"
            function = next(node for node in ast.parse(official.read_bytes()).body
                            if isinstance(node, ast.FunctionDef) and node.name == "finish_video")
            scope = {"torch": torch, "correct_video_color": pipeline.correct_video_color}
            exec(compile(ast.Module(body=[function], type_ignores=[]), "official-032-finish-video", "exec"), scope)
            decoded = torch.linspace(-1.3, 1.3, 1 * 3 * 5 * 16 * 16).reshape(1, 3, 5, 16, 16)
            decoded[:, :, :, :4, :4] = 0
            sample = decoded.flip(-1).clone()
            for method in ("none", "lab", "wavelet"):
                with self.subTest(method=method), torch.inference_mode():
                    before = scope["finish_video"](decoded, sample, 5, method)
                    after = pipeline.finish_video(decoded, sample, 5, method)
                    self.assertEqual(after.dtype, torch.uint8)
                    self.assertTrue(torch.equal(before, after))

        def test_actual_telemetry_uses_new_three_source_identities(self):
            telemetry = importlib.import_module("vllm_omni.diffusion.models.seedvr2.sr_telemetry")
            expected = {
                "pipeline_seedvr2.py": sr.lock()["changes"][sr.PREFIX + "pipeline_seedvr2.py"]["after_sha256"],
                "ipc.py": sr.lock()["changes"]["vllm_omni/diffusion/ipc.py"]["after_sha256"],
                "sr_telemetry.py": sr.lock()["additions"][sr.PREFIX + "sr_telemetry.py"],
            }
            self.assertEqual(telemetry._source_hashes(), expected)

        def test_real_gpu_gate_is_resolvable_with_new_identity_only(self):
            adapter = load("_installed_omni032_gpu_adapter", HERE / "gpu_roundtrip.py")
            with patch.dict(sys.modules):
                sys.modules.pop("runtime", None)
                sys.modules.pop("torch_contract", None)
                sys.modules.pop("_omni032_inherited_gpu_gate", None)
                gate = adapter.resolve_gate()
                self.assertEqual(Path(gate.main.__code__.co_filename).resolve(), adapter.PARENT / "gpu_roundtrip.py")
                self.assertEqual(Path(sys.modules["runtime"].__file__).resolve(), HERE / "runtime.py")
                self.assertEqual(Path(sys.modules["torch_contract"].__file__).resolve(), adapter.PARENT / "torch_contract.py")
                self.assertEqual(sys.modules["runtime"].verify_installed()["status"], "PASS")
                self.assertEqual(sys.modules["torch_contract"].verify()["status"], "PASS")

    return unittest.defaultTestLoader.loadTestsFromTestCase(InstalledRuntime)
