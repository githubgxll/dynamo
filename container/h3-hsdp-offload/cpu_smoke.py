# SPDX-License-Identifier: Apache-2.0
"""Required real-image CPU checks. HSDP/CUDA remains a separate acceptance."""
import argparse
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent


class InstalledCPU(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.migration = importlib.import_module("vllm_omni.diffusion.offloader.hsdp_migration")
        self.sequential = importlib.import_module("vllm_omni.diffusion.offloader.sequential_backend")

    def model(self):
        torch = self.torch
        result = torch.nn.Module()
        result.weight = torch.nn.Parameter(torch.arange(8, dtype=torch.float32), requires_grad=False)
        result.alias = result.weight
        result.register_buffer("buffer", torch.arange(3, dtype=torch.float32))
        result.register_buffer("buffer_alias", result.buffer)
        result.child = torch.nn.Module()
        result.child.parent = result  # intentional Cache-DiT-like module cycle
        return result

    def test_installed_cpu_noop_preserves_cycle_and_aliases(self):
        model = self.model()
        weight, buffer = model.weight, model.buffer
        with self.torch.no_grad():
            for _ in range(3):
                self.assertFalse(self.sequential.SequentialOffloadHook._move_params(model, self.torch.device("cpu")))
                self.assertIs(model.weight, weight)
                self.assertIs(model.alias, weight)
                self.assertIs(model.buffer, buffer)
                self.assertIs(model.buffer_alias, buffer)
                self.assertIs(model.child.parent, model)
        self.assertTrue(self.torch.equal(model.weight, self.torch.arange(8, dtype=self.torch.float32)))

    def test_snapshot_restore_uses_owners_on_cycle(self):
        model = self.model()
        with self.torch.no_grad():
            saved = self.sequential._capture_tensor_devices([model])
            self.sequential._restore_tensor_devices(saved)
        self.assertEqual(model.weight.device.type, "cpu")
        self.assertIs(model.alias, model.weight)

    def test_unsupported_meta_target_rejected_without_mutation(self):
        model = self.model()
        old = model.weight.clone()
        with self.torch.no_grad(), self.assertRaises((RuntimeError, ValueError)):
            self.migration.move_component(model, self.torch.device("meta"))
        self.assertEqual(model.weight.device.type, "cpu")
        self.assertTrue(self.torch.equal(model.weight, old))

    def test_native_torch_conversion_contract(self):
        from torch_contract import verify
        self.assertEqual(verify()["status"], "PASS")

    def test_actual_installed_diffuse_entry_control_flow(self):
        # Execute the installed method, while keeping model work and migration
        # simulated here. Real nested FSDP cold-start execution is a GPU gate.
        module = importlib.import_module("vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3")
        installed = inspect.unwrap(module.MiniMaxH3Pipeline.diffuse)
        spec = importlib.util.spec_from_file_location("_image_dit_entry_contract", HERE / "tests/test_dit_entry.py")
        fixture_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture_module)
        for selected in (None, {"dit", "text_encoder"}, {"text_encoder"}, {"dit"}):
            for task, target in (("fl2va", "transformer"), ("ref2va", "transformers_ref")):
                with self.subTest(selected=selected, task=task):
                    fixture = fixture_module.Fixture(selected=selected)
                    scope = dict(installed.__globals__, minimax_h3_denoise_loop=fixture.denoise)
                    method = types.FunctionType(installed.__code__, scope, installed.__name__,
                                                installed.__defaults__, installed.__closure__)
                    method.__kwdefaults__ = installed.__kwdefaults__
                    fixture.pipeline.diffuse = types.MethodType(method, fixture.pipeline)
                    self.assertEqual(fixture.run(task), ("finished-video", "finished-audio"))
                    self.assertEqual(fixture.events.count(("forward", target)), 50)
                    enters = fixture.events.count(("component-enter", target))
                    self.assertEqual(enters, int(selected is not None))
                    if selected is not None and "dit" in selected:
                        self.assertEqual(fixture.events.count(("move", target, "cuda:0")), 1)
                        self.assertLess(fixture.events.index(("move", target, "cuda:0")),
                                        fixture.events.index(("native-pre", target)))

    def test_actual_vllm_parameter_metadata_and_noop(self):
        from vllm.model_executor import parameter
        loader = lambda *args: None
        # Only rank discovery is stubbed on the CPU builder. The installed
        # Parameter class/constructor are real; actual TP8 runs in GPU gate.
        with mock.patch.object(parameter, "get_tensor_model_parallel_rank", return_value=0), \
             mock.patch.object(parameter, "get_tensor_model_parallel_world_size", return_value=1):
            weight = parameter.ModelWeightParameter(data=self.torch.ones(4, 4), input_dim=1,
                                                    output_dim=0, weight_loader=loader)
        weight.requires_grad_(False)
        model = self.torch.nn.Module()
        model.weight = weight
        model.alias = weight
        before = dict(weight.__dict__)
        self.assertFalse(self.migration.move_component(model, self.torch.device("cpu")))
        with self.torch.no_grad(), self.migration._preserve_parameter_identity():
            def clone(tensor):
                copied = tensor.clone()
                self.assertTrue(self.torch._has_compatible_shallow_copy_type(tensor, copied))
                return copied
            model._apply(clone, recurse=False)
        self.assertIs(model.weight, weight)
        self.assertIs(type(model.weight), parameter.ModelWeightParameter)
        self.assertIs(model.alias, weight)
        self.assertEqual(weight.__dict__.keys(), before.keys())
        for key, value in before.items():
            self.assertIs(weight.__dict__[key], value)

    def _installed_scope_forward(self, mode, *, fail=False):
        # Real installed HookRegistry, torch module forwards, helper no-op
        # traversal and telemetry wiring. This does not exercise CUDA or FSDP.
        torch, backend = self.torch, self.sequential
        dit = torch.nn.Linear(4, 4, bias=False).requires_grad_(False)
        encoder = torch.nn.Linear(4, 4, bias=False).requires_grad_(False)
        values = torch.arange(4, dtype=torch.float32).reshape(1, 4)
        weight = dit.weight
        with torch.no_grad():
            expected = dit(values).clone()
        platform = types.SimpleNamespace(
            is_xpu=lambda: False,
            synchronize=mock.Mock(),
            empty_cache=mock.Mock(),
            get_free_memory=mock.Mock(side_effect=AssertionError("Unexpected memory query")),
        )
        failure = RuntimeError("installed CPU scope body fixture")
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.dict(os.environ, {"H3_OFFLOAD_FASTPATH": mode,
                                         "H3_OFFLOAD_TIMING": "1",
                                         "H3_OFFLOAD_TIMING_DIR": directory}), \
             mock.patch.object(backend, "current_omni_platform", platform), \
             mock.patch.object(backend.logger, "isEnabledFor", return_value=False), \
             mock.patch.object(backend, "move_component", wraps=self.migration.move_component) as moves, \
             torch.no_grad():
            self.assertIsNone(backend._ACTIVE_DIT_SCOPE)
            self.assertIsNone(self.migration._OFFLOAD_PROFILE.get())
            backend.apply_sequential_offload([dit], [encoder], torch.device("cpu"),
                                             pin_memory=False, use_hsdp=True)
            try:
                try:
                    with backend.sequential_offload_component(dit):
                        self.assertEqual(moves.call_count, 2)
                        for _ in range(50):
                            self.assertTrue(torch.equal(dit(values), expected))
                            self.assertIs(dit.weight, weight)
                        self.assertEqual(moves.call_count, 2 if mode == "1" else 102)
                        if fail:
                            raise failure
                except RuntimeError as error:
                    if not fail:
                        raise
                    self.assertIs(error, failure)
                else:
                    self.assertFalse(fail, "The body exception was swallowed")
                self.assertIsNone(backend._ACTIVE_DIT_SCOPE)
                self.assertIsNone(self.migration._OFFLOAD_PROFILE.get())
                self.assertEqual(moves.call_count, 3 if mode == "1" else 103)
                self.assertEqual(platform.synchronize.call_count, 1 if mode == "1" else 51)
                platform.empty_cache.assert_not_called()
                platform.get_free_memory.assert_not_called()
                records = list(Path(directory).rglob("*.json"))
                self.assertEqual(len(records), 1)
                self.assertEqual(list(Path(directory).rglob("*.tmp")), [])
                record = json.loads(records[0].read_text(encoding="utf-8"))
                self.assertEqual(record["schema"], "h3-offload-timing/1")
                self.assertEqual(record["label"], "component:Linear")
                self.assertEqual(record["fastpath_env"], mode)
                self.assertEqual(record["status"], "failed" if fail else "succeeded")
                if fail:
                    self.assertEqual(record["error_type"], "RuntimeError")
                self.assertEqual(record["counts"].get("hook_fastpath", 0), 50 if mode == "1" else 0)
                self.assertEqual(record["counts"]["hook_slowpath"], 1 if mode == "1" else 51)
                self.assertEqual(record["counts"]["hook_synchronize_calls"], 1 if mode == "1" else 51)
                self.assertEqual(record["counts"]["component_body_calls"], 1)
                self.assertGreaterEqual(record["seconds"]["component_body"], 0)
            finally:
                backend.remove_sequential_offload([dit, encoder])

    def test_installed_scope_hook_and_profile_wiring_for_both_modes(self):
        for mode in ("1", "0"):
            with self.subTest(fastpath=mode):
                self._installed_scope_forward(mode)

    def test_installed_scope_exception_clears_lease_and_profile(self):
        self._installed_scope_forward("1", fail=True)

    def test_conversion_future_flags_restored_after_exception(self):
        future = self.torch.__future__
        old_swap = future.get_swap_module_params_on_conversion()
        old_overwrite = future.get_overwrite_module_params_on_conversion()
        try:
            future.set_swap_module_params_on_conversion(True)
            future.set_overwrite_module_params_on_conversion(True)
            with self.assertRaisesRegex(RuntimeError, "fixture"):
                with self.migration._preserve_parameter_identity():
                    self.assertFalse(future.get_swap_module_params_on_conversion())
                    self.assertFalse(future.get_overwrite_module_params_on_conversion())
                    raise RuntimeError("fixture")
            self.assertTrue(future.get_swap_module_params_on_conversion())
            self.assertTrue(future.get_overwrite_module_params_on_conversion())
        finally:
            future.set_swap_module_params_on_conversion(old_swap)
            future.set_overwrite_module_params_on_conversion(old_overwrite)

    def test_distinct_shared_storage_rejected_but_tied_objects_allowed(self):
        module = self.model()
        self.migration._check_storage_aliases(self.migration._postorder(module))
        module.view_parameter = self.torch.nn.Parameter(module.weight.detach(), requires_grad=False)
        with self.assertRaisesRegex(RuntimeError, "share storage"):
            self.migration._check_storage_aliases(self.migration._postorder(module))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import torch  # required, never skipped in final image
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(InstalledCPU)])
    suite.addTests(unittest.defaultTestLoader.discover(str(HERE / "tests"), pattern="test_*.py"))
    log = args.output_dir / "hsdp-cpu-tests.txt"
    with log.open("w", encoding="utf-8") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = {"schema": "h3-hsdp-offload-cpu/1", "status": "PASS" if result.wasSuccessful() and not result.skipped else "FAIL",
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
        "torch_version": torch.__version__, "gpu_validation_status": "NOT_RUN",
        "scope": "Installed CPU no-op/owner snapshot and source invariants; no CUDA migration or H3 inference claim"}
    (args.output_dir / "regression.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "PASS":
        print(log.read_text(encoding="utf-8")[-16000:], flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
