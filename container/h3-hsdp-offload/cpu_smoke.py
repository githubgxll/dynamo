# SPDX-License-Identifier: Apache-2.0
"""Required real-image CPU checks. HSDP/CUDA remains a separate acceptance."""
import argparse
import importlib
import json
from pathlib import Path
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
