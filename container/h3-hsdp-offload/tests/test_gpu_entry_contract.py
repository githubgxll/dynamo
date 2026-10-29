# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the GPU launcher/control-flow binding; never GPU proof."""
import ast
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_gpu_entry_contract", ROOT / "gpu_roundtrip.py")
gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu)


def minimax_h3_denoise_loop(value):
    return "original", value


class PipelineFixture:
    def diffuse(self, *, task, num_steps=50):
        with self._component_on_device(self._transformer_for_task(task)):
            with self._resident_dit_layers_on_device(enabled=True):
                return minimax_h3_denoise_loop(num_steps)

    @contextmanager
    def _component_on_device(self, component):
        self.events.append(("component-enter", component))
        try:
            yield
        finally:
            self.events.append(("component-exit", component))

    def _transformer_for_task(self, task):
        return self.ref if task == "ref2va" else self.main

    @contextmanager
    def _resident_dit_layers_on_device(self, *, enabled):
        self.events.append(("dlo", enabled))
        yield


class InstalledEntryBinding(unittest.TestCase):
    def test_binding_reuses_code_and_contexts_without_mutating_original_globals(self):
        shell = SimpleNamespace(events=[], main="main", ref="ref")
        original = PipelineFixture.diffuse
        function = gpu.bind_installed_diffuse(PipelineFixture, shell, lambda n: ("synthetic", n))
        self.assertIs(function.__func__.__code__, original.__code__)
        self.assertEqual(function(task="ref2va"), ("synthetic", 50))
        self.assertEqual(shell.events, [("component-enter", "ref"), ("dlo", True), ("component-exit", "ref")])
        self.assertIs(original.__globals__["minimax_h3_denoise_loop"], minimax_h3_denoise_loop)

    def test_synthetic_failure_runs_installed_cleanup_and_keeps_source_global(self):
        def fail(_):
            raise RuntimeError("fixture failure")
        shell = SimpleNamespace(events=[], main="main", ref="ref")
        function = gpu.bind_installed_diffuse(PipelineFixture, shell, fail)
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            function(task="fl2va")
        self.assertEqual(shell.events[-1], ("component-exit", "main"))
        self.assertIs(PipelineFixture.diffuse.__globals__["minimax_h3_denoise_loop"], minimax_h3_denoise_loop)


class VllmLauncherContext(unittest.TestCase):
    def fixture(self, *, available=True, count=8):
        events, active = [], []
        cuda = SimpleNamespace(is_available=lambda: available, device_count=lambda: count,
                               set_device=lambda rank: events.append(("device", rank)))
        def parallel(**kwargs):
            if not events or events[0] != ("device", 2):
                raise AssertionError("Constructed vLLM configuration before selecting assigned CUDA device")
            events.append(("parallel", kwargs))
            return kwargs
        def config(**kwargs):
            return SimpleNamespace(model_config=None, **kwargs)
        @contextmanager
        def current(value, *, check_compile):
            events.append(("context-enter", check_compile))
            active.append(value)
            try:
                yield
            finally:
                active.pop()
                events.append(("context-exit",))
        modules = {"torch": SimpleNamespace(cuda=cuda), "vllm": SimpleNamespace(),
                   "vllm.config": SimpleNamespace(ParallelConfig=parallel, VllmConfig=config,
                                                  set_current_vllm_config=current,
                                                  get_current_vllm_config=lambda: active[-1] if active else None)}
        return events, active, modules

    def test_current_config_wraps_run_and_restores_even_on_failure(self):
        for fail in (False, True):
            events, active, modules = self.fixture()
            def run(args, report):
                self.assertEqual(active[-1].parallel_config["tensor_parallel_size"], 8)
                self.assertEqual(report["vllm_context"]["model_config"], None)
                events.append(("run",))
                if fail:
                    raise RuntimeError("test run error")
            with mock.patch.dict(os.environ, {"WORLD_SIZE": "8", "LOCAL_RANK": "2"}), \
                 mock.patch.dict(sys.modules, modules), mock.patch.object(gpu, "run", run):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "test run error"):
                        gpu.run_with_vllm_config(None, {})
                else:
                    gpu.run_with_vllm_config(None, {})
            self.assertEqual(active, [])
            self.assertEqual(events[-1], ("context-exit",))
            self.assertIn(("context-enter", False), events)

    def test_cpu_or_wrong_visible_gpu_count_is_rejected(self):
        for available, count in ((False, 8), (True, 1), (True, 9)):
            _, _, modules = self.fixture(available=available, count=count)
            with mock.patch.dict(os.environ, {"WORLD_SIZE": "8", "LOCAL_RANK": "2"}), \
                 mock.patch.dict(sys.modules, modules), mock.patch.object(gpu, "run") as run, \
                 self.assertRaisesRegex(RuntimeError, "Eight visible"):
                gpu.run_with_vllm_config(None, {})
            run.assert_not_called()

    def test_wrong_rank_configuration_is_rejected_before_initialization(self):
        for world, local in (("1", "0"), ("8", "-1"), ("8", "8")):
            events, _, modules = self.fixture()
            with mock.patch.dict(os.environ, {"WORLD_SIZE": world, "LOCAL_RANK": local}), \
                 mock.patch.dict(sys.modules, modules), self.assertRaisesRegex(RuntimeError, "eight torchrun"):
                gpu.run_with_vllm_config(None, {})
            self.assertEqual(events, [])


class ScopeFastpathEvidence(unittest.TestCase):
    def test_environment_control_is_exact_and_defaults_enabled(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(gpu.scope_fastpath_enabled())
        for value, expected in (("0", False), ("1", True)):
            with mock.patch.dict(os.environ, {"H3_OFFLOAD_FASTPATH": value}):
                self.assertIs(gpu.scope_fastpath_enabled(), expected)
        for value in ("", "true", "2"):
            with mock.patch.dict(os.environ, {"H3_OFFLOAD_FASTPATH": value}), \
                 self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                gpu.scope_fastpath_enabled()

    def test_zero_migrations_alone_cannot_pass_fastpath_when_scans_or_sync_remain(self):
        gpu.assert_forward_admin(True, 0, 0)
        for scans, sync in ((1, 0), (0, 1), (2, 1)):
            with self.subTest(scans=scans, sync=sync), self.assertRaisesRegex(AssertionError, "scanned parameters"):
                gpu.assert_forward_admin(True, scans, sync)

    def test_disabled_control_must_prove_scans_and_sync_are_exercised(self):
        gpu.assert_forward_admin(False, 2, 1)
        for scans, sync in ((0, 0), (1, 0), (0, 1)):
            with self.subTest(scans=scans, sync=sync), self.assertRaisesRegex(AssertionError, "Disabled fastpath"):
                gpu.assert_forward_admin(False, scans, sync)

    def test_invalid_or_regressed_counters_are_rejected(self):
        for scans, sync in ((-1, 0), (0, -1), (True, 0), (0, 0.0)):
            with self.subTest(scans=scans, sync=sync), self.assertRaisesRegex(AssertionError, "Invalid per-forward"):
                gpu.assert_forward_admin(True, scans, sync)

    def test_observer_does_not_patch_shared_platform_or_count_external_sync(self):
        calls = []
        def synchronize(*args, **kwargs):
            calls.append((args, kwargs))
            return "synchronized"
        platform = SimpleNamespace(synchronize=synchronize, marker=object())
        counts = {"backend_synchronize_calls": 0}
        observer = gpu.BackendPlatformObserver(platform, counts)
        self.assertEqual(observer.synchronize("device", option=True), "synchronized")
        self.assertEqual(counts["backend_synchronize_calls"], 1)
        self.assertEqual(calls, [(("device",), {"option": True})])
        self.assertIs(observer.marker, platform.marker)
        self.assertIs(platform.synchronize, synchronize)
        platform.synchronize()
        self.assertEqual(counts["backend_synchronize_calls"], 1)


class GateSourceContract(unittest.TestCase):
    def test_main_uses_context_launcher_and_requires_three_cycles(self):
        tree = ast.parse((ROOT / "gpu_roundtrip.py").read_text(encoding="utf-8"))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        calls = [node.func.id for node in ast.walk(main) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        self.assertIn("run_with_vllm_config", calls)
        self.assertNotIn("run", calls)
        self.assertIn("At least three complete offload cycles are required", ast.unparse(main))

    def test_cold_gate_is_separate_and_must_check_backing_during_each_step(self):
        tree = ast.parse((ROOT / "gpu_roundtrip.py").read_text(encoding="utf-8"))
        cold = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                    and node.name == "cold_installed_pipeline_entry")
        denoise = next(node for node in cold.body if isinstance(node, ast.FunctionDef) and node.name == "synthetic_denoise")
        step_loop = next(node for node in denoise.body if isinstance(node, ast.For))
        self.assertIn("backing_addresses() != addresses", ast.unparse(step_loop))
        self.assertIn("len(moves) != move_count_at_entry", ast.unparse(step_loop))
        self.assertIn("admin_at_entry = dict(admin_counts)", ast.unparse(step_loop))
        self.assertIn("assert_forward_admin(fastpath, **forward_counts)", ast.unparse(step_loop))
        self.assertIn("reference = fresh_model()", ast.unparse(cold))
        self.assertIn("cold = fresh_model()", ast.unparse(cold))

    def test_cold_gate_observes_backend_variable_not_shared_platform_method(self):
        tree = ast.parse((ROOT / "gpu_roundtrip.py").read_text(encoding="utf-8"))
        cold = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                    and node.name == "cold_installed_pipeline_entry")
        source = ast.unparse(cold)
        self.assertIn("mock.patch.object(backend, 'current_omni_platform', platform_observer)", source)
        self.assertNotIn("mock.patch.object(backend.current_omni_platform", source)
        self.assertIn("forward_move_params_calls=sum(", source)
        self.assertIn("forward_backend_synchronize_calls=sum(", source)


if __name__ == "__main__":
    unittest.main()
