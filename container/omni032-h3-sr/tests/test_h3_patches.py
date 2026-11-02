# SPDX-License-Identifier: Apache-2.0
"""Actual 0.32 source/control-flow checks; all device operations here are stubs.

These preserve the existing lease/cleanup safety regression cases while adding
0.32 native-stager coverage. They do not test Torch, CUDA or HSDP transfers.
"""
from __future__ import annotations

import ast
from collections import Counter
from contextlib import contextmanager, nullcontext
import copy
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parents[1]
UPSTREAM = HERE / "tests/fixtures/upstream"
spec = importlib.util.spec_from_file_location("_omni032_h3_patches", HERE / "h3_patches.py")
patches = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patches)


def source(relative):
    return (UPSTREAM / relative).read_text(encoding="utf-8")


def generated_backend():
    return patches.planned_patch(patches.SEQUENTIAL, source(patches.SEQUENTIAL))


def method(data, class_name, name):
    cls = next(node for node in ast.parse(data).body
               if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)


def fingerprint(node):
    return ast.dump(node, include_attributes=False)


class Device:
    def __init__(self, value):
        self.name = value.name if isinstance(value, Device) else str(value)
        self.type, _, index = self.name.partition(":")
        self.index = int(index) if index else None

    def __eq__(self, other):
        return isinstance(other, Device) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name


class Fixture:
    """All scans/synchronizations here are observable stubs, never benchmarks."""
    def __init__(self, *, selected=("dit", "text_encoder"), park=True, debug=False):
        self.scans = []
        self.moves = []
        self.syncs = 0
        self.queries = 0
        self.empty_cache_calls = 0
        self.counts = Counter()
        self.errors = []
        self.fail_move = None
        self.debug = debug
        self.cuda, self.cpu = Device("cuda:0"), Device("cpu")
        fixture = self

        class Registry:
            def __init__(self):
                self.hooks = {}

            @staticmethod
            def get_or_create(module):
                return module._hook_registry

            def get_hook(self, name):
                return self.hooks.get(name)

            def register_hook(self, name, hook):
                self.hooks[name] = hook

            def remove_hook(self, name):
                self.hooks.pop(name, None)

        class Module:
            def __init__(self, name):
                self.name = name
                self.residency = fixture.cuda
                self.active = False
                self._hook_registry = Registry()

        self.dit, self.encoder = Module("dit"), Module("encoder")

        def move(module, target, **kwargs):
            target = Device(target)
            self.scans.append((module.name, target.name))
            if self.fail_move == (module.name, target.name):
                raise RuntimeError("fixture migration failure")
            if module.residency == target:
                return False
            if module.active:
                raise RuntimeError("attempted movement after native FSDP pre-forward")
            self.moves.append((module.name, module.residency.name, target.name))
            module.residency = target
            return True

        def sync():
            self.syncs += 1

        def query():
            self.queries += 1
            return 1024**3

        def empty_cache():
            self.empty_cache_calls += 1

        def count(name, amount=1):
            self.counts[name] += amount

        def snapshot(modules):
            return [(module, module.residency) for module in modules]

        def restore(placements):
            for module, device in placements:
                move(module, device)

        logger = SimpleNamespace(
            isEnabledFor=lambda level: self.debug,
            debug=lambda *args, **kwargs: None,
            info=lambda *args, **kwargs: None,
            exception=lambda *args, **kwargs: self.errors.append(args),
        )
        namespace = {
            "__name__": "_isolated_scope_backend",
            "contextmanager": contextmanager, "nullcontext": nullcontext,
            "logging": logging, "os": os, "threading": threading,
            "torch": SimpleNamespace(device=Device), "nn": SimpleNamespace(Module=Module),
            "ModelHook": object, "HookRegistry": Registry,
            "init_logger": lambda name: logger,
            "current_omni_platform": SimpleNamespace(
                is_xpu=lambda: False, synchronize=sync, get_free_memory=query, empty_cache=empty_cache),
            "capture_components": snapshot, "restore_components": restore, "move_component": move,
            "offload_profile_scope": lambda *a, **kw: nullcontext(),
            "timed_offload": lambda *a, **kw: nullcontext(), "count_offload": count,
        }
        tree = ast.parse(generated_backend())
        nodes = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "ModelLevelOffloadBackend":
                break
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                nodes.append(node)
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        code = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
        exec(compile(code, "<generated-offload-scope-backend>", "exec"), namespace)
        self.namespace = namespace
        namespace["apply_sequential_offload"](
            [self.dit], [self.encoder], self.cuda, pin_memory=True, use_hsdp=True,
            offload_initial_dits=park,
            offload_dit_modules=[self.dit] if "dit" in selected else [],
            offload_encoder_modules=[self.encoder] if "text_encoder" in selected else [])
        self.hook = self.dit._hook_registry.get_hook("sequential_offload")
        self.encoder_hook = self.encoder._hook_registry.get_hook("sequential_offload")
        self.reset_counts()

    def reset_counts(self):
        self.scans.clear()
        self.moves.clear()
        self.counts.clear()
        self.syncs = self.queries = self.empty_cache_calls = 0

    def scope(self, module=None):
        return self.namespace["sequential_offload_component"](self.dit if module is None else module)

    def forward(self, module=None):
        module = self.dit if module is None else module
        if module.residency != self.cuda:
            raise RuntimeError("component was not restored before native FSDP pre-forward")
        module.active = True
        try:
            module._hook_registry.get_hook("sequential_offload").pre_forward(module)
            return "unchanged-output"
        finally:
            module.active = False

    def assert_no_scope(self, test):
        test.assertIsNone(self.namespace["_ACTIVE_DIT_SCOPE"])


class ScopeFastpathContracts(unittest.TestCase):
    def setUp(self):
        self.environment = mock.patch.dict(os.environ, {"H3_OFFLOAD_FASTPATH": "1"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_fifty_steps_use_one_activation_and_no_step_scans_or_sync(self):
        f = Fixture()
        with f.scope():
            self.assertEqual(f.dit.residency, f.cuda)
            self.assertEqual(f.encoder.residency, f.cpu)
            scans, syncs = list(f.scans), f.syncs
            for _ in range(50):
                self.assertEqual(f.forward(), "unchanged-output")
            self.assertEqual(f.scans, scans)
            self.assertEqual(f.syncs, syncs)
        self.assertEqual(f.counts["hook_fastpath"], 50)
        self.assertEqual(f.moves, [("encoder", "cuda:0", "cpu"),
                                   ("dit", "cpu", "cuda:0"), ("dit", "cuda:0", "cpu")])
        f.assert_no_scope(self)

    def test_flag_zero_retains_full_hook_work_inside_scope(self):
        with mock.patch.dict(os.environ, {"H3_OFFLOAD_FASTPATH": "0"}):
            f = Fixture()
            with f.scope():
                scans, syncs = len(f.scans), f.syncs
                for _ in range(50):
                    f.forward()
                self.assertEqual(len(f.scans)-scans, 100)
                self.assertEqual(f.syncs-syncs, 50)
        self.assertEqual(f.counts["hook_fastpath"], 0)
        self.assertEqual(sum(x[0] == "dit" and x[2] == "cuda:0" for x in f.moves), 1)
        f.assert_no_scope(self)

    def test_no_scope_never_fastpaths_an_already_resident_module(self):
        f = Fixture(park=False)
        for _ in range(3):
            f.forward()
        self.assertEqual(len(f.scans), 6)
        self.assertEqual(f.syncs, 3)
        self.assertEqual(f.counts["hook_fastpath"], 0)

    def test_reentrant_same_scope_does_not_reload_or_unload_early(self):
        f = Fixture()
        with f.scope():
            entry = list(f.moves)
            with f.scope():
                f.forward()
            self.assertEqual(f.moves, entry)
            self.assertEqual(f.dit.residency, f.cuda)
            f.forward()
        self.assertEqual(f.dit.residency, f.cpu)
        self.assertEqual(sum(x[0] == "dit" and x[2] == "cpu" for x in f.moves), 1)
        f.assert_no_scope(self)

    def test_swallowed_nested_body_failure_leaves_outer_scope_valid(self):
        f = Fixture()
        with f.scope():
            with self.assertRaisesRegex(RuntimeError, "inner failed"):
                with f.scope():
                    raise RuntimeError("inner failed")
            self.assertEqual(f.dit.residency, f.cuda)
            f.forward()
        f.assert_no_scope(self)

    def test_body_failure_releases_component_and_clears_scope(self):
        f = Fixture()
        with self.assertRaisesRegex(ValueError, "original failure"):
            with f.scope():
                f.forward()
                raise ValueError("original failure")
        self.assertEqual(f.dit.residency, f.cpu)
        f.assert_no_scope(self)
        with f.scope():
            f.forward()

    def test_cleanup_failure_does_not_mask_original_body_failure(self):
        f = Fixture()
        with self.assertRaisesRegex(ValueError, "original failure"):
            with f.scope():
                f.fail_move = ("dit", "cpu")
                raise ValueError("original failure")
        self.assertTrue(f.errors)
        f.assert_no_scope(self)

    def test_cleanup_failure_after_success_is_raised_and_lease_cleared(self):
        f = Fixture()
        with self.assertRaisesRegex(RuntimeError, "fixture migration failure"):
            with f.scope():
                f.fail_move = ("dit", "cpu")
        f.assert_no_scope(self)

    def test_activation_failure_does_not_leave_a_fastpath_lease(self):
        f = Fixture()
        f.fail_move = ("dit", "cuda:0")
        with self.assertRaisesRegex(RuntimeError, "fixture migration failure"):
            with f.scope():
                self.fail("failed activation reached body")
        f.assert_no_scope(self)
        f.fail_move = None
        with f.scope():
            f.forward()

    def test_other_component_cannot_evict_active_dit(self):
        f = Fixture()
        with f.scope():
            before = list(f.scans)
            with self.assertRaises(RuntimeError):
                f.encoder_hook.pre_forward(f.encoder)
            self.assertEqual(f.scans, before)
            self.assertEqual(f.dit.residency, f.cuda)
            f.forward()

    def test_other_context_cannot_enter_active_dit_scope(self):
        f = Fixture()
        with f.scope():
            before = list(f.scans)
            with self.assertRaises(RuntimeError):
                with f.scope(f.encoder):
                    self.fail("conflicting component entered scope")
            self.assertEqual(f.scans, before)
            f.forward()

    def test_cross_thread_forward_is_rejected_before_migration(self):
        f = Fixture()
        errors = []
        def worker():
            try:
                f.hook.pre_forward(f.dit)
            except BaseException as exc:
                errors.append(exc)
        with f.scope():
            before = list(f.scans)
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive(), "cross-thread guard deadlocked")
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], RuntimeError)
            self.assertEqual(f.scans, before)
            f.forward()

    def test_changed_scope_configuration_fails_before_fastpath(self):
        f = Fixture()
        with f.scope():
            before = list(f.scans)
            original = f.hook.offload_targets
            try:
                f.hook.offload_targets = []
                with self.assertRaises(RuntimeError):
                    f.forward()
                self.assertEqual(f.scans, before)
            finally:
                f.hook.offload_targets = original
            f.forward()

    def test_changed_release_policy_cannot_silently_skip_exit_cleanup(self):
        f = Fixture()
        with self.assertRaises(RuntimeError):
            with f.scope():
                f.hook.offload_after_context = False
        f.assert_no_scope(self)

    def test_direct_eviction_cannot_leave_ready_scope_pointing_at_cpu(self):
        f = Fixture()
        with f.scope():
            before = list(f.scans)
            with self.assertRaises(RuntimeError):
                f.hook._to_cpu(f.dit)
            self.assertEqual(f.scans, before)
            self.assertEqual(f.dit.residency, f.cuda)
            f.forward()

    def test_hook_removal_is_rejected_during_active_scope(self):
        f = Fixture()
        with f.scope():
            with self.assertRaises(RuntimeError):
                f.namespace["remove_sequential_offload"]([f.dit, f.encoder])
            self.assertIs(f.dit._hook_registry.get_hook("sequential_offload"), f.hook)
            f.forward()

    def test_invalid_fastpath_flag_is_rejected_before_activation(self):
        f = Fixture()
        with mock.patch.dict(os.environ, {"H3_OFFLOAD_FASTPATH": "yes"}):
            with self.assertRaises(ValueError):
                with f.scope():
                    self.fail("invalid control flag reached the model body")
        self.assertEqual(f.moves, [])
        self.assertEqual(f.scans, [])
        f.assert_no_scope(self)

    def test_debug_disabled_does_not_query_free_memory(self):
        f = Fixture(debug=False)
        with f.scope():
            f.forward()
        self.assertEqual(f.queries, 0)

    def test_debug_enabled_queries_only_slow_activation(self):
        f = Fixture(debug=True)
        with f.scope():
            for _ in range(50):
                f.forward()
        self.assertEqual(f.queries, 1)

    def test_dit_only_keeps_encoder_resident(self):
        f = Fixture(selected=("dit",))
        with f.scope():
            for _ in range(2):
                f.forward()
            self.assertEqual(f.encoder.residency, f.cuda)
        self.assertEqual(f.encoder.residency, f.cuda)
        self.assertEqual(f.moves, [("dit", "cpu", "cuda:0"), ("dit", "cuda:0", "cpu")])

    def test_encoder_only_does_not_unload_unselected_dit(self):
        f = Fixture(selected=("text_encoder",), park=False)
        with f.scope():
            f.forward()
        self.assertEqual(f.dit.residency, f.cuda)
        self.assertEqual(f.moves, [("encoder", "cuda:0", "cpu")])
        f.assert_no_scope(self)


class OfficialSourceLocks(unittest.TestCase):
    def test_actual_official_files_before_after_and_compile(self):
        for relative in patches.CHANGES:
            with self.subTest(relative=relative):
                before = source(relative)
                after = patches.planned_patch(relative, before)
                item = patches.policy()["changes"][relative]
                self.assertEqual(hashlib.sha256(before.encode()).hexdigest(), item["before_sha256"])
                self.assertEqual(hashlib.sha256(after.encode()).hexdigest(), item["after_sha256"])
                compile(after, relative, "exec")

    def test_unknown_changed_or_already_patched_input_rejected(self):
        with self.assertRaises(ValueError):
            patches.planned_patch("vllm_omni/unknown.py", "pass\n")
        for relative in patches.CHANGES:
            before = source(relative)
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                patches.planned_patch(relative, before + "# unreviewed edit\n")
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                patches.planned_patch(relative, patches.planned_patch(relative, before))

    def test_h3_diffuse_only_changes_entry_context(self):
        before = source(patches.PIPELINE)
        after = patches.planned_patch(patches.PIPELINE, before)
        old_tree, new_tree = ast.parse(before), ast.parse(after)
        old_cls = next(n for n in old_tree.body if isinstance(n, ast.ClassDef) and n.name == "MiniMaxH3Pipeline")
        new_cls = next(n for n in new_tree.body if isinstance(n, ast.ClassDef) and n.name == "MiniMaxH3Pipeline")
        old_methods = {n.name: fingerprint(n) for n in old_cls.body if isinstance(n, ast.FunctionDef)}
        new_methods = {n.name: fingerprint(n) for n in new_cls.body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(set(old_methods), set(new_methods))
        self.assertEqual([name for name in old_methods if old_methods[name] != new_methods[name]], ["diffuse"])
        old_diffuse = method(before, "MiniMaxH3Pipeline", "diffuse")
        new_diffuse = copy.deepcopy(method(after, "MiniMaxH3Pipeline", "diffuse"))
        assignments = [n for n in new_diffuse.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "component_context" for t in n.targets)]
        self.assertEqual(len(assignments), 1)
        self.assertEqual(ast.unparse(assignments[0].value),
                         "self._component_on_device(transformer) if getattr(self, '_model_cpu_offload_modules', None) else nullcontext()")
        new_diffuse.body.remove(assignments[0])
        withs = [n for n in ast.walk(new_diffuse) if isinstance(n, ast.With)
                 and any(isinstance(i.context_expr, ast.Name) and i.context_expr.id == "component_context" for i in n.items)]
        self.assertEqual(len(withs), 1)
        self.assertIsInstance(withs[0].items[0].context_expr, ast.Name)
        withs[0].items = withs[0].items[1:]
        self.assertEqual(fingerprint(new_diffuse), fingerprint(old_diffuse))
        old_cls.body = [n for n in old_cls.body if not (isinstance(n, ast.FunctionDef) and n.name == "diffuse")]
        new_cls.body = [n for n in new_cls.body if not (isinstance(n, ast.FunctionDef) and n.name == "diffuse")]
        self.assertEqual(fingerprint(old_tree), fingerprint(new_tree))

    def test_native_stager_methods_and_entire_backend_class_preserved(self):
        before, after = source(patches.SEQUENTIAL), generated_backend()
        for name in ("initialize_hook", "_module_stager"):
            with self.subTest(method=name):
                self.assertEqual(fingerprint(method(before, "SequentialOffloadHook", name)),
                                 fingerprint(method(after, "SequentialOffloadHook", name)))
        old_tree, new_tree = ast.parse(before), ast.parse(after)
        cls = lambda t: next(n for n in t.body if isinstance(n, ast.ClassDef) and n.name == "ModelLevelOffloadBackend")
        self.assertEqual(fingerprint(cls(old_tree)), fingerprint(cls(new_tree)))
        methods = lambda t: {n.name for c in t.body if isinstance(c, ast.ClassDef)
                             and c.name == "SequentialOffloadHook" for n in c.body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(methods(old_tree), methods(new_tree))

    def test_stager_branch_survives_to_cpu_and_to_gpu_before_plain_migration(self):
        before, after = source(patches.SEQUENTIAL), generated_backend()
        for name in ("_to_cpu", "_to_gpu"):
            old = method(before, "SequentialOffloadHook", name)
            new = method(after, "SequentialOffloadHook", name)
            # New lock wraps the old stager check and migration, after a scope guard.
            wrapped = new.body[0]
            self.assertIsInstance(wrapped, ast.With)
            stager_start = next(index for index, node in enumerate(wrapped.body)
                                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
                                    and t.id == "stager" for t in node.targets))
            self.assertEqual([fingerprint(n) for n in wrapped.body[stager_start:stager_start+2]],
                             [fingerprint(n) for n in old.body[:2]])


class NewStagerControlFlow(unittest.TestCase):
    def setUp(self):
        self.environment = mock.patch.dict(os.environ, {"H3_OFFLOAD_FASTPATH": "1"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    @staticmethod
    def stager_fixture(fixture):
        class Stager:
            def __init__(self, module, device, **kwargs):
                self.module, self.device, self.kwargs = module, device, kwargs
                self.loads = self.offloads = 0
            def load(self):
                self.loads += 1
                self.module.residency = self.device
            def offload(self):
                self.offloads += 1
                self.module.residency = fixture.cpu
        return Stager

    def test_hsdp_never_initializes_native_persistent_stager(self):
        f = Fixture()
        f.namespace["PinnedModuleStager"] = mock.Mock(side_effect=AssertionError("HSDP cannot use native persistent staging"))
        f.hook.persistent_staging = True
        self.assertIs(f.hook.initialize_hook(f.dit), f.dit)
        self.assertIsNone(f.hook._stager)

    def test_native_initialize_keeps_retain_storage_and_cross_target_resolution(self):
        f = Fixture(park=False)
        f.namespace["PinnedModuleStager"] = self.stager_fixture(f)
        f.hook.use_hsdp = False
        f.hook.persistent_staging = True
        self.assertIs(f.hook.initialize_hook(f.dit), f.dit)
        stager = f.hook._stager
        self.assertEqual(stager.kwargs, {"pin_memory": True, "retain_device_storage": True})
        # Resolve the target DiT's stager even through the running encoder hook.
        f.encoder_hook._to_cpu(f.dit)
        self.assertEqual(stager.offloads, 1)
        f.encoder_hook._to_gpu(f.dit)
        self.assertEqual(stager.loads, 1)
        self.assertEqual(f.scans, [])

    def test_native_persistent_path_has_no_new_fastpath_lease(self):
        f = Fixture(park=False)
        f.namespace["PinnedModuleStager"] = self.stager_fixture(f)
        f.namespace["apply_sequential_offload"]([f.dit], [f.encoder], f.cuda,
            pin_memory=True, use_hsdp=False, persistent_dit_staging=True)
        f.hook = f.dit._hook_registry.get_hook("sequential_offload")
        self.assertFalse(f.hook.scope_fastpath)
        f.hook.initialize_hook(f.dit)
        stager = f.hook._stager
        with f.scope():
            self.assertIsNone(f.namespace["_ACTIVE_DIT_SCOPE"])
            for _ in range(3): f.forward()
        self.assertEqual(stager.loads, 4)
        self.assertEqual(stager.offloads, 1)
        self.assertEqual(f.counts["hook_fastpath"], 0)

    def test_mid_scope_stager_mutation_is_rejected(self):
        for name, value in (("persistent_staging", True), ("_stager", object())):
            with self.subTest(name=name):
                f = Fixture()
                with f.scope():
                    old = getattr(f.hook, name)
                    try:
                        setattr(f.hook, name, value)
                        with self.assertRaisesRegex(RuntimeError, "configuration changed"):
                            f.forward()
                    finally:
                        setattr(f.hook, name, old)
                    f.forward()


if __name__ == "__main__":
    unittest.main()
