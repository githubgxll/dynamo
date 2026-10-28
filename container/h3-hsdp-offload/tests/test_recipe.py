# SPDX-License-Identifier: Apache-2.0
"""Local source/contract regressions. Deliberately not a CUDA substitute."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
from collections import namedtuple
from contextlib import nullcontext
from types import SimpleNamespace

HERE = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location("_test_hsdp_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExactSourcePatch(unittest.TestCase):
    def setUp(self):
        self.patch = load("source_patches")
        self.original = (HERE / "vendor/original" / self.patch.TARGET).read_bytes()

    def test_locked_transform_and_helper(self):
        rel, before, after, helper = self.patch.locked_changes(HERE / "vendor/original")
        self.assertEqual(rel, self.patch.TARGET)
        self.assertEqual(before, self.original)
        self.assertNotEqual(before, after)
        self.assertEqual(helper, (HERE / "hsdp_migration.py").read_bytes())

    def test_only_migration_and_snapshot_functions_change(self):
        def definitions(body):
            tree = ast.parse(body)
            found = {}
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    found[node.name] = ast.dump(node)
                if isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            found[node.name + "." + item.name] = ast.dump(item)
            return found
        before = definitions(self.original)
        after = definitions(self.patch.transform(self.original))
        changed = {name for name in set(before) | set(after) if before.get(name) != after.get(name)}
        self.assertEqual(changed, {"_capture_tensor_devices", "_restore_tensor_devices", "SequentialOffloadHook._move_params"})

    def test_hook_still_controls_full_components(self):
        after = self.patch.transform(self.original).decode()
        self.assertIn("offload_initial_dits", after)
        self.assertIn("offload_after_context", after)
        self.assertIn("return move_component(", after)
        self.assertNotIn(".data =", after)

    def test_duplicate_or_absent_import_anchor_fails(self):
        anchor = b"from .plan_resolver import resolve_offload_plan\n"
        for body in (self.original.replace(anchor, b""), self.original + anchor):
            with self.subTest(body_length=len(body)), self.assertRaises(ValueError):
                self.patch.transform(body)

    def test_repeat_application_rejected(self):
        with self.assertRaises(ValueError):
            self.patch.transform(self.patch.transform(self.original))

    def test_unchanged_hsdp_reference_and_original_pipeline_are_retained(self):
        locked = json.loads((HERE / "source-patches.lock.json").read_text())
        for relative, sha in locked["unchanged_reference_sha256"].items():
            self.assertEqual(hashlib.sha256((HERE / "vendor/original" / relative).read_bytes()).hexdigest(), sha)
        relative, before, after = self.patch.locked_pipeline_change(HERE / "vendor/original")
        self.assertNotIn(relative, locked["unchanged_reference_sha256"])
        self.assertEqual(hashlib.sha256(before).hexdigest(), locked["pipeline"]["before_sha256"])
        self.assertEqual(hashlib.sha256(after).hexdigest(), locked["pipeline"]["after_sha256"])

    def test_no_install_or_external_frontend_in_dockerfile(self):
        text = (HERE / "Dockerfile").read_text()
        self.assertNotIn("# syntax=", text)
        for command in ("pip install", "uv pip install", "apt-get", "wget", "curl "):
            self.assertNotIn(command, text)
        self.assertIn("GPU_NOT_RUN", text)
        self.assertIn("/opt/seedvr2-c14/runtime.py validate", text)
        self.assertIn("USER dynamo\nRUN --network=none", text)

    def test_contract_hash_tracks_behavior_but_not_formatting(self):
        contract = load("torch_contract")
        first = "def f(self):\n    return 1\n"
        self.assertEqual(contract.method_hash(first), contract.method_hash("def f(self):\n    # note\n    return 1\n"))
        self.assertNotEqual(contract.method_hash(first), contract.method_hash("def f(self):\n    return 2\n"))

    def test_contract_pins_native_conversion_and_reset(self):
        policy = json.loads((HERE / "torch-contract.json").read_text())
        self.assertEqual(policy["version"], "2.13.0+cu130")
        self.assertEqual({(x["class"], x["method"]) for x in policy["methods"]},
                         {("Module", "_apply"), ("FSDPModule", "_apply"), ("FSDPParam", "reset_sharded_param")})


def isolated_function(name, scope=None):
    # No fake claim of Torch execution: exercise only pure graph/control flow.
    tree = ast.parse((HERE / "hsdp_migration.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    body = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    namespace = {} if scope is None else scope
    exec(compile(body, "isolated-control-flow", "exec"), namespace)
    return namespace[name]


class MigrationControlFlow(unittest.TestCase):
    def test_same_device_forward_noop_does_not_require_idle_fsdp(self):
        Device = namedtuple("Device", "type index")
        cuda = Device("cuda", 0)
        def active(*args):
            raise RuntimeError("active FSDP must not be inspected for a no-op")
        move = isolated_function("_move_component", {"torch": SimpleNamespace(device=lambda x: x),
            "_postorder": lambda module: [], "_validate": lambda nodes, **kw: {cuda}, "_fsdp_units": active})
        self.assertFalse(move(object(), cuda, non_blocking=False, pin_memory=False, allow_mixed=False))

    def test_active_fsdp_still_rejects_actual_device_change(self):
        Device = namedtuple("Device", "type index")
        cuda, cpu = Device("cuda", 0), Device("cpu", None)
        def active(*args):
            raise RuntimeError("active FSDP")
        move = isolated_function("_move_component", {"torch": SimpleNamespace(device=lambda x: x),
            "_postorder": lambda module: [], "_validate": lambda nodes, **kw: {cuda}, "_fsdp_units": active})
        with self.assertRaisesRegex(RuntimeError, "active FSDP"):
            move(object(), cpu, non_blocking=False, pin_memory=False, allow_mixed=False)

    def test_cycle_and_shared_child_are_visited_once_children_first(self):
        leaf = SimpleNamespace(_modules={})
        root = SimpleNamespace(_modules={})
        left = SimpleNamespace(_modules={"leaf": leaf})
        right = SimpleNamespace(_modules={"leaf": leaf, "back": root})
        root._modules.update(left=left, right=right, empty=None)
        order = [m for _, m in isolated_function("_postorder")(root)]
        self.assertEqual([id(m) for m in order], [id(leaf), id(left), id(right), id(root)])

    def test_deep_graph_does_not_recurse(self):
        leaf = SimpleNamespace(_modules={})
        root = leaf
        for _ in range(2000):
            root = SimpleNamespace(_modules={"child": root})
        nodes = isolated_function("_postorder")(root)
        self.assertEqual(len(nodes), 2001)
        self.assertIs(nodes[0][1], leaf)
        self.assertIs(nodes[-1][1], root)

    def test_rollback_attempts_every_component_and_reports_failure(self):
        calls = []
        def move(module, device, **kwargs):
            calls.append((module, device, kwargs))
            if module == "broken":
                raise RuntimeError("fixture failure")
        restore = isolated_function("restore_components", {"_CONVERSION_LOCK": nullcontext(), "_move_component": move})
        placements = [SimpleNamespace(module="first", device="cuda:0"),
                      SimpleNamespace(module="broken", device="cuda:0"),
                      SimpleNamespace(module="empty", device=None)]
        with self.assertRaisesRegex(RuntimeError, "discard worker"):
            restore(placements)
        self.assertEqual([c[0] for c in calls], ["broken", "first"])
        self.assertTrue(all(c[2]["allow_mixed"] and not c[2]["non_blocking"] for c in calls))


if __name__ == "__main__":
    unittest.main()
