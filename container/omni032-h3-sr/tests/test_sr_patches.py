# SPDX-License-Identifier: Apache-2.0
"""Real official-source migration checks; no installed framework is required."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures/upstream"
spec = importlib.util.spec_from_file_location("_test_omni032_sr_patches", HERE / "sr_patches.py")
sr = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sr
spec.loader.exec_module(sr)


def before(relative):
    return (FIXTURES / relative).read_bytes()


def fixture_hashes(root=FIXTURES):
    return {path.relative_to(root).as_posix(): sr.digest(path.read_bytes())
            for path in root.rglob("*") if path.is_file()}


class StripObservations(ast.NodeTransformer):
    """Recover upstream AST from wrappers, keeping upstream control flow."""
    def visit_ImportFrom(self, node):
        if any(item.name in {"sr_telemetry", "c14_capture"} for item in node.names):
            return None
        return node

    @staticmethod
    def observation_call(node):
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"_sr_metrics", "_c14"})

    def visit_With(self, node):
        self.generic_visit(node)
        if len(node.items) == 1 and self.observation_call(node.items[0].context_expr):
            return node.body
        return node

    def visit_Expr(self, node):
        if self.observation_call(node.value):
            return None
        return self.generic_visit(node)


class SourceMigration(unittest.TestCase):
    def test_official_fixtures_have_exact_release_hashes(self):
        expected = {name: item["before_sha256"] for name, item in sr.lock()["changes"].items()}
        expected.update({name: item["sha256"] for name, item in sr.lock()["absorbed"].items()})
        # H3 migration fixtures share this tree and are independently owned.
        actual = fixture_hashes()
        self.assertEqual({name: actual[name] for name in expected}, expected)

    def test_all_deltas_produce_reviewed_final_hashes(self):
        for name in sr.CHANGES:
            with self.subTest(source=name):
                result = sr.transform(name, before(name))
                self.assertEqual(sr.digest(result), sr.lock()["changes"][name]["after_sha256"])

    def test_prepare_validates_all_inputs_without_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "wheel"
            shutil.copytree(FIXTURES, root)
            initial = fixture_hashes(root)
            result = sr.locked_changes(root)
            self.assertEqual(set(result), set(sr.CHANGES))
            self.assertEqual(fixture_hashes(root), initial)
            for name, (original, changed) in result.items():
                self.assertEqual(original, before(name))
                self.assertNotEqual(changed, original)

    def test_rejects_unreviewed_upstream_source(self):
        name = "vllm_omni/diffusion/ipc.py"
        with self.assertRaisesRegex(ValueError, "before source differs"):
            sr.transform(name, before(name) + b"\n")

    def test_rejects_repeat_patch_application(self):
        name = "vllm_omni/diffusion/ipc.py"
        with self.assertRaisesRegex(ValueError, "before source differs"):
            sr.transform(name, sr.transform(name, before(name)))

    def test_rejects_unknown_source_without_running_a_generator(self):
        with patch.object(sr, "_generator", side_effect=AssertionError("must not import")):
            with self.assertRaisesRegex(ValueError, "Unknown"):
                sr.transform("vllm_omni/__init__.py", b"")

    def test_rejects_existing_helper_before_preparing_deltas(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "wheel"
            shutil.copytree(FIXTURES, root)
            (root / sr.ADDITIONS[0]).write_bytes(b"unexpected existing helper")
            with self.assertRaisesRegex(ValueError, "existing SR helper"):
                sr.locked_changes(root)

    def test_absorbed_rmsnorm_is_verified_and_not_in_changes(self):
        self.assertEqual(set(sr.verify_absorbed(FIXTURES)), set(sr.ABSORBED))
        self.assertTrue(set(sr.ABSORBED).isdisjoint(sr.CHANGES))

    def test_rejects_mismatched_absorbed_rmsnorm(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "wheel"
            shutil.copytree(FIXTURES, root)
            (root / sr.ABSORBED[0]).write_bytes(b"unpatched norm")
            with self.assertRaisesRegex(ValueError, "absorbed RMSNorm source differs"):
                sr.locked_changes(root)

    def test_historical_generator_is_hash_checked_before_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "seedvr2-c14/source_patches.py"
            path.parent.mkdir()
            path.write_text("raise AssertionError('must not execute')\n")
            with patch.object(sr, "LEGACY_DIR", root):
                with self.assertRaisesRegex(ValueError, "generator differs"):
                    sr._generator("seedvr2-c14/source_patches.py")

    def test_no_generic_source_patches_module_binding(self):
        sentinel = types.ModuleType("source_patches")
        with patch.dict(sys.modules, {"source_patches": sentinel}):
            sr.transform(sr.CHANGES[0], before(sr.CHANGES[0]))
            self.assertIs(sys.modules["source_patches"], sentinel)

    def test_complete_manifest_preserves_other_official_sources(self):
        original = fixture_hashes()
        original["vllm_omni/other_upstream_file.py"] = "1" * 64
        result = sr.expected_sources(original)
        self.assertEqual(result["vllm_omni/other_upstream_file.py"], "1" * 64)
        self.assertEqual(set(result), set(original) | set(sr.ADDITIONS))
        self.assertEqual(result[sr.ABSORBED[0]], original[sr.ABSORBED[0]])

    def test_manifest_rejects_collision_and_wrong_base(self):
        original = fixture_hashes()
        original[sr.ADDITIONS[0]] = "1" * 64
        with self.assertRaisesRegex(ValueError, "overwrite"):
            sr.expected_sources(original)
        original = fixture_hashes()
        original[sr.CHANGES[0]] = "1" * 64
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            sr.expected_sources(original)

    def test_ipc_original_ast_including_failure_atomicity_is_preserved(self):
        name = "vllm_omni/diffusion/ipc.py"
        original = ast.parse(before(name))
        observed = StripObservations().visit(ast.parse(sr.transform(name, before(name))))
        self.assertEqual(ast.dump(original), ast.dump(observed))

    def test_api_original_asts_including_release_fixes_are_preserved(self):
        for name in ("vllm_omni/entrypoints/openai/serving_video.py",
                     "vllm_omni/entrypoints/openai/video_api_utils.py"):
            with self.subTest(source=name):
                original = ast.parse(before(name))
                observed = StripObservations().visit(ast.parse(sr.transform(name, before(name))))
                self.assertEqual(ast.dump(original), ast.dump(observed))

    def test_exact_copy_stream_and_sync_calls_preserved(self):
        name = "vllm_omni/diffusion/ipc.py"
        original, changed = before(name).decode(), sr.transform(name, before(name)).decode()
        for expression in ("cpu.copy_(t, non_blocking=True)", "host_tensor.copy_(source, non_blocking=True)",
                           "tensor.detach().cpu().contiguous()", "d2h_stream.synchronize()",
                           "torch.accelerator.set_stream(d2h_stream)", "torch.accelerator.set_stream(old_stream)"):
            self.assertEqual(changed.count(expression), original.count(expression), expression)

    def test_metrics_helpers_are_final_v2_bytes(self):
        helpers = sr.additions()
        self.assertEqual(set(helpers), set(sr.ADDITIONS))
        for name, data in helpers.items():
            self.assertEqual(sr.digest(data), sr.lock()["additions"][name])
        for name in (sr.PREFIX + "sr_telemetry.py", sr.PREFIX + "long_video_support.py"):
            self.assertIn(b'sr-timing-memory/2', helpers[name])

    def test_c14_capture_stays_disabled_without_operator_profile(self):
        module = types.ModuleType("_omni032_test_capture")
        exec(compile(sr.additions()[sr.PREFIX + "c14_capture.py"], "capture-fixture", "exec"), module.__dict__)
        env = {name: value for name, value in os.environ.items() if name != "SEEDVR2_C14_PROFILE"}
        with patch.dict(os.environ, env, clear=True):
            self.assertIsNone(module.profile())

    def test_window_and_aspect_contracts_do_not_silently_fallback(self):
        module = types.ModuleType("_omni032_test_long_support")
        exec(compile(sr.additions()[sr.PREFIX + "long_video_support.py"], "support-fixture", "exec"), module.__dict__)
        for window in (33, 65):
            self.assertEqual(module.select_window(65, window), window)
        with self.assertRaisesRegex(ValueError, "no automatic fallback"):
            module.select_window(33, 65)
        for delivery in ("1800x1080", "2400x1440"):
            settings = module.options("single_encode", delivery, 33)
            module.validate_source(settings, 1680, 1008, 1840, 1104)
        settings = module.options("single_encode", "1920x1080", 33)
        with self.assertRaisesRegex(ValueError, "exact source aspect ratio"):
            module.validate_source(settings, 1680, 1008, 1840, 1104)


if __name__ == "__main__":
    unittest.main()
