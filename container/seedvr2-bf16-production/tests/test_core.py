"""Source/protocol checks requiring only Python's standard library."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
CORE = HERE.parent / "core"


def load(name):
    spec = importlib.util.spec_from_file_location("production_test_" + name, CORE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SourcePatchTests(unittest.TestCase):
    def setUp(self):
        self.module = load("source_patch")
        self.source = (HERE / "fixtures" / "pipeline_seedvr2.py").read_bytes()

    def test_exact_three_byte_splices_and_ast(self):
        result, proof = self.module.apply_source(self.source)
        self.assertEqual(hashlib.sha256(self.source).hexdigest(), self.module.PARENT_PIPELINE_SHA256)
        self.assertTrue(proof["compile_and_ast_verified"])
        self.assertTrue(proof["non_anchor_bytes_unchanged"])
        self.assertEqual(len(proof["changes"]), 3)
        self.assertEqual(set(proof["coverage"].values()), {1})
        # Independently reverse only the inserted spans, recovering every byte.
        restored = result
        for _name, old, new in reversed(self.module._PATCHES):
            self.assertEqual(restored.count(new), 1)
            restored = restored.replace(new, old, 1)
        self.assertEqual(restored, self.source)
        tree = ast.parse(result)
        pipeline = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                        and node.name == "SeedVR2Pipeline")
        forward = next(node for node in pipeline.body if isinstance(node, ast.FunctionDef)
                       and node.name == "forward")
        self.assertEqual(ast.unparse(forward.body[0]), "_sr_bf16.configure_once(self)")
        self.assertEqual(proof["patched_sha256"], hashlib.sha256(result).hexdigest())
        json.dumps(proof, allow_nan=False)

    def test_drift_duplicate_reapply_and_wrong_type_fail_closed(self):
        for changed in (self.source + b"\n", self.source.replace(b"velocity =", b"velocity  =", 1),
                        self.source + self.module._IMPORT):
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                self.module.apply_source(changed)
        patched, _proof = self.module.apply_source(self.source)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.module.apply_source(patched)
        with self.assertRaises(TypeError):
            self.module.apply_source(self.source.decode())


class FlagTests(unittest.TestCase):
    def setUp(self):
        self.runtime = load("bf16_runtime")

    def test_default_on_frozen_first_read_and_copy(self):
        with patch.dict(os.environ, {}, clear=True):
            first = self.runtime.configuration()
            self.assertTrue(first["enabled"])
            first["enabled"] = False
            os.environ["SEEDVR2_DIT_BF16"] = "0"
            self.assertTrue(self.runtime.configuration()["enabled"])

    def test_disabled_exact_original_call_and_identity_no_torch(self):
        sentinel = object()
        calls = []
        pipeline = SimpleNamespace(transformer=lambda *args, **kwargs:
                                   calls.append((args, kwargs)) or SimpleNamespace(vid_sample=sentinel))
        with patch.dict(os.environ, {"SEEDVR2_DIT_BF16": "0"}), \
                patch.object(self.runtime.importlib, "import_module", side_effect=AssertionError("torch imported")):
            self.runtime.configure_once(pipeline)
            self.assertIs(self.runtime.run_transformer(pipeline, 7, example=8), sentinel)
            self.assertIs(self.runtime.velocity_to_fp16(pipeline, sentinel), sentinel)
        self.assertEqual(calls, [((7,), {"example": 8})])
        self.assertFalse(hasattr(pipeline, "_srbf16_configured"))

    def test_invalid_initial_flag(self):
        for flag in ("true", "", "01", " 1", "2"):
            self.runtime._FROZEN = None
            with patch.dict(os.environ, {"SEEDVR2_DIT_BF16": flag}), self.assertRaises(ValueError):
                self.runtime.configuration()

    def test_precision_report_is_cached_metadata_and_return_copy(self):
        class TensorMetadata:
            dtype = "bf16"

            def is_floating_point(self):
                return True

            def numel(self):
                return 9

        class Model:
            def __init__(self):
                self.reads = 0

            def named_parameters(self):
                self.reads += 1
                return [("weight", TensorMetadata())]

            def named_buffers(self):
                self.reads += 1
                return []

        model, vae = Model(), Model()
        pipeline = SimpleNamespace(transformer=model, vae=vae,
                                   text=SimpleNamespace(dtype="bf16"), _srbf16_configured=True)
        first = self.runtime.precision_report(pipeline)
        first["dit_parameters"]["bf16"]["numel"] = 0
        second = self.runtime.precision_report(pipeline)
        self.assertEqual(second["dit_parameters"]["bf16"]["numel"], 9)
        self.assertEqual((model.reads, vae.reads), (2, 1))
        self.assertEqual(second["gpu_validation"], "CONFIGURATION_ONLY_NOT_INFERENCE_ACCEPTANCE")


class FlagTensor:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeTorch:
    float16 = "fp16"
    bfloat16 = "bf16"
    int32 = "int32"

    def __init__(self, remote_failures=()):
        self.reductions = []
        self.remote_failures = iter(remote_failures)
        self.distributed = SimpleNamespace(ReduceOp=SimpleNamespace(MAX="MAX"), all_reduce=self.reduce)
        self.cuda = SimpleNamespace(is_bf16_supported=lambda: True)

    def tensor(self, values, **kwargs):
        return FlagTensor(values[0])

    def reduce(self, flag, *, op, group):
        self.reductions.append((flag.value, op, group))
        flag.value = max(flag.value, next(self.remote_failures, 0))

    def finfo(self, dtype):
        assert dtype == self.float16
        return SimpleNamespace(max=65504.0)


class Velocity:
    def __init__(self, dtype="bf16", *, finite=True, maximum=1.0, converted_finite=True):
        self.dtype = dtype
        self.finite = finite
        self.maximum = maximum
        self.converted_finite = converted_finite
        self.casts = []

    def to(self, *, dtype):
        self.casts.append(dtype)
        return Velocity(dtype, finite=self.converted_finite, maximum=self.maximum)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.runtime = load("bf16_runtime")
        self.runtime._FROZEN = True
        self.pipeline = SimpleNamespace(device=SimpleNamespace(type="cuda"),
                                        _srbf16_configured=True, _srbf16_group="sp-group")

    @staticmethod
    def stats(value, _torch):
        return {"all_finite": value.finite, "finite_absmax": value.maximum if value.finite else None}

    def bridge(self, value, torch):
        with patch.object(self.runtime.importlib, "import_module", return_value=torch), \
                patch.object(self.runtime, "_numeric_stats", side_effect=self.stats):
            return self.runtime.velocity_to_fp16(self.pipeline, value)

    def test_velocity_two_numerical_gates_only_then_fp16(self):
        torch, value = FakeTorch(), Velocity(maximum=65280.0)
        result = self.bridge(value, torch)
        self.assertEqual(result.dtype, "fp16")
        self.assertEqual(value.casts, ["fp16"])
        self.assertEqual(torch.reductions, [(0, "MAX", "sp-group"), (0, "MAX", "sp-group")])

    def test_native_failure_collective_precedes_raise_and_prevents_cast(self):
        for value in (Velocity(dtype="fp16"), Velocity(finite=False), Velocity(maximum=65536.0)):
            torch = FakeTorch()
            with self.assertRaisesRegex(RuntimeError, "velocity_native_bf16"):
                self.bridge(value, torch)
            self.assertEqual(torch.reductions, [(1, "MAX", "sp-group")])
            self.assertEqual(value.casts, [])

    def test_remote_native_failure_prevents_local_cast(self):
        torch, value = FakeTorch([1]), Velocity()
        with self.assertRaisesRegex(RuntimeError, "velocity_native_bf16"):
            self.bridge(value, torch)
        self.assertEqual(torch.reductions, [(0, "MAX", "sp-group")])
        self.assertEqual(value.casts, [])

    def test_local_or_remote_cast_failure_reaches_second_gate(self):
        for local_good, remote_bad in ((False, 0), (True, 1)):
            torch = FakeTorch([0, remote_bad])
            value = Velocity(converted_finite=local_good)
            with self.assertRaisesRegex(RuntimeError, "velocity_fp16_bridge"):
                self.bridge(value, torch)
            self.assertEqual(value.casts, ["fp16"])
            self.assertEqual(len(torch.reductions), 2)
            self.assertEqual(torch.reductions[1][0], int(not local_good))

    def test_configuration_two_gates_and_log_failure_cannot_add_collective(self):
        pipeline = SimpleNamespace(device="cuda:0", transformer=object(), vae=object(),
                                   text=SimpleNamespace(dtype="fp16"))
        torch = FakeTorch()
        logger = SimpleNamespace(info=lambda *args: (_ for _ in ()).throw(OSError("log unavailable")))
        converted_text = SimpleNamespace(dtype="bf16")
        with patch.object(self.runtime.importlib, "import_module", return_value=torch), \
                patch.object(self.runtime, "_group", return_value="sp-group"), \
                patch.object(self.runtime, "_check_fp_dtype"), \
                patch.object(self.runtime, "_convert_transformer", return_value=(converted_text, ["rope"])) as convert, \
                patch.object(self.runtime, "precision_report", return_value={"configured": True}), \
                patch.object(self.runtime, "_LOGGER", logger):
            self.runtime.configure_once(pipeline)
            self.runtime.configure_once(pipeline)
        self.assertIs(pipeline.text, converted_text)
        self.assertTrue(pipeline._srbf16_configured)
        self.assertEqual(pipeline._srbf16_rope_names, ("rope",))
        self.assertEqual(convert.call_count, 1)
        self.assertEqual(torch.reductions, [(0, "MAX", "sp-group"), (0, "MAX", "sp-group")])

    def test_failed_configuration_does_not_mark_configured(self):
        for stage in ("configuration_precheck", "configuration_postcheck"):
            pipeline = SimpleNamespace(device="cuda:0", transformer=object(), vae=object(),
                                       text=SimpleNamespace(dtype="fp16"))
            torch = FakeTorch()
            with patch.object(self.runtime.importlib, "import_module", return_value=torch), \
                    patch.object(self.runtime, "_group", return_value="sp-group"), \
                    patch.object(self.runtime, "_check_fp_dtype",
                                 side_effect=ValueError("bad parameter") if stage.endswith("precheck") else None), \
                    patch.object(self.runtime, "_convert_transformer", side_effect=ValueError("bad conversion")), \
                    self.assertRaisesRegex(RuntimeError, stage):
                self.runtime.configure_once(pipeline)
            self.assertFalse(getattr(pipeline, "_srbf16_configured", False))
            self.assertEqual(len(torch.reductions), 1 if stage.endswith("precheck") else 2)

    def test_group_checks_actual_configured_sp_size(self):
        pipeline = SimpleNamespace(od_config=SimpleNamespace(
            parallel_config=SimpleNamespace(sequence_parallel_size=4)))
        torch = FakeTorch()
        torch.distributed.is_initialized = lambda: False
        with self.assertRaisesRegex(RuntimeError, "initialized SP group"):
            self.runtime._group(pipeline, torch)
        pipeline.od_config.parallel_config.sequence_parallel_size = 1
        self.assertIsNone(self.runtime._group(pipeline, torch))


class StaticBoundaryTests(unittest.TestCase):
    def test_no_probe_hooks_io_rng_or_value_repair(self):
        tree = ast.parse((CORE / "bf16_runtime.py").read_bytes())
        imports, called = set(), set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module or "")
            elif isinstance(node, ast.Call):
                called.add(ast.unparse(node.func))
        self.assertFalse(any("probe" in name or "sitecustomize" in name or "precision_contract" in name
                             for name in imports))
        self.assertFalse(any(name.split(".")[-1] in {
            "clamp", "clamp_", "nan_to_num", "nan_to_num_", "manual_seed", "randn", "rand",
            "seed", "set_rng_state", "open", "write_text", "write_bytes", "mkdir"
        } for name in called))
        self.assertNotIn("set_default_dtype", " ".join(called))


if __name__ == "__main__":
    unittest.main()
