# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import ast
import importlib.util
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import pipeline_patch


def parent_bytes():
    c14 = HERE.parent / "seedvr2-c14"
    spec = importlib.util.spec_from_file_location("_test_sr_parent_patch", c14 / "source_patches.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    relative = "vllm_omni/diffusion/models/seedvr2/pipeline_seedvr2.py"
    return module.transform(relative, (c14 / "vendor/original" / relative).read_bytes())


class StripObservations(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if any(item.name == "sr_telemetry" for item in node.names):
            return None
        return node

    def visit_With(self, node):
        self.generic_visit(node)
        expression = node.items[0].context_expr
        if isinstance(expression, ast.Call) and isinstance(expression.func, ast.Attribute):
            if isinstance(expression.func.value, ast.Name) and expression.func.value.id in ("observation", "_sr_metrics"):
                return node.body
        return node

    def visit_Expr(self, node):
        call = node.value
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
            if isinstance(call.func.value, ast.Name) and call.func.value.id == "_sr_metrics":
                return None
        return node


class PipelinePatchTests(unittest.TestCase):
    def test_observation_removal_restores_entire_original_ast(self):
        before = parent_bytes()
        after = pipeline_patch.transform(before)
        stripped = StripObservations().visit(ast.parse(after))
        self.assertEqual(ast.dump(ast.parse(before), include_attributes=False), ast.dump(stripped, include_attributes=False))

    def test_requires_c14_parent_and_rejects_repeat(self):
        before = parent_bytes()
        with self.assertRaises(ValueError):
            pipeline_patch.transform(pipeline_patch.transform(before))
        with self.assertRaises(ValueError):
            pipeline_patch.transform(before.replace(b"from vllm_omni.diffusion.models.seedvr2 import c14_capture as _c14\n", b""))

    def test_all_contract_stages_once(self):
        source = pipeline_patch.transform(parent_bytes()).decode()
        for stage in ("prepare_video_gpu", "vae_encode", "latent_prepare", "dit", "solver", "vae_decode", "color_uint8", "device_to_host"):
            self.assertEqual(source.count(f'observation.stage("{stage}")'), 1)
        self.assertNotIn("torch.cuda.synchronize", source)
        self.assertEqual(source.count(".cpu().numpy()"), parent_bytes().decode().count(".cpu().numpy()"))


if __name__ == "__main__":
    unittest.main()
