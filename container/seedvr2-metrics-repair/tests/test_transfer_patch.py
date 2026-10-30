# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import ast
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import transfer_patch


def original():
    local = HERE / "vendor/original/vllm_omni/diffusion/ipc.py"
    fallback = Path("D:/AI/_workspace/H3_Omni031rc1_Research_20261012/wheel031/vllm_omni/diffusion/ipc.py")
    return (local if local.is_file() else fallback).read_bytes()


class StripObservation(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        return None if any(item.name == "sr_telemetry" for item in node.names) else node

    def visit_With(self, node):
        self.generic_visit(node)
        expr = node.items[0].context_expr
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and isinstance(expr.func.value, ast.Name) and expr.func.value.id == "_sr_metrics":
            return node.body
        return node


class TransferPatchTests(unittest.TestCase):
    def test_stripping_observations_recovers_entire_original_ast(self):
        before = original()
        after = transfer_patch.transform(before)
        self.assertEqual(ast.dump(ast.parse(before), include_attributes=False), ast.dump(StripObservation().visit(ast.parse(after)), include_attributes=False))

    def test_exact_original_copy_and_stream_calls_preserved(self):
        before = original().decode()
        after = transfer_patch.transform(original()).decode()
        for expression in ("cpu.copy_(t, non_blocking=True)", "host_tensor.copy_(source, non_blocking=True)", "tensor.detach().cpu().contiguous()", "d2h_stream.synchronize()", "torch.accelerator.set_stream(d2h_stream)", "torch.accelerator.set_stream(old_stream)"):
            self.assertEqual(after.count(expression), before.count(expression), expression)

    def test_repeat_application_rejected(self):
        with self.assertRaises(ValueError):
            transfer_patch.transform(transfer_patch.transform(original()))


if __name__ == "__main__":
    unittest.main()
