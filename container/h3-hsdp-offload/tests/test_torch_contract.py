# SPDX-License-Identifier: Apache-2.0
"""Pure-source identity regressions, runnable on both local and image Python."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_native_torch_contract", HERE / "torch_contract.py")
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


class PortableTorchContract(unittest.TestCase):
    def setUp(self):
        self.policy = contract.load_policy()

    def test_all_three_official_excerpts_match_locked_fingerprints(self):
        for entry in self.policy["methods"]:
            with self.subTest(method=entry["class"]):
                raw = (HERE / entry["reference_method_file"]).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), entry["reference_method_sha256"])
                self.assertEqual(contract.method_hash(raw.decode(), method_name=entry["method"]), entry["ast_sha256"])

    def test_class_and_inspect_style_source_match(self):
        for entry in self.policy["methods"]:
            source = (HERE / entry["reference_method_file"]).read_text(encoding="utf-8")
            wrapped = "class " + entry["class"] + ":\n" + textwrap.indent(source, "    ")
            self.assertEqual(contract.method_hash(wrapped, entry["class"], entry["method"]), entry["ast_sha256"])

    def test_comments_newlines_and_indentation_are_not_behavior(self):
        for entry in self.policy["methods"]:
            source = (HERE / entry["reference_method_file"]).read_text(encoding="utf-8")
            changed = textwrap.indent("# fixture comment\n\n" + source, "    ").replace("\n", "\r\n")
            self.assertEqual(contract.method_hash(changed), entry["ast_sha256"])

    def test_changes_to_each_native_method_are_detected(self):
        for entry in self.policy["methods"]:
            tree = ast.parse((HERE / entry["reference_method_file"]).read_text(encoding="utf-8"))
            tree.body[0].body.insert(0, ast.Return(value=ast.Constant(value=None)))
            changed = ast.unparse(ast.fix_missing_locations(tree))
            self.assertNotEqual(contract.method_hash(changed), entry["ast_sha256"])

    def test_signature_branch_call_operator_and_decorator_changes_are_detected(self):
        source = "def f(self, recurse=True):\n    if recurse:\n        self.reshard()\n    return 1 + 2\n"
        for changed in (source.replace("recurse=True", "recurse=False"),
                        source.replace("if recurse:", "if not recurse:"),
                        source.replace("self.reshard()", "self.reset_sharded_param()"),
                        source.replace("1 + 2", "1 - 2"), "@wrapped\n" + source):
            self.assertNotEqual(contract.method_hash(changed), contract.method_hash(source))

    def test_literal_types_remain_distinct(self):
        literals = ["True", "1", "1.0", "1j", "'1'", "b'1'", "None", "..."]
        hashes = {contract.method_hash("def f():\n    return " + literal) for literal in literals}
        self.assertEqual(len(hashes), len(literals))

    def test_empty_and_new_fields_are_not_discarded(self):
        node = ast.parse("def f():\n    pass").body[0]
        encoded = contract.canonical_ast(node)
        fields = dict(encoded[1])
        self.assertEqual(fields["decorator_list"], [])
        self.assertEqual(fields["returns"], ["none"])
        node._fields = (*node._fields, "future_semantic_field")
        node.future_semantic_field = []
        self.assertNotEqual(contract.canonical_ast(node), encoded)

    def test_missing_or_duplicate_definitions_are_rejected(self):
        for source, cls, method in [("x = 1", None, None),
                                    ("def f(): pass\ndef f(): pass", None, "f"),
                                    ("class A:\n def f(self): pass\n def f(self): pass", "A", "f"),
                                    ("class A: pass\nclass A: pass", "A", "f"),
                                    ("class A: pass", "A", "f")]:
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "exactly one"):
                contract.method_hash(source, cls, method)

    def test_unknown_schema_or_algorithm_is_rejected(self):
        for key in ("schema", "fingerprint"):
            invalid = dict(self.policy, **{key: "unknown"})
            with mock.patch.object(Path, "read_text", return_value=json.dumps(invalid)), \
                 self.assertRaisesRegex(ValueError, "schema/fingerprint"):
                contract.load_policy()

    def test_actual_verification_rejects_changed_source(self):
        # Simulated identity inputs only; actual Torch runs in the image gate.
        fake_torch = SimpleNamespace(__version__=self.policy["version"], version=SimpleNamespace(git_version="fixture"))
        fake_module = SimpleNamespace(**{entry["class"]: SimpleNamespace(**{entry["method"]: object()})
                                        for entry in self.policy["methods"]})
        sources = [(HERE / entry["reference_method_file"]).read_text(encoding="utf-8") for entry in self.policy["methods"]]
        with mock.patch.dict(sys.modules, {"torch": fake_torch}), \
             mock.patch.object(contract.importlib, "import_module", return_value=fake_module), \
             mock.patch.object(contract.inspect, "getsource", side_effect=sources):
            self.assertEqual(contract.verify()["status"], "PASS")
        for index in range(len(sources)):
            changed = list(sources)
            changed[index] = "@changed\n" + changed[index]
            with self.subTest(index=index), mock.patch.dict(sys.modules, {"torch": fake_torch}), \
                 mock.patch.object(contract.importlib, "import_module", return_value=fake_module), \
                 mock.patch.object(contract.inspect, "getsource", side_effect=changed), \
                 self.assertRaisesRegex(RuntimeError, "Native Torch method differs"):
                contract.verify()


if __name__ == "__main__":
    unittest.main()
