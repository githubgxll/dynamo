"""Regress diagnostics and the copied-source projection boundary."""
import argparse
import ast
import hashlib
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("worker_runtime", Path(__file__).resolve().parents[1] / "runtime.py")
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


class RuntimeDiagnostics(unittest.TestCase):
    def cli_fixture(self, changed=None, convert=None, tracking=False):
        namespace = SimpleNamespace(num_gpus=4, tensor_parallel_size=4, text_encoder_tp_size=4,
                                    ulysses_degree=1, diffusion_offload_config=None, additional_config={},
                                    enable_multithread_weight_load=False, tokenizer=None,
                                    model_tag="/opt/models/minimax-h3-image", model="Qwen/default")
        for name, value in (changed or {}).items():
            setattr(namespace, name, value)
        if tracking:
            path = Path(__file__).parent / "fixtures/tracking_parser.py"
            source = path.read_bytes()
            self.assertEqual(hashlib.sha256(source).hexdigest(),
                             "a47ff3fea71fd64905d06c1a7f633ed2e21a180013c3732c64a3dd87f3943120")
            module = ast.parse(source)
            actual_class = next(item for item in module.body if isinstance(item, ast.ClassDef)
                                and item.name == "TrackingNamespace")
            scope = {"argparse": argparse, "Any": Any}
            exec(compile(ast.Module(body=[actual_class], type_ignores=[]), str(path), "exec"), scope)
            namespace = scope["TrackingNamespace"](argparse.Namespace(**vars(namespace)),
                                                   frozenset({"model_tag", "num_gpus"}))
        parser = SimpleNamespace(add_subparsers=lambda **kwargs: object(), parse_args=lambda argv: namespace)
        command = SimpleNamespace(name="serve", subparser_init=lambda subcommands: None)
        serve = SimpleNamespace(_ensure_vllm_platform=lambda: None, TrackingArgumentParser=lambda: parser,
                                cmd_init=lambda: [command])
        engine = SimpleNamespace(OmniEngineArgs=SimpleNamespace(from_cli_args=convert or (lambda args: args)))
        profile = {"id": "tp4", "gpus": 4, "dit_tp": 4, "te_tp": 4, "sp": 1, "argv": ["vllm", "serve"]}
        with patch.object(runtime.importlib, "import_module", side_effect=[serve, engine]), \
                patch.object(runtime, "read", return_value={"profiles": [profile], "model_root": "/opt/models/minimax-h3-image"}):
            report = runtime.check_cli()
        return report, namespace

    def test_cli_uses_the_actual_store_false_destination(self):
        self.assertEqual(self.cli_fixture()[0][0]["status"], "PASS")

    def test_cli_checks_positional_model_and_uses_a_temporary_local_fixture(self):
        observed = []
        def convert(args):
            self.assertEqual(args.model_tag, "/opt/models/minimax-h3-image")
            self.assertTrue(Path(args.model).is_dir())
            self.assertEqual(list(Path(args.model).iterdir()), [])
            observed.append(args.model)
            return args
        _, original = self.cli_fixture(convert=convert)
        self.assertEqual(original.model, original.model_tag)
        self.assertFalse(Path(observed[0]).exists())

    def test_cli_rejects_wrong_model_before_dataclass_construction(self):
        def unexpected(args):
            self.fail("Wrong model must not reach dataclass construction")
        with self.assertRaisesRegex(ValueError, "positional model"):
            self.cli_fixture({"model_tag": "remote/unexpected"}, unexpected)

    def test_cli_isolates_the_actual_tracking_namespace_inner_object(self):
        observed = []
        def convert(args):
            self.assertEqual(args.model_tag, "/opt/models/minimax-h3-image")
            self.assertTrue(Path(args.model).is_dir())
            observed.append(args.model)
            return args
        _, original = self.cli_fixture(convert=convert, tracking=True)
        self.assertEqual(original.model, original.model_tag)
        self.assertEqual(original.unfiltered_ns.model, "/opt/models/minimax-h3-image")
        self.assertFalse(Path(observed[0]).exists())

    def test_cli_rejects_remote_tokenizer_resolution(self):
        with self.assertRaisesRegex(ValueError, "must not resolve a tokenizer"):
            self.cli_fixture({"tokenizer": "remote/tokenizer"})

    def test_cli_rejects_dataclass_model_path_changes(self):
        def changed(args):
            args.model = "remote/unexpected"
            return args
        with self.assertRaisesRegex(ValueError, "isolated local model fixture"):
            self.cli_fixture(convert=changed)

    def test_probe_surfaces_actual_child_stderr(self):
        result = subprocess.CompletedProcess([], 1, "partial stdout\n", "actual inventory error\n")
        with patch.object(runtime.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "actual inventory error") as caught:
                runtime.probe()
        self.assertIn("partial stdout", str(caught.exception))
        self.assertIn("exit 1", str(caught.exception))

    def test_probe_reads_successful_json(self):
        result = subprocess.CompletedProcess([], 0, '{"source_sha256": {"known.py": "value"}}', "")
        with patch.object(runtime.subprocess, "run", return_value=result):
            self.assertEqual(runtime.probe()["source_sha256"], {"known.py": "value"})

    def test_snapshot_accepts_complete_installed_subset_of_copy(self):
        inventory = {"source_sha256": {"installed.py": "a"}}
        with patch.object(runtime, "probe", return_value=inventory), \
                patch.object(runtime, "expected_sources", return_value={"installed.py": "a", "not_packaged.py": "b"}), \
                patch.object(runtime, "dump") as write:
            self.assertEqual(runtime.snapshot()["files"], 1)
        write.assert_called_once_with(runtime.BUILD / "parent-inventory.json", inventory)

    def test_snapshot_rejects_changed_or_unknown_installed_source(self):
        for installed in ({"installed.py": "changed"}, {"unknown.py": "a"}):
            with self.subTest(installed=installed), \
                    patch.object(runtime, "probe", return_value={"source_sha256": installed}), \
                    patch.object(runtime, "expected_sources", return_value={"installed.py": "a"}), \
                    patch.object(runtime, "dump") as write:
                with self.assertRaises(ValueError):
                    runtime.snapshot()
                write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
