"""Regress diagnostics and the copied-source projection boundary."""
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("worker_runtime", Path(__file__).resolve().parents[1] / "runtime.py")
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


class RuntimeDiagnostics(unittest.TestCase):
    def test_cli_uses_the_actual_store_false_destination(self):
        namespace = SimpleNamespace(num_gpus=4, tensor_parallel_size=4, text_encoder_tp_size=4,
                                    ulysses_degree=1, diffusion_offload_config=None, additional_config={},
                                    enable_multithread_weight_load=False)
        parser = SimpleNamespace(add_subparsers=lambda **kwargs: object(), parse_args=lambda argv: namespace)
        command = SimpleNamespace(name="serve", subparser_init=lambda subcommands: None)
        serve = SimpleNamespace(_ensure_vllm_platform=lambda: None, TrackingArgumentParser=lambda: parser,
                                cmd_init=lambda: [command])
        engine = SimpleNamespace(OmniEngineArgs=SimpleNamespace(from_cli_args=lambda args: args))
        profile = {"id": "tp4", "gpus": 4, "dit_tp": 4, "te_tp": 4, "sp": 1, "argv": ["vllm", "serve"]}
        with patch.object(runtime.importlib, "import_module", side_effect=[serve, engine]), \
                patch.object(runtime, "read", return_value={"profiles": [profile]}):
            self.assertEqual(runtime.check_cli()[0]["status"], "PASS")

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
