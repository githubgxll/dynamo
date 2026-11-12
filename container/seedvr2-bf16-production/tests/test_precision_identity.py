"""The one-time receipt uses local communicator queries, never collectives."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


def load_runtime():
    path = Path(__file__).resolve().parents[1] / "core/bf16_runtime.py"
    spec = importlib.util.spec_from_file_location("precision_identity_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PrecisionIdentity(unittest.TestCase):
    def test_records_global_and_sp_identity_without_collective(self):
        runtime = load_runtime()
        group = object()
        dist = SimpleNamespace(is_available=lambda: True, is_initialized=lambda: True,
                               get_rank=lambda g=None: 3 if g is None else 2,
                               get_world_size=lambda g: 8, all_reduce=Mock())
        pipeline = SimpleNamespace(_srbf16_group=group,
                    od_config=SimpleNamespace(parallel_config=SimpleNamespace(sequence_parallel_size=8)))
        with patch.dict(sys.modules, {"torch": SimpleNamespace(distributed=dist)}):
            result = runtime._rank_identity(pipeline)
        self.assertEqual((result["global_rank"], result["sp_rank"], result["sp_world_size"]), (3, 2, 8))
        self.assertEqual(result["configured_sp_size"], 8)
        self.assertGreater(result["pid"], 0)
        dist.all_reduce.assert_not_called()

    def test_uninitialized_cpu_process_reports_no_rank(self):
        runtime = load_runtime()
        pipeline = SimpleNamespace(od_config=SimpleNamespace(
                    parallel_config=SimpleNamespace(sequence_parallel_size=8)))
        dist = SimpleNamespace(is_available=lambda: True, is_initialized=lambda: False)
        with patch.dict(sys.modules, {"torch": SimpleNamespace(distributed=dist)}):
            result = runtime._rank_identity(pipeline)
        self.assertIsNone(result["global_rank"])
        self.assertIsNone(result["sp_world_size"])

    def test_metadata_failure_is_reported_without_raising(self):
        runtime = load_runtime()
        result = runtime._rank_identity(SimpleNamespace())
        self.assertEqual(result["rank_metadata_error"], "AttributeError")


if __name__ == "__main__":
    unittest.main()
