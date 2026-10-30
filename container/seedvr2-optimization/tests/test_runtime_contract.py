# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_sr_runtime_contract_test", HERE / "runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class RuntimeContract(unittest.TestCase):
    def test_runtime_metrics_configuration_does_not_change_source_identity(self):
        parent = SimpleNamespace(contract=lambda: ("c14", "lite", "lock"))
        source = {"parent_image": runtime.PARENT, "parent_source_commit": runtime.PARENT_COMMIT}
        environment = {"H3_HSDP_SOURCE_COMMIT": runtime.PARENT_COMMIT, "SEEDVR2_OPT_PARENT_IMAGE": runtime.PARENT,
            "SEEDVR2_OPT_SOURCE_COMMIT": "a" * 40, "SEEDVR2_METRICS_DIR": "/work/metrics"}
        with patch.object(runtime, "parent_runtime", return_value=parent), patch.object(runtime, "read", return_value=source), patch.dict(runtime.os.environ, environment):
            self.assertEqual(runtime.contract()[1:], ("c14", "lite", "lock"))
            with self.assertRaisesRegex(ValueError, "by default"):
                runtime.contract(require_default_observations=True)


if __name__ == "__main__":
    unittest.main()
