# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_sr_submission", HERE / "ops.py")
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)


class Submission(unittest.TestCase):
    def test_only_own_sources_and_workflow_permitted(self):
        self.assertTrue(ops.permitted("container/seedvr2-metrics-repair/runtime.py"))
        self.assertTrue(ops.permitted(".github/workflows/dingo-router-ci.yml"))
        for name in ("container/h3-hsdp-offload/runtime.py", "pyproject.toml", "README.md", "delivery/h3-experimental/ops.py"):
            self.assertFalse(ops.permitted(name))
        with self.assertRaises(ValueError):
            ops.check_scope(["pyproject.toml"])

    def test_marker_and_github_skip_semantics(self):
        self.assertTrue(ops.triggers_sr("build [seedvr2-metrics-repair]"))
        for suffix in ("[skip ci]", "[actions skip]", "\nskip-checks: true"):
            self.assertFalse(ops.triggers_sr("build [seedvr2-metrics-repair] " + suffix))
        self.assertFalse(ops.triggers_sr("build [h3-hsdp-offload]"))

    def test_fixed_parent_and_existing_branch(self):
        lock = json.loads((HERE / "source-lock.json").read_text())
        self.assertEqual(lock["remote_branch"], "Dingorouter-H3-vllm-omni-jirx")
        self.assertTrue(lock["parent_image"].endswith("9547a5e924383dcaba4cdb8cc4fc9c5b5b3b80c24a5290edbd64e4cf96992d69"))
        self.assertEqual(lock["recipe_base_commit"], lock["working_tree_base_commit"])

    def test_missing_existing_remote_cannot_create_branch(self):
        with patch.object(ops, "check", return_value=(ops.load_lock(), "t")), patch.object(ops.LEGACY, "remote_head", return_value=None), patch.object(ops, "git") as git:
            with self.assertRaisesRegex(ValueError, "no new branch"):
                ops.submit(True)
            git.assert_not_called()

    def test_submission_requires_explicit_entrypoint(self):
        with self.assertRaises(ValueError):
            ops.submit(False)


if __name__ == "__main__":
    unittest.main()
