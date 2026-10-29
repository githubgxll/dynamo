"""Patch publication safeguards; generated tests exercise the real experiment."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


TOOL = Path(__file__).resolve().parents[1] / "integrate_experiment.py"
spec = importlib.util.spec_from_file_location("integrate_experiment", TOOL)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

RESIDENT = '''def validate_config(config):
    required = set()
    allowed = required | {"sr_window_frames", "sr_frame_pixels", "tiers"}
    require(type(config["sr_capable"]) is bool, "sr_capable must be boolean")
    return config

def environment(work, stage, config):
    env = {}
    if stage == "sr":
        env.update(VLLM_OMNI_SEEDVR2_SHARDED_FRAME_PIXELS="1")
    return env

def build_h3(config):
    return ["unchanged-h3"]

def build_sr(config):
    return ["unchanged-sr"]
'''
WORKFLOW = '''def prepare(args):
    run_id = identifier(args.id)
    identifier(args.namespace, 63)
    runtime = {}
    payload['config.json'] = json.dumps(runtime, indent=2)
    launch = {}
    save(directory / 'launch-argv.json', launch)

def build_parser():
    p = parser
    p.add_argument('--sr-capable', action='store_true', help='Mount and CPU-convert SR weights; still load SR only on explicit sr-start')

def load_run(run):
    return "unchanged-run-validation"

def environment_contract(value):
    return "unchanged-env-validation"

def validate_pod_spec(value):
    return "unchanged-admission-validation"
'''


class IntegrationSafeguardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        for name, content in {"resident.py": RESIDENT, "workflow.py": WORKFLOW,
                              "image_binding.py": "# unchanged identity\n",
                              "image-repair-lock.json": '{"image": "existing-baseline"}',
                              "console_output.py": "# local fixture\n"}.items():
            (self.source / name).write_text(content, encoding="utf-8")
        (self.source / "runs").mkdir()
        (self.source / "runs" / "frozen.json").write_text("frozen baseline", encoding="utf-8")
        self.output = self.root / "output"

    def tearDown(self):
        self.temp.cleanup()

    def test_precise_patch_refuses_ambiguous_or_already_changed_sources(self):
        changed = module.transform(RESIDENT, WORKFLOW)
        with self.assertRaises(ValueError):
            module.transform(changed["resident.py"], changed["workflow.py"])
        with self.assertRaises(ValueError):
            module.transform(RESIDENT + RESIDENT, WORKFLOW)

    def test_locked_contract_changes_are_rejected(self):
        changed = module.transform(RESIDENT, WORKFLOW)
        changed["resident.py"] = changed["resident.py"].replace("unchanged-h3", "different-h3")
        with self.assertRaisesRegex(ValueError, "locked contract"):
            module.verify_unchanged_contracts({"resident.py": RESIDENT, "workflow.py": WORKFLOW}, changed)

    def test_existing_output_and_nested_output_are_rejected(self):
        self.output.mkdir()
        (self.output / "runs").mkdir()
        with self.assertRaises(ValueError):
            module.generate(self.source, self.output)
        with self.assertRaises(ValueError):
            module.generate(self.source, self.source / "candidate")

    def test_publishes_only_non_runnable_patch_and_keeps_source(self):
        before = {path.relative_to(self.source): path.read_bytes() for path in self.source.rglob("*") if path.is_file()}
        with patch.object(module.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="7 tests OK")) as run:
            report = module.generate(self.source, self.output)
        self.assertEqual(report["status"], "LOCAL_PATCH_TESTED_NOT_DEPLOYABLE")
        self.assertFalse(report["new_image_bound"])
        self.assertEqual(set(path.name for path in self.output.iterdir()),
                         {"experiment.patch", "LOCAL-VALIDATION.log", "integration.json", "README.md"})
        self.assertEqual(before, {path.relative_to(self.source): path.read_bytes() for path in self.source.rglob("*") if path.is_file()})
        diff = (self.output / "experiment.patch").read_text(encoding="utf-8")
        self.assertIn("--- /dev/null\n+++ b/tests/test_offload_controls.py", diff)
        self.assertIn("--offload-fastpath", diff)
        self.assertNotIn("existing-baseline", diff)
        self.assertIn("unittest", run.call_args.args[0])
        self.assertEqual(run.call_args.kwargs["encoding"], "utf-8")
        self.assertEqual(json.loads((self.output / "integration.json").read_text())["cluster_calls"], 0)

    def test_failed_local_validation_leaves_no_output(self):
        with patch.object(module.subprocess, "run", return_value=SimpleNamespace(returncode=1, stdout="", stderr="failed")):
            with self.assertRaisesRegex(RuntimeError, "no output published"):
                module.generate(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_concurrent_source_change_prevents_publication(self):
        def mutate(*args, **kwargs):
            (self.source / "resident.py").write_text(RESIDENT + "\n# concurrent change\n", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="passed")
        with patch.object(module.subprocess, "run", side_effect=mutate):
            with self.assertRaisesRegex(RuntimeError, "Source changed"):
                module.generate(self.source, self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
