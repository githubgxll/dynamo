"""Real backend subprocess path regressions with explicit CPU import doubles."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from service import launcher, profile


class BackendEntryPath(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.doubles = self.directory / "import-doubles"
        modules = {
            "vllm_omni/__init__.py": "",
            "vllm_omni/entrypoints/__init__.py": "",
            "vllm_omni/entrypoints/cli/__init__.py": "",
            "vllm_omni/entrypoints/cli/main.py": "import cProfile\ndef main():\n    raise AssertionError('Model dispatch forbidden in import check')\n",
            "torch/__init__.py": "",
            "torch/_dynamo.py": "import cProfile\n",
        }
        for name, content in modules.items():
            path = self.doubles / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    def prepare(self, name, role):
        value = profile.load(ROOT / "profiles" / (name + ".json"))
        for stage in ("h3", "sr"):
            for key, item in value[stage]["stage_environment"].items():
                if item.startswith("/work/"):
                    value[stage]["stage_environment"][key] = str(self.directory / stage / key)
        path = self.directory / (name + ".json")
        path.write_text(json.dumps(value), encoding="utf-8")
        environment = profile.environment(value, role, profile_path=path, epoch="path-test",
            root=self.directory, inherited={key: item for key, item in os.environ.items()
                                           if key not in ("PYTHONPATH", "PYTHONSAFEPATH")})
        environment["PYTHONPATH"] += os.pathsep + str(self.doubles)
        report = self.directory / (name + "-" + role + ".json")
        return path, environment, report

    def test_all_four_real_entry_commands_keep_stdlib_profile_in_service_cwd(self):
        for name in ("fl2va-sr", "ref2va-sr"):
            for role in ("h3", "sr"):
                with self.subTest(profile=name, role=role):
                    path, environment, report = self.prepare(name, role)
                    # The command's -P must work without a preconfigured parent.
                    environment.pop("PYTHONSAFEPATH")
                    command = launcher.backend_command(path, role) + ["--check-imports", "--import-report", str(report)]
                    result = subprocess.run(command, env=environment, cwd=ROOT / "service", capture_output=True,
                                            text=True, timeout=20)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    value = json.loads(report.read_text(encoding="utf-8"))
                    self.assertTrue(value["safe_path"])
                    self.assertEqual(value["status"], "PASS_ACTUAL_BACKEND_IMPORTS_ONLY")
                    self.assertEqual(Path(value["service_profile"]).resolve(), (ROOT / "service/profile.py").resolve())
                    self.assertNotEqual(Path(value["stdlib_profile"]).resolve(), (ROOT / "service/profile.py").resolve())

    def test_legacy_file_entry_reproduces_the_exact_cprofile_failure(self):
        path, environment, report = self.prepare("fl2va-sr", "h3")
        environment.pop("PYTHONSAFEPATH")
        command = [sys.executable, "-B", str(ROOT / "service/backend.py"), "--profile", str(path),
                   "--role", "h3", "--check-imports", "--import-report", str(report)]
        result = subprocess.run(command, env=environment, cwd=self.directory, capture_output=True,
                                text=True, timeout=20)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("module 'profile' has no attribute 'run'", result.stderr)
        self.assertFalse(report.exists())

    def test_environment_forces_safe_path_for_all_owned_roles(self):
        for name in ("fl2va-sr", "ref2va-sr"):
            path, environment, _ = self.prepare(name, "h3")
            self.assertEqual(environment["PYTHONSAFEPATH"], "1")
            value = profile.load(path)
            env = profile.gateway_environment(value, profile_path=path, epoch="safe", root=self.directory,
                inherited={"PYTHONSAFEPATH": "0", "PYTHONPATH": "untrusted"})
            self.assertEqual(env["PYTHONSAFEPATH"], "1")
            self.assertEqual(env["PYTHONPATH"], str(ROOT))


class BF16StartupAssets(unittest.TestCase):
    def test_baked_profiles_reject_quantization_and_keep_bf16(self):
        for name in ("fl2va-sr", "ref2va-sr"):
            value = profile.load(ROOT / "profiles" / (name + ".json"))
            self.assertIsNone(value["h3"]["quantization_config"])
            corrupted = copy.deepcopy(value)
            corrupted["h3"]["native_argv"] += ["--diffusion-quantization-config", '{"method":"fp8"}']
            with self.assertRaises(ValueError):
                profile.validate(corrupted)

    def test_startup_sources_match_frozen_handoff_hashes(self):
        startup = ROOT.parent / "h3-sr-startup"
        if not startup.is_dir():
            startup = Path("/opt/h3-sr-startup")
        lock = json.loads((startup / "SOURCE-MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual(len(lock["files"]), 9)
        for name, digest in lock["files"].items():
            self.assertEqual(hashlib.sha256((startup / name).read_bytes()).hexdigest(), digest, name)


if __name__ == "__main__":
    unittest.main()
