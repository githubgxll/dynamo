"""Local metadata/preservation regressions; no Linux installation or GPU claim."""
from __future__ import annotations

import hashlib
import base64
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from contextlib import ExitStack
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("omni032_environment_tests", HERE / "environment.py")
environment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(environment)
POLICY = environment.read_json(HERE / "environment-policy.json")


def record(name, version="1.0", requires=(), location="/usr/local/lib/python3.12/dist-packages"):
    return {"name": name, "canonical_name": name.lower().replace("_", "-"), "version": version,
            "requires_dist": list(requires), "location": location, "metadata_path": location + "/" + name + ".dist-info"}


class SolveInputs(unittest.TestCase):
    def baseline(self):
        names = [record("ai-dingo", "1.3.0"), record("ai-dingo-runtime", "1.3.0"), record("gpu-memory-service", "0.9.0"),
                 record("vllm", "0.31.0", ["tokenizers>=0.21.1"]), record("vllm-omni", "0.31.0rc1", ["transformers<5.15"]),
                 record("torch", "2.13.0+cu130"), record("nvidia-nccl-cu13", "2.30.7"),
                 record("nvidia-cutlass-dsl", "4.7.1"), record("nvidia-cutlass-dsl-libs-cu12", "4.7.1"),
                 record("transformers", "5.14.1"), record("tokenizers", "0.22.2"),
                 record("lmcache", "0.5.5", ["opentelemetry-api<=1.40.0"]), record("opentelemetry-api", "1.40.0"),
                 record("opentelemetry-exporter-prometheus", "0.61b0"),
                 record("python-apt", "2.7.7", location="/usr/lib/python3/dist-packages")]
        return {"distributions": names}

    def test_apt_packages_are_not_pypi_roots(self):
        plan = environment.solve_plan(self.baseline(), POLICY)
        self.assertNotIn("python-apt", plan["roots"])
        self.assertIn("ai-dingo[video-gateway]", plan["roots"])
        self.assertNotIn("ai-dingo[vllm]", plan["roots"])

    def test_same_torch_cuda_and_service_exact_constraints(self):
        plan = environment.solve_plan(self.baseline(), POLICY)
        for line in ("torch==2.13.0+cu130", "nvidia-nccl-cu13==2.30.7", "ai-dingo==1.3.0", "lmcache==0.5.5"):
            self.assertIn(line, plan["constraints"])

    def test_old_framework_api_bounds_are_replaced(self):
        plan = environment.solve_plan(self.baseline(), POLICY)
        self.assertIn("transformers==5.16.1", plan["constraints"])
        self.assertIn("tokenizers==0.23.2", plan["constraints"])
        self.assertIn("vllm==0.32.0", plan["constraints"])
        self.assertFalse(any("<5.15" in line or "==0.31.0" in line for line in plan["constraints"]))

    def test_cutlass_family_upgrade_does_not_drop_required_cu12_split(self):
        plan = environment.solve_plan(self.baseline(), POLICY)
        self.assertNotIn("nvidia-cutlass-dsl", plan["protected_names"])
        for name in ("nvidia-cutlass-dsl", "nvidia-cutlass-dsl-libs-base", "nvidia-cutlass-dsl-libs-core", "nvidia-cutlass-dsl-libs-cu12", "nvidia-cutlass-dsl-libs-cu13"):
            self.assertIn(name + "==4.8.0", plan["constraints"])

    def test_monitoring_capabilities_remain_joint_roots(self):
        plan = environment.solve_plan(self.baseline(), POLICY)
        self.assertIn("opentelemetry-exporter-prometheus", plan["roots"])
        self.assertIn("opentelemetry-api", plan["roots"])

    def test_only_reviewed_unconsumed_internal_split_may_be_pruned(self):
        inventory = self.baseline()
        inventory["distributions"].append(record("opentelemetry-exporter-otlp-common", "0.66b0"))
        plan = environment.solve_plan(inventory, POLICY)
        self.assertEqual(plan["omitted_orphans"], ["opentelemetry-exporter-otlp-common"])
        inventory["distributions"].append(record("consumer", "1.0", ['opentelemetry-exporter-otlp-common; extra == "optional"']))
        plan = environment.solve_plan(inventory, POLICY)
        self.assertEqual(plan["omitted_orphans"], [])
        self.assertIn("opentelemetry-exporter-otlp-common", plan["roots"])

    def test_multiple_effective_wheels_fail(self):
        inventory = self.baseline()
        inventory["distributions"].append(record("tokenizers", "0.23.2"))
        with self.assertRaisesRegex(ValueError, "Multiple effective"):
            environment.solve_plan(inventory, POLICY)

    def test_unknown_system_only_install_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "No inherited"):
            environment.pip_records({"distributions": [record("python-apt", location="/usr/lib/python3/dist-packages")]})

    def test_target_and_preserved_native_sets_do_not_overlap(self):
        self.assertFalse(set(POLICY["target_versions"]) & set(POLICY["preserved_versions"]))

    def test_all_changed_helpers_have_immutable_public_artifact_inputs(self):
        requirements = [environment.Requirement(line) for line in (HERE / "requirements.lock").read_text().splitlines()
                        if line.strip() and not line.startswith("#")]
        self.assertEqual({environment.canonicalize_name(item.name) for item in requirements}, set(POLICY["target_versions"]) - {"vllm", "vllm-omni"})
        for requirement in requirements:
            self.assertTrue(requirement.url.startswith("https://files.pythonhosted.org/packages/"))
            self.assertRegex(requirement.url, r"#sha256=[a-f0-9]{64}$")


class WheelAndPreservation(unittest.TestCase):
    def test_same_directory_then_final_audit_preserves_parent_snapshot(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as patches:
            root = Path(directory)
            inputs, final = root / "inputs", root / "final"
            inputs.mkdir()
            final.mkdir()
            native = root / "native.so"
            native.write_bytes(b"protected original bytes")
            version = "2.13.0+cu130"
            dist = types.SimpleNamespace(files=[Path("native.so")], version=version,
                                         locate_file=lambda entry: root / entry)
            inventory = {"distributions": [record("torch", version)]}
            policy = {**POLICY, "parent_versions": {"torch": version}, "target_versions": {},
                      "preserved_versions": {"torch": version}, "native_names": ["torch"],
                      "native_prefixes": [], "mutable_native_prefixes": [], "root_extras": {},
                      "reviewed_orphans": [], "forbidden_components": [], "wheel_artifacts": {}}
            def report(**kwargs):
                return {"packages": {"torch": version}, "errors": [], "frameworks": {}, "known_metadata_exceptions": []}
            def finalize(value):
                value["status"] = "FAIL" if value["errors"] else "PASS"
            auditor = types.SimpleNamespace(audit_dependencies=report, require_framework_versions=lambda _: None,
                                             finalize_status=finalize)
            for name, replacement in {
                    "legacy_module": lambda _: types.SimpleNamespace(snapshot=lambda **kw: inventory),
                    "dependency_auditor": lambda **kw: auditor,
                    "nccl_evidence": lambda *args: {"verified": True},
                    "verify_provider": lambda _: {}, "core_record_evidence": lambda: {},
                    "helper_artifact_evidence": lambda: {}, "run_uv_check": lambda *args: None}.items():
                patches.enter_context(patch.object(environment, name, replacement))
            patches.enter_context(patch.object(environment.md, "distribution", side_effect={"torch": dist}.__getitem__))
            patches.enter_context(patch.object(environment.platform, "system", return_value="Linux"))
            patches.enter_context(patch.object(environment.platform, "machine", return_value="x86_64"))
            patches.enter_context(patch.object(environment.platform, "python_version_tuple", return_value=("3", "12", "3")))
            patches.enter_context(patch.object(environment.os, "geteuid", return_value=1000, create=True))
            environment.prepare(inputs, policy)
            expected = environment.protected_file_manifest(["torch"])
            snapshot = next(path for path in inputs.glob("*.json") if environment.read_json(path) == expected)
            frozen_bytes = snapshot.read_bytes()
            self.assertEqual(environment.audit(inputs, inputs, policy)["status"], "PASS")
            self.assertEqual(environment.audit(final, inputs, policy)["status"], "PASS")
            self.assertEqual(snapshot.read_bytes(), frozen_bytes)
            self.assertEqual(environment.read_json(inputs / "native-preservation.json")["status"], "PASS")
            # Keeping the original input must still detect genuine native changes.
            native.write_bytes(b"changed binary, same package version")
            with self.assertRaisesRegex(RuntimeError, "environment audit failed"):
                environment.audit(final, inputs, policy)
            self.assertIn("native-preservation", environment.read_json(final / "environment-check.json")["errors"][0])
            self.assertEqual(snapshot.read_bytes(), frozen_bytes)

    def test_audit_receipt_cannot_be_treated_as_parent_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment.write_json(root / "parent-native-manifest.json", {"status": "PASS", "evidence": {}})
            with patch.object(environment, "protected_file_manifest") as lookup:
                with self.assertRaisesRegex(RuntimeError, "Invalid parent native snapshot"):
                    environment.compare_native(root)
                lookup.assert_not_called()

    def wheel_fixture(self, root):
        wheels = root / "wheels"
        wheels.mkdir()
        wheel = wheels / "release.whl"
        wheel.write_bytes(b"test fixture, not a published artifact")
        artifact = {"filename": wheel.name, "size_bytes": wheel.stat().st_size,
                    "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(), "url": "https://files.pythonhosted.org/release.whl"}
        return wheel, artifact

    def test_wheel_tampering_fails_with_same_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            wheel, artifact = self.wheel_fixture(root)
            environment.wheel_evidence(root, artifact)
            wheel.write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "content differs"):
                environment.wheel_evidence(root, artifact)

    def test_absent_direct_url_is_rejected_even_with_right_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _, artifact = self.wheel_fixture(root)
            dist = types.SimpleNamespace(read_text=lambda _: None)
            with self.assertRaisesRegex(RuntimeError, "origin differs"):
                environment.wheel_evidence(root, artifact, dist=dist)

    def test_verified_local_wheel_can_supply_missing_installer_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            wheel, artifact = self.wheel_fixture(root)
            dist = types.SimpleNamespace(read_text=lambda _: json.dumps({"url": wheel.as_uri(), "archive_info": {}}))
            self.assertEqual(environment.wheel_evidence(root, artifact, dist=dist)["sha256"], artifact["sha256"])

    def test_conflicting_installer_hash_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            wheel, artifact = self.wheel_fixture(root)
            dist = types.SimpleNamespace(read_text=lambda _: json.dumps({"url": wheel.as_uri(), "archive_info": {"hashes": {"sha256": "0" * 64}}}))
            with self.assertRaisesRegex(RuntimeError, "conflicting"):
                environment.wheel_evidence(root, artifact, dist=dist)

    def test_unrecorded_service_code_is_preserved_by_actual_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dingo").mkdir()
            unrecorded = root / "dingo" / "sr_worker.py"
            unrecorded.write_text("original")
            meta = root / "METADATA"
            meta.write_text("Name: ai-dingo\nVersion: 1.3.0\n")
            dist = types.SimpleNamespace(files=[Path("METADATA")], version="1.3.0", locate_file=lambda entry: root / entry)
            first = environment.protected_file_manifest(["ai-dingo"], {"ai-dingo": dist})
            unrecorded.write_text("changed")
            second = environment.protected_file_manifest(["ai-dingo"], {"ai-dingo": dist})
            self.assertNotEqual(first, second)

    def test_version_label_cannot_hide_native_byte_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native = root / "native.so"
            native.write_bytes(b"first")
            dist = types.SimpleNamespace(files=[Path("native.so")], version="2.13.0+cu130", locate_file=lambda entry: root / entry)
            first = environment.protected_file_manifest(["torch"], {"torch": dist})
            native.write_bytes(b"other")
            second = environment.protected_file_manifest(["torch"], {"torch": dist})
            self.assertNotEqual(first, second)

    def test_missing_protected_binary_fails(self):
        dist = types.SimpleNamespace(files=[Path("missing.so")], version="1.0", locate_file=lambda entry: Path(tempfile.gettempdir()) / "nonexistent-omni032-test" / entry)
        with self.assertRaisesRegex(RuntimeError, "Protected installed file is missing"):
            environment.protected_file_manifest(["torch"], {"torch": dist})

    def test_core_record_validates_compiled_bytes_and_import_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "vllm").mkdir()
            entries = []
            class Entry:
                def __init__(self, name):
                    self.name = name
                    digest = hashlib.sha256((root / name).read_bytes()).digest()
                    self.hash = types.SimpleNamespace(mode="sha256", value=base64.urlsafe_b64encode(digest).rstrip(b"=").decode())
                def __str__(self):
                    return self.name
                def __fspath__(self):
                    return self.name
            for name, data in (("vllm/__init__.py", b"source"), ("vllm/_C.abi3.so", b"native")):
                (root / name).write_bytes(data)
                entries.append(Entry(name))
            dist = types.SimpleNamespace(files=entries, locate_file=lambda entry: root / str(entry))
            origin = types.SimpleNamespace(origin=str(root / "vllm/__init__.py"))
            self.assertEqual(environment.core_record_evidence(dist, origin)["status"], "PASS")
            (root / "vllm/_C.abi3.so").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "differs from RECORD"):
                environment.core_record_evidence(dist, origin)
            with self.assertRaisesRegex(RuntimeError, "import does not belong"):
                environment.core_record_evidence(dist, types.SimpleNamespace(origin=str(root / "shadow.py")))


class VersionAndToolGates(unittest.TestCase):
    def test_new_framework_pair_is_configured_without_changing_parent_base_provenance(self):
        inherited = environment.dependency_auditor()
        target = environment.dependency_auditor(target=True)
        self.assertEqual(inherited.EXPECTED_FRAMEWORK_VERSIONS["vllm"], "0.31.0")
        self.assertEqual(target.EXPECTED_FRAMEWORK_VERSIONS, {"vllm": "0.32.0", "vllm-omni": "0.32.0"})
        self.assertEqual(target.BASE_DIGEST, inherited.BASE_DIGEST)

    def test_exact_version_mismatch_fails(self):
        with self.assertRaisesRegex(RuntimeError, "torch: expected"):
            environment.require_versions({"torch": "2.13.0"}, {"torch": "2.13.0+cu130"})

    def test_uv_check_always_uses_writable_private_cache_and_same_interpreter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            auditor = types.SimpleNamespace(review_uv_check=lambda report, code, text: report.update(exit=code))
            report = {}
            def run(command, **kwargs):
                self.assertEqual(command, ["uv", "--no-config", "--no-cache", "--offline", "--no-python-downloads", "pip", "check", "--python", environment.sys.executable])
                self.assertFalse(any(name.upper().startswith("UV_") for name in kwargs["env"]))
                cache = Path(kwargs["cwd"])
                self.assertTrue(cache.is_dir())
                self.assertNotEqual(str(cache), "/opt/uv/cache")
                return types.SimpleNamespace(returncode=2, stdout="permission error fixture")
            with patch.object(environment.subprocess, "run", side_effect=run), patch.object(environment.os, "geteuid", return_value=1000, create=True):
                environment.run_uv_check(root, auditor, report)
            self.assertEqual(report["exit"], 2)
            self.assertEqual(environment.read_json(root / "uv-pip-check-context.json")["euid"], 1000)

    def test_installer_preserves_one_joint_solve_without_upgrade_all(self):
        script = (HERE / "install_environment.sh").read_text()
        self.assertNotIn("--upgrade ", script)
        self.assertIn('--requirement "$INFO/roots.in"', script)
        self.assertIn('--constraint "$INFO/constraints.txt"', script)
        self.assertIn("--reinstall-package vllm --reinstall-package vllm-omni", script)
        self.assertIn("--no-deps --only-binary :all: --reinstall-package opencv-python-headless", script)


if __name__ == "__main__":
    unittest.main()
