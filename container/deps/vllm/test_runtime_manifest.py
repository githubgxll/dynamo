"""CPU regression tests for real build failures, provenance and dependency scope.

Synthetic metadata verifies the gate, not the Linux installed solve or GPU ABI.
"""
from __future__ import annotations
import json
import hashlib
import io
import importlib.metadata as md
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path, PurePosixPath
from unittest.mock import patch
from collect_runtime_manifest import (
    BASE_DIGEST, audit_dependencies, main, review_uv_check, upstream_nccl_evidence,
)

LINUX = {"sys_platform": "linux", "platform_system": "Linux", "platform_machine": "x86_64"}
BASE = f"vllm/vllm-openai:v0.30.0-ubuntu2404@{BASE_DIGEST}"
OVERRIDE = "nvidia-nccl-cu13==2.30.7\n"
PROTECTED = "torch==2.13.0+cu130\nnvidia-nccl-cu13==2.30.7\n"
NCCL_REQUIREMENT = 'nvidia-nccl-cu13==2.29.7; platform_system == "Linux"'
KNOWN_UV_OUTPUT = """Using Python 3.12.3 environment at: /usr
Checked 318 packages in 58ms
Found 1 incompatibility
The package `torch` requires `nvidia-nccl-cu13==2.29.7 ; sys_platform == 'linux'`, but `2.30.7` is installed
"""


class Distribution:
    def __init__(self, name, version="1.0", requires=(), requires_python=None, path=None, files=()):
        self.metadata = {"Name": name}
        if requires_python is not None:
            self.metadata["Requires-Python"] = requires_python
        self.version, self.requires, self._path, self.files = version, list(requires), path, list(files)

    def locate_file(self, name):
        return str(PurePosixPath(self._path).parent / name)


def audit(*distributions, evidence=None, **kwargs):
    return audit_dependencies(distributions, python_version="3.12.3", environment=LINUX,
                              nccl_evidence=evidence, **kwargs)


def evidence(**overrides):
    values = {"base_reference": BASE, "overrides_text": OVERRIDE, "protected_text": PROTECTED}
    values.update(overrides)
    return upstream_nccl_evidence(**values)


def official_stack(*, torch="2.13.0+cu130", nccl="2.30.7", requirement=NCCL_REQUIREMENT):
    return [Distribution("vllm", "0.30.0", ["torch==2.13.0"]),
            Distribution("vllm-omni", "0.30.0", ["torch>=2.0"]),
            Distribution("torch", torch, [requirement]), Distribution("nvidia-nccl-cu13", nccl)]


def shadow_pair(name="cryptography"):
    module = {"cryptography": "cryptography/__init__.py", "pyjwt": "jwt/__init__.py",
              "six": "six.py", "oauthlib": "oauthlib/__init__.py"}[name]
    system = Distribution(name, "1.0", path=f"/usr/lib/python3/dist-packages/{name}-1.0.egg-info")
    wheel = Distribution(name, "2.0", path=f"/usr/local/lib/python3.12/dist-packages/{name}-2.0.dist-info", files=[module])
    return system, wheel, wheel.locate_file(module)


class DependencyAuditTests(unittest.TestCase):
    def test_compatible_environment(self):
        self.assertEqual(audit(Distribution("app", requires=["dependency>=1"]), Distribution("dependency"))["status"], "PASS")

    def test_exact_nccl_override_requires_provenance(self):
        report = audit(*official_stack(), evidence=evidence())
        review_uv_check(report, 1, KNOWN_UV_OUTPUT)
        self.assertEqual(report["status"], "PASS_WITH_UPSTREAM_NCCL_OVERRIDE")
        for project in ("vllm", "vllm-omni"):
            scope = report["frameworks"][project]
            self.assertEqual(scope["default_requirements"]["status"], "PASS")
            self.assertEqual(scope["recursive_dependency_closure"]["status"], "PASS_WITH_UPSTREAM_NCCL_OVERRIDE")

    def test_absent_wrong_or_incomplete_provenance_fails(self):
        cases = [None, evidence(base_reference=BASE.replace("vllm/vllm-openai", "untrusted/vllm-openai")),
                 evidence(base_reference="vllm/vllm-openai:v0.30.0-ubuntu2404"),
                 evidence(base_reference=BASE.replace(BASE_DIGEST, "sha256:" + "0" * 64)),
                 evidence(overrides_text=""), evidence(overrides_text=OVERRIDE + OVERRIDE),
                 evidence(overrides_text='nvidia-nccl-cu13==2.30.7; platform_system == "Linux"'),
                 evidence(protected_text="torch==2.13.0+cu130\n"),
                 evidence(protected_text=PROTECTED.replace("2.30.7", "2.29.7"))]
        for source in cases:
            with self.subTest(source=source):
                report = audit(*official_stack(), evidence=source)
                review_uv_check(report, 1, KNOWN_UV_OUTPUT)
                self.assertEqual(report["status"], "FAIL")
                self.assertFalse(report["known_metadata_exceptions"])

    def test_docker_io_normalization_does_not_accept_other_registry(self):
        self.assertTrue(evidence(base_reference="docker.io/" + BASE)["verified"])
        self.assertFalse(evidence(base_reference="mirror.example/" + BASE)["verified"])

    def test_no_other_nccl_mismatch_is_waived(self):
        for changes in ({"torch": "2.13.1+cu130"}, {"nccl": "2.30.8"},
                        {"requirement": "nvidia-nccl-cu13==2.29.7"},
                        {"requirement": 'nvidia-nccl-cu13>=2.31; platform_system == "Linux"'},
                        {"requirement": 'nvidia-nccl-cu13[extra]==2.29.7; platform_system == "Linux"'}):
            with self.subTest(changes=changes):
                self.assertEqual(audit(*official_stack(**changes), evidence=evidence())["status"], "FAIL")

    def test_platform_mismatch_cannot_use_override(self):
        report = audit_dependencies(official_stack(), python_version="3.12.3",
            environment={**LINUX, "platform_machine": "aarch64"}, nccl_evidence=evidence())
        self.assertEqual(report["status"], "FAIL")

    def test_all_six_real_failure_types_stay_visible(self):
        distributions = official_stack() + [
            Distribution("ai-dingo-runtime", "1.3.0", ["pydantic<=2.13,>=2.10.6"]), Distribution("pydantic", "2.13.5"),
            Distribution("ai-dingo", "1.3.0", ["kubernetes<33.0.0,>=32.0.1", "zstandard<1.0,>=0.23.0"]),
            Distribution("kvbm", "1.3.0", ["nixl[cu13]==1.0.1"]), Distribution("nixl", "1.4.1"),
            Distribution("modelexpress", "0.4.0", ["protobuf<6.0.0,>=5.27.0"]), Distribution("protobuf", "6.33.6"),
        ]
        report = audit(*distributions)
        self.assertEqual(len(report["errors"]), 6)
        for name in ("pydantic", "kubernetes", "zstandard", "nixl", "protobuf", "nvidia-nccl-cu13"):
            self.assertTrue(any(name in item for item in report["errors"]), name)
        reviewed = audit(*distributions, evidence=evidence())
        self.assertEqual(len(reviewed["errors"]), 5)
        self.assertEqual(reviewed["status"], "FAIL")

    def test_old_kvbm_exception_is_removed(self):
        for version in ("1.3.1", "1.4.1"):
            with self.subTest(nixl=version):
                report = audit(Distribution("kvbm", "1.3.0", ["nixl[cu13]==1.0.1"]), Distribution("nixl", version))
                self.assertEqual(report["status"], "FAIL")
                self.assertEqual(report["known_metadata_exceptions"], [])

    def test_omni_default_requirement_cannot_be_waived(self):
        stack = official_stack()
        stack[1] = Distribution("vllm-omni", "0.30.0", [NCCL_REQUIREMENT, "torch>=2.0"])
        report = audit(*stack, evidence=evidence())
        self.assertEqual(report["frameworks"]["vllm-omni"]["default_requirements"]["status"], "FAIL")
        self.assertEqual(report["status"], "FAIL")

    def test_omni_transitive_conflict_is_separate_from_vllm(self):
        stack = official_stack()
        stack[1] = Distribution("vllm-omni", "0.30.0", ["torch>=2.0", "s3tokenizer==0.3.0"])
        report = audit(*stack, Distribution("s3tokenizer", "0.3.0", ["onnx"]),
            Distribution("onnx", "1.23.1", ["protobuf>=6.31.1"]), Distribution("protobuf", "5.29.0"), evidence=evidence())
        self.assertEqual(report["frameworks"]["vllm"]["status"], "PASS_WITH_UPSTREAM_NCCL_OVERRIDE")
        self.assertEqual(report["frameworks"]["vllm-omni"]["default_requirements"]["status"], "PASS")
        self.assertEqual(report["frameworks"]["vllm-omni"]["recursive_dependency_closure"]["status"], "FAIL")

    def test_recursive_extras_reach_fixpoint(self):
        report = audit(Distribution("vllm-omni", "0.30.0", ["second[media]"]),
            Distribution("third", requires=['missing==1; extra == "codec"']),
            Distribution("second", requires=['third[codec]; extra == "media"']))
        tree = report["frameworks"]["vllm-omni"]["recursive_dependency_closure"]
        self.assertEqual(tree["active_extras"]["third"], ["codec"])
        self.assertTrue(any("missing missing" in error for error in tree["errors"]))

    def test_inactive_markers_and_unrequested_extras_skipped(self):
        report = audit(Distribution("app", requires=['windows-only; sys_platform == "win32"',
            'developer-only; extra == "dev"', 'old-python; python_version < "3.12"']))
        self.assertEqual(report["status"], "PASS")

    def test_extra_cycle_terminates(self):
        report = audit(Distribution("first", requires=["second[media]"]),
                       Distribution("second", requires=['first[codec]; extra == "media"']))
        self.assertEqual(report["status"], "PASS")

    def test_requires_python_and_invalid_metadata_fail(self):
        for distribution in (Distribution("app", requires_python=">=3.13"),
                             Distribution("bad", requires=["not valid !!!"]),
                             Distribution("bad", requires_python="not valid"), Distribution("bad", version="not valid")):
            with self.subTest(distribution=distribution.__dict__):
                self.assertEqual(audit(distribution)["status"], "FAIL")

    def test_framework_closure_includes_transitive_python_requirement(self):
        report = audit(Distribution("vllm-omni", "0.30.0", ["child"]),
                       Distribution("child", requires_python=">=3.13"))
        self.assertEqual(report["frameworks"]["vllm-omni"]["recursive_dependency_closure"]["status"], "FAIL")

    def test_uv_unknown_tool_output_and_extra_diagnostic_fail(self):
        variants = [(2, "error: Python missing"), (0, "All installed packages compatible"),
            (1, KNOWN_UV_OUTPUT + "error: unexpected failure\n"), (1, "error: unknown failure"),
            (1, KNOWN_UV_OUTPUT.replace("Found 1 incompatibility", "Found 2 incompatibilities")),
            (1, KNOWN_UV_OUTPUT.replace("sys_platform == 'linux'", "sys_platform != 'win32'")),
            (1, KNOWN_UV_OUTPUT.replace("==2.29.7", ">=2.31"))]
        for status, output in variants:
            with self.subTest(status=status, output=output):
                report = audit(*official_stack(), evidence=evidence())
                review_uv_check(report, status, output)
                self.assertEqual(report["status"], "FAIL")

    def test_uv_wheel_linux_marker_is_also_recognized(self):
        report = audit(*official_stack(), evidence=evidence())
        review_uv_check(report, 1, KNOWN_UV_OUTPUT.replace("sys_platform == 'linux'", "platform_system == 'Linux'"))
        self.assertEqual(report["status"], "PASS_WITH_UPSTREAM_NCCL_OVERRIDE")

    def test_uv_failure_without_matching_metadata_fails(self):
        report = audit(Distribution("app"))
        review_uv_check(report, 1, KNOWN_UV_OUTPUT)
        self.assertEqual(report["status"], "FAIL")

    def test_same_realpath_repeated_distribution_is_not_duplicate_installation(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = Distribution("some-package", path=Path(temporary) / "some.dist-info")
            second = Distribution("some_package", path=Path(temporary) / "x" / ".." / "some.dist-info")
            report = audit(first, second)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["distribution_records"]["some-package"]["same_realpath_duplicates"], 1)

    def test_same_path_different_metadata_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "some.dist-info"
            report = audit(Distribution("some", "1.0", path=target), Distribution("some", "2.0", path=target))
        self.assertEqual(report["status"], "FAIL")

    def test_unknown_or_pathless_duplicate_fails(self):
        self.assertEqual(audit(Distribution("some"), Distribution("some"))["status"], "FAIL")
        self.assertEqual(audit(Distribution("some", path="/one"), Distribution("some", path="/two"))["status"], "FAIL")

    def test_known_shadow_pairs_require_matching_effective_import_and_wheel_files(self):
        for name in ("cryptography", "pyjwt", "six", "oauthlib"):
            with self.subTest(name=name), patch("collect_runtime_manifest.resolved_path", side_effect=str):
                system, wheel, origin = shadow_pair(name)
                report = audit(system, wheel, effective_distribution=lambda _: wheel, origin_resolver=lambda _: origin)
                self.assertEqual(report["status"], "PASS")
                self.assertEqual(report["packages"][name], "2.0")
                self.assertEqual(report["shadowed_distributions"][0]["shadowed_version"], "1.0")
                self.assertEqual(report["distribution_records"][name]["import_origin"], origin)

    def test_logged_cryptography_system_egg_and_dist_info_shadow(self):
        with patch("collect_runtime_manifest.resolved_path", side_effect=str):
            wheel = Distribution("cryptography", "50.0.1",
                path="/usr/local/lib/python3.12/dist-packages/cryptography-50.0.1.dist-info",
                files=["cryptography/__init__.py"])
            system = [Distribution("cryptography", "41.0.7",
                path=f"/usr/lib/python3/dist-packages/{name}")
                for name in ("cryptography.egg-info", "cryptography-41.0.7.dist-info")]
            report = audit(*system, wheel, effective_distribution=lambda _: wheel,
                origin_resolver=lambda _: wheel.locate_file("cryptography/__init__.py"))
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["packages"]["cryptography"], "50.0.1")
        shadow = report["shadowed_distributions"][0]
        self.assertEqual(shadow["shadowed_version"], "41.0.7")
        self.assertEqual(len(shadow["shadowed_metadata_paths"]), 2)
        self.assertIsNone(shadow["shadowed_metadata_path"])

    def test_logged_oauthlib_shadow_with_verified_nccl_override(self):
        with patch("collect_runtime_manifest.resolved_path", side_effect=str):
            system, wheel, origin = shadow_pair("oauthlib")
            system.version, wheel.version = "3.2.2", "4.0.0"
            system._path = "/usr/lib/python3/dist-packages/oauthlib-3.2.2.egg-info"
            wheel._path = "/usr/local/lib/python3.12/dist-packages/oauthlib-4.0.0.dist-info"
            report = audit(*official_stack(), system, wheel, evidence=evidence(),
                effective_distribution=lambda _: wheel, origin_resolver=lambda _: origin)
            review_uv_check(report, 1, KNOWN_UV_OUTPUT)
        self.assertEqual(report["status"], "PASS_WITH_UPSTREAM_NCCL_OVERRIDE")
        self.assertEqual(report["packages"]["oauthlib"], "4.0.0")

    def test_ambiguous_system_or_local_metadata_still_fails(self):
        with patch("collect_runtime_manifest.resolved_path", side_effect=str):
            system, wheel, origin = shadow_pair()
            additions = [
                Distribution("cryptography", "0.9", path="/usr/lib/python3/dist-packages/cryptography-0.9.dist-info"),
                Distribution("cryptography", "1.0", path="/usr/lib/python3/dist-packages/cryptography.egg-info"),
                Distribution("cryptography", "1.0", path="/usr/lib/python3/dist-packages/cryptography.metadata"),
                Distribution("cryptography", "2.0", path="/usr/local/lib/python3.12/dist-packages/cryptography-extra.dist-info"),
            ]
            for extra in additions:
                with self.subTest(path=extra._path):
                    report = audit(system, wheel, extra, effective_distribution=lambda _: wheel,
                                   origin_resolver=lambda _: origin)
                    self.assertEqual(report["status"], "FAIL")

    def test_reviewed_shadow_without_wheel_record_still_fails(self):
        with patch("collect_runtime_manifest.resolved_path", side_effect=str):
            for name in ("cryptography", "oauthlib"):
                system, wheel, origin = shadow_pair(name)
                wheel.files = []
                report = audit(system, wheel, effective_distribution=lambda _: wheel,
                               origin_resolver=lambda _: origin)
                self.assertEqual(report["status"], "FAIL")

    def test_real_metadata_discovery_and_import_precedence(self):
        # Actual importlib metadata/RECORD/find_spec behavior on disk; remap only
        # the temporary directory prefix to emulate Ubuntu roots on Windows.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            local = root / "usr/local/lib/python3.12/dist-packages"
            system = root / "usr/lib/python3/dist-packages"
            local.mkdir(parents=True)
            system.mkdir(parents=True)
            for base, name, version, suffix in (
                (local, "oauthlib", "4.0.0", ".dist-info"),
                (system, "oauthlib", "3.2.2", ".egg-info"),
                (local, "cryptography", "50.0.1", ".dist-info"),
                (system, "cryptography", "41.0.7", ".egg-info"),
                (system, "cryptography", "41.0.7", ".dist-info"),
            ):
                metadata = base / f"{name}-{version}{suffix}"
                metadata.mkdir()
                filename = "METADATA" if suffix == ".dist-info" else "PKG-INFO"
                (metadata / filename).write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
                module = base / name / "__init__.py"
                module.parent.mkdir(exist_ok=True)
                module.write_text("# metadata/import precedence fixture\n")
                if base == local:
                    (metadata / "RECORD").write_text(f"{name}/__init__.py,,\n")

            def ubuntu_path(path):
                return "/" + Path(path).resolve().relative_to(root).as_posix()

            distributions = list(md.distributions(path=[str(local), str(system)]))
            self.assertEqual(len(distributions), 5)
            original_path = list(sys.path)
            with patch("collect_runtime_manifest.resolved_path", side_effect=ubuntu_path), patch.object(
                    sys, "path", [str(local), str(system), *original_path]):
                report = audit(*distributions)
            self.assertEqual(report["status"], "PASS", report["errors"])
            self.assertEqual(report["packages"]["oauthlib"], "4.0.0")
            self.assertEqual(report["packages"]["cryptography"], "50.0.1")
            self.assertEqual(len(report["shadowed_distributions"]), 2)
            # If the system path takes priority, no shadow exception is valid.
            with patch("collect_runtime_manifest.resolved_path", side_effect=ubuntu_path), patch.object(
                    sys, "path", [str(system), str(local), *original_path]):
                report = audit(*distributions)
            self.assertEqual(report["status"], "FAIL")

    def test_known_names_do_not_blindly_allow_duplicate_paths(self):
        with patch("collect_runtime_manifest.resolved_path", side_effect=str):
            system, wheel, origin = shadow_pair()
            scenarios = [([system, wheel], system, origin),
                         ([system, wheel], wheel, "/usr/lib/python3/dist-packages/cryptography/__init__.py"),
                         ([system, wheel], wheel, "/usr/local/lib/python3.12/dist-packages/unowned.py"),
                         ([system, wheel, Distribution("cryptography", path="/third")], wheel, origin)]
            for distributions, selected, path in scenarios:
                with self.subTest(path=path, distributions=len(distributions)):
                    result = audit(*distributions, effective_distribution=lambda _: selected, origin_resolver=lambda _: path)
                    self.assertEqual(result["status"], "FAIL")
            wheel._path = "/unexpected/cryptography-2.0.dist-info"
            self.assertEqual(audit(system, wheel, effective_distribution=lambda _: wheel, origin_resolver=lambda _: origin)["status"], "FAIL")

    def test_shadowed_metadata_requirements_do_not_override_active_wheel(self):
        with patch("collect_runtime_manifest.resolved_path", side_effect=str):
            system, wheel, origin = shadow_pair()
            system.requires = ["old-system-only-missing-package"]
            report = audit(system, wheel, effective_distribution=lambda _: wheel, origin_resolver=lambda _: origin)
            self.assertEqual(report["status"], "PASS")
            wheel.requires = ["active-wheel-missing-package"]
            report = audit(system, wheel, effective_distribution=lambda _: wheel, origin_resolver=lambda _: origin)
            self.assertEqual(report["status"], "FAIL")

    def test_main_writes_audit_before_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "uv-pip-check.exit-code").write_text("2\n")
            (directory / "uv-pip-check.txt").write_text("error: failed to find Python")
            with patch("collect_runtime_manifest.BUILD_INFO_DIRECTORY", directory), patch(
                "collect_runtime_manifest.md.distributions", return_value=official_stack()), patch(
                "collect_runtime_manifest.read_upstream_nccl_evidence", return_value=evidence()), redirect_stdout(io.StringIO()) as output:
                with self.assertRaisesRegex(RuntimeError, "uv pip check failed as a tool"):
                    main()
            report = json.loads((directory / "dependency-audit.json").read_text())
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["uv_pip_check_exit_code"], 2)
            self.assertIn("vllm-omni", report["frameworks"])
            self.assertIn("Runtime dependency audit summary:", output.getvalue())
            self.assertIn("upstream_nccl_evidence", output.getvalue())

    def test_main_success_preserves_artifacts_source_hashes_and_framework_statuses(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "uv-pip-check.exit-code").write_text("1\n")
            (directory / "uv-pip-check.txt").write_text(KNOWN_UV_OUTPUT)
            omni = Distribution("vllm-omni", "0.30.0")
            omni.locate_file = lambda name: directory / name
            source_names = ["pipeline_minimax_h3.py", "vae.py", "quality_policy.py", "time_request.py"]
            for name in source_names:
                source = directory / "vllm_omni/diffusion/models/minimax_h3" / name
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(b"# synthetic H3 source for build inventory test\n")
            with patch("collect_runtime_manifest.BUILD_INFO_DIRECTORY", directory), patch(
                "collect_runtime_manifest.md.distributions", return_value=official_stack()), patch(
                "collect_runtime_manifest.md.distribution", return_value=omni), patch(
                "collect_runtime_manifest.default_environment", return_value=LINUX.copy()), patch(
                "collect_runtime_manifest.read_upstream_nccl_evidence", return_value=evidence()), redirect_stdout(io.StringIO()) as output:
                main()
            audit_report = json.loads((directory / "dependency-audit.json").read_text())
            manifest = json.loads((directory / "runtime-manifest.json").read_text())
            self.assertEqual(audit_report["status"], "PASS_WITH_UPSTREAM_NCCL_OVERRIDE")
            self.assertEqual(manifest["frameworks"]["vllm-omni"]["default_requirements"]["status"], "PASS")
            self.assertEqual(manifest["dependency_audit_status"], audit_report["status"])
            self.assertEqual(len(manifest["h3_source_sha256"]), 4)
            for name, digest in manifest["h3_source_sha256"].items():
                self.assertEqual(digest, hashlib.sha256((directory / name).read_bytes()).hexdigest())
            self.assertIn("vllm-omni==0.30.0\n", (directory / "installed-versions.txt").read_text())
            self.assertIn("PASS_WITH_UPSTREAM_NCCL_OVERRIDE", output.getvalue())
            self.assertIn('"default_requirements": "PASS"', output.getvalue())


if __name__ == "__main__":
    unittest.main()
