# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small metadata fixtures exercise the dependency solve's input boundaries."""

from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import runtime_dependencies as runtime
from runtime_dependencies import (
    inherited_telemetry_requirements, protected_constraints, protected_runtime_requirements,
    runtime_requirements, snapshot, verify_protected,
)


class Distribution:
    def __init__(self, name, version="1.0", requirements=(), location="/site"):
        self.metadata = {"Name": name}
        self.version = version
        self.requires = list(requirements)
        self._path = Path(location) / f"{name}-{version}.dist-info"

    def locate_file(self, filename):
        return self._path.parent / filename


def inventory(*extra):
    return snapshot([
        Distribution("ai-dingo", "1.3.0", [
            "ai-dingo-runtime==1.3.0", "kubernetes>=32.0.1,<33.0.0",
            "zstandard>=0.23.0,<1.0", "transformers>=4.56.0",
            'nixl[cu13]==1.3.1; extra == "vllm"',
            'av>=14,<17; extra == "video-gateway"',
            'PyYAML>=6,<7; extra == "video-gateway"',
        ]),
        Distribution("ai-dingo-runtime", "1.3.0", ["pydantic>=2.10.6,<=2.13"]),
        Distribution("vllm", "0.30.0", [
            "torch==2.13.0", "transformers>=5.12.0",
            'win32-only; platform_system == "Windows"',
            'nixl[cu13]>=1.4.0; platform_system == "Linux"',
        ]),
        Distribution("torch", "2.13.0+cu130"),
        Distribution("gpu-memory-service", "0.9.0", ["uvloop>=0.21.0"]),
        *extra,
    ], phase="before")


class RuntimeRequirementsTests(unittest.TestCase):
    def test_frozen_lmcache_dependencies_bound_otherwise_unconstrained_telemetry(self):
        data = inventory(Distribution("lmcache", "0.5.5", [
            "opentelemetry-api>=1.20.0,<=1.40.0",
            "opentelemetry-exporter-prometheus>=0.50b0,<=0.61b0",
        ]), Distribution("opentelemetry-api", "1.40.0"),
            Distribution("opentelemetry-exporter-prometheus", "0.61b0"))
        constraints = protected_constraints(data, ["lmcache"], omni_version="0.31.0rc1")
        self.assertIn("lmcache==0.5.5", constraints)
        # The old default roots omit these upper bounds despite freezing LMCache.
        self.assertFalse(any("<=1.40.0" in line for line in runtime_requirements(data)))
        lines = protected_runtime_requirements(data, constraints)
        self.assertIn("opentelemetry-api<=1.40.0,>=1.20.0", lines)
        self.assertIn("opentelemetry-exporter-prometheus<=0.61b0,>=0.50b0", lines)
        self.assertEqual(inherited_telemetry_requirements(data), [
            "opentelemetry-api", "opentelemetry-exporter-prometheus"])

    def test_every_protected_component_contributes_not_just_lmcache(self):
        data = inventory(Distribution("vendor-kernel", "2.0+cuda", ["shared-api<3"]),
                         Distribution("nvidia-future-cu13", "13.5", ["runtime-loader>=2"]))
        constraints = protected_constraints(data, ["vendor-kernel"], omni_version="0.31.0rc1")
        lines = protected_runtime_requirements(data, constraints)
        self.assertIn("shared-api<3", lines)
        self.assertIn("runtime-loader>=2", lines)
        # Feed installed metadata edges without asking an index for vendor wheels.
        self.assertNotIn("vendor-kernel==2.0+cuda", lines)

    def test_protected_dependencies_preserve_markers_and_requested_edge_extras(self):
        data = inventory(Distribution("lmcache", "0.5.5", [
            'transport[metrics]>=2; sys_platform == "linux"',
            'win-only>=1; sys_platform == "win32"',
            'optional-store>=1; extra == "redis"',
        ]))
        lines = protected_runtime_requirements(data, ["lmcache==0.5.5"],
                                               environment={"sys_platform": "linux"})
        self.assertIn('transport[metrics]>=2; sys_platform == "linux"', lines)
        self.assertFalse(any("win-only" in line or "optional-store" in line for line in lines))

    def test_mutable_api_and_telemetry_old_dependency_edges_are_not_frozen(self):
        data = inventory(Distribution("transformers", "5.15.0", ["future-api>=9"]),
                         Distribution("opentelemetry-exporter-otlp-common", "0.66b0", ["opentelemetry-sdk~=1.45.0"]),
                         Distribution("lmcache", "0.5.5", ["opentelemetry-api<=1.40.0"]))
        constraints = protected_constraints(data, ["lmcache", "transformers"], omni_version="0.31.0rc1")
        lines = protected_runtime_requirements(data, constraints)
        self.assertFalse(any("future-api" in line or "~=1.45.0" in line for line in lines))
        self.assertIn("opentelemetry-api<=1.40.0", lines)

    def test_protected_dependency_inventory_rejects_missing_or_different_versions(self):
        for pins in (["lmcache==0.5.5"], ["torch==2.12.0"]):
            with self.subTest(pins=pins), self.assertRaises(ValueError):
                protected_runtime_requirements(inventory(), pins)

    def test_protected_dependency_inventory_rejects_conflicting_metadata(self):
        data = inventory(Distribution("lmcache", "0.5.5", ["opentelemetry-api<=1.40.0"]),
                         Distribution("lmcache", "0.5.5", ["opentelemetry-api>=1.45"], "/other"))
        with self.assertRaisesRegex(ValueError, "Conflicting installed requirements for lmcache"):
            protected_runtime_requirements(data, ["lmcache==0.5.5"])

    def test_protected_torch_edge_remains_visible_to_reviewed_nccl_override(self):
        data = snapshot([Distribution("torch", "2.13.0+cu130", [
            'nvidia-nccl-cu13==2.29.7; platform_system == "Linux"'])], phase="before")
        self.assertIn('nvidia-nccl-cu13==2.29.7; platform_system == "Linux"',
                      protected_runtime_requirements(data, ["torch==2.13.0+cu130"],
                                                      environment={"platform_system": "Linux"}))

    def test_before_phase_writes_protected_requirements_and_source_evidence(self):
        data = inventory(Distribution("lmcache", "0.5.5", ["opentelemetry-api<=1.40.0"]),
                         Distribution("opentelemetry-api", "1.40.0"))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            names = directory / "protected-packages.txt"
            names.write_text("lmcache\n", encoding="utf-8")
            argv = ["runtime_dependencies.py", "before", "--directory", str(directory),
                    "--protected-packages", str(names), "--omni-version", "0.31.0rc1"]
            with patch.object(runtime, "snapshot", return_value=data), patch.object(runtime.sys, "argv", argv):
                runtime.main()
            lines = (directory / "protected-runtime-requirements.txt").read_text().splitlines()
            evidence = json.loads((directory / "protected-runtime-solve-inputs.json").read_text())
            self.assertIn("opentelemetry-api<=1.40.0", lines)
            self.assertIn("lmcache==0.5.5", evidence["protected_roots"])
            self.assertEqual(evidence["requirements"], lines)
            self.assertEqual(evidence["source_inventory"], "runtime-before-omni.json")
            self.assertEqual(json.loads((directory / "runtime-before-omni.json").read_text()), data)

    def test_new_internal_split_is_not_forced_into_older_compatible_family(self):
        data = inventory(
            Distribution("opentelemetry-exporter-otlp-common", "0.66b0", ["opentelemetry-sdk~=1.45.0"]),
            Distribution("opentelemetry-exporter-otlp-proto-common", "1.40.0"),
            Distribution("opentelemetry-exporter-otlp-proto-grpc", "1.40.0"),
            Distribution("opentelemetry-exporter-otlp-proto-http", "1.40.0"),
            Distribution("opentelemetry-exporter-prometheus", "0.61b0"),
            Distribution("opentelemetry-sdk", "1.40.0", ["opentelemetry-api==1.40.0"]),
            Distribution("opentelemetry-api", "1.40.0"),
            Distribution("opentelemetry-semantic-conventions", "0.61b0"),
        )
        self.assertEqual(inherited_telemetry_requirements(data), [
            "opentelemetry-api", "opentelemetry-exporter-otlp-proto-common",
            "opentelemetry-exporter-otlp-proto-grpc", "opentelemetry-exporter-otlp-proto-http",
            "opentelemetry-exporter-prometheus",
            "opentelemetry-sdk", "opentelemetry-semantic-conventions",
        ])

    def test_internal_split_remains_a_real_requirement_when_a_protected_owner_needs_it(self):
        data = inventory(Distribution("fixed-service", "1.0", ["opentelemetry-exporter-otlp-common>=0.65b0"]),
                         Distribution("opentelemetry-exporter-otlp-common", "0.66b0"))
        self.assertNotIn("opentelemetry-exporter-otlp-common", inherited_telemetry_requirements(data))
        self.assertIn("opentelemetry-exporter-otlp-common>=0.65b0",
                      protected_runtime_requirements(data, ["fixed-service==1.0"]))

    def test_before_phase_records_internal_split_for_reconciliation_without_dropping_exporter(self):
        data = inventory(Distribution("opentelemetry-exporter-otlp-common", "0.66b0"),
                         Distribution("opentelemetry-exporter-otlp-proto-common", "1.40.0"))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            names = directory / "protected-packages.txt"
            names.write_text("torch\n", encoding="utf-8")
            argv = ["runtime_dependencies.py", "before", "--directory", str(directory),
                    "--protected-packages", str(names), "--omni-version", "0.31.0rc1"]
            with patch.object(runtime, "snapshot", return_value=data), patch.object(runtime.sys, "argv", argv):
                runtime.main()
            evidence = json.loads((directory / "telemetry-solve-inputs.json").read_text())
            self.assertEqual(evidence["resolver_managed_internal_packages"],
                             {"opentelemetry-exporter-otlp-common": "0.66b0"})
            self.assertEqual(evidence["roots"], ["opentelemetry-exporter-otlp-proto-common"])

    def test_reconciliation_removes_only_reviewed_orphan_not_monitoring_exporters(self):
        data = inventory(Distribution("opentelemetry-exporter-otlp-common", "0.66b0"),
                         Distribution("opentelemetry-exporter-otlp-proto-common", "1.40.0"),
                         Distribution("opentelemetry-exporter-prometheus", "0.61b0"),
                         Distribution("opentelemetry-exporter-http-transport", "0.66b1"))
        plan = runtime.telemetry_reconciliation_plan(data)
        self.assertEqual(plan["remove"], ["opentelemetry-exporter-otlp-common"])
        self.assertEqual(plan["candidates"][plan["remove"][0]]["consumers"], [])

    def test_reconciliation_preserves_internal_package_if_any_consumer_mentions_it(self):
        for suffix in ("", '; extra == "optional"', '; sys_platform == "win32"'):
            with self.subTest(marker=suffix):
                data = inventory(Distribution("opentelemetry-exporter-otlp-common", "0.66b0"),
                                 Distribution("other-exporter", "1", ["opentelemetry-exporter-otlp-common>=0.65b0" + suffix]))
                plan = runtime.telemetry_reconciliation_plan(data)
                self.assertEqual(plan["remove"], [])
                item = plan["candidates"]["opentelemetry-exporter-otlp-common"]
                self.assertEqual(item["action"], "retain-required")
                self.assertEqual(len(item["consumers"]), 1)

    def test_reconciliation_absent_split_is_noop(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(runtime.subprocess, "run") as run:
            runtime.reconcile_telemetry(inventory(), Path(temporary))
            run.assert_not_called()
            result = json.loads((Path(temporary) / "telemetry-reconciliation.json").read_text())
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(result["remove"], [])

    def test_reconciliation_uses_package_manager_and_records_failure(self):
        data = inventory(Distribution("opentelemetry-exporter-otlp-common", "0.66b0"))
        for exit_code in (0, 1):
            with self.subTest(exit_code=exit_code), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                with patch.object(runtime.subprocess, "run", return_value=SimpleNamespace(returncode=exit_code, stdout="synthetic uninstall result")) as run:
                    if exit_code:
                        with self.assertRaisesRegex(RuntimeError, "Cannot reconcile"):
                            runtime.reconcile_telemetry(data, directory)
                    else:
                        runtime.reconcile_telemetry(data, directory)
                self.assertEqual(run.call_args.args[0], ["uv", "pip", "uninstall", "--python", runtime.sys.executable,
                                                        "opentelemetry-exporter-otlp-common"])
                result = json.loads((directory / "telemetry-reconciliation.json").read_text())
                self.assertEqual(result["status"], "FAIL" if exit_code else "COMPLETE")

    def test_telemetry_roots_preserve_extra_installed_instrumentation(self):
        data = inventory(Distribution("OpenTelemetry_Instrumentation_FastAPI", "0.61b0"),
                         Distribution("unrelated-package", "1.0"))
        self.assertEqual(inherited_telemetry_requirements(data), ["opentelemetry-instrumentation-fastapi"])
        self.assertEqual(inherited_telemetry_requirements(inventory()), [])

    def test_conflicting_inherited_telemetry_versions_are_not_silently_hidden(self):
        with self.assertRaisesRegex(ValueError, "Conflicting installed versions"):
            inherited_telemetry_requirements(inventory(
                Distribution("opentelemetry-sdk", "1.40.0"),
                Distribution("opentelemetry-sdk", "1.45.0", location="/other")))

    def test_complete_roots_include_dynamo_and_vllm_default_requirements(self):
        lines = runtime_requirements(inventory(), environment={"platform_system": "Linux"})
        active = [line for line in lines if not line.startswith("#")]
        self.assertIn("kubernetes<33.0.0,>=32.0.1", active)
        self.assertIn("zstandard<1.0,>=0.23.0", active)
        self.assertIn("pydantic<=2.13,>=2.10.6", active)
        self.assertIn("transformers>=4.56.0", active)
        self.assertIn("transformers>=5.12.0", active)
        self.assertIn("torch==2.13.0", active)
        self.assertIn("av<17,>=14", active)
        self.assertIn("PyYAML<7,>=6", active)
        self.assertFalse(any("extra ==" in line for line in active))

    def test_backend_extra_is_not_activated_but_default_edge_extras_survive(self):
        lines = runtime_requirements(inventory(), environment={"platform_system": "Linux"})
        self.assertFalse(any("==1.3.1" in line or "win32-only" in line for line in lines))
        self.assertTrue(any("nixl[cu13]>=1.4.0" in line for line in lines))

    def test_missing_installed_root_fails_instead_of_silently_skipping(self):
        data = snapshot([Distribution("vllm")], phase="before")
        with self.assertRaisesRegex(ValueError, "missing: ai-dingo"):
            runtime_requirements(data)

    def test_conflicting_root_metadata_is_not_hidden(self):
        data = inventory(Distribution("vllm", "0.30.0", ["torch==2.12.0"], "/other"))
        with self.assertRaisesRegex(ValueError, "Conflicting installed requirements"):
            runtime_requirements(data)

    def test_every_nvidia_library_is_frozen_and_python_shared_apis_can_resolve(self):
        data = inventory(
            Distribution("nvidia-nccl-cu13", "2.30.7"),
            Distribution("nvidia-new-library-cu13", "13.0.4"),
            Distribution("nixl", "1.4.1"),
            Distribution("transformers", "5.15.0"),
            Distribution("tokenizers", "0.22.2"),
        )
        for version in ("0.31.0rc1",):
            with self.subTest(omni_version=version):
                constraints = protected_constraints(
                    data, ["nixl", "transformers", "tokenizers"], omni_version=version
                )
                self.assertIn("nvidia-nccl-cu13==2.30.7", constraints)
                self.assertIn("nvidia-new-library-cu13==13.0.4", constraints)
                self.assertIn("nixl==1.4.1", constraints)
                self.assertIn("torch==2.13.0+cu130", constraints)
                self.assertIn("vllm==0.30.0", constraints)
                self.assertFalse(any(line.startswith(("transformers==", "tokenizers==")) for line in constraints))

    def test_shared_api_exception_does_not_expand_to_unreviewed_omni_versions(self):
        constraints = protected_constraints(
            inventory(Distribution("transformers", "5.15.0")),
            ["transformers"], omni_version="0.31.0",
        )
        self.assertIn("transformers==5.15.0", constraints)

    def test_snapshot_keeps_duplicate_paths_for_downstream_audit(self):
        data = inventory(
            Distribution("six", "1.17.0", location="/site"),
            Distribution("six", "1.16.0", location="/system"),
        )
        entries = [item for item in data["distributions"] if item["canonical_name"] == "six"]
        self.assertEqual(len(entries), 2)
        self.assertEqual({item["location"] for item in entries}, {str(Path("/site")), str(Path("/system"))})
        self.assertTrue(all(item["metadata_path"].endswith(".dist-info") for item in entries))

    def test_protected_version_change_or_removal_fails(self):
        with self.assertRaisesRegex(ValueError, "Protected package changed: torch"):
            verify_protected(inventory(), ["torch==2.12.0"])
        with self.assertRaisesRegex(ValueError, "missing: nvidia-nccl-cu13"):
            verify_protected(inventory(), ["nvidia-nccl-cu13==2.30.7"])

    def test_duplicate_protected_versions_fail_before_resolve(self):
        with self.assertRaisesRegex(ValueError, "Conflicting installed versions for torch"):
            protected_constraints(inventory(Distribution("torch", "2.12.0")), [], omni_version="0.30.0")

    def test_repeated_identical_discovery_does_not_hide_or_invent_version_changes(self):
        data = inventory(Distribution("torch", "2.13.0+cu130"))
        verify_protected(data, ["torch==2.13.0+cu130"])
        self.assertEqual(sum(item["canonical_name"] == "torch" for item in data["distributions"]), 2)


if __name__ == "__main__":
    unittest.main()
