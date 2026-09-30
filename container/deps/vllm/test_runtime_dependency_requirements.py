# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small metadata fixtures exercise the dependency solve's input boundaries."""

from pathlib import Path
import unittest

from runtime_dependency_requirements import (
    protected_constraints, runtime_requirements, snapshot, verify_protected,
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
        ]),
        Distribution("ai-dingo-runtime", "1.3.0", ["pydantic>=2.10.6,<=2.13"]),
        Distribution("vllm", "0.30.0", [
            "torch==2.13.0", "transformers>=5.12.0",
            'win32-only; platform_system == "Windows"',
            'nixl[cu13]>=1.4.0; platform_system == "Linux"',
        ]),
        Distribution("torch", "2.13.0+cu130"),
        *extra,
    ], phase="before")


class RuntimeRequirementsTests(unittest.TestCase):
    def test_complete_roots_include_dynamo_and_vllm_default_requirements(self):
        lines = runtime_requirements(inventory(), environment={"platform_system": "Linux"})
        active = [line for line in lines if not line.startswith("#")]
        self.assertIn("kubernetes<33.0.0,>=32.0.1", active)
        self.assertIn("zstandard<1.0,>=0.23.0", active)
        self.assertIn("pydantic<=2.13,>=2.10.6", active)
        self.assertIn("transformers>=4.56.0", active)
        self.assertIn("transformers>=5.12.0", active)
        self.assertIn("torch==2.13.0", active)

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
        constraints = protected_constraints(
            data, ["nixl", "transformers", "tokenizers"], omni_version="0.30.0"
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
