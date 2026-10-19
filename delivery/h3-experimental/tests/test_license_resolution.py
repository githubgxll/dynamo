"""Verify the released Harmony metadata through the real NOTICE/policy pipeline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "container"))
from compliance import overrides
from compliance.generators import python as python_gen, rust as rust_gen
from compliance.generators import common
from compliance.policy import validate

FIXTURE = Path(__file__).resolve().parents[1] / "license-evidence/oss-harmony-0.0.11"
POLICY = ROOT / "container/compliance/policy/licenses.toml"


class LicenseResolutionTests(unittest.TestCase):
    def setUp(self):
        overrides.reset_caches()
        self.addCleanup(overrides.reset_caches)

    def test_existing_versionless_mappings_keep_their_behavior(self):
        data = yaml.safe_load(overrides._DEFAULT_OVERRIDES_PATH.read_text(encoding="utf-8"))
        legacy = {(item["ecosystem"], item["name"]): item["license"]
                  for item in data["overrides"] if "version" not in item}
        for (ecosystem, name), license_id in legacy.items():
            for version in ("", "unreviewed-version"):
                self.assertEqual(overrides.lookup(ecosystem, name, version), license_id)

    def test_exact_entry_precedes_general_entry_without_expanding_scope(self):
        entries = {("python", "example", ""): "MIT",
                   ("python", "example", "2.0"): "Apache-2.0",
                   ("python", "exact-only", "1.0"): "BSD-2-Clause"}
        with patch.object(overrides, "_overrides", return_value=entries):
            self.assertEqual(overrides.lookup("python", "example", "2.0"), "Apache-2.0")
            self.assertEqual(overrides.lookup("python", "example", "3.0"), "MIT")
            self.assertIsNone(overrides.lookup("python", "exact-only"))
            self.assertIsNone(overrides.lookup("python", "exact-only", "2.0"))
            self.assertIsNone(overrides.lookup("rust", "exact-only", "1.0"))

    def test_invalid_explicit_version_never_becomes_a_blanket_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "overrides.yaml"
            for version in (None, "", " ", 11):
                with self.subTest(version=version):
                    path.write_text(yaml.safe_dump({"overrides": [{"ecosystem": "python",
                        "name": "example", "version": version, "license": "MIT"}]}), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        overrides._load_overrides(path)

    def test_exact_release_evidence_keeps_original_artifact_bytes(self):
        record = json.loads((FIXTURE / "source.json").read_text(encoding="utf-8"))
        self.assertEqual((record["name"], record["version"], record["spdx"]),
                         ("oss-harmony", "0.0.11", "Apache-2.0"))
        for filename, item in record["files"].items():
            self.assertEqual(hashlib.sha256((FIXTURE / filename).read_bytes()).hexdigest(), item["sha256"])
        parsed = python_gen._extract_metadata_field((FIXTURE / "METADATA").read_text(encoding="utf-8"))
        self.assertEqual(parsed[:3], ("oss-harmony", "0.0.11", None))
        self.assertEqual(parsed[3], [])
        self.assertFalse(any(item.startswith("License ::") for item in parsed[4]))
        with patch.object(overrides, "lookup", return_value=None):
            self.assertEqual(python_gen._resolve_spdx(*parsed), "UNKNOWN")
        self.assertEqual(python_gen._resolve_spdx(*parsed), "Apache-2.0")

    def test_release_notice_csv_sbom_and_real_policy_pass_without_exception(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            site, output = directory / "site-packages", directory / "legal"
            dist = site / "oss_harmony-0.0.11.dist-info"
            (dist / "licenses").mkdir(parents=True)
            (dist / "sboms").mkdir()
            output.mkdir()
            shutil.copyfile(FIXTURE / "METADATA", dist / "METADATA")
            shutil.copyfile(FIXTURE / "LICENSE", dist / "licenses/LICENSE")
            shutil.copyfile(FIXTURE / "oss-harmony.cyclonedx.json", dist / "sboms/oss-harmony.cyclonedx.json")
            components = python_gen.generate([site], output)
            self.assertEqual(len(components), 1)
            self.assertEqual(components[0].spdx, "Apache-2.0")
            original = (FIXTURE / "LICENSE").read_text(encoding="utf-8")
            self.assertEqual(components[0].license_text, original)
            self.assertFalse(components[0].license_text_is_canonical)
            self.assertIn(original.strip(), (output / "NOTICES-Python.txt").read_text(encoding="utf-8"))
            rust = rust_gen.collect_components([site])
            self.assertEqual(len(rust), 84)
            self.assertTrue(any(item.spdx != "Apache-2.0" for item in rust))
            common.write_merged_csv(components + rust, {}, output / "osrb-deps.csv")
            common.write_cyclonedx(components + rust, output / "osrb.cdx.json")
            bom = json.loads((output / "osrb.cdx.json").read_text(encoding="utf-8"))
            root = next(item for item in bom["components"] if item["name"] == "oss-harmony")
            self.assertEqual(root["licenses"], [{"license": {"id": "Apache-2.0"}}])
            self.assertEqual(validate.main(["--policy", str(POLICY), "--image", "vllm-runtime",
                                            "--input", str(output / "osrb-deps.csv")]), 0)

    def test_other_versions_and_unknown_packages_still_fail_closed(self):
        policy = validate.load_policy(POLICY)
        self.assertEqual(policy.unknown_action, "deny")
        self.assertFalse(any(item.get("name") == "oss-harmony" for item in policy.exceptions))
        for name, version in (("oss-harmony", "0.0.12"), ("oss-harmony", "0.0.10"),
                              ("unreviewed-package", "0.0.11")):
            spdx = python_gen._resolve_spdx(name, version, None, [], [])
            self.assertEqual(spdx, "UNKNOWN")
            self.assertIsNotNone(validate.validate_row(policy, "python", name, version, spdx, "vllm-runtime"))

    def test_structured_upstream_license_is_not_overwritten(self):
        self.assertEqual(python_gen._resolve_spdx("oss-harmony", "0.0.11", "GPL-3.0-only", [], []),
                         "GPL-3.0-only")
        self.assertEqual(overrides.lookup("python", "openai-harmony", "0.0.4"), "MIT")


if __name__ == "__main__":
    unittest.main()
