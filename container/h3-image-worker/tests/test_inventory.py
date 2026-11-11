import base64
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch


SPEC = importlib.util.spec_from_file_location("worker_inventory", Path(__file__).resolve().parents[1] / "inventory.py")
inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory)


class InstalledInventory(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.site = self.root / "site"
        self.site.mkdir()
        self.expected = {}
        for name in (*inventory.CRITICAL_SOURCE_PATHS, "vllm_omni/__init__.py"):
            self.write_source(name, b"# installed source\n")
            self.expected[name] = hashlib.sha256(b"# installed source\n").hexdigest()
        self.expected["vllm_omni/unpackaged_example.py"] = "0" * 64
        self.write_source(inventory.GENERATED_SOURCE, b"__version__ = 'test'\n")
        self.omni = self.add_distribution(self.site, "vllm_omni-1.dist-info", "vllm-omni", "1")
        self.write_record()
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({"files": self.expected}), encoding="utf-8")
        self.instances = [self.omni]
        self.active = {"vllm-omni": self.omni}

    def write_source(self, name, payload, site=None):
        path = (site or self.site) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def add_distribution(self, site, directory, name, version, legacy=False):
        path = site / directory
        path.mkdir(parents=True)
        filename = "PKG-INFO" if legacy else "METADATA"
        (path / filename).write_text("Name: " + name + "\nVersion: " + version + "\n", encoding="utf-8")
        return importlib.metadata.PathDistribution(path)

    def write_record(self, distribution=None, site=None):
        distribution = distribution or self.omni
        site = site or self.site
        rows = []
        for path in sorted((site / "vllm_omni").rglob("*.py")):
            payload = path.read_bytes()
            encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode().rstrip("=")
            rows.append([path.relative_to(site).as_posix(), "sha256=" + encoded, str(len(payload))])
        metadata_path = Path(distribution._path)
        rows.append([metadata_path.name + "/RECORD", "", ""])
        with (metadata_path / "RECORD").open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream, lineterminator="\n").writerows(rows)

    def collect(self):
        with patch.object(inventory.metadata, "distributions", return_value=self.instances), patch.object(
            inventory.metadata, "distribution", side_effect=lambda name: self.active[inventory.canonical_name(name)],
        ):
            return inventory.collect(self.manifest)

    def test_copy_source_subset_and_generated_source_are_recorded(self):
        value = self.collect()
        self.assertEqual(value["source_coverage"]["status"], "PASS")
        self.assertEqual(value["source_coverage"]["omitted_from_wheel"], ["vllm_omni/unpackaged_example.py"])
        self.assertEqual(set(value["source_sha256"]), set(self.expected) - {"vllm_omni/unpackaged_example.py"})
        self.assertEqual(set(value["generated_source_sha256"]), {inventory.GENERATED_SOURCE})
        self.assertEqual(value["source_coverage"]["recorded_source_files"], len(value["source_sha256"]) + 1)

    def test_shadow_metadata_preserves_all_instances_and_selected_one(self):
        shadow = self.add_distribution(self.root / "shadow", "vllm_omni-0.dist-info", "VLLM_Omni", "0")
        self.instances.insert(0, shadow)
        value = self.collect()
        self.assertEqual(len(value["distributions"]), 2)
        self.assertEqual(value["packages"]["vllm-omni"]["version"], "1")
        self.assertEqual(value["packages"]["vllm-omni"]["metadata_directory"], str(Path(self.omni._path).resolve()))

    def test_active_selection_change_is_visible(self):
        other_site = self.root / "other"
        other = self.add_distribution(other_site, "vllm_omni-2.dist-info", "vllm-omni", "2")
        for path in (self.site / "vllm_omni").rglob("*.py"):
            self.write_source(path.relative_to(self.site).as_posix(), path.read_bytes(), other_site)
        self.write_record(other, other_site)
        self.instances.append(other)
        before = self.collect()
        self.active["vllm-omni"] = other
        after = self.collect()
        self.assertNotEqual(before["packages"], after["packages"])
        self.assertEqual(before["distributions"], after["distributions"])
        self.assertNotEqual(before["omni_root"], after["omni_root"])

    def test_same_physical_metadata_directory_is_deduplicated(self):
        self.instances.append(importlib.metadata.PathDistribution(Path(self.omni._path)))
        self.assertEqual(len(self.collect()["distributions"]), 1)

    def test_legacy_pkg_info_is_hashed_without_record(self):
        legacy = self.add_distribution(self.site, "helper.egg-info", "My_Helper", "3", legacy=True)
        self.instances.append(legacy)
        self.active["my-helper"] = legacy
        record = self.collect()["packages"]["my-helper"]
        self.assertEqual(record["metadata_sha256"], hashlib.sha256(legacy.read_text("PKG-INFO").encode()).hexdigest())
        self.assertIsNone(record["record_sha256"])

    def test_single_file_legacy_egg_info_is_hashed_and_empty_metadata_is_rejected(self):
        path = self.site / "single_helper.egg-info"
        payload = "Name: Single_Helper\nVersion: 4\n"
        path.write_text(payload, encoding="utf-8")
        legacy = importlib.metadata.PathDistribution(path)
        self.instances.append(legacy)
        self.active["single-helper"] = legacy
        record = self.collect()["packages"]["single-helper"]
        self.assertEqual(record["metadata_sha256"], hashlib.sha256(payload.encode()).hexdigest())
        self.assertEqual(record["metadata_directory"], str(path.resolve()))
        empty = Mock(metadata={"Name": "Single_Helper"}, _path=path)
        empty.read_text.return_value = ""
        with self.assertRaisesRegex(ValueError, "Missing or empty"):
            inventory.distribution_record(empty)

    def test_recorded_missing_source_is_rejected(self):
        (self.site / inventory.CRITICAL_SOURCE_PATHS[0]).unlink()
        with self.assertRaisesRegex(ValueError, "RECORD source file is missing"):
            self.collect()

    def test_modified_recorded_source_is_rejected(self):
        self.write_source(inventory.CRITICAL_SOURCE_PATHS[0], b"# changed bytes\n")
        with self.assertRaisesRegex(ValueError, "RECORD hash differs"):
            self.collect()

    def test_recorded_wrong_size_is_rejected(self):
        record = Path(self.omni._path) / "RECORD"
        rows = list(csv.reader(record.read_text(encoding="utf-8").splitlines()))
        rows[0][2] = "999"
        with record.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)
        with self.assertRaisesRegex(ValueError, "RECORD size differs"):
            self.collect()

    def test_unrecorded_source_is_rejected(self):
        self.write_source("vllm_omni/unrecorded.py", b"# added\n")
        with self.assertRaisesRegex(ValueError, "filesystem/RECORD source mismatch"):
            self.collect()

    def test_unknown_recorded_source_is_rejected(self):
        self.write_source("vllm_omni/unknown.py", b"# added\n")
        self.write_record()
        with self.assertRaisesRegex(ValueError, "absent from COPY manifest"):
            self.collect()

    def test_other_generated_filename_is_rejected(self):
        self.write_source("vllm_omni/_generated_version.py", b"version = 1\n")
        self.write_record()
        with self.assertRaisesRegex(ValueError, "absent from COPY manifest"):
            self.collect()

    def test_missing_critical_source_is_rejected_even_if_record_agrees(self):
        (self.site / inventory.CRITICAL_SOURCE_PATHS[0]).unlink()
        self.write_record()
        with self.assertRaisesRegex(ValueError, "Required installed Omni source is missing"):
            self.collect()

    def test_generated_file_changes_are_visible_for_preservation(self):
        before = self.collect()
        self.write_source(inventory.GENERATED_SOURCE, b"__version__ = 'changed'\n")
        self.write_record()
        after = self.collect()
        self.assertNotEqual(before["generated_source_sha256"], after["generated_source_sha256"])
        self.assertEqual(before["source_sha256"], after["source_sha256"])

    def test_copy_hash_comparison_is_reserved_for_snapshot(self):
        name = inventory.CRITICAL_SOURCE_PATHS[0]
        self.write_source(name, b"# approved repair\n")
        self.write_record()
        value = self.collect()
        self.assertNotEqual(value["source_sha256"][name], self.expected[name])

    def test_unsafe_record_path_is_rejected(self):
        record = Path(self.omni._path) / "RECORD"
        with record.open("a", encoding="utf-8") as stream:
            stream.write("vllm_omni/../escape.py,sha256=bad,0\n")
        with self.assertRaisesRegex(ValueError, "Unsafe Omni source RECORD path"):
            self.collect()

    def test_duplicate_record_source_is_rejected(self):
        record = Path(self.omni._path) / "RECORD"
        first = record.read_text(encoding="utf-8").splitlines()[0]
        with record.open("a", encoding="utf-8") as stream:
            stream.write(first + "\n")
        with self.assertRaisesRegex(ValueError, "Duplicate Omni source RECORD row"):
            self.collect()


if __name__ == "__main__":
    unittest.main()
