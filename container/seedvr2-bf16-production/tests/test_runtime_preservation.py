"""Exact source/RECORD/projection checks; standard library only, no GPU claim."""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bf16_installer_test", HERE / "runtime.py")
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)
SPEC = importlib.util.spec_from_file_location("bf16_inventory_test", HERE / "inventory.py")
inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory)


class RecordPreservation(unittest.TestCase):
    def setUp(self):
        self.changes = {runtime.REL: b"new pipeline\n", runtime.HELPER: b"new helper\n"}
        self.before = (b'"unrelated,quoted.py",sha256=old,8\r\n'
                       + (runtime.REL + ",sha256=old,7\r\n").encode()
                       + b"package.dist-info/RECORD,,")

    def test_existing_rows_raw_bytes_order_and_final_termination_preserved(self):
        changed = runtime.update_record(self.before, self.changes)
        self.assertTrue(changed.startswith(b'"unrelated,quoted.py",sha256=old,8\r\n'))
        self.assertTrue(changed.endswith(b"package.dist-info/RECORD,,"))
        rows = list(csv.reader(io.StringIO(changed.decode(), newline="")))
        self.assertEqual([r[0] for r in rows], ["unrelated,quoted.py", runtime.REL, runtime.HELPER, "package.dist-info/RECORD"])
        self.assertEqual(rows[1][2], str(len(self.changes[runtime.REL])))
        self.assertEqual(rows[2][2], str(len(self.changes[runtime.HELPER])))

    def test_quoted_multiline_unrelated_csv_entry_preserved(self):
        prefix = b'"odd\nfilename.py",sha256=same,9\n'
        after = runtime.update_record(prefix + self.before, self.changes)
        self.assertTrue(after.startswith(prefix))

    def test_duplicate_malformed_existing_helper_and_missing_pipeline_rejected(self):
        cases = [self.before + ("\n" + runtime.REL + ",x,1\n").encode(), b"malformed\n",
                 self.before + ("\n" + runtime.HELPER + ",x,1\n").encode(), b"other.py,x,1\n"]
        for data in cases:
            with self.subTest(data=data), self.assertRaises(ValueError):
                runtime.update_record(data, self.changes)

    def test_unreviewed_record_change_disallowed(self):
        with self.assertRaises(ValueError):
            runtime.update_record(self.before, {**self.changes, "dingo/any.py": b"x"})


class MetadataPreservation(unittest.TestCase):
    def setUp(self):
        self.before = {"schema": "seedvr2-bf16-clean-metadata/1", "status": "PASS",
                       "packages": [{"name": "vllm-omni", "version": "0.32.0", "location": "/site"}],
                       "files": {"/site/omni/RECORD": "old", "/site/omni/METADATA": "same",
                                 "/site/dingo/RECORD": "dingo", "/site/omni/direct_url.json": "direct"}}

    def test_only_omni_record_may_change(self):
        after = copy.deepcopy(self.before)
        after["files"]["/site/omni/RECORD"] = "new"
        self.assertTrue(inventory.require_preserved(self.before, after, record_path="/site/omni/RECORD", record_sha256="new"))
        for name in ("/site/omni/METADATA", "/site/dingo/RECORD", "/site/omni/direct_url.json"):
            changed = copy.deepcopy(after)
            changed["files"][name] = "tampered"
            with self.subTest(file=name), self.assertRaises(ValueError):
                inventory.require_preserved(self.before, changed, record_path="/site/omni/RECORD", record_sha256="new")

    def test_versions_locations_duplicates_or_added_metadata_fail(self):
        for change in ("version", "location", "duplicate", "file"):
            after = copy.deepcopy(self.before)
            after["files"]["/site/omni/RECORD"] = "new"
            if change in ("version", "location"):
                after["packages"][0][change] = "changed"
            elif change == "duplicate":
                after["packages"].append(copy.deepcopy(after["packages"][0]))
            else:
                after["files"]["/site/new/METADATA"] = "new"
            with self.subTest(change=change), self.assertRaises(ValueError):
                inventory.require_preserved(self.before, after, record_path="/site/omni/RECORD", record_sha256="new")


class Projection(unittest.TestCase):
    def test_projection_preserves_version_startup_and_historical_fields(self):
        original = {"schema": "h3-sr-production-source-lock/1", "versions": {"torch": "2.13.0+cu130"},
                    "startup_sources": {"startup.sh": "unchanged"}, "dingo_changes": {"dingo/old.py": {}},
                    "parent_source_commit": "old", "parent_gateway_commit": "historical"}
        saved = copy.deepcopy(original)
        sources, preserved = {runtime.REL: "pipeline"}, {"dingo/current.py": "current", "opt/history/file": "historical"}
        result = runtime.projected_lock(original, sources, preserved)
        self.assertEqual(original, saved)
        self.assertEqual(result["parent_image"], runtime.PARENT_IMAGE)
        self.assertEqual(result["parent_source_commit"], runtime.PARENT_COMMIT)
        self.assertEqual(result["parent_source_manifest_sha256"], runtime.map_sha(sources))
        self.assertEqual(result["parent_dingo_sources"], {"dingo/current.py": "current"})
        self.assertEqual(result["dingo_changes"], {})
        for name in ("versions", "startup_sources", "parent_gateway_commit"):
            self.assertEqual(result[name], original[name])

    def test_identity_and_manifest_fail_closed(self):
        with self.assertRaises(ValueError):
            runtime.identity(runtime.PARENT_COMMIT)
        with self.assertRaises(ValueError):
            runtime.identity("short")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.json"
            runtime.dump(path, {"source_sha256": {runtime.REL: "a" * 64}, "source_manifest_sha256": "b" * 64})
            with self.assertRaises(ValueError):
                runtime.manifest(path)

    def test_source_target_does_not_allow_dingo_h3_or_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(runtime.target(root, runtime.HELPER), root / runtime.HELPER)
            for name in ("dingo/minimax_h3.py", "vllm_omni/diffusion/models/minimax_h3.py", "../pipeline.py"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    runtime.target(root, name)

    def test_plan_changes_exactly_two_sources_and_checks_all_five_parents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = {}
            for index, relative in enumerate(runtime.PARENT_FIVE):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                data = (HERE / "tests/fixtures/pipeline_seedvr2.py").read_bytes() if relative == runtime.REL else ("# source " + str(index)).encode()
                path.write_bytes(data)
                parent[relative] = hashlib.sha256(data).hexdigest()
            original = dict(parent)
            with patch.object(runtime, "PARENT_FIVE", dict(parent)):
                changes, planned, proof = runtime.checked_plan(root, parent)
                self.assertEqual(set(changes), {runtime.REL, runtime.HELPER})
                self.assertEqual(parent, original)
                self.assertEqual(set(k for k in parent if parent[k] != planned["source_sha256"][k]), {runtime.REL})
                self.assertTrue(proof["non_anchor_bytes_unchanged"])
                unchanged = next(k for k in parent if k != runtime.REL)
                (root / unchanged).write_bytes(b"source drift")
                with self.assertRaisesRegex(ValueError, "diagnostic parent"):
                    runtime.checked_plan(root, parent)

    def test_docker_retains_fixed_parent_inherited_identity_until_apply(self):
        source = (HERE / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("FROM " + runtime.PARENT_IMAGE + "\n", source)
        self.assertLess(source.index("runtime.py apply"), source.index("ENV SEEDVR2_BF16_SOURCE_COMMIT="))
        copies = [line for line in source.splitlines() if line.startswith("COPY ")]
        self.assertEqual(len(copies), 2)
        self.assertTrue(all("seedvr2-bf16-production" in line for line in copies))
        self.assertNotIn("ENTRYPOINT ", source)
        self.assertNotIn("LD_LIBRARY_PATH=", source)
        self.assertIn("USER dynamo\nRUN --network=none", source)
        self.assertIn('org.opencontainers.image.revision="${SEEDVR2_BF16_SOURCE_COMMIT}"', source)
        self.assertIn('org.opencontainers.image.base.name="' + runtime.PARENT_IMAGE + '"', source)

    def test_windows_or_wrong_uid_cannot_emit_linux_pass(self):
        with patch.object(runtime.os, "name", "nt"):
            with self.assertRaisesRegex(ValueError, "Linux Python3.12 UID1000"):
                runtime.validate(Path("unused"))


class RealInheritedVerifierProjection(unittest.TestCase):
    """Run the unchanged verifier against a projected filesystem, not a clone.

    Only installed-package discovery and the historical snapshot provider are
    replaced with a bounded fixture. The real verify/source/recipe/startup gates
    execute, including their failure paths. Actual-image discovery runs in CI.
    """
    def setUp(self):
        original = Path("/opt/h3-sr/runtime.py")
        if not original.is_file():
            original = HERE.parent / "h3-sr-production/runtime.py"
        self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(), runtime.BASE_RUNTIME_SHA256)
        spec = importlib.util.spec_from_file_location("test_actual_inherited_verifier", original)
        self.base = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.base)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.base.HERE, self.base.INFO = root / "h3-sr", root / "h3-sr/build-info"
        self.site = root / "site"
        self.omni_record, self.dingo_record = root / "omni-RECORD", root / "dingo-RECORD"
        self.omni_record.write_bytes(b"new reviewed RECORD")
        self.dingo_record.write_bytes(b"unchanged Dingo RECORD")
        for relative in (runtime.REL, runtime.HELPER, "vllm_omni/unchanged_native.so"):
            path = self.site / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("installed " + relative).encode())
        startup = root / "h3-sr-startup/start.sh"
        startup.parent.mkdir()
        startup.write_bytes(b"unaltered startup")
        self.preserved = {"dingo/current.py": "actual26d", "opt/history/evidence": "unchanged"}
        self.packages = [("vllm-omni", "0.32.0"), ("ai-dingo", "1.3.0")]
        self.sources = self.base.source_map(self.site, "vllm_omni")
        self.startup = self.base.startup_sources()
        original_lock = {"schema": "h3-sr-production-source-lock/1", "versions": {"vllm-omni": "0.32.0"},
                         "startup_sources": self.startup}
        lock = runtime.projected_lock(original_lock, {runtime.REL: "original"}, self.preserved)
        runtime.dump(self.base.HERE / "inputs/source-lock.json", lock)
        runtime.dump(self.base.HERE / "inputs/expected-source-manifest.json",
                     {"source_sha256": self.sources, "source_manifest_sha256": runtime.map_sha(self.sources)})
        runtime.dump(self.base.INFO / "preserved-parent-sources.json", self.preserved)
        runtime.dump(self.base.INFO / "package-inventory.json", self.packages)
        self.receipt = {"schema": "h3sr-production-application/1", "status": "PASS", "source_commit": "a" * 40,
                        "parent_image": runtime.PARENT_IMAGE, "source_manifest_sha256": runtime.map_sha(self.sources),
                        "omni_changes": {name: self.sources[name] for name in (runtime.REL, runtime.HELPER)},
                        "dingo_changes": {}, "recipe_sources": self.base.recipe_map(), "startup_sources": self.startup,
                        "omni_record_sha256": runtime.sha(self.omni_record.read_bytes()),
                        "dingo_record_sha256": runtime.sha(self.dingo_record.read_bytes()),
                        "delta_lineage": {"method": "projection; historical apply NOT replayed"}}
        runtime.dump(self.base.INFO / "application.json", self.receipt)
        self.addCleanup(patch.stopall)
        patch.object(self.base, "parent_module", return_value=SimpleNamespace(preserved_snapshot=lambda: dict(self.preserved))).start()
        patch.object(self.base, "distribution", side_effect=lambda name: (None, self.site,
                     self.omni_record if name == "vllm-omni" else self.dingo_record)).start()
        patch.object(self.base, "inventory", side_effect=lambda: list(self.packages)).start()
        patch.object(self.base.md, "version", return_value="0.32.0").start()
        patch.dict(os.environ, {"H3_SR_PRODUCTION_SOURCE_COMMIT": "a" * 40,
                                "H3_SR_PRODUCTION_PARENT_IMAGE": runtime.PARENT_IMAGE}).start()
        real_read = Path.read_bytes

        def pth_read(path):
            if path.as_posix() == "/usr/local/lib/python3.12/dist-packages/h3_sr_production.pth":
                return b"/opt/h3-sr\n"
            return real_read(path)

        patch.object(Path, "read_bytes", pth_read).start()

    def test_projected_child_passes_unchanged_original_verifier(self):
        proof = self.base.verify()
        self.assertEqual(proof["status"], "PASS")
        self.assertEqual(proof["source_sha256"], self.sources)
        self.assertEqual(proof["changed_sources"], sorted((runtime.REL, runtime.HELPER)))
        self.assertEqual(proof["dingo_changes"], {})

    def test_unrelated_source_or_native_byte_change_still_rejected(self):
        (self.site / "vllm_omni/unchanged_native.so").write_bytes(b"tampered native")
        with self.assertRaisesRegex(ValueError, "Complete installed Omni"):
            self.base.verify()

    def test_legacy_grandparent_preserved_snapshot_cannot_be_reused(self):
        runtime.dump(self.base.INFO / "preserved-parent-sources.json", {"dingo/current.py": "oldc88"})
        with self.assertRaisesRegex(ValueError, "Gateway or historical"):
            self.base.verify()

    def test_unrelated_omni_record_change_rejected(self):
        self.omni_record.write_bytes(b"unreviewed unrelated row")
        with self.assertRaisesRegex(ValueError, "Omni RECORD"):
            self.base.verify()

    def test_dingo_record_and_package_inventory_still_guarded(self):
        self.dingo_record.write_bytes(b"changed Dingo RECORD")
        with self.assertRaisesRegex(ValueError, "package/RECORD"):
            self.base.verify()
        self.dingo_record.write_bytes(b"unchanged Dingo RECORD")
        self.packages.append(("unreviewed", "1"))
        with self.assertRaisesRegex(ValueError, "package/RECORD"):
            self.base.verify()

    def test_recipe_input_and_startup_bytes_still_guarded(self):
        path = self.base.HERE / "inputs/unreviewed.json"
        path.write_bytes(b"unexpected")
        with self.assertRaisesRegex(ValueError, "helper/profile/recipe"):
            self.base.verify()
        path.unlink()
        (self.base.HERE.parent / "h3-sr-startup/start.sh").write_bytes(b"changed startup")
        with self.assertRaisesRegex(ValueError, "startup bundle"):
            self.base.verify()

    def test_full_apply_projects_real_verifier_and_retains_original_evidence(self):
        # Turn the bounded child fixture into the original 26d fixture, then run
        # the real installer including source/RECORD writes and both verifiers.
        parent_image = "registry.example/old@sha256:" + "c" * 64
        parent_sources = {}
        for index, relative in enumerate(runtime.PARENT_FIVE):
            path = self.site / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            data = (HERE / "tests/fixtures/pipeline_seedvr2.py").read_bytes() if relative == runtime.REL else ("# original " + str(index)).encode()
            path.write_bytes(data)
            parent_sources[relative] = runtime.sha(data)
        (self.site / runtime.HELPER).unlink()
        self.omni_record.write_bytes((runtime.REL + ",sha256=old,1\nother.py,sha256=same,5\n").encode())
        original_module = Path("/opt/h3-sr/runtime.py")
        if not original_module.is_file():
            original_module = HERE.parent / "h3-sr-production/runtime.py"
        (self.base.HERE / "runtime.py").write_bytes(original_module.read_bytes())
        original_map = self.base.source_map(self.site, "vllm_omni")
        old_lock = {"schema": "h3-sr-production-source-lock/1", "parent_image": parent_image,
                    "versions": {"vllm-omni": "0.32.0"}, "startup_sources": self.startup}
        runtime.dump(self.base.HERE / "inputs/source-lock.json", old_lock)
        runtime.dump(self.base.HERE / "inputs/expected-source-manifest.json",
                     {"source_sha256": original_map, "source_manifest_sha256": runtime.map_sha(original_map)})
        grandparent_snapshot = {**self.preserved, "dingo/current.py": "historicalc88"}
        runtime.dump(self.base.INFO / "preserved-parent-sources.json", grandparent_snapshot)
        old_receipt = {**self.receipt, "source_commit": runtime.PARENT_COMMIT, "parent_image": parent_image,
                       "source_manifest_sha256": runtime.map_sha(original_map), "omni_changes": {},
                       "dingo_changes": {"dingo/current.py": "actual26d"},
                       "omni_record_sha256": runtime.sha(self.omni_record.read_bytes())}
        old_receipt["recipe_sources"] = self.base.recipe_map()
        runtime.dump(self.base.INFO / "application.json", old_receipt)
        child_root = self.base.HERE.parent / "seedvr2-bf16"
        for relative in ("runtime.py", "inventory.py", "core/bf16_runtime.py", "core/source_patch.py"):
            path = child_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((HERE / relative).read_bytes())
        parent_manifest = {"source_sha256": original_map, "source_manifest_sha256": runtime.map_sha(original_map)}
        runtime.dump(child_root / "inputs/parent-source-manifest.json", parent_manifest)
        before_record = self.omni_record.read_bytes()
        metadata = {"schema": "seedvr2-bf16-clean-metadata/1", "status": "PASS",
                    "packages": [{"name": name, "version": version} for name, version in self.packages],
                    "files": {str(self.omni_record): runtime.sha(before_record),
                              str(self.dingo_record): runtime.sha(self.dingo_record.read_bytes())}}

        def clean(_directory):
            return {**copy.deepcopy(metadata), "files": {**metadata["files"],
                    str(self.omni_record): runtime.sha(self.omni_record.read_bytes())}}

        child_info = child_root / "build-info"
        with patch.object(runtime, "HERE", child_root), patch.object(runtime, "INFO", child_info), \
             patch.object(runtime, "BASE", self.base.HERE), patch.object(runtime, "PARENT_FIVE", parent_sources), \
             patch.object(runtime, "inherited", return_value=self.base), \
             patch.object(runtime, "base_fresh_verify", side_effect=lambda _directory: self.base.verify()), \
             patch.object(runtime, "clean_inventory", side_effect=clean), \
             patch.object(runtime.os, "geteuid", return_value=0, create=True), \
             patch.dict(os.environ, {"H3_SR_PRODUCTION_SOURCE_COMMIT": runtime.PARENT_COMMIT,
                                     "H3_SR_PRODUCTION_PARENT_IMAGE": parent_image}):
            changes, expected, _ = runtime.checked_plan(self.site, original_map)
            runtime.dump(child_root / "inputs/expected-source-manifest.json", expected)
            release = {"versions": old_lock["versions"], "parent_source_manifest_sha256": parent_manifest["source_manifest_sha256"],
                       "expected_source_manifest_sha256": expected["source_manifest_sha256"],
                       "pipeline_patched_sha256": runtime.sha(changes[runtime.REL])}
            with patch.object(runtime, "policy", return_value=release):
                proof = runtime.apply("a" * 40)
            self.assertEqual(proof["status"], "PASS")
            self.assertEqual(proof["source_sha256"], expected["source_sha256"])
            self.assertEqual(runtime.read(child_info / "base-build-info/application.json"), old_receipt)
            self.assertEqual(runtime.read(child_info / "base-build-info/preserved-parent-sources.json"), grandparent_snapshot)
            self.assertEqual(runtime.read(self.base.INFO / "preserved-parent-sources.json"), self.preserved)
            projected = runtime.read(self.base.INFO / "application.json")
            self.assertEqual(projected["dingo_changes"], {})
            self.assertEqual(set(projected["omni_changes"]), {runtime.REL, runtime.HELPER})
            self.assertIn("NOT replayed", projected["delta_lineage"]["method"])
            self.assertIn("base-inputs/source-lock.json", runtime.read(child_info / "application.json")["base_evidence_sources"])
            self.assertEqual((self.base.HERE / "runtime.py").read_bytes(), original_module.read_bytes())


if __name__ == "__main__":
    unittest.main()
