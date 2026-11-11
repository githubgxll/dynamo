"""Regression fixtures for publication identity, unknown state and retry handling."""
from __future__ import annotations

import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import subprocess
from contextlib import redirect_stdout
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("h3_image_worker_ci_under_test", ROOT / ".github/scripts/h3_image_worker_ci.py")
ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ci)
COMMIT = "a" * 40
DIGEST = "sha256:" + "b" * 64
CHILD_ID = "sha256:" + "c" * 64
CURRENT = {"commit": COMMIT, "run_id": "431", "run_attempt": 3}
ENV = {"GITHUB_REF_NAME": ci.BRANCH, "GITHUB_EVENT_NAME": "push", "GITHUB_SHA": COMMIT,
       "GITHUB_RUN_ID": "431", "GITHUB_RUN_ATTEMPT": "3"}
IMAGE = ci.image_tag(COMMIT)
# Keep atomic-receipt fixtures outside the Windows virtualized temporary root.
TEMP_ROOT = ROOT / ".h3-experimental-evidence/h3-image-worker/ci-unit-fixtures"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)


def configs():
    parent = {"Id": ci.PARENT_ID, "Os": "linux", "Architecture": "amd64", "RootFS": {"Layers": ["parent"]},
              "Config": {"User": "", "Env": ["PATH=/usr/bin", "LD_PRELOAD=/opt/frozen-cudnn/libcudnn.so.9"],
                         "Entrypoint": None, "Cmd": None, "WorkingDir": "/app"}}
    child = copy.deepcopy(parent)
    child["Id"] = CHILD_ID
    child["RootFS"]["Layers"].append("worker")
    child["Config"]["User"] = "1000:1000"
    child["Config"]["Env"] += ["H3_IMAGE_WORKER_SOURCE_COMMIT=" + COMMIT,
                               "H3_IMAGE_WORKER_PARENT_IMAGE=" + ci.PARENT]
    child["Config"]["Labels"] = {"org.opencontainers.image.revision": COMMIT,
                                "org.opencontainers.image.base.name": ci.PARENT}
    return parent, child


def source_inventory(patched=False):
    copied, patches = ci.source_inputs()
    names = sorted(patches) + [next(name for name in copied if name not in patches)]
    files = {name: patches[name] if patched and name in patches else copied[name] for name in names}
    generated = {ci.GENERATED_SOURCE: "d" * 64}
    coverage = {"status": "PASS", "copied_source_files": len(copied),
                "installed_record_source_files": len(files) + len(generated),
                "recorded_source_files": len(files) + len(generated),
                "parent_projected_source_files": len(files), "missing_record_sources": [],
                "unexpected_installed_sources": [], "critical_paths_verified": sorted(patches),
                "generated_source_files": sorted(generated), "omitted_from_wheel": sorted(set(copied) - set(files))}
    return {"source_sha256": files, "generated_source_sha256": generated, "source_coverage": coverage}


def parent_inventory():
    packages = {name: {"canonical_name": name, "version": "1.0", "metadata_directory": "/site-packages/" + name + ".dist-info",
                       "metadata_sha256": "e" * 64, "record_sha256": "f" * 64, "direct_url_sha256": "0" * 64}
                for name in ("torch", "vllm", "vllm-omni", "lmcache")}
    return {"packages": packages, "distributions": [{"canonical_name": name, **item} for name, item in packages.items()],
            **source_inventory()}


def parent_probe_files(directory):
    ci.dump(directory / "parent-inventory.json", parent_inventory())
    ci.dump(directory / "parent-probe-result.json", {"status": "PASS"})
    return {"status": "PASS"}


def native_evidence():
    path = ROOT / "container/h3-image-worker/native.lock.json"
    lock = ci.read(path)
    root = "/site-packages/lmcache"
    files = [{"path": item["path"], "resolved_path": root + "/" + item["path"].split("/", 1)[1],
              "sha256": item["sha256"], "bytes": item["bytes"], "record_verified": True} for item in lock["files"]]
    imports = [{"module": "torch", "origin": "/site-packages/torch/__init__.py"},
               {"module": "lmcache", "origin": root + "/__init__.py"}]
    for name in ("lmcache_native", "cuda_ops"):
        origin = next(item["resolved_path"] for item in files if item["path"].split("/", 1)[1].startswith(name + "."))
        imports.append({"module": "lmcache." + name, "origin": origin, "locked_origin_verified": True})
    return {"status": "PASS_STRICT_NATIVE_CPU_LOADER", "distribution": "lmcache", "version": lock["version"],
            "distribution_root": root, "native_lock_sha256": ci.hashlib.sha256(path.read_bytes()).hexdigest(),
            "files": files, "imports": imports, "gpu_validation": "NOT_RUN", "driver_error_skips": False}


def reports(directory):
    sources = source_inventory(patched=True)
    files = sources["source_sha256"]
    source_hash = ci.hash_value(files)
    profiles = ci.read(ROOT / "container/h3-image-worker/profiles.json")["profiles"]
    runtime = {"status": "PASS_CPU_ONLY", "source_commit": COMMIT, "parent_image": ci.PARENT,
               "parent_image_id": ci.PARENT_ID, "euid": 1000, "inventory_preserved": True,
               "errors": [], "source_manifest_sha256": source_hash,
               "cpu_checks": {"status": "PASS_CPU_RUNTIME_ONLY", "tests_run": 12,
                              "failures": 0, "errors": 0, "skipped": 0},
               "cli_profiles": [{"profile": item["id"], "status": "PASS", "gpus": item["gpus"],
                                 "dit_tp": item["dit_tp"], "te_tp": item["te_tp"]} for item in profiles],
               "native_validation": native_evidence()}
    proof = {"status": "PASS", "source_commit": COMMIT, "parent_image": ci.PARENT,
             **sources, "source_manifest_sha256": source_hash}
    ci.dump(directory / "runtime-validation.json", runtime)
    ci.dump(directory / "source-verification.json", proof)
    ci.dump(directory / "native-validation.json", runtime["native_validation"])
    return runtime, proof


def transaction(directory, reused=False):
    parent, child = configs()
    prepared = {"schema": "h3-image-worker-ci-preparation/1", **CURRENT, "image": IMAGE,
                "reused": reused, "existing_digest": DIGEST if reused else None, "recipe_sha256": "e" * 64}
    runtime, proof = reports(directory / "final")
    candidate = {"schema": "h3-image-worker-candidate/1", **CURRENT, "status": "PASS_CPU_ONLY",
                 "image": IMAGE, "image_id": CHILD_ID, "recipe_sha256": prepared["recipe_sha256"],
                 "source_manifest_sha256": runtime["source_manifest_sha256"]}
    ci.dump(directory / "preparation.json", prepared)
    ci.dump(directory / "candidate.json", candidate)
    return prepared, candidate, parent, child


class ManifestClassification(unittest.TestCase):
    def classify(self, text, returncode=1):
        return ci.classify_manifest_lookup(SimpleNamespace(returncode=returncode, stdout="", stderr=text), IMAGE)

    def test_success_reports_exists(self):
        self.assertEqual(self.classify("", 0), "EXISTS")

    def test_precise_missing_manifest_reports_absence(self):
        for text in ("manifest unknown", "no such manifest", "manifest unknown status 404",
                     "unknown: artifact " + IMAGE.split("/", 1)[1] + " not found"):
            with self.subTest(text=text): self.assertEqual(self.classify(text), "ABSENT_CONFIRMED")

    def test_tls_network_auth_rate_limit_never_report_absence(self):
        for text in ("manifest unknown TLS EOF", "manifest unknown unauthorized", "manifest unknown forbidden",
                     "manifest unknown status code 500", "manifest unknown status: 429", "manifest unknown rate limit",
                     "no such manifest: proxyconnect failed", "manifest unknown timed out"):
            with self.subTest(text=text): self.assertEqual(self.classify(text), "UNKNOWN")

    def test_unrelated_404_and_wrong_harbor_repository_are_unknown(self):
        for text in ("404 not found", "unknown: artifact other/repo:tag not found", "repository not found"):
            with self.subTest(text=text): self.assertEqual(self.classify(text), "UNKNOWN")

    def test_tool_failure_is_unknown(self):
        self.assertEqual(self.classify("manifest unknown", 2), "UNKNOWN")

    def test_pull_digest_requires_one_full_value(self):
        self.assertEqual(ci.pull_digest("Digest: " + DIGEST + "\n"), DIGEST)
        for text in ("", "Digest: sha256:short", "Digest: " + DIGEST + "\nDigest: sha256:" + "f" * 64):
            with self.subTest(text=text), self.assertRaises(ValueError): ci.pull_digest(text)


class IdentityAndConfig(unittest.TestCase):
    def test_real_attempt_three_is_valid(self):
        self.assertEqual(ci.run_identity(ENV), CURRENT)

    def test_wrong_branch_event_and_nonpositive_identity_fail(self):
        for key, value in (("GITHUB_REF_NAME", "main"), ("GITHUB_EVENT_NAME", "workflow_dispatch"),
                           ("GITHUB_RUN_ATTEMPT", "0"), ("GITHUB_RUN_ID", ""), ("GITHUB_SHA", "short")):
            with self.subTest(key=key), self.assertRaises(ValueError): ci.run_identity({**ENV, key: value})

    def test_reports_from_another_attempt_fail(self):
        with mock.patch.dict(os.environ, ENV), self.assertRaises(ValueError):
            ci.require_current({**CURRENT, "run_attempt": 2})

    def test_reviewed_nonroot_child_extends_exact_parent(self):
        parent, child = configs()
        ci.compare_config(parent, child, COMMIT, {"expected_image_config": {"User": "1000:1000"}})

    def test_cudnn_preload_change_fails(self):
        parent, child = configs()
        child["Config"]["Env"][1] = "LD_PRELOAD="
        with self.assertRaises(ValueError):
            ci.compare_config(parent, child, COMMIT, {"expected_image_config": {"User": "1000:1000"}})

    def test_reviewed_environment_value_can_override_parent(self):
        parent, child = configs()
        parent["Config"]["Env"].append("HOME=/root")
        child["Config"]["Env"].append("HOME=/work")
        ci.compare_config(parent, child, COMMIT,
                          {"expected_image_config": {"User": "1000:1000"},
                           "expected_environment_additions": {"HOME": "/work"}})

    def test_parent_layer_replacement_fails(self):
        parent, child = configs()
        child["RootFS"]["Layers"][0] = "different-parent"
        with self.assertRaises(ValueError):
            ci.compare_config(parent, child, COMMIT, {"expected_image_config": {"User": "1000:1000"}})

    def test_unreviewed_entrypoint_and_root_fail(self):
        parent, child = configs()
        for field, value in (("Entrypoint", ["unreviewed"]), ("User", "0")):
            changed = copy.deepcopy(child)
            changed["Config"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                ci.compare_config(parent, changed, COMMIT, {"expected_image_config": {"User": "1000:1000"}})


class RuntimeProof(unittest.TestCase):
    def test_matching_installed_proof_passes(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            directory = Path(temp)
            runtime, proof = reports(directory)
            self.assertEqual(ci.validate_reports(directory, COMMIT), (runtime, proof))

    def test_complete_record_projection_may_be_smaller_than_copy_tree(self):
        sources = source_inventory(patched=True)
        self.assertLess(len(sources["source_sha256"]), sources["source_coverage"]["copied_source_files"])
        self.assertEqual(ci.validate_source_coverage(sources, patched=True)["status"], "PASS")

    def test_root_changed_inventory_or_wrong_parent_fails(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            directory = Path(temp)
            for key, value in (("euid", 0), ("inventory_preserved", False), ("parent_image_id", CHILD_ID),
                               ("errors", ["actual failure"]), ("source_commit", "f" * 40)):
                runtime, _ = reports(directory)
                runtime[key] = value
                ci.dump(directory / "runtime-validation.json", runtime)
                with self.subTest(key=key), self.assertRaises(ValueError): ci.validate_reports(directory, COMMIT)

    def test_empty_or_malformed_source_proof_fails(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            directory = Path(temp)
            for files in ({}, {"file.py": "short"}, {"file.py": None}):
                _, proof = reports(directory)
                proof["source_sha256"] = files
                ci.dump(directory / "source-verification.json", proof)
                with self.subTest(files=files), self.assertRaises(ValueError): ci.validate_reports(directory, COMMIT)

    def test_source_hash_or_identity_change_fails(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            directory = Path(temp)
            for key, value in (("source_manifest_sha256", "f" * 64), ("source_commit", "f" * 40),
                               ("parent_image", "other-parent")):
                _, proof = reports(directory)
                proof[key] = value
                ci.dump(directory / "source-verification.json", proof)
                with self.subTest(key=key), self.assertRaises(ValueError): ci.validate_reports(directory, COMMIT)

    def test_missing_failed_or_empty_actual_cpu_checks_fail(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            directory = Path(temp)
            for key, value in (("status", "PENDING"), ("tests_run", 0), ("tests_run", "12"),
                               ("failures", 1), ("errors", 1), ("skipped", 1)):
                runtime, _ = reports(directory)
                runtime["cpu_checks"][key] = value
                ci.dump(directory / "runtime-validation.json", runtime)
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    ci.validate_reports(directory, COMMIT)

    def test_incomplete_cli_profiles_and_source_inventory_fail(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            directory = Path(temp)
            runtime, _ = reports(directory)
            runtime["cli_profiles"].pop()
            ci.dump(directory / "runtime-validation.json", runtime)
            with self.assertRaises(ValueError): ci.validate_reports(directory, COMMIT)
            _, proof = reports(directory)
            proof["source_sha256"].pop(next(iter(proof["source_sha256"])))
            proof["source_manifest_sha256"] = ci.hash_value(proof["source_sha256"])
            ci.dump(directory / "source-verification.json", proof)
            with self.assertRaises(ValueError): ci.validate_reports(directory, COMMIT)

    def test_source_coverage_requires_counts_omission_and_both_critical_modules(self):
        for key, value in (("copied_source_files", 0), ("recorded_source_files", 0),
                           ("installed_record_source_files", 0), ("parent_projected_source_files", 0),
                           ("omitted_from_wheel", []), ("critical_paths_verified", []),
                           ("missing_record_sources", ["missing.py"]),
                           ("unexpected_installed_sources", ["extra.py"]), ("generated_source_files", [])):
            inventory = source_inventory(patched=True)
            inventory["source_coverage"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): ci.validate_source_coverage(inventory, patched=True)

    def test_unreviewed_copied_or_generated_module_is_rejected(self):
        inventory = source_inventory(patched=True)
        name = next(iter(inventory["source_sha256"]))
        inventory["source_sha256"][name] = "0" * 64
        with self.assertRaises(ValueError): ci.validate_source_coverage(inventory, patched=True)
        inventory = source_inventory(patched=True)
        inventory["generated_source_sha256"]["vllm_omni/unreviewed.py"] = "0" * 64
        with self.assertRaises(ValueError): ci.validate_source_coverage(inventory, patched=True)


class ParentProbe(unittest.TestCase):
    def invoke(self, directory, result=None, error=None):
        response = result or SimpleNamespace(returncode=0, stdout=json.dumps(parent_inventory()), stderr="")
        console = io.StringIO()
        with mock.patch.dict(os.environ, ENV), redirect_stdout(console), \
                mock.patch.object(ci, "command", side_effect=error, return_value=response) as command:
            value = ci.parent_probe(directory)
        return value, command, console.getvalue()

    def test_parent_probe_is_uid1000_offline_readonly_and_keeps_both_streams(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            directory = Path(temp)
            record, command, _ = self.invoke(directory)
            arguments = command.call_args.args[0]
            self.assertIn("--read-only", arguments)
            self.assertIn("--network", arguments)
            self.assertIn("none", arguments)
            self.assertIn("1000:1000", arguments)
            self.assertIn(ci.PARENT_ID, arguments)
            self.assertNotIn("--privileged", arguments)
            self.assertEqual(command.call_args.kwargs["timeout"], ci.PARENT_PROBE_TIMEOUT_SECONDS)
            self.assertEqual(record["status"], "PASS")
            self.assertTrue((directory / "parent-inventory.json").is_file())
            self.assertEqual((directory / "parent-inventory.stderr.txt").read_text(), "")
            self.assertEqual(ci.read(directory / "parent-probe-result.json")["returncode"], 0)

    def test_nonzero_subprocess_preserves_root_cause_and_stops(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            directory = Path(temp)
            response = SimpleNamespace(returncode=7, stdout="partial evidence", stderr="the actual metadata traceback")
            with self.assertRaises(RuntimeError): self.invoke(directory, result=response)
            result = ci.read(directory / "parent-probe-result.json")
            self.assertEqual(result["returncode"], 7)
            self.assertEqual((directory / "parent-inventory.stderr.txt").read_text(), response.stderr)
            self.assertFalse((directory / "parent-inventory.json").exists())

    def test_timeout_preserves_partial_stdout_and_stderr(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            directory = Path(temp)
            error = subprocess.TimeoutExpired(["docker", "run"], 180, output=b"partial JSON", stderr=b"blocked read")
            with self.assertRaises(RuntimeError): self.invoke(directory, error=error)
            result = ci.read(directory / "parent-probe-result.json")
            self.assertEqual(result["error_type"], "TimeoutExpired")
            self.assertEqual((directory / "parent-inventory.stdout.txt").read_text(), "partial JSON")
            self.assertEqual((directory / "parent-inventory.stderr.txt").read_text(), "blocked read")

    def test_invalid_json_preserves_process_output_and_diagnostic(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            directory = Path(temp)
            response = SimpleNamespace(returncode=0, stdout="not JSON", stderr="a warning")
            with self.assertRaises(RuntimeError): self.invoke(directory, result=response)
            self.assertEqual(ci.read(directory / "parent-probe-result.json")["error_type"], "JSONDecodeError")
            self.assertEqual((directory / "parent-inventory.stdout.txt").read_text(), "not JSON")
            self.assertEqual((directory / "parent-inventory.stderr.txt").read_text(), "a warning")

    def test_missing_tool_still_produces_failure_record(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            directory = Path(temp)
            with self.assertRaises(RuntimeError): self.invoke(directory, error=FileNotFoundError("docker unavailable"))
            self.assertEqual(ci.read(directory / "parent-probe-result.json")["error_type"], "FileNotFoundError")
            self.assertTrue((directory / "parent-inventory.stdout.txt").exists())
            self.assertTrue((directory / "parent-inventory.stderr.txt").exists())

    def test_active_and_shadow_metadata_instances_are_retained(self):
        inventory = parent_inventory()
        shadow = {**inventory["distributions"][0], "metadata_directory": "/other-site/torch.dist-info", "version": "old"}
        inventory["distributions"].append(shadow)
        ci.validate_package_inventory(inventory)
        for broken in ({**inventory, "packages": {}}, {**inventory, "distributions": []}):
            with self.assertRaises(ValueError): ci.validate_package_inventory(broken)
        inventory["packages"]["torch"]["metadata_directory"] = "/unobserved/torch.dist-info"
        with self.assertRaises(ValueError): ci.validate_package_inventory(inventory)

    def test_missing_optional_metadata_files_remain_distinct_from_empty_file_hash(self):
        inventory = parent_inventory()
        for item in inventory["distributions"]:
            item["record_sha256"] = None
            item["direct_url_sha256"] = None
        for item in inventory["packages"].values():
            item["record_sha256"] = None
            item["direct_url_sha256"] = None
        ci.validate_package_inventory(inventory)
        inventory["packages"]["torch"]["record_sha256"] = ci.hashlib.sha256(b"").hexdigest()
        with self.assertRaises(ValueError): ci.validate_package_inventory(inventory)


class NativeEvidence(unittest.TestCase):
    def check(self, directory, evidence):
        ci.dump(directory / "native-validation.json", evidence)
        return ci.validate_native_reports(directory, {"native_validation": evidence})

    def test_four_record_verified_artifacts_and_two_actual_imports_pass(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            evidence = native_evidence()
            self.assertEqual(self.check(Path(temp), evidence), evidence)

    def test_actual_native_file_and_runtime_copy_must_match(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            evidence = native_evidence()
            ci.dump(Path(temp) / "native-validation.json", evidence)
            changed = {**evidence, "status": "PENDING"}
            with self.assertRaises(ValueError): ci.validate_native_reports(Path(temp), {"native_validation": changed})

    def test_native_lock_identity_version_and_no_driver_skip_are_required(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            for key, value in (("status", "FAILED_NATIVE_CHECK"), ("distribution", "other"), ("version", "old"),
                               ("native_lock_sha256", "0" * 64), ("gpu_validation", "PASS"),
                               ("driver_error_skips", True), ("driver_error_skips", None)):
                evidence = native_evidence()
                evidence[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError): self.check(Path(temp), evidence)

    def test_missing_duplicate_or_extra_native_artifacts_fail(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            original = native_evidence()
            files = original["files"]
            for changed in (files[:3], files + [files[0]], [files[0], files[0], files[2], files[3]]):
                evidence = {**original, "files": changed}
                with self.subTest(length=len(changed)), self.assertRaises(ValueError): self.check(Path(temp), evidence)

    def test_native_bytes_record_and_active_origin_are_locked(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            for key, value in (("sha256", "0" * 64), ("bytes", 1), ("record_verified", False),
                               ("record_verified", 1), ("resolved_path", "/unrelated/cuda_ops.so")):
                evidence = native_evidence()
                evidence["files"][0][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError): self.check(Path(temp), evidence)

    def test_real_cuda_and_native_imports_need_unique_locked_origins(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            for field, value in (("origin", "/other/native.so"), ("locked_origin_verified", False),
                                 ("module", "lmcache.c_ops")):
                evidence = native_evidence()
                evidence["imports"][-1][field] = value
                with self.subTest(field=field), self.assertRaises(ValueError): self.check(Path(temp), evidence)
            evidence = native_evidence()
            evidence["imports"].append(copy.deepcopy(evidence["imports"][-1]))
            with self.assertRaises(ValueError): self.check(Path(temp), evidence)

    def test_active_distribution_root_must_be_canonical_absolute_linux_path(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            for root in ("relative/lmcache", "/site/../lmcache", "/site//lmcache", ""):
                evidence = native_evidence()
                evidence["distribution_root"] = root
                with self.subTest(root=root), self.assertRaises(ValueError): self.check(Path(temp), evidence)


class TarObservation(unittest.TestCase):
    def invoke(self, directory, observation=None, error=None, returncode=0, stderr=""):
        if observation is None:
            observation = {"euid": 1000, "tar_path": "/usr/bin/tar", "version_returncode": 0,
                           "version_stdout": "tar (GNU tar) 1.34\n", "version_stderr": ""}
        result = SimpleNamespace(returncode=returncode, stdout=json.dumps(observation), stderr=stderr)
        with mock.patch.dict(os.environ, ENV), mock.patch.object(ci, "command", side_effect=error, return_value=result) as command:
            record = ci.observe_tar(directory, ci.REPOSITORY + "@" + DIGEST, CHILD_ID)
        return record, command

    def test_available_tar_is_observed_readonly_offline_and_bound_to_pulled_identity(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            directory = Path(temp)
            record, command = self.invoke(directory)
            arguments = command.call_args.args[0]
            self.assertEqual(record["status"], "AVAILABLE")
            self.assertEqual(record["image_id"], CHILD_ID)
            self.assertEqual(record["published_image"], ci.REPOSITORY + "@" + DIGEST)
            self.assertIn(CHILD_ID, arguments)
            self.assertIn("--read-only", arguments)
            self.assertIn("1000:1000", arguments)
            self.assertIn("none", arguments)
            self.assertEqual(record["tar_path"], "/usr/bin/tar")
            self.assertIn("GNU tar", record["version_stdout"])
            self.assertTrue((directory / "tar-validation.json").is_file())

    def test_missing_tar_is_information_and_never_raises(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            missing = {"euid": 1000, "tar_path": None, "version_returncode": None,
                       "version_stdout": "", "version_stderr": ""}
            record, _ = self.invoke(Path(temp), observation=missing)
            self.assertEqual(record["status"], "MISSING")

    def test_docker_failure_is_preserved_without_changing_existing_publication(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            directory = Path(temp)
            publication = {"status": "PASS_PUBLISHED_AND_PULLED_CPU_VALIDATED", "digest": DIGEST, "image_id": CHILD_ID}
            ci.dump(directory / "publication.json", publication)
            record, _ = self.invoke(directory, returncode=125, stderr="OCI runtime unavailable")
            self.assertEqual(record["status"], "OBSERVATION_ERROR")
            self.assertEqual(record["docker_returncode"], 125)
            self.assertEqual((directory / "tar-validation.stderr.txt").read_text(), "OCI runtime unavailable")
            self.assertEqual(ci.read(directory / "publication.json"), publication)

    def test_timeout_keeps_partial_diagnostics_and_pulled_identity(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            error = subprocess.TimeoutExpired(["docker", "run"], 30, output=b"partial", stderr=b"daemon blocked")
            record, _ = self.invoke(Path(temp), error=error)
            self.assertEqual(record["status"], "OBSERVATION_ERROR")
            self.assertEqual(record["error_type"], "TimeoutExpired")
            self.assertEqual(record["image_id"], CHILD_ID)
            self.assertEqual((Path(temp) / "tar-validation.stderr.txt").read_text(), "daemon blocked")

    def test_malformed_json_or_wrong_uid_never_fabricates_tar_availability(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            for number, text in enumerate(("not JSON", json.dumps({"euid": 0}))):
                result = SimpleNamespace(returncode=0, stdout=text, stderr="")
                directory = Path(temp) / str(number)
                with mock.patch.dict(os.environ, ENV), mock.patch.object(ci, "command", return_value=result):
                    record = ci.observe_tar(directory, ci.REPOSITORY + "@" + DIGEST, CHILD_ID)
                self.assertEqual(record["status"], "OBSERVATION_ERROR")

    def test_container_output_cannot_replace_the_publication_identity(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            observation = {"euid": 1000, "tar_path": None, "version_returncode": None,
                           "version_stdout": "", "version_stderr": "", "image_id": "injected"}
            record, _ = self.invoke(Path(temp), observation=observation)
            self.assertEqual(record["status"], "OBSERVATION_ERROR")
            self.assertEqual(record["image_id"], CHILD_ID)


class Preparation(unittest.TestCase):
    def test_missing_approval_stops_before_registry_or_parent_pull(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp, mock.patch.dict(os.environ, ENV), \
                mock.patch.object(ci.ops, "check_seal", side_effect=ValueError("approval missing")), \
                mock.patch.object(ci, "docker") as docker, mock.patch.object(ci, "lookup") as lookup:
            with self.assertRaises(ValueError): ci.prepare(Path(temp))
            docker.assert_not_called()
            lookup.assert_not_called()

    def test_unknown_registry_state_stops_before_build(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp, mock.patch.dict(os.environ, ENV), \
                mock.patch.object(ci.ops, "check_seal", return_value=({}, "tree", {"recipe_sha256": "e" * 64, "review_tree": "tree"})), \
                mock.patch.object(ci.ops, "git", side_effect=["worker [h3-image-worker]", COMMIT]), \
                mock.patch.object(ci.ops, "valid_request_message", return_value=True), \
                mock.patch.object(ci, "lookup", return_value="UNKNOWN"), mock.patch.object(ci, "docker") as docker:
            with self.assertRaises(RuntimeError): ci.prepare(Path(temp))
            docker.assert_not_called()

    def test_preparation_is_exclusive_for_this_attempt(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp, mock.patch.dict(os.environ, ENV), \
                mock.patch.object(ci.ops, "check_seal", return_value=({}, "tree", {})), \
                mock.patch.object(ci.ops, "git", side_effect=["worker [h3-image-worker]", COMMIT]), \
                mock.patch.object(ci.ops, "valid_request_message", return_value=True):
            ci.dump(Path(temp) / "preparation.json", {})
            with self.assertRaises(ValueError): ci.prepare(Path(temp))

    def test_parent_probe_failure_never_emits_build_permission(self):
        parent, _ = configs()
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp, mock.patch.dict(os.environ, ENV), \
                mock.patch.object(ci.ops, "check_seal", return_value=({}, "tree", {"recipe_sha256": "e" * 64, "review_tree": "tree"})), \
                mock.patch.object(ci.ops, "git", side_effect=["worker [h3-image-worker]", COMMIT]), \
                mock.patch.object(ci.ops, "valid_request_message", return_value=True), \
                mock.patch.object(ci, "lookup", return_value="ABSENT_CONFIRMED"), \
                mock.patch.object(ci, "inspect", return_value=parent), \
                mock.patch.object(ci, "docker", return_value="Digest: " + DIGEST + "\n") as docker, \
                mock.patch.object(ci, "parent_probe", side_effect=RuntimeError("parent metadata failure")):
            directory = Path(temp)
            output = directory / "github-output.txt"
            with self.assertRaises(RuntimeError): ci.prepare(directory, output)
            self.assertFalse(output.exists())
            self.assertFalse((directory / "preparation.json").exists())
            self.assertEqual(docker.call_args.args[0], "pull")
            self.assertEqual(docker.call_count, 1)

    def test_existing_tag_disables_build_and_pins_existing_digest(self):
        parent, _ = configs()
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp, mock.patch.dict(os.environ, ENV), \
                mock.patch.object(ci.ops, "check_seal", return_value=({}, "tree", {"recipe_sha256": "e" * 64, "review_tree": "tree"})), \
                mock.patch.object(ci.ops, "git", side_effect=["worker [h3-image-worker]", COMMIT]), \
                mock.patch.object(ci.ops, "valid_request_message", return_value=True), \
                mock.patch.object(ci, "lookup", return_value="EXISTS"), \
                mock.patch.object(ci, "inspect", return_value=parent), \
                mock.patch.object(ci, "parent_probe", side_effect=parent_probe_files), \
                mock.patch.object(ci, "docker", return_value="Digest: " + DIGEST + "\n"):
            output = Path(temp) / "github-output.txt"
            result = ci.prepare(Path(temp), output)
            self.assertTrue(result["reused"])
            self.assertEqual(result["existing_digest"], DIGEST)
            self.assertIn("build_required=false", output.read_text())


class Publication(unittest.TestCase):
    def invoke(self, directory, child, reused=False, pull_error=False, tag_state="ABSENT_CONFIRMED", tar_status="AVAILABLE"):
        calls = []
        def docker(*args, **kwargs):
            calls.append(args)
            if args[0] == "push": return "worker: digest: " + DIGEST + " size: 123\n"
            if args[0] == "pull" and pull_error: raise RuntimeError("TLS EOF after completed push")
            return "Digest: " + DIGEST + "\n"
        def tar_observation(directory, reference, image_id):
            result = {"status": tar_status, "published_image": reference, "image_id": image_id}
            ci.dump(directory / "tar-validation.json", result)
            return result
        with mock.patch.dict(os.environ, ENV), mock.patch.object(ci, "inspect", return_value=child), \
                mock.patch.object(ci, "lookup", return_value=tag_state), mock.patch.object(ci, "docker", side_effect=docker), \
                mock.patch.object(ci, "observe_tar", side_effect=tar_observation):
            result = ci.publish(directory)
        return result, calls

    def test_publishes_once_and_hashes_deployment_receipt(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            directory = Path(temp)
            _, _, _, child = transaction(directory)
            result, calls = self.invoke(directory, child)
            self.assertEqual(result["status"], "PASS_PUBLISHED_AND_PULLED_CPU_VALIDATED")
            self.assertEqual(result["push_count"], 1)
            self.assertEqual(sum(args[0] == "push" for args in calls), 1)
            self.assertIn("deployment-image.txt", result["evidence_sha256"])
            self.assertIn("tar-validation.json", result["evidence_sha256"])
            self.assertEqual(result["published_image"], ci.REPOSITORY + "@" + DIGEST)

    def test_missing_tar_or_observation_failure_does_not_invalidate_a_published_image(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            for status in ("MISSING", "OBSERVATION_ERROR"):
                directory = Path(temp) / status
                _, _, _, child = transaction(directory)
                result, calls = self.invoke(directory, child, tar_status=status)
                self.assertEqual(result["status"], "PASS_PUBLISHED_AND_PULLED_CPU_VALIDATED")
                self.assertEqual(result["tar_observation_status"], status)
                self.assertEqual(sum(args[0] == "push" for args in calls), 1)

    def test_existing_publication_is_revalidated_and_reused_without_push(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            directory = Path(temp)
            _, _, _, child = transaction(directory, reused=True)
            result, calls = self.invoke(directory, child, reused=True)
            self.assertEqual(result["push_count"], 0)
            self.assertTrue(result["reused_existing"])
            self.assertFalse(any(args[0] == "push" for args in calls))

    def test_tail_pull_failure_preserves_actual_published_identity(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            directory = Path(temp)
            _, _, _, child = transaction(directory)
            with self.assertRaises(RuntimeError): self.invoke(directory, child, pull_error=True)
            record = ci.read(directory / "publication.json")
            self.assertEqual(record["status"], "PUBLISHED_PULL_PENDING")
            self.assertEqual(record["digest"], DIGEST)
            self.assertEqual(record["push_count"], 1)

    def test_second_publish_in_same_transaction_is_excluded(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            directory = Path(temp)
            _, _, _, child = transaction(directory)
            self.invoke(directory, child)
            with self.assertRaises(FileExistsError): self.invoke(directory, child)

    def test_unknown_or_existing_prepush_tag_stops_publication(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            for state in ("UNKNOWN", "EXISTS"):
                directory = Path(temp) / state
                _, _, _, child = transaction(directory)
                with self.subTest(state=state), self.assertRaises(RuntimeError):
                    self.invoke(directory, child, tag_state=state)
                self.assertFalse((directory / "publication.json").exists())

    def test_changed_candidate_identity_stops_publication(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            directory = Path(temp)
            _, candidate, _, child = transaction(directory)
            candidate["recipe_sha256"] = "f" * 64
            ci.dump(directory / "candidate.json", candidate)
            with self.assertRaises(ValueError): self.invoke(directory, child)
            self.assertFalse((directory / "publication-operation.json").exists())

    def test_pulled_imageid_must_match_tested_image(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            directory = Path(temp)
            _, _, _, child = transaction(directory)
            wrong = {**child, "Id": "sha256:" + "f" * 64}
            with mock.patch.dict(os.environ, ENV), mock.patch.object(ci, "inspect", side_effect=[child, wrong]), \
                    mock.patch.object(ci, "lookup", return_value="ABSENT_CONFIRMED"), \
                    mock.patch.object(ci, "docker", side_effect=["digest: " + DIGEST, "Digest: " + DIGEST]):
                with self.assertRaises(ValueError): ci.publish(directory)
            self.assertEqual(ci.read(directory / "publication.json")["status"], "PUBLISHED_PULL_PENDING")


class WorkflowRouting(unittest.TestCase):
    def test_worker_push_is_excluded_from_legacy_lite(self):
        text = (ROOT / ".github/workflows/dingo-router-ci.yml").read_text(encoding="utf-8")
        condition = text.split("  omni-gateway-lite:\n", 1)[1].splitlines()[0]
        self.assertIn("!contains(github.event.head_commit.message, '[h3-image-worker]')", condition)

    def test_dedicated_workflow_has_one_unpublished_build_and_terminal_evidence(self):
        text = (ROOT / ".github/workflows/h3-image-worker.yml").read_text(encoding="utf-8")
        self.assertIn("contains(github.event.head_commit.message, '[h3-image-worker]')", text)
        self.assertEqual(text.count("uses: docker/build-push-action@"), 1)
        self.assertIn("push: false", text)
        self.assertIn("load: true", text)
        self.assertIn("driver: docker", text)
        self.assertNotIn("workflow_dispatch:", text)
        self.assertNotIn("matrix:", text)
        self.assertIn("cancel-in-progress: false", text)
        self.assertEqual(text.count("if: always()"), 2)


if __name__ == "__main__":
    unittest.main()
