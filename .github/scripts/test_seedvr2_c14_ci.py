"""Publication regression fixtures; these tests never build or publish an image."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("c14_ci", HERE / "seedvr2_c14_ci.py")
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)
COMMIT = "a" * 40
IMAGE_ID = "sha256:" + "a" * 64
ENV = {"GITHUB_SHA": COMMIT, "GITHUB_REF_NAME": ci.BRANCH, "GITHUB_EVENT_NAME": "workflow_dispatch",
       "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "2"}


def image_fixtures():
    parent = {"Os": "linux", "Architecture": "amd64", "Id": ci.PARENT_IMAGE_ID, "Size": 100,
        "RootFS": {"Layers": ["sha256:one", "sha256:two"]},
        "Config": {"User": "dynamo", "WorkingDir": "/workspace", "Entrypoint": None, "Cmd": None,
                   "Env": ["PATH=/usr/bin", "DYNAMO_COMMIT_SHA=" + ci.FRAMEWORK_COMMIT,
                           "DYNAMO_GATEWAY_SOURCE_SHA=" + ci.GATEWAY_BEFORE,
                           "H3_EXPERIMENTAL_SOURCE_COMMIT=" + ci.PARENT_COMMIT], "Labels": {"custom": "keep"}}}
    child = copy.deepcopy(parent)
    child["Id"] = IMAGE_ID
    child["Size"] = 101
    child["RootFS"]["Layers"].append("sha256:patch")
    child["Config"]["Env"] += ["H3_C14_SOURCE_COMMIT=" + COMMIT, "H3_C14_PARENT_IMAGE=" + ci.PARENT]
    child["Config"]["Labels"].update({"org.opencontainers.image.revision": COMMIT,
                                    "org.opencontainers.image.base.name": ci.PARENT})
    return parent, child


def inventory_fixtures():
    parent = {"packages": [["torch", "2.13.0+cu130"]], "dpkg": ["libc6\tv"],
              "module_path": "/installed/nadit.py", "module_sha256": ci.BEFORE,
              "source_commit": None, "dynamo_commit": ci.FRAMEWORK_COMMIT, "lite_source_commit": ci.PARENT_COMMIT,
              "gateway_commit": ci.GATEWAY_BEFORE, "adapters_present": False, "euid": 1000}
    child = {**parent, "module_sha256": ci.AFTER, "source_commit": COMMIT, "gateway_commit": ci.GATEWAY}
    return parent, child


def good_reports(directory):
    summary = {"schema": "seedvr2-c14-runtime/1", "status": "PASS", "euid": 1000,
               "gateway_commit": ci.GATEWAY, "parent_image": ci.PARENT, "source_commit": COMMIT,
               "retained_packages_unchanged": True, "adapters_present": False, "ref8_patch_present": False,
               "diagnostics_default_enabled": False, "source_manifest_sha256": "e" * 64}
    norm = {"status": "PASS", "device": "cpu", "euid": 1000, "source_commit": COMMIT,
            "parent_image": ci.PARENT, "module_sha256": ci.AFTER, "upstream_commit": ci.UPSTREAM,
            "six_case_count": 6, "upstream_pytest_case_count": 6, "inventory_unchanged": True,
            "module_path": "/installed/vllm_omni/diffusion/models/seedvr2/nadit.py", "source_manifest_sha256": "e" * 64,
            "six_cases": [{"dtype": dtype, "affine": affine, "device": "cpu", "allclose": True, "finite": True}
                          for dtype in ("torch.float16", "torch.bfloat16", "torch.float32") for affine in (False, True)],
            "all_300": {"passed": True, "old_formula_all_zero": True}}
    ci.dump(directory / "runtime-validation.json", summary)
    ci.dump(directory / "rmsnorm/runtime-validation.json", norm)
    ci.dump(directory / "gateway-regression.json", {"status": "PASS", "case_count": 9})
    ci.dump(directory / "diagnostics-regression.json", {"status": "PASS", "tests_run": 12,
                                                       "failures": 0, "errors": 0, "skipped": 0})
    for name, status in {"source-verification.json": "PASS", "preservation.json": "PASS",
        "environment/environment-check.json": "PASS_EXPERIMENTAL_ENVIRONMENT_CPU_ONLY",
        "environment/media-smoke.json": "PASS_REAL_CPU_MEDIA_SMOKE",
        "environment/service-smoke.json": "PASS_SERVICE_IMPORTS_AND_HELP_ONLY"}.items():
        ci.dump(directory / name, {"status": status, **({"source_manifest_sha256": "e" * 64} if name == "source-verification.json" else {})})
    return summary, norm


class InheritanceTests(unittest.TestCase):
    def test_parent_id_must_match_historical_publication(self):
        parent, child = image_fixtures()
        parent["Id"] = "sha256:" + "b" * 64
        with self.assertRaisesRegex(ValueError, "parent image ID"):
            ci.compare_images(parent, child, COMMIT)

    def test_complete_framework_parent_and_added_markers(self):
        report = ci.compare_images(*image_fixtures(), COMMIT)
        self.assertEqual(report["added_layers"], 1)
        self.assertEqual(report["parent_size_bytes"], 100)

    def test_parent_does_not_need_final_distilled_marker(self):
        parent, child = image_fixtures()
        self.assertEqual(ci.env_map(parent["Config"])["H3_EXPERIMENTAL_SOURCE_COMMIT"], ci.PARENT_COMMIT)
        ci.compare_images(parent, child, COMMIT)

    def test_layer_rebase_or_no_patch_is_rejected(self):
        for layers in ([], ["different"], ["sha256:one", "sha256:two"]):
            parent, child = image_fixtures()
            child["RootFS"]["Layers"] = layers
            with self.subTest(layers=layers), self.assertRaises(ValueError):
                ci.compare_images(parent, child, COMMIT)

    def test_source_native_and_proxy_changes_rejected(self):
        for replacement in ("DYNAMO_COMMIT_SHA=" + COMMIT, "HTTPS_PROXY=http://unknown"):
            parent, child = image_fixtures()
            child["Config"]["Env"][1] = replacement
            with self.subTest(value=replacement), self.assertRaises(ValueError):
                ci.compare_images(parent, child, COMMIT)

    def test_entrypoint_user_workdir_and_arch_must_be_preserved(self):
        for field, value in (("Entrypoint", ["bash"]), ("User", "root"), ("WorkingDir", "/tmp")):
            parent, child = image_fixtures()
            child["Config"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                ci.compare_images(parent, child, COMMIT)
        parent, child = image_fixtures()
        child["Architecture"] = "arm64"
        with self.assertRaises(ValueError): ci.compare_images(parent, child, COMMIT)

    def test_inventory_must_not_gain_packages_or_adapters(self):
        parent, child = inventory_fixtures()
        ci.compare_inventories(parent, child, COMMIT)
        for key, value in (("packages", [["torch", "different"]]), ("dpkg", ["new"]),
                           ("adapters_present", True), ("euid", 0), ("gateway_commit", "wrong")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                ci.compare_inventories(parent, {**child, key: value}, COMMIT)


class ReportTests(unittest.TestCase):
    def test_source_lock_has_full_omni_and_exact_parent(self):
        lock = ci.read(ci.ROOT / ci.LOCK_PATH)
        ci.validate_lock(lock)
        for key, value in (("parent_image", ci.REPOSITORY + ":latest"),
                           ("versions", {**lock["versions"], "vllm-omni": "0.29.0rc1"}),
                           ("gateway_commit", "wrong")):
            with self.subTest(key=key), self.assertRaises(ValueError): ci.validate_lock({**lock, key: value})

    def test_complete_evidence_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            good_reports(directory)
            ci.validate_reports(directory, COMMIT)

    def test_retained_core_gate_failure_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            good_reports(directory)
            ci.dump(directory / "source-verification.json", {"status": "FAIL"})
            with self.assertRaises(ValueError): ci.validate_reports(directory, COMMIT)

    def test_all_retained_capabilities_require_fresh_passing_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ("environment/environment-check.json", "environment/media-smoke.json",
                         "environment/service-smoke.json", "gateway-regression.json"):
                good_reports(directory)
                ci.dump(directory / name, {"status": "FAIL", "errors": ["real check failed"]})
                with self.subTest(name=name), self.assertRaises(ValueError):
                    ci.validate_reports(directory, COMMIT)

    def test_diagnostic_tests_must_run_without_skips_failures_or_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for key, value in (("tests_run", 0), ("tests_run", True), ("failures", 1),
                               ("errors", 1), ("skipped", 1), ("status", "SKIPPED")):
                good_reports(directory)
                record = ci.read(directory / "diagnostics-regression.json")
                ci.dump(directory / "diagnostics-regression.json", {**record, key: value})
                with self.subTest(key=key), self.assertRaises(ValueError):
                    ci.validate_reports(directory, COMMIT)

    def test_old_or_unbound_numerical_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for key, value in (("source_commit", ci.PARENT_COMMIT), ("parent_image", "old-parent"),
                               ("source_manifest_sha256", "f" * 64)):
                _, norm = good_reports(directory)
                ci.dump(directory / "rmsnorm/runtime-validation.json", {**norm, key: value})
                with self.subTest(key=key), self.assertRaises(ValueError):
                    ci.validate_reports(directory, COMMIT)

    def test_diagnostic_default_must_remain_disabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report, _ = good_reports(directory)
            report["diagnostics_default_enabled"] = True
            ci.dump(directory / "runtime-validation.json", report)
            with self.assertRaises(ValueError): ci.validate_reports(directory, COMMIT)

    def test_duplicate_or_failed_numeric_case_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, norm = good_reports(directory)
            norm["six_cases"][-1] = norm["six_cases"][0]
            ci.dump(directory / "rmsnorm/runtime-validation.json", norm)
            with self.assertRaises(ValueError): ci.validate_reports(directory, COMMIT)
            _, norm = good_reports(directory)
            norm["six_cases"][0]["finite"] = False
            ci.dump(directory / "rmsnorm/runtime-validation.json", norm)
            with self.assertRaises(ValueError): ci.validate_reports(directory, COMMIT)

    def test_distilled_mode_or_wrong_gateway_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            summary, _ = good_reports(directory)
            for key, value in (("ref8_patch_present", True), ("adapters_present", True),
                               ("gateway_commit", "wrong"), ("retained_packages_unchanged", False)):
                ci.dump(directory / "runtime-validation.json", {**summary, key: value})
                with self.subTest(key=key), self.assertRaises(ValueError): ci.validate_reports(directory, COMMIT)

    def test_noninstalled_module_and_missing_old_failure_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, norm = good_reports(directory)
            for key, value in (("module_path", "/opt/seedvr2-c14/rmsnorm/nadit.py"),
                               ("all_300", {"passed": True, "old_formula_all_zero": False}), ("euid", 0)):
                ci.dump(directory / "rmsnorm/runtime-validation.json", {**norm, key: value})
                with self.subTest(key=key), self.assertRaises(ValueError): ci.validate_reports(directory, COMMIT)


class PublicationTests(unittest.TestCase):
    def test_push_and_manual_existing_branch_allowed(self):
        for event in ("push", "workflow_dispatch"):
            self.assertEqual(ci.identity({**ENV, "GITHUB_EVENT_NAME": event}), COMMIT)
        for key, value in (("GITHUB_EVENT_NAME", "pull_request"), ("GITHUB_EVENT_NAME", "schedule"), ("GITHUB_REF_NAME", "main"), ("GITHUB_SHA", "abc")):
            with self.subTest(key=key), self.assertRaises(ValueError): ci.identity({**ENV, key: value})

    def test_this_push_digest_is_required(self):
        digest = "sha256:" + "d" * 64
        self.assertEqual(ci.publication_digest("tag: digest: " + digest + " size: 123\n"), digest)
        for output in ("", ci.REPOSITORY + "@" + digest, "tag: digest: " + digest + " size: 1\nsecond: digest: sha256:" + "c" * 64 + " size: 2\n"):
            with self.assertRaises(ValueError): ci.publication_digest(output)

    def test_failed_or_retagged_candidate_never_pushes(self):
        inputs = {"source_commit": COMMIT, "candidate_image": "candidate", "final_image": "final"}
        verified = {"status": "FAIL", "image_id": IMAGE_ID, "source_commit": COMMIT,
                    "candidate_image": "candidate", "parent_image": ci.PARENT}
        with patch.object(ci, "checked_inputs", return_value=inputs), patch.object(ci, "read", return_value=verified), patch.object(ci, "docker") as docker:
            with self.assertRaises(ValueError): ci.publish(ENV, Path("unused"))
            docker.assert_not_called()
        verified["status"] = "PASS_CANDIDATE_CPU_ONLY"
        with patch.object(ci, "checked_inputs", return_value=inputs), patch.object(ci, "read", return_value=verified), patch.object(ci, "inspect", return_value={"Id": "wrong"}), patch.object(ci, "docker") as docker:
            with self.assertRaises(ValueError): ci.publish(ENV, Path("unused"))
            docker.assert_not_called()

    def test_publishes_exact_tested_id_and_records_current_push_digest(self):
        inputs = {"source_commit": COMMIT, "candidate_image": "candidate", "final_image": "final"}
        verified = {"status": "PASS_CANDIDATE_CPU_ONLY", "image_id": IMAGE_ID, "source_commit": COMMIT,
                    "candidate_image": "candidate", "parent_image": ci.PARENT}
        digest = "sha256:" + "d" * 64
        calls = []
        def fake(*args, log=None):
            calls.append(args)
            if log:
                Path(log).write_text("final: digest: " + digest + " size: 123\n", encoding="utf-8")
            return ""
        with tempfile.TemporaryDirectory() as temporary, patch.object(ci, "checked_inputs", return_value=inputs), patch.object(ci, "read", return_value=verified), patch.object(ci, "inspect", return_value={"Id": IMAGE_ID}), patch.object(ci, "validate_reports"), patch.object(ci, "compare_images"), patch.object(ci, "docker", side_effect=fake):
            directory = Path(temporary)
            with patch("builtins.print"):
                ci.publish(ENV, directory)
            actual = json.loads((directory / "image.json").read_text(encoding="utf-8"))
        self.assertEqual(calls[0], ("tag", IMAGE_ID, "final"))
        self.assertEqual(calls[2], ("pull", "--platform", "linux/amd64", ci.REPOSITORY + "@" + digest))
        self.assertEqual(actual["digest"], digest)
        self.assertEqual(actual["image_id"], IMAGE_ID)
        self.assertEqual(actual["parent_image"], ci.PARENT)

    def test_wrong_immutable_registry_pull_does_not_create_success_record(self):
        inputs = {"source_commit": COMMIT, "candidate_image": "candidate", "final_image": "final"}
        verified = {"status": "PASS_CANDIDATE_CPU_ONLY", "image_id": IMAGE_ID, "source_commit": COMMIT,
                    "candidate_image": "candidate", "parent_image": ci.PARENT}
        digest = "sha256:" + "d" * 64
        def fake(*args, log=None):
            if log: Path(log).write_text("final: digest: " + digest + " size: 123\n", encoding="utf-8")
            return ""
        with tempfile.TemporaryDirectory() as temporary, patch.object(ci, "checked_inputs", return_value=inputs), patch.object(ci, "read", return_value=verified), patch.object(ci, "inspect", side_effect=[{"Id": IMAGE_ID}, {"Id": "wrong"}]), patch.object(ci, "validate_reports"), patch.object(ci, "docker", side_effect=fake):
            directory = Path(temporary)
            with self.assertRaises(ValueError): ci.publish(ENV, directory)
            self.assertFalse((directory / "image.json").exists())

    def test_final_probe_does_not_lose_reports_in_tmpfs(self):
        calls = []
        def fake(*args, **kwargs):
            calls.append(args)
            if args[0] == "create": return "container-id"
            if args[0] == "container": return json.dumps([{"State": {"ExitCode": 0, "OOMKilled": False}}])
            return ""
        with tempfile.TemporaryDirectory() as temporary, patch.object(ci, "docker", side_effect=fake):
            ci.run_final_validation(IMAGE_ID, Path(temporary))
        create = calls[0]
        self.assertNotIn("--tmpfs", create)
        self.assertNotIn("--read-only", create)
        self.assertNotIn("--volume", create)
        self.assertIn("1000:0", create)
        self.assertIn("none", create)
        self.assertLess([a[0] for a in calls].index("start"), [a[0] for a in calls].index("cp"))
        self.assertEqual(calls[-1], ("rm", "container-id"))

    def test_failed_probe_still_exports_and_removes_container(self):
        calls = []
        def fake(*args, **kwargs):
            calls.append(args)
            if args[0] == "create": return "failed-container"
            if args[0] == "container": return json.dumps([{"State": {"ExitCode": 1, "OOMKilled": False}}])
            return ""
        with tempfile.TemporaryDirectory() as temporary, patch.object(ci, "docker", side_effect=fake):
            with self.assertRaises(RuntimeError): ci.run_final_validation(IMAGE_ID, Path(temporary))
        self.assertIn("cp", [a[0] for a in calls])
        self.assertEqual(calls[-1], ("rm", "failed-container"))

    def test_c14_candidate_is_verified_before_publication(self):
        workflow = yaml.load((ci.ROOT / ".github/workflows/dingo-router-ci.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        job = workflow["jobs"]["seedvr2-c14"]
        steps = job["steps"]
        build = next(step for step in steps if step.get("uses", "").startswith("docker/build-push-action@"))
        self.assertEqual(build["with"]["push"], "false")
        self.assertEqual(build["with"]["load"], "true")
        self.assertIn("H3_C14_SOURCE_COMMIT=", build["with"]["build-args"])
        self.assertLess(next(i for i,s in enumerate(steps) if " verify-candidate " in s.get("run", "")),
                        next(i for i,s in enumerate(steps) if " publish " in s.get("run", "")))

    def test_workflow_routes_c14_marked_push_only_to_c14(self):
        workflow = yaml.load((ci.ROOT / ".github/workflows/dingo-router-ci.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        def selected(event, branch, recipe, message):
            result = set()
            for name, job in workflow["jobs"].items():
                expression = job["if"].replace("contains(github.event.head_commit.message, '[seedvr2-c14]')", str('[seedvr2-c14]' in message))
                for field, value in {"github.event_name": event, "github.ref_name": branch, "inputs.build_recipe": recipe}.items():
                    expression = expression.replace(field, repr(value))
                expression = expression.replace("&&", "and").replace("||", "or").replace("!", "not ")
                tree = ast.parse(expression, mode="eval")
                if any(not isinstance(node, (ast.Expression, ast.BoolOp, ast.UnaryOp, ast.Not, ast.And, ast.Or, ast.Compare, ast.Eq, ast.Constant)) for node in ast.walk(tree)):
                    raise AssertionError("Unexpected routing expression")
                if eval(compile(tree, "routing", "eval"), {"__builtins__": {}}): result.add(name)
            return result
        scenarios = [
            ("push", ci.BRANCH, "", "build [seedvr2-c14]", {"seedvr2-c14"}),
            ("push", ci.BRANCH, "", "lite rebuild", {"omni-gateway-lite"}),
            ("workflow_dispatch", ci.BRANCH, "seedvr2-c14", "", {"seedvr2-c14"}),
            ("workflow_dispatch", ci.BRANCH, "omni-gateway-lite", "", {"omni-gateway-lite"}),
            ("workflow_dispatch", ci.BRANCH, "seedvr2-rmsnorm", "", {"h3-seedvr2-rmsnorm"}),
            ("workflow_dispatch", ci.BRANCH, "full", "", {"h3-experimental"}),
            ("push", "main", "", "[seedvr2-c14]", set()),
            ("pull_request", ci.BRANCH, "", "[seedvr2-c14]", set()),
        ]
        for event, branch, recipe, message, expected in scenarios:
            with self.subTest(event=event, message=message, recipe=recipe):
                self.assertEqual(selected(event, branch, recipe, message), expected)
        self.assertIn("[seedvr2-c14]", workflow["run-name"])
        self.assertIn("github.sha", workflow["run-name"])

if __name__ == "__main__":
    unittest.main()
