"""Local publication/inheritance regression fixtures; never build/push images."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sr_ci", HERE / "h3_seedvr2_rmsnorm_ci.py")
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)
COMMIT = "a" * 40
IMAGE_ID = "sha256:" + "a" * 64


def fixtures():
    parent = {"Os": "linux", "Architecture": "amd64", "Id": "sha256:" + "b" * 64,
              "RootFS": {"Layers": ["sha256:parent1", "sha256:parent2"]},
              "Config": {"User": "dynamo", "WorkingDir": "/workspace", "Entrypoint": None, "Cmd": None,
                         "Env": ["PATH=/usr/bin", "H3_EXPERIMENTAL_SOURCE_COMMIT=" + ci.PARENT_COMMIT,
                                 "H3_EXPERIMENTAL_BASE_IMAGE=original-framework@sha256:original"],
                         "Labels": {"gateway.custom": "preserved"}}}
    child = copy.deepcopy(parent)
    child["Id"] = IMAGE_ID
    child["RootFS"]["Layers"].append("sha256:patch")
    child["Config"]["Env"][1] = "H3_EXPERIMENTAL_SOURCE_COMMIT=" + COMMIT
    child["Config"]["Env"] += ["H3_RMSNORM_PARENT_IMAGE=" + ci.PARENT, "H3_RMSNORM_UPSTREAM_COMMIT=" + ci.UPSTREAM]
    child["Config"]["Labels"].update({"org.opencontainers.image.revision": COMMIT, "org.opencontainers.image.base.name": ci.PARENT})
    return parent, child


def report():
    return {"status": "PASS", "device": "cpu", "module_sha256": ci.AFTER,
            "module_path": "/usr/local/lib/python3.12/dist-packages/vllm_omni/diffusion/models/seedvr2/nadit.py",
            "six_case_count": 6, "six_cases": [
                {"dtype": dtype, "affine": affine, "finite": True, "allclose": True, "device": "cpu"}
                for dtype in ("torch.float16", "torch.bfloat16", "torch.float32") for affine in (False, True)],
            "upstream_pytest_case_count": 6, "inventory_unchanged": True, "euid": 1000,
            "source_commit": COMMIT, "all_300": {"passed": True}}


class InheritanceTests(unittest.TestCase):
    def test_valid_derivative_preserves_complete_parent(self):
        result = ci.compare_images(*fixtures(), COMMIT)
        self.assertEqual(result["inherited_layers"], 2)
        self.assertEqual(result["added_layers"], 1)

    def test_rebased_or_flattened_image_rejected(self):
        parent, child = fixtures()
        for layers in ([], ["sha256:other", "sha256:patch"], parent["RootFS"]["Layers"]):
            with self.subTest(layers=layers), self.assertRaises(ValueError):
                child["RootFS"]["Layers"] = layers
                ci.compare_images(parent, child, COMMIT)

    def test_command_user_and_working_directory_changes_rejected(self):
        for name, value in (("Cmd", ["sleep", "infinity"]), ("Entrypoint", ["bash"]), ("User", "root"), ("WorkingDir", "/tmp")):
            parent, child = fixtures()
            child["Config"][name] = value
            with self.subTest(field=name), self.assertRaises(ValueError):
                ci.compare_images(parent, child, COMMIT)

    def test_new_proxy_and_library_path_changes_rejected(self):
        for entry in ("HTTPS_PROXY=http://unexpected:8080", "LD_LIBRARY_PATH=/other"):
            parent, child = fixtures()
            child["Config"]["Env"].append(entry)
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                ci.compare_images(parent, child, COMMIT)

    def test_inherited_parent_marker_is_not_rewritten(self):
        parent, child = fixtures()
        child["Config"]["Env"][2] = "H3_EXPERIMENTAL_BASE_IMAGE=" + ci.PARENT
        with self.assertRaises(ValueError):
            ci.compare_images(parent, child, COMMIT)

    def test_wrong_architecture_and_custom_label_rejected(self):
        parent, child = fixtures()
        child["Architecture"] = "arm64"
        with self.assertRaises(ValueError): ci.compare_images(parent, child, COMMIT)
        child["Architecture"] = "amd64"
        child["Config"]["Labels"]["gateway.custom"] = "changed"
        with self.assertRaises(ValueError): ci.compare_images(parent, child, COMMIT)

    def test_package_and_source_inventory_drift_rejected(self):
        base = {"packages": [["torch", "2.13.0+cu130"]], "dpkg": ["libc6\tversion"],
                "module_path": "/installed/nadit.py", "module_sha256": ci.BEFORE,
                "source_commit": ci.PARENT_COMMIT, "gateway_commit": "e8c9195672f761ec09344a641730844c9e6c9e99", "euid": 1000}
        child = {**base, "module_sha256": ci.AFTER, "source_commit": COMMIT}
        ci.compare_inventories(base, child, COMMIT)
        for key, bad in (("packages", [["torch", "2.14"]]), ("dpkg", ["libc6\tnew"]), ("module_sha256", ci.BEFORE), ("euid", 0)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                ci.compare_inventories(base, {**child, key: bad}, COMMIT)


class EvidenceTests(unittest.TestCase):
    def test_source_lock_rejects_parent_or_framework_drift(self):
        lock = ci.read(ci.ROOT / ci.LOCK_PATH)
        ci.validate_lock(lock)
        for key, value in (("parent_image", ci.REPOSITORY + ":latest"), ("upstream_commit", "b" * 40),
                           ("versions", {**lock["versions"], "torch": "2.14.0"})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                ci.validate_lock({**lock, key: value})

    def test_six_real_combinations_and_nonzero_regression_required(self):
        ci.validate_runtime_report(report(), COMMIT)
        for key, value in (("status", "FAIL"), ("euid", 0), ("upstream_pytest_case_count", 5), ("all_300", {"passed": False}), ("source_commit", "b" * 40)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                ci.validate_runtime_report({**report(), key: value}, COMMIT)

    def test_duplicate_numerical_cases_do_not_count_as_six(self):
        value = report()
        value["six_cases"][-1] = value["six_cases"][0]
        with self.assertRaises(ValueError): ci.validate_runtime_report(value, COMMIT)

    def test_nonfinite_or_failed_comparison_not_accepted(self):
        for key in ("finite", "allclose"):
            value = report()
            value["six_cases"][0][key] = False
            with self.assertRaises(ValueError): ci.validate_runtime_report(value, COMMIT)

    def test_vendored_source_import_not_accepted(self):
        value = report()
        value["module_path"] = "/opt/h3-seedvr2-rmsnorm/upstream/nadit.py"
        with self.assertRaises(ValueError): ci.validate_runtime_report(value, COMMIT)

    def test_digest_selection_uses_this_push_receipt(self):
        digest = "sha256:" + "c" * 64
        self.assertEqual(ci.publication_digest("layer: Layer already exists\ntag: digest: " + digest + " size: 1234\n"), digest)
        for output in ("", "reference " + ci.REPOSITORY + "@" + digest,
                       "tag: digest: " + digest + " size: 12\nother: digest: sha256:" + "d" * 64 + " size: 13\n"):
            with self.assertRaises(ValueError): ci.publication_digest(output)

    def test_final_evidence_survives_container_exit(self):
        calls = []
        exited = False
        def fake_docker(*args, **kwargs):
            nonlocal exited
            calls.append(args)
            if args[0] == "create": return "c" * 64
            if args[0] == "start":
                exited = True
                return ""
            if args[0] == "container":
                return json.dumps([{"State": {"ExitCode": 0, "OOMKilled": False}}])
            if args[0] == "cp":
                self.assertTrue(exited)
                ci.dump(Path(args[-1]) / "runtime-validation.json", report())
            return ""
        with tempfile.TemporaryDirectory() as temp, patch.object(ci, "docker", side_effect=fake_docker):
            output = Path(temp)
            ci.run_final_validation("candidate:test", output)
            create = calls[0]
            self.assertNotIn("--tmpfs", create)
            self.assertNotIn("--read-only", create)
            self.assertNotIn("--rm", create)
            self.assertEqual(create[create.index("--network") + 1], "none")
            self.assertEqual(create[create.index("--user") + 1], "1000:0")
            self.assertEqual(ci.read(output / "final/runtime-validation.json")["status"], "PASS")
            self.assertEqual(calls[-1][0], "rm")

    def test_changed_candidate_stops_before_docker_push(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            ci.dump(path / "candidate-verified.json", {"status": "PASS_CANDIDATE_CPU_ONLY", "source_commit": COMMIT,
                "candidate_image": "candidate:test", "image_id": IMAGE_ID})
            inputs = {"source_commit": COMMIT, "candidate_image": "candidate:test", "final_image": ci.REPOSITORY + ":test"}
            with patch.object(ci, "checked_inputs", return_value=inputs), patch.object(ci, "inspect", return_value={"Id": "changed"}), patch.object(ci, "docker") as call:
                with self.assertRaisesRegex(ValueError, "changed after validation"):
                    ci.publish(path, {})
                call.assert_not_called()

    def test_publication_tags_tested_id_and_checks_immutable_pull(self):
        parent, candidate = fixtures()
        digest = "sha256:" + "c" * 64
        published = {**candidate, "RepoDigests": [ci.REPOSITORY + "@" + digest]}
        inputs = {"source_commit": COMMIT, "candidate_image": "candidate:test", "final_image": ci.REPOSITORY + ":test"}
        env = {"GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1"}
        def fake_docker(*args, log=None):
            if args[0] == "push":
                Path(log).write_text("test: digest: " + digest + " size: 1234\n", encoding="utf-8")
            return ""
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            ci.dump(path / "candidate-verified.json", {"status": "PASS_CANDIDATE_CPU_ONLY", "source_commit": COMMIT,
                "candidate_image": "candidate:test", "image_id": IMAGE_ID})
            ci.dump(path / "final/runtime-validation.json", report())
            with patch.object(ci, "checked_inputs", return_value=inputs), patch.object(ci, "inspect", side_effect=[candidate, published, candidate, parent]), patch.object(ci, "validate_build_evidence"), patch.object(ci, "docker", side_effect=fake_docker) as call:
                with patch("builtins.print"):
                    result = ci.publish(path, env)
                self.assertEqual(call.call_args_list[0].args, ("tag", IMAGE_ID, inputs["final_image"]))
                self.assertEqual(call.call_args_list[2].args, ("pull", "--platform", "linux/amd64", ci.REPOSITORY + "@" + digest))
                self.assertEqual(result["deployment_image"], ci.REPOSITORY + "@" + digest)
                self.assertEqual(result["image_id"], IMAGE_ID)

    def test_registry_pull_mismatch_never_records_success(self):
        _, candidate = fixtures()
        published = {**candidate, "RepoDigests": [ci.REPOSITORY + "@sha256:" + "c" * 64]}
        inputs = {"source_commit": COMMIT, "candidate_image": "candidate:test", "final_image": ci.REPOSITORY + ":test"}
        def fake_docker(*args, log=None):
            if args[0] == "push":
                Path(log).write_text("test: digest: sha256:" + "c" * 64 + " size: 1234\n", encoding="utf-8")
            return ""
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            ci.dump(path / "candidate-verified.json", {"status": "PASS_CANDIDATE_CPU_ONLY", "source_commit": COMMIT,
                "candidate_image": "candidate:test", "image_id": IMAGE_ID})
            ci.dump(path / "final/runtime-validation.json", report())
            with patch.object(ci, "checked_inputs", return_value=inputs), patch.object(ci, "inspect", side_effect=[candidate, published, {"Id": "wrong"}]), patch.object(ci, "validate_build_evidence"), patch.object(ci, "docker", side_effect=fake_docker):
                with self.assertRaisesRegex(ValueError, "does not match"):
                    ci.publish(path, {})
            self.assertFalse((path / "image.json").exists())

    def test_wrong_event_branch_and_run_identity_rejected(self):
        env = {"GITHUB_SHA": COMMIT, "GITHUB_REF_NAME": ci.BRANCH, "GITHUB_EVENT_NAME": "push", "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1"}
        self.assertEqual(ci.identity(env), COMMIT)
        for key, value in (("GITHUB_REF_NAME", "main"), ("GITHUB_SHA", "abc"), ("GITHUB_EVENT_NAME", "pull_request"), ("GITHUB_RUN_ID", "../123")):
            with self.subTest(key=key), self.assertRaises(ValueError): ci.identity({**env, key: value})

    def test_publication_destination_cannot_drift_after_preparation(self):
        env = {"GITHUB_SHA": COMMIT, "GITHUB_REF_NAME": ci.BRANCH, "GITHUB_EVENT_NAME": "push", "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1"}
        value = {"source_commit": COMMIT, "parent_image": ci.PARENT, "run_id": "123", "run_attempt": "1",
                 "source_lock_sha256": ci.sha256(ci.ROOT / ci.LOCK_PATH),
                 "candidate_image": "h3-srnorm-ci:" + COMMIT[:12] + "-123-1",
                 "final_image": ci.REPOSITORY + ":omni031rc1-gateway-distilled-ref8shift12-srnormfix-" + COMMIT[:12]}
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            ci.dump(directory / "inputs.json", value)
            self.assertEqual(ci.checked_inputs(directory, env), value)
            for key in ("candidate_image", "final_image"):
                ci.dump(directory / "inputs.json", {**value, key: "other/repository:tag"})
                with self.assertRaisesRegex(ValueError, "tag does not match"):
                    ci.checked_inputs(directory, env)


class WorkflowTests(unittest.TestCase):
    def test_previous_recipes_require_explicit_dispatch(self):
        workflow = yaml.load((ci.ROOT / ".github/workflows/dingo-router-ci.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        self.assertEqual(workflow["on"]["workflow_dispatch"]["inputs"]["build_recipe"]["default"], "omni-gateway-lite")
        self.assertEqual(workflow["jobs"]["h3-experimental"]["if"], f"github.ref_name == '{ci.BRANCH}' && github.event_name == 'workflow_dispatch' && inputs.build_recipe == 'full'")
        job = workflow["jobs"]["h3-seedvr2-rmsnorm"]
        self.assertEqual(job["if"], f"github.ref_name == '{ci.BRANCH}' && github.event_name == 'workflow_dispatch' && inputs.build_recipe == 'seedvr2-rmsnorm'")
        self.assertEqual(job["runs-on"], ["self-hosted", "linux", "x64", "dingo"])
        builds = [s for s in job["steps"] if s.get("uses", "").startswith("docker/build-push-action@")]
        self.assertEqual(len(builds), 1)
        self.assertEqual(builds[0]["with"]["load"], "true")
        self.assertEqual(builds[0]["with"]["push"], "false")
        commands = [s.get("run", "") for s in job["steps"]]
        self.assertLess(next(i for i, c in enumerate(commands) if "verify-candidate" in c), next(i for i, c in enumerate(commands) if "_ci.py publish" in c))
        self.assertNotIn("container/render.py", "\n".join(commands))


if __name__ == "__main__":
    unittest.main(verbosity=2)
