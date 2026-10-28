"""Publication boundary tests: no network, Docker daemon or GPU required."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("sr_native_ci", Path(__file__).with_name("sr_native_ci.py"))
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)
COMMIT = "a" * 40
ENV = {"GITHUB_SHA": COMMIT, "GITHUB_REF_NAME": ci.BRANCH, "GITHUB_EVENT_NAME": "push",
       "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1"}


def inputs(backend="seedvr2"):
    return ci.expected_inputs(backend, ENV)


def image(backend="seedvr2"):
    value = inputs(backend)
    return {"Id": "sha256:" + "b" * 64, "Os": "linux", "Architecture": "amd64",
            "Config": {"User": "1000:0", "Env": ["DYNAMO_COMMIT_SHA=" + COMMIT],
                       "Labels": {"org.opencontainers.image.revision": COMMIT,
                                  "org.opencontainers.image.base.name": value["base_image"]}}}


def inventory(backend="seedvr2"):
    return {"euid": 1000, "source_commit": COMMIT, "dpkg": ["libc6\t2.35"],
            "packages": [{"name": "torch", "version": ci.TORCH[backend], "metadata": "metadata",
                          "direct_url": None, "license": "BSD-3-Clause"}]}


def report(backend="seedvr2"):
    return {"status": "PASS_CPU_ONLY", "backend": backend, "euid": 1000,
            "source_commit": COMMIT, "gpu_validation_status": "NOT_RUN"}


def validated_evidence(root):
    values = inputs()
    ci.dump(root / "inputs.json", values)
    ci.dump(root / "final/runtime-validation.json", report())
    ci.dump(root / "installed-inventory.json", inventory())
    ci.dump(root / "candidate-verified.json", {"status": "PASS_CANDIDATE_CPU_ONLY", "backend": "seedvr2",
        "source_commit": COMMIT, "candidate_image": values["candidate_image"], "base_image": values["base_image"],
        "gpu_validation_status": "NOT_RUN", "image_id": image()["Id"], "evidence_sha256": ci.evidence_hashes(root)})


class Identity(unittest.TestCase):
    def test_current_source_lock_is_reviewed(self):
        ci.validate_lock(ci.read(ci.ROOT / ci.LOCK_PATH))

    def test_moving_base_or_extra_backend_rejected(self):
        lock = ci.read(ci.ROOT / ci.LOCK_PATH)
        lock["models"]["seedvr2"]["base_image"] = "nvidia/cuda:latest"
        with self.assertRaises(ValueError):
            ci.validate_lock(lock)
        lock = ci.read(ci.ROOT / ci.LOCK_PATH)
        lock["models"]["other"] = {}
        with self.assertRaises(ValueError):
            ci.validate_lock(lock)

    def test_wrong_repository_or_branch_rejected(self):
        for key in ("remote_branch", "repository", "harbor_repository", "build_recipe"):
            lock = ci.read(ci.ROOT / ci.LOCK_PATH)
            lock[key] = ("registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm"
                         if key == "harbor_repository" else "other")
            with self.subTest(key=key), self.assertRaises(ValueError):
                ci.validate_lock(lock)

    def test_run_identity_and_distinct_tags(self):
        self.assertEqual(ci.identity(ENV), COMMIT)
        for backend in ci.BACKENDS:
            expected = "registry.hd-04.alayanew.com:8443/openclaw/vllm-omni:"
            self.assertTrue(inputs(backend)["final_image"].startswith(expected))
            self.assertEqual(inputs(backend)["registry"], "registry.hd-04.alayanew.com:8443")
        a = inputs()
        b = ci.expected_inputs("seedvr2", {**ENV, "GITHUB_RUN_ATTEMPT": "2"})
        self.assertNotEqual(a["candidate_image"], b["candidate_image"])
        self.assertNotEqual(a["final_image"], b["final_image"])
        self.assertNotEqual(a["final_image"], inputs("flashvsr-plus")["final_image"])

    def test_malformed_actions_identity_rejected(self):
        for key, value in (("GITHUB_SHA", "short"), ("GITHUB_REF_NAME", "main"),
                           ("GITHUB_EVENT_NAME", "pull_request"), ("GITHUB_RUN_ID", "0"),
                           ("GITHUB_RUN_ATTEMPT", "1\nBAD=true")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                ci.identity({**ENV, key: value})

    def test_stale_attempt_or_cross_backend_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ci.dump(root / "inputs.json", inputs())
            ci.checked_inputs("seedvr2", ENV, root)
            with self.assertRaises(ValueError):
                ci.checked_inputs("flashvsr-plus", ENV, root)
            with self.assertRaises(ValueError):
                ci.checked_inputs("seedvr2", {**ENV, "GITHUB_RUN_ATTEMPT": "2"}, root)

    def test_multiline_github_output_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                ci.write_outputs(Path(temp) / "output", {"a": "ok\nb=wrong"})


class BaseAndImage(unittest.TestCase):
    def test_exact_base_passes_and_dockerhub_prefix_normalizes(self):
        value = {"Os": "linux", "Architecture": "amd64", "Id": "sha256:" + "b" * 64,
                 "RepoDigests": ["docker.io/" + ci.BASES["seedvr2"]]}
        ci.validate_base(value, ci.BASES["seedvr2"])
        for key, wrong in (("Architecture", "arm64"), ("RepoDigests", []), ("Id", "missing")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                ci.validate_base({**value, key: wrong}, ci.BASES["seedvr2"])

    def test_transient_pull_retries_same_digest(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(ci, "docker", side_effect=[RuntimeError("EOF"), "ok"]) as docker, patch.object(ci.time, "sleep"):
            ci.pull_with_retry(ci.BASES["seedvr2"], Path(temp), "base")
            self.assertEqual(docker.call_count, 2)
            self.assertEqual(docker.call_args_list[0].args, docker.call_args_list[1].args)

    def test_pull_never_falls_back_to_tag(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(ci, "docker") as docker:
            with self.assertRaises(ValueError):
                ci.pull_with_retry("nvidia/cuda:latest", Path(temp), "base")
            docker.assert_not_called()

    def test_access_denied_does_not_repeat(self):
        def denied(*args, log=None):
            Path(log).write_text("denied: requested access", encoding="utf-8")
            raise RuntimeError("denied")
        with tempfile.TemporaryDirectory() as temp, patch.object(ci, "docker", side_effect=denied) as docker, patch.object(ci.time, "sleep") as sleep:
            with self.assertRaises(RuntimeError):
                ci.pull_with_retry(ci.BASES["seedvr2"], Path(temp), "base")
            self.assertEqual(docker.call_count, 1)
            sleep.assert_not_called()

    def test_candidate_metadata_is_not_inherited_omni_identity(self):
        for backend in ci.BACKENDS:
            ci.validate_image(image(backend), inputs(backend))

    def test_root_or_wrong_platform_candidate_rejected(self):
        value = image()
        for key, wrong in (("Os", "windows"), ("Architecture", "arm64"), ("Id", "short")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                ci.validate_image({**value, key: wrong}, inputs())
        value["Config"]["User"] = "root"
        with self.assertRaises(ValueError):
            ci.validate_image(value, inputs())

    def test_wrong_or_duplicate_source_marker_rejected(self):
        for env in (["DYNAMO_COMMIT_SHA=old"], ["DYNAMO_COMMIT_SHA=" + COMMIT] * 2):
            value = image()
            value["Config"]["Env"] = env
            with self.assertRaises(ValueError):
                ci.validate_image(value, inputs())

    def test_wrong_base_and_revision_labels_rejected(self):
        for key in ("org.opencontainers.image.revision", "org.opencontainers.image.base.name"):
            value = image()
            value["Config"]["Labels"][key] = "old"
            with self.subTest(key=key), self.assertRaises(ValueError):
                ci.validate_image(value, inputs())


class RuntimeAndPublication(unittest.TestCase):
    def test_inventory_and_sbom_preserve_declared_license_boundary(self):
        for backend in ci.BACKENDS:
            ci.validate_inventory(inventory(backend), backend, COMMIT)
        value = ci.sbom(inventory())
        self.assertEqual(value["bomFormat"], "CycloneDX")
        self.assertIn("Not a license-policy approval", value["metadata"]["properties"][0]["value"])

    def test_duplicate_distribution_and_wrong_torch_rejected(self):
        value = inventory()
        value["packages"].append(copy.deepcopy(value["packages"][0]))
        with self.assertRaises(ValueError):
            ci.validate_inventory(value, "seedvr2", COMMIT)
        value = inventory()
        value["packages"][0]["version"] = "2.13.0"
        with self.assertRaises(ValueError):
            ci.validate_inventory(value, "seedvr2", COMMIT)

    def test_omni_mixture_rejected(self):
        for name in ("vllm", "vllm_omni", "ai.dingo", "ai-dingo-runtime"):
            value = inventory()
            value["packages"].append({"name": name, "version": "1"})
            with self.subTest(name=name), self.assertRaises(ValueError):
                ci.validate_inventory(value, "seedvr2", COMMIT)

    def test_runtime_does_not_authorize_gpu_or_other_source(self):
        ci.validate_report(report(), inputs())
        for key, wrong in (("status", "PASS"), ("euid", 0), ("backend", "flashvsr-plus"),
                           ("source_commit", "old"), ("gpu_validation_status", "PASS"), ("errors", ["failed"])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                ci.validate_report({**report(), key: wrong}, inputs())

    def test_bad_publication_proof_never_calls_docker(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(ci, "docker") as docker:
            root = Path(temp)
            ci.dump(root / "inputs.json", inputs())
            ci.dump(root / "candidate-verified.json", {"status": "FAIL"})
            with self.assertRaises(ValueError):
                ci.publish("seedvr2", ENV, root)
            docker.assert_not_called()

    def test_changed_evidence_never_reaches_publication(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(ci, "docker") as docker:
            root = Path(temp)
            values = inputs()
            ci.dump(root / "inputs.json", values)
            proof = {"status": "PASS_CANDIDATE_CPU_ONLY", "backend": "seedvr2", "source_commit": COMMIT,
                     "candidate_image": values["candidate_image"], "base_image": values["base_image"],
                     "gpu_validation_status": "NOT_RUN", "evidence_sha256": ci.evidence_hashes(root)}
            ci.dump(root / "candidate-verified.json", proof)
            (root / "unexpected.txt").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                ci.publish("seedvr2", ENV, root)
            docker.assert_not_called()

    def test_push_digest_not_zip_hash(self):
        digest = "sha256:" + "d" * 64
        self.assertEqual(ci.publication_digest("tag: digest: " + digest + " size: 123"), digest)
        for text in ("", "zip " + digest, "digest: " + digest + "\ndigest: sha256:" + "e" * 64):
            with self.assertRaises(ValueError):
                ci.publication_digest(text)

    def test_changed_candidate_tag_cannot_publish(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(ci, "docker") as docker:
            root = Path(temp)
            validated_evidence(root)
            candidate = image()
            candidate["Id"] = "sha256:" + "c" * 64
            with patch.object(ci, "inspect", return_value=candidate):
                with self.assertRaisesRegex(ValueError, "tag changed"):
                    ci.publish("seedvr2", ENV, root)
            docker.assert_not_called()

    def test_registry_pull_must_match_exact_tested_id(self):
        def registry(*args, log=None):
            if args[0] == "push":
                Path(log).write_text("digest: sha256:" + "d" * 64 + " size: 123", encoding="utf-8")
            return ""
        with tempfile.TemporaryDirectory() as temp, patch.object(ci, "docker", side_effect=registry) as docker, patch.object(ci, "pull_with_retry"):
            root = Path(temp)
            validated_evidence(root)
            pulled = image()
            pulled["Id"] = "sha256:" + "c" * 64
            with patch.object(ci, "inspect", side_effect=[image(), pulled]):
                with self.assertRaisesRegex(ValueError, "Registry pull differs"):
                    ci.publish("seedvr2", ENV, root)
            self.assertFalse((root / "image.json").exists())
            self.assertEqual(docker.call_args_list[0].args[0:2], ("tag", image()["Id"]))

    def test_successful_publication_receipt_keeps_gpu_unqualified(self):
        def registry(*args, log=None):
            if args[0] == "push":
                Path(log).write_text("digest: sha256:" + "d" * 64 + " size: 123", encoding="utf-8")
            return ""
        with tempfile.TemporaryDirectory() as temp, patch.object(ci, "docker", side_effect=registry), patch.object(ci, "pull_with_retry"), patch.object(ci, "inspect", return_value=image()):
            root = Path(temp)
            validated_evidence(root)
            with patch("builtins.print"):
                ci.publish("seedvr2", ENV, root)
            receipt = ci.read(root / "image.json")
            self.assertEqual(receipt["gpu_validation_status"], "NOT_RUN")
            self.assertEqual(receipt["deployment_image"], ci.REPOSITORY + "@sha256:" + "d" * 64)
            self.assertIn("complete 17-frame inference", receipt["not_run"])

    def test_workflow_routes_only_native_job_and_validates_before_push(self):
        import yaml
        workflow = yaml.load((ci.ROOT / ".github/workflows/dingo-router-ci.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        self.assertIn("sr-native", workflow["on"]["workflow_dispatch"]["inputs"]["build_recipe"]["options"])
        self.assertIn("sr-native-seedvr2-and-flashvsr-plus", workflow["run-name"])
        for name in ("omni-gateway-lite", "seedvr2-c14", "h3-hsdp-offload"):
            self.assertIn("!contains(github.event.head_commit.message, '[sr-native]')", workflow["jobs"][name]["if"])
        job = workflow["jobs"]["sr-native"]
        self.assertEqual(job["strategy"]["matrix"]["backend"], list(ci.BACKENDS))
        self.assertEqual(job["strategy"]["max-parallel"], "1")
        self.assertEqual(job["strategy"]["fail-fast"], "false")
        steps = job["steps"]
        base = next(i for i, s in enumerate(steps) if "_ci.py pull-base" in s.get("run", ""))
        build = next(i for i, s in enumerate(steps) if s.get("uses", "").startswith("docker/build-push-action"))
        verify = next(i for i, s in enumerate(steps) if "_ci.py verify-candidate" in s.get("run", ""))
        publish = next(i for i, s in enumerate(steps) if "_ci.py publish" in s.get("run", ""))
        self.assertLess(base, build)
        self.assertLess(build, verify)
        self.assertLess(verify, publish)
        self.assertEqual(steps[build]["with"]["push"], "false")
        self.assertEqual(steps[build]["with"]["load"], "true")
        self.assertIn("GPU_NOT_RUN", steps[publish]["run"])


if __name__ == "__main__":
    unittest.main()
