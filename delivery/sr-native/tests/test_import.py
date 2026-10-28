"""Imported artifacts are consistency evidence, never an independent registry/GPU check."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("sr_native_import", HERE / "import_evidence.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ci = module.ci
COMMIT = "a" * 40
ENV = {"GITHUB_SHA": COMMIT, "GITHUB_REF_NAME": ci.BRANCH, "GITHUB_EVENT_NAME": "push",
       "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1"}


def fixture(directory):
    values = ci.expected_inputs("seedvr2", ENV)
    ci.dump(directory / "inputs.json", values)
    base = {"Os": "linux", "Architecture": "amd64", "Id": "sha256:" + "f" * 64,
            "RepoDigests": [values["base_image"]]}
    ci.dump(directory / "base-image-inspect.json", base)
    ci.dump(directory / "base-resolved.json", {"status": "PASS_IMMUTABLE_BASE_PULL", "backend": "seedvr2",
        "base_image": values["base_image"], "image_id": base["Id"]})
    ci.dump(directory / "final-container-state.json", {"exit_code": 0, "oom_killed": False})
    ci.dump(directory / "candidate-image-inspect.json", {"Id": "sha256:" + "b" * 64,
        "Os": "linux", "Architecture": "amd64", "Config": {"User": "1000",
        "Env": ["DYNAMO_COMMIT_SHA=" + COMMIT], "Labels": {"org.opencontainers.image.revision": COMMIT,
        "org.opencontainers.image.base.name": values["base_image"]}}})
    ci.dump(directory / "installed-inventory.json", {"euid": 1000, "source_commit": COMMIT,
        "dpkg": ["libc6\t2.35"], "packages": [{"name": "torch", "version": "2.3.0+cu121"}]})
    ci.dump(directory / "packages.cdx.json", {"bomFormat": "CycloneDX"})
    (directory / "pip-check.txt").write_text("No broken requirements found.\n", encoding="utf-8")
    ci.dump(directory / "final/runtime-validation.json", {"status": "PASS_CPU_ONLY", "backend": "seedvr2",
        "euid": 1000, "source_commit": COMMIT, "gpu_validation_status": "NOT_RUN"})
    seal = {"shared_inputs": {".gitattributes": "hash"}, "delta": {
        "delivery/sr-native/source-lock.json": {"status": "A"},
        "container/sr-native/shared/example.py": {"status": "A"}}}
    sources = {"delivery/sr-native/source-lock.json": (ci.ROOT / ci.LOCK_PATH).read_bytes(),
               "delivery/sr-native/input-seal.json": (json.dumps(seal) + "\n").encode(),
               ".gitattributes": b"* text eol=lf\n", "container/sr-native/shared/example.py": b"# fixture\n"}
    for name, data in sources.items():
        path = directory / "recipe" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    digest = "sha256:" + "c" * 64
    record = {"schema": "sr-native-publication/1", "status": "PASS_PUBLISHED_AND_PULLED_CPU_VALIDATED",
        "backend": "seedvr2", "branch": ci.BRANCH, "source_commit": COMMIT, "base_image": values["base_image"],
        "image": values["final_image"], "image_id": "sha256:" + "b" * 64, "digest": digest,
        "deployment_image": ci.REPOSITORY + "@" + digest, "gpu_validation_status": "NOT_RUN",
        "run_id": "123", "run_attempt": "1", "actions_url": "https://github.com/githubgxll/dynamo/actions/runs/123"}
    (directory / "deployment-image.txt").write_text(record["deployment_image"] + "\n", encoding="utf-8")
    ci.dump(directory / "image.json", record)
    refresh(directory)

    def git(*args, binary=False):
        if args == ("rev-parse", "HEAD"):
            return COMMIT
        if len(args) == 2 and args[0] == "show" and args[1].startswith(COMMIT + ":"):
            return sources[args[1].split(":", 1)[1]]
        raise AssertionError(args)
    return git


def refresh(directory):
    record = ci.read(directory / "image.json")
    record["evidence_sha256"] = ci.evidence_hashes(directory)
    ci.dump(directory / "image.json", record)


class ImportGates(unittest.TestCase):
    def test_consistent_artifact_passes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(module.ops, "git", side_effect=fixture(root)):
                result = module.verify(root)
            self.assertEqual(result["gpu_validation_status"], "NOT_RUN")

    def test_cross_attempt_inputs_rejected_even_when_hashes_match(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            git = fixture(root)
            value = ci.read(root / "inputs.json")
            value["run_attempt"] = "2"
            ci.dump(root / "inputs.json", value)
            refresh(root)
            with patch.object(module.ops, "git", side_effect=git), self.assertRaisesRegex(ValueError, "Inputs and publication"):
                module.verify(root)

    def test_missing_sealed_recipe_file_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            git = fixture(root)
            (root / "recipe/container/sr-native/shared/example.py").unlink()
            refresh(root)
            with patch.object(module.ops, "git", side_effect=git), self.assertRaisesRegex(ValueError, "exactly the sealed"):
                module.verify(root)

    def test_altered_recipe_rejected_against_git(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            git = fixture(root)
            (root / "recipe/container/sr-native/shared/example.py").write_text("# changed\n", encoding="utf-8")
            refresh(root)
            with patch.object(module.ops, "git", side_effect=git), self.assertRaisesRegex(ValueError, "Recipe differs"):
                module.verify(root)

    def test_wrong_base_architecture_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            git = fixture(root)
            value = ci.read(root / "base-image-inspect.json")
            value["Architecture"] = "arm64"
            ci.dump(root / "base-image-inspect.json", value)
            refresh(root)
            with patch.object(module.ops, "git", side_effect=git), self.assertRaisesRegex(ValueError, "Base is not"):
                module.verify(root)

    def test_failed_container_state_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            git = fixture(root)
            ci.dump(root / "final-container-state.json", {"exit_code": 137, "oom_killed": True})
            refresh(root)
            with patch.object(module.ops, "git", side_effect=git), self.assertRaisesRegex(ValueError, "did not finish"):
                module.verify(root)

    def test_unmanifested_file_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            git = fixture(root)
            (root / "unexpected.txt").write_text("extra", encoding="utf-8")
            with patch.object(module.ops, "git", side_effect=git), self.assertRaisesRegex(ValueError, "evidence manifest"):
                module.verify(root)

    def test_registry_identity_rejects_bad_digest_and_old_repository(self):
        for wrong_repository in (False, True):
            with self.subTest(old_repository=wrong_repository), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                git = fixture(root)
                value = ci.read(root / "image.json")
                if wrong_repository:
                    value["deployment_image"] = (
                        "registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm@" + value["digest"])
                    (root / "deployment-image.txt").write_text(value["deployment_image"] + "\n", encoding="utf-8")
                    error = "Deployment image identity"
                else:
                    value["digest"] = "abc123"
                    error = "Invalid registry digest"
                ci.dump(root / "image.json", value)
                with patch.object(module.ops, "git", side_effect=git), self.assertRaisesRegex(ValueError, error):
                    module.verify(root)


if __name__ == "__main__":
    unittest.main()
