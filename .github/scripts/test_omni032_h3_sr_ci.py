"""Submission/publication negative tests with simulated Git/Docker evidence."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("_tested_omni032_ci", ROOT / ".github/scripts/omni032_h3_sr_ci.py")
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)
ops = ci.ops
COMMIT = "a" * 40
FILES = ci.expected_sources()


def lock():
    return {"repository": "https://github.com/githubgxll/dynamo.git", "remote_branch": ci.BRANCH,
            "harbor_repository": ci.REPOSITORY, "parent_image": ci.PARENT, "parent_source_commit": ci.BASE_COMMIT,
            "recipe_base_commit": ci.BASE_COMMIT, "working_tree_base_commit": ci.BASE_COMMIT,
            "platform": "linux/amd64", "omni_release_commit": "5917730a4b02aedb44d374173cdc1b10484ccd52",
            "versions": {"vllm": "0.32.0", "vllm-omni": "0.32.0", "torch": "2.13.0+cu130", "cache-dit": "1.5.0"},
            "tag_prefix": "omni032-gateway-h3-sr"}


def contract(recipe):
    return {"schema": "h3-sr-omni032-upgrade/1", "repository": lock()["repository"], "branch": ci.BRANCH,
            "functional_parent_image": ci.PARENT, "harbor_repository": ci.REPOSITORY,
            "sole_online_submitter_thread": "01a0eb06-bc59-7ee0-b2d8-b2467e740cd1", "build_request_count": 0,
            "reviews": {name: {"status": "PASS"} for name in ("image", "k8s", "independent_audit")},
            "freeze": {"recipe_sha256": recipe["recipe_sha256"], "tree": recipe["tree"]}}


def reports(directory):
    legal_files = {"NOTICES-Python.txt": "d" * 64, "policy-validation.txt": "e" * 64}
    legal_hash = ci.hash_value(legal_files)
    ci.dump(directory / "runtime-validation.json", {"status": "PASS_CPU_ONLY", "source_commit": COMMIT,
            "euid": 1000, "gpu_validation_status": "NOT_RUN", "source_manifest_sha256": ci.hash_value(FILES),
            "legal_status": "PASS", "legal_manifest_sha256": legal_hash})
    ci.dump(directory / "source-verification.json", {"status": "PASS", "source_sha256": FILES,
            "source_manifest_sha256": ci.hash_value(FILES)})
    ci.dump(directory / "environment/dependency-audit.json", {"status": "PASS", "errors": [], "known_metadata_exceptions": []})
    ci.dump(directory / "cpu/regression.json", {"status": "PASS", "tests_run": 30, "failures": 0, "errors": 0, "skipped": 0})
    ci.dump(directory / "rmsnorm/runtime-validation.json", {"status": "PASS", "source_commit": COMMIT,
            "device": "cpu", "six_case_count": 6,
            "six_cases": [{"dtype": dtype, "affine": affine, "finite": True, "allclose": True}
                          for dtype in ("torch.float16", "torch.bfloat16", "torch.float32") for affine in (False, True)],
            "all_300": {"passed": True}, "module_sha256": FILES["vllm_omni/diffusion/models/seedvr2/nadit.py"]})
    policy = ci.read(ROOT / "container/h3-hsdp-offload/torch-contract.json")
    ci.dump(directory / "torch-contract.json", {"status": "PASS", "schema": policy["schema"],
            "fingerprint": policy["fingerprint"], "torch_version": "2.13.0+cu130",
            "methods": {item["module"] + "." + item["class"] + "." + item["method"]: item["ast_sha256"] for item in policy["methods"]}})
    ci.dump(directory / "legal-validation.json", {"status": "PASS", "source_commit": COMMIT, "files": legal_files,
            "legal_manifest_sha256": legal_hash, "changed_python_components": 2,
            "policy_sha256": ci.hashlib.sha256((ROOT / "container/compliance/policy/licenses.toml").read_bytes()).hexdigest()})
    argv_path = ROOT / "container/omni032-h3-sr/native-cli-argv.json"
    argv = ci.read(argv_path)
    ci.dump(directory / "native-cli/native-cli.json", {"schema": "omni032-native-cli-validation/1",
            "status": "PASS", "source_commit": COMMIT, "euid": 1000, "dispatch_executed": False,
            "engine_started": False, "model_loaded": False, "unknown_options_rejected": True,
            "argv_contract_sha256": ci.hashlib.sha256(argv_path.read_bytes()).hexdigest(),
            "parser_sources_sha256": argv["parser_sources_sha256"],
            "profiles": [{"name": item["name"], "argv_sha256": item["argv_sha256"], "status": "PASS"} for item in argv["profiles"]]})


class ReviewAndSubmission(unittest.TestCase):
    def test_strict_scope(self):
        for path in ("container/omni032-h3-sr/h3_patches.py", ".github/workflows/omni032-h3-sr.yml"):
            self.assertTrue(ops.permitted(path))
        for path in ("dingo/vllm/omni/main.py", "container/seedvr2-optimization/runtime.py", "README.md"):
            self.assertFalse(ops.permitted(path))
            with self.assertRaises(ValueError): ops.check_scope([path])

    def test_request_marker_cannot_route_other_recipes_or_skip_workflow(self):
        self.assertTrue(ops.valid_request_message("upgrade " + ops.MARKER))
        for suffix in ("[seedvr2-metrics-repair]", "[sr-native]", "[skip ci]", "\nskip-checks: true", ops.MARKER):
            with self.subTest(suffix=suffix):
                self.assertFalse(ops.valid_request_message("upgrade " + ops.MARKER + suffix))

    def test_all_three_reviews_and_exact_freeze_required(self):
        recipe = {"recipe_sha256": "c" * 64, "tree": "d" * 40}
        value = contract(recipe)
        self.assertEqual(ops.review_contract(value, lock(), recipe), value)
        for name in value["reviews"]:
            changed = copy.deepcopy(value)
            changed["reviews"][name]["status"] = "PENDING"
            with self.subTest(name=name), self.assertRaises(ValueError): ops.review_contract(changed, lock(), recipe)
        for key in ("recipe_sha256", "tree"):
            changed = copy.deepcopy(value)
            changed["freeze"][key] = "0" * len(changed["freeze"][key])
            with self.subTest(key=key), self.assertRaises(ValueError): ops.review_contract(changed, lock(), recipe)

    def test_nonnegative_request_history_requires_terminal_unpublished_result(self):
        recipe = {"recipe_sha256": "c" * 64, "tree": "d" * 40}
        for count in (-1, False, 0.0, "0", None):
            value = contract(recipe)
            value["build_request_count"] = count
            with self.subTest(count=count), self.assertRaises(ValueError): ops.review_contract(value, lock(), recipe)
        previous = {"repository": lock()["repository"], "branch": ci.BRANCH,
                    "workflow_path": ".github/workflows/omni032-h3-sr.yml", "commit": COMMIT,
                    "run_id": 123, "run_attempt": 2, "status": "completed", "conclusion": "failure",
                    "publication_status": "not_started"}
        for count in (1, 3, 17):
            value = {**contract(recipe), "build_request_count": count, "submitted_commit": COMMIT,
                     "previous_build_result": previous}
            with self.subTest(count=count):
                self.assertEqual(ops.review_contract(value, lock(), recipe), value)
        value = {**contract(recipe), "build_request_count": 1, "submitted_commit": COMMIT}
        with self.assertRaises(ValueError): ops.review_contract(value, lock(), recipe)

    def test_atomic_journal_never_overwrites_existing_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "request.json"
            ops.claim_request(path, {"status": "REQUESTED"})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError): ops.claim_request(path, {"status": "REQUESTED_AGAIN"})
            self.assertEqual(path.read_bytes(), before)

    def test_unaccounted_existing_journal_stops_before_any_network(self):
        recipe = {"recipe_sha256": "c" * 64, "tree": "d" * 40}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = root / "shared-contract.json"
            ops.dump(shared, contract(recipe))
            ops.dump(root / "one-build-journal.json", {"status": "UNKNOWN"})
            with mock.patch.object(ops, "EVIDENCE", root), mock.patch.object(ops, "CONTRACT", shared), \
                 mock.patch.object(ops, "check", return_value=(lock(), "full-tree", recipe)), \
                 mock.patch.object(ops.LEGACY, "remote_head") as remote, mock.patch.object(ops, "observe_run") as observe:
                with self.assertRaises(ValueError): ops.submit(True)
                remote.assert_not_called()
                observe.assert_not_called()

    def test_ambiguous_push_is_claimed_once_and_never_automatically_retried(self):
        recipe = {"recipe_sha256": "c" * 64, "tree": "d" * 40}
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            journal = Path(tmp) / "one-build-journal.json"
            shared = Path(tmp) / "shared-contract.json"
            ops.dump(shared, contract(recipe))
            def git(*arguments, **kwargs):
                calls.append(arguments)
                if arguments[0] == "push":
                    self.assertEqual(ops.read(journal)["status"], "REQUESTED")
                    raise RuntimeError("network response lost")
                if arguments[:2] == ("log", "-1"): return "upgrade " + ops.MARKER
                if arguments[0] == "status": return ""
                if arguments[:2] == ("rev-parse", "HEAD^{tree}"): return "reviewed-full-tree"
                if arguments[:2] == ("rev-parse", "HEAD"): return COMMIT
                raise AssertionError(arguments)
            with mock.patch.object(ops, "EVIDENCE", Path(tmp)), \
                 mock.patch.object(ops, "CONTRACT", shared), \
                 mock.patch.object(ops, "check", return_value=(lock(), "reviewed-full-tree", recipe)), \
                 mock.patch.object(ops.LEGACY, "remote_head", return_value=ci.BASE_COMMIT), \
                 mock.patch.object(ops, "prospective", return_value=("reviewed-full-tree", recipe)), \
                 mock.patch.object(ops, "check_seal"), mock.patch.object(ops, "git", side_effect=git):
                with self.assertRaisesRegex(RuntimeError, "response lost"): ops.submit(True)
                self.assertEqual(ops.read(journal)["status"], "UNKNOWN")
                with self.assertRaises(ValueError): ops.submit(True)
                self.assertEqual(sum(args[0] == "push" for args in calls), 1)

    def test_no_dispatch_or_empty_commit_retry_api(self):
        self.assertFalse(hasattr(ops, "dispatch"))
        source = (ROOT / "delivery/omni032-h3-sr/ops.py").read_text(encoding="utf-8")
        self.assertNotIn('"--allow-empty"', source)

    def test_symlink_recipe_delta_rejected(self):
        raw = (":000000 120000 " + "0" * 40 + " " + "1" * 40 + " A\0container/omni032-h3-sr/link\0").encode()
        with mock.patch.object(ops, "git", return_value=raw), self.assertRaises(ValueError):
            ops.raw_delta("base", "tree")

    def test_working_bytes_must_match_the_publishable_git_blob(self):
        recipe = {"recipe_sha256": "c" * 64, "tree": "d" * 40}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = "container/omni032-h3-sr/input.py"
            target = root / name
            target.parent.mkdir(parents=True)
            target.write_bytes(b"unreviewed bytes")
            def read(path):
                return recipe if str(path).endswith("input-seal.json") else contract(recipe)
            with mock.patch.object(ops, "ROOT", root), mock.patch.object(ops, "lock", return_value=lock()), \
                 mock.patch.object(ops, "identity"), mock.patch.object(ops, "read", side_effect=read), \
                 mock.patch.object(ops, "prospective", return_value=("full-tree", recipe)), \
                 mock.patch.object(ops, "raw_delta", return_value={name: {"status": "M", "blob": "e" * 40}}), \
                 mock.patch.object(ops, "_blob_contents", return_value={"e" * 40: b"frozen bytes"}):
                with self.assertRaisesRegex(ValueError, "Working bytes differ"): ops.check_seal()
                target.write_bytes(b"frozen bytes")
                self.assertEqual(ops.check_seal()[2], recipe)


class LocalGitRecipeIntegration(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        binary = shutil.which("git") or r"D:\workingtools\Git\cmd\git.exe"
        def git(*arguments, env=None, network=False, binary=False):
            if network: raise AssertionError("No network in local recipe integration")
            command = [str(git_binary), "-C", str(self.root), "-c", "safe.directory=" + self.root.as_posix(), *arguments]
            result = subprocess.run(command, env=env, capture_output=True, check=True)
            return result.stdout if binary else result.stdout.decode("utf-8").strip()
        git_binary = binary
        self.git = git
        git("init", "-b", ci.BRANCH)
        git("config", "user.name", "Local Fixture")
        git("config", "user.email", "fixture@example.invalid")
        helper = self.root / "container/h3-hsdp-offload/helper.py"
        helper.parent.mkdir(parents=True)
        helper.write_bytes(b"# fixed shared input\n")
        git("add", ".")
        git("commit", "-m", "fixture baseline")
        self.value = lock()
        base = git("rev-parse", "HEAD")
        self.value.update(recipe_base_commit=base, working_tree_base_commit=base)
        self.source = self.root / "container/omni032-h3-sr/new.py"
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(b"value = 1\n")
        for name, value in (("ROOT", self.root), ("EVIDENCE", self.root / ".evidence"), ("git", git)):
            patch = mock.patch.object(ops, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        # Temporary indexes/evidence must not enter the recipe's untracked set.
        (self.root / ".git/info/exclude").write_text(".evidence/\n", encoding="utf-8")

    def test_seal_approval_are_nonrecursive_and_user_index_is_unchanged(self):
        before_index = self.git("diff", "--cached", "--name-only")
        first_tree, first = ops.prospective(self.value)
        self.assertEqual(set(first["delta"]), {"container/omni032-h3-sr/new.py"})
        self.assertIn("container/h3-hsdp-offload/helper.py", first["shared_inputs"])
        ops.dump(self.root / ops.SEAL, first)
        ops.dump(self.root / ops.APPROVAL, contract(first))
        second_tree, second = ops.prospective(self.value)
        self.assertNotEqual(first_tree, second_tree)
        self.assertEqual(first, second)
        self.assertEqual(self.git("diff", "--cached", "--name-only"), before_index)
        self.assertEqual(ops.review_contract(contract(first), self.value, second)["freeze"]["tree"], first["tree"])

    def test_actual_source_edit_invalidates_previous_review(self):
        _, first = ops.prospective(self.value)
        self.source.write_text("value = 2\n", encoding="utf-8")
        _, changed = ops.prospective(self.value)
        self.assertNotEqual(first["recipe_sha256"], changed["recipe_sha256"])
        self.assertNotEqual(first["tree"], changed["tree"])
        with self.assertRaises(ValueError): ops.review_contract(contract(first), self.value, changed)

    def test_unrelated_and_shared_source_edits_are_not_staged(self):
        unrelated = self.root / "README.md"
        unrelated.write_text("do not stage\n", encoding="utf-8")
        with self.assertRaises(ValueError): ops.prospective(self.value)
        self.assertEqual(self.git("diff", "--cached", "--name-only"), "")
        unrelated.unlink()
        (self.root / "container/h3-hsdp-offload/helper.py").write_text("changed shared source\n", encoding="utf-8")
        with self.assertRaises(ValueError): ops.prospective(self.value)
        self.assertEqual(self.git("diff", "--cached", "--name-only"), "")

    def test_crlf_working_bytes_under_lf_attributes_block_seal_before_dump(self):
        attributes = self.source.parent / ".gitattributes"
        attributes.write_bytes(b"* text eol=lf\n")
        self.source.write_bytes(b"value = 1\r\n")
        before_index = self.git("diff", "--cached", "--name-only")
        tree, recipe = ops.prospective(self.value)
        blob = recipe["delta"]["container/omni032-h3-sr/new.py"]["blob"]
        self.assertEqual(ops._blob_contents([blob])[blob], b"value = 1\n")
        with self.assertRaisesRegex(ValueError, "CRLF/LF/filter differences"):
            ops.check_export(self.value, tree, recipe)
        with mock.patch.object(ops, "lock", return_value=self.value), mock.patch.object(ops, "identity"), \
             mock.patch.object(ops, "dump") as writer:
            with self.assertRaisesRegex(ValueError, "Working bytes differ"):
                ops.seal()
            writer.assert_not_called()
        self.assertFalse((self.root / ops.SEAL).exists())
        self.assertFalse((self.root / ops.APPROVAL).exists())
        self.source.write_bytes(b"value = 1\n")
        tree, recipe = ops.prospective(self.value)
        self.assertEqual(ops.check_export(self.value, tree, recipe)["status"], "PASS_LOCAL_EXPORT_BYTES_ONLY")
        self.assertEqual(self.git("diff", "--cached", "--name-only"), before_index)

    def test_export_checks_locked_shared_raw_bytes_even_when_git_hides_crlf_difference(self):
        self.git("config", "core.autocrlf", "true")
        shared = self.root / "container/h3-hsdp-offload/helper.py"
        shared.write_bytes(b"# fixed shared input\r\n")
        tree, recipe = ops.prospective(self.value)
        self.assertNotIn("container/h3-hsdp-offload/helper.py", recipe["delta"])
        with self.assertRaisesRegex(ValueError, "container/h3-hsdp-offload/helper.py"):
            ops.check_export(self.value, tree, recipe)
        shared.write_bytes(b"# fixed shared input\n")
        tree, recipe = ops.prospective(self.value)
        report = ops.check_export(self.value, tree, recipe)
        self.assertEqual(report["shared_paths"], 1)
        self.assertGreater(report["checked_input_paths"], report["shared_paths"])
        self.assertFalse((self.root / ops.APPROVAL).exists())
        self.assertEqual(self.git("diff", "--cached", "--name-only"), "")


class Publication(unittest.TestCase):
    def environment(self):
        return {"GITHUB_SHA": COMMIT, "GITHUB_REF_NAME": ci.BRANCH, "GITHUB_EVENT_NAME": "push",
                "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1"}

    def test_positive_attempts_on_the_designated_push_branch(self):
        value = self.environment()
        for attempt in ("1", "2", "3"):
            self.assertEqual(ci.identity({**value, "GITHUB_RUN_ATTEMPT": attempt}), COMMIT)
        for key, changed in (("GITHUB_RUN_ATTEMPT", "0"), ("GITHUB_EVENT_NAME", "workflow_dispatch"),
                             ("GITHUB_REF_NAME", "main"), ("GITHUB_SHA", "short")):
            with self.subTest(key=key), self.assertRaises(ValueError): ci.identity({**value, key: changed})

    def test_empty_commit_fails_before_any_registry_or_build_operation(self):
        recipe = {"recipe_sha256": "c" * 64, "tree": "d" * 40}
        def git(*arguments, **kwargs):
            if arguments[0] == "log": return "upgrade " + ops.MARKER
            if arguments[0] == "rev-parse": return COMMIT
            if arguments[0] == "diff-tree": return ""
            raise AssertionError(arguments)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(ci.os.environ, self.environment()), \
             mock.patch.object(ops, "check_seal", return_value=(lock(), "tree", recipe)), \
             mock.patch.object(ops, "git", side_effect=git), mock.patch.object(ci, "docker") as tool:
            with self.assertRaisesRegex(ValueError, "Empty/approval-only"): ci.prepare(Path(tmp))
            tool.assert_not_called()

    def test_parent_and_framework_lock(self):
        ci.validate_lock(lock())
        for key, changed in (("parent_image", "nvidia/cuda:latest"), ("runtime_base_image", "new-base"),
                             ("working_tree_base_commit", "short"), ("tag_prefix", "../../other")):
            with self.subTest(key=key), self.assertRaises(ValueError): ci.validate_lock({**lock(), key: changed})
        value = lock()
        value["versions"]["torch"] = "2.14.0"
        with self.assertRaises(ValueError): ci.validate_lock(value)

    def test_only_exact_verified_inherited_nccl_exception_passes(self):
        value = {"status": "PASS_WITH_UPSTREAM_NCCL_OVERRIDE", "errors": [], "known_metadata_exceptions": [ci.NCCL_DIAGNOSTIC],
                 "upstream_nccl_evidence": {"verified": True, "errors": [], "inherited_from_parent": ci.PARENT,
                    "base_reference": ci.INHERITED_BASE, "override_file": "/etc/uv-overrides.txt",
                    "overrides_sha256": "969805720e16848021d2bdc6be10c970b0c5d4f25f31d76b655eaaf6b595d83c",
                    "protected_before_omni_sha256": "f14b55d884f304fe97ed7aee13caf4d693bbeeec5e26eb20c68ede4089a7b6fe"}}
        ci.validate_dependency_report(value)
        for key, changed in (("verified", False), ("errors", ["unverified"]),
                ("inherited_from_parent", "other"), ("base_reference", "vllm/vllm-openai:v0.32.0"),
                ("overrides_sha256", "0" * 64), ("protected_before_omni_sha256", "0" * 64)):
            broken = copy.deepcopy(value)
            broken["upstream_nccl_evidence"][key] = changed
            with self.subTest(key=key), self.assertRaises(ValueError): ci.validate_dependency_report(broken)
        for status, known in (("PASS_EXPERIMENTAL_DECLARED_EXCEPTION", [ci.NCCL_DIAGNOSTIC]),
                ("PASS", [ci.NCCL_DIAGNOSTIC]), ("PASS_WITH_UPSTREAM_NCCL_OVERRIDE", []),
                ("PASS_WITH_UPSTREAM_NCCL_OVERRIDE", [ci.NCCL_DIAGNOSTIC, "protobuf conflict"])):
            broken = {**value, "status": status, "known_metadata_exceptions": known}
            with self.subTest(status=status, known=known), self.assertRaises(ValueError): ci.validate_dependency_report(broken)

    def test_real_environment_producer_and_auditor_feed_the_ci_consumer(self):
        # Project the two effective native records from published b315's
        # installed-environment inventory. This is metadata contract testing,
        # not a current Linux install or the full dependency solve.
        spec = importlib.util.spec_from_file_location("_ci_environment_producer", ROOT / "container/omni032-h3-sr/environment.py")
        environment = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(environment)
        policy = ci.read(ROOT / "container/omni032-h3-sr/environment-policy.json")
        before = {"distributions": [{"name": name, "canonical_name": name, "version": version,
                  "location": "/usr/local/lib/python3.12/dist-packages", "requires_dist": []}
                 for name, version in (("nvidia-nccl-cu13", "2.30.7"), ("torch", "2.13.0+cu130"))]}
        auditor = environment.dependency_auditor(target=True)
        with mock.patch.dict(os.environ, {"DYNAMO_VLLM_BASE_IMAGE": ci.INHERITED_BASE}), \
             mock.patch.object(Path, "read_text", return_value=ci.NCCL_OVERRIDE_TEXT):
            evidence = environment.nccl_evidence(policy, before, auditor)
        def distribution(name, version, requires):
            return types.SimpleNamespace(metadata={"Name": name, "Requires-Python": ">=3.11"}, version=version,
                requires=requires, _path=Path("/usr/local/lib/python3.12/dist-packages") / (name + ".dist-info"))
        # The real edge parser, Linux marker evaluator, exception decision,
        # status finalizer and new provenance producer all execute here.
        distributions = [distribution("torch", "2.13.0+cu130", ['nvidia-nccl-cu13==2.29.7; platform_system == "Linux"']),
                         distribution("nvidia-nccl-cu13", "2.30.7", []),
                         distribution("vllm", "0.32.0", ["torch==2.13.0"]),
                         distribution("vllm-omni", "0.32.0", ["vllm==0.32.0"])]
        report = auditor.audit_dependencies(distributions, python_version="3.12.3", nccl_evidence=evidence,
                    environment={"platform_system": "Linux", "sys_platform": "linux", "platform_machine": "x86_64"})
        auditor.require_framework_versions(report)
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["known_metadata_exceptions"], [ci.NCCL_DIAGNOSTIC])
        ci.validate_dependency_report(report)
        report["upstream_nccl_evidence"]["inherited_from_parent"] = "wrong parent"
        with self.assertRaises(ValueError): ci.validate_dependency_report(report)

    def test_failed_external_command_retains_complete_stdout_and_stderr(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failed-command.txt"
            result = subprocess.CompletedProcess(["docker"], 1, "stdout-start" + "x" * 12000, "stderr-end")
            with mock.patch.object(ci.subprocess, "run", return_value=result), self.assertRaises(RuntimeError):
                ci.execute(["docker", "push", "image"], log=path)
            self.assertEqual(path.read_text(encoding="utf-8"), result.stdout + result.stderr)

    def test_required_reports_and_rejection_of_skips_or_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            reports(directory)
            self.assertEqual(ci.validate_reports(directory, COMMIT)["status"], "PASS")
            for file, key, value in (("runtime-validation.json", "euid", 0),
                    ("runtime-validation.json", "source_commit", "0" * 40),
                    ("source-verification.json", "source_manifest_sha256", "0" * 64),
                    ("environment/dependency-audit.json", "errors", ["conflict"]),
                    ("cpu/regression.json", "skipped", 1)):
                reports(directory)
                changed = ci.read(directory / file)
                changed[key] = value
                ci.dump(directory / file, changed)
                with self.subTest(file=file, key=key), self.assertRaises(ValueError): ci.validate_reports(directory, COMMIT)
            reports(directory)
            (directory / "cpu/regression.json").unlink()
            with self.assertRaises(FileNotFoundError): ci.validate_reports(directory, COMMIT)

    def test_all_named_source_numeric_torch_legal_and_cli_gates_are_mandatory(self):
        changes = (("rmsnorm/runtime-validation.json", "six_case_count", 5),
                   ("torch-contract.json", "torch_version", "2.14.0"),
                   ("torch-contract.json", "methods", {}),
                   ("legal-validation.json", "source_commit", "0" * 40),
                   ("legal-validation.json", "legal_manifest_sha256", "0" * 64),
                   ("native-cli/native-cli.json", "dispatch_executed", True),
                   ("native-cli/native-cli.json", "unknown_options_rejected", False),
                   ("native-cli/native-cli.json", "profiles", []),
                   ("native-cli/native-cli.json", "argv_contract_sha256", "0" * 64))
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for name, key, changed in changes:
                reports(directory)
                value = ci.read(directory / name)
                value[key] = changed
                ci.dump(directory / name, value)
                with self.subTest(name=name, key=key), self.assertRaises(ValueError): ci.validate_reports(directory, COMMIT)
            for name in ("rmsnorm/runtime-validation.json", "torch-contract.json", "legal-validation.json", "native-cli/native-cli.json"):
                reports(directory)
                (directory / name).unlink()
                with self.subTest(name=name), self.assertRaises(FileNotFoundError): ci.validate_reports(directory, COMMIT)

    def test_self_consistent_but_unreviewed_source_map_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            reports(directory)
            proof = ci.read(directory / "source-verification.json")
            proof["source_sha256"]["vllm_omni/unknown.py"] = "1" * 64
            proof["source_manifest_sha256"] = ci.hash_value(proof["source_sha256"])
            runtime = ci.read(directory / "runtime-validation.json")
            runtime["source_manifest_sha256"] = proof["source_manifest_sha256"]
            ci.dump(directory / "source-verification.json", proof)
            ci.dump(directory / "runtime-validation.json", runtime)
            with self.assertRaisesRegex(ValueError, "complete frozen expected"): ci.validate_reports(directory, COMMIT)

    def test_exact_inherited_service_config(self):
        parent = {"Id": ci.PARENT_ID, "Os": "linux", "Architecture": "amd64",
                  "Config": {"User": "dynamo", "Entrypoint": ["/entry"], "Cmd": ["serve"],
                             "Env": ["PRECISION=fp16", "PARENT_SOURCE=old"]}, "RootFS": {"Layers": ["base"]}}
        child = copy.deepcopy(parent)
        child["Id"] = "sha256:" + "c" * 64
        child["RootFS"]["Layers"].append("upgrade")
        child["Config"]["Labels"] = {"org.opencontainers.image.revision": COMMIT, "org.opencontainers.image.base.name": ci.PARENT}
        self.assertEqual(ci.compare_config(parent, child, COMMIT)["status"], "PASS")
        for key, value in (("User", "root"), ("Entrypoint", ["other"]), ("Env", ["PRECISION=bf16"])):
            changed = copy.deepcopy(child)
            changed["Config"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): ci.compare_config(parent, changed, COMMIT)

    def test_digest_parser_requires_single_explicit_digest(self):
        digest = "sha256:" + "e" * 64
        self.assertEqual(ci.digest_from_push("finaltag: digest: " + digest + " size: 123"), digest)
        for text in ("sha256:" + "a" * 64, "digest: short", "digest: " + digest + "\ndigest: sha256:" + "f" * 64):
            with self.subTest(text=text), self.assertRaises(ValueError): ci.digest_from_push(text)

    def test_publication_uses_one_push_and_verifies_same_image_id(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(ci.os.environ, self.environment()):
            directory = Path(tmp)
            reports(directory / "final")
            image = ci.REPOSITORY + ":omni032-test-" + COMMIT[:12]
            image_id = "sha256:" + "c" * 64
            digest = "sha256:" + "e" * 64
            ci.dump(directory / "candidate.json", {"status": "PASS_CPU_ONLY", "commit": COMMIT, "image": image,
                    "image_id": image_id, "source_manifest_sha256": ci.hash_value(FILES),
                    "run_id": self.environment()["GITHUB_RUN_ID"], "run_attempt": int(self.environment()["GITHUB_RUN_ATTEMPT"])})
            def docker(*args, **kwargs):
                return "digest: " + digest if args[0] == "push" else "pulled"
            with mock.patch.object(ci, "inspect", return_value={"Id": image_id}), mock.patch.object(ci, "docker", side_effect=docker) as tool:
                result = ci.publish(directory)
                self.assertEqual(result["published_image"], ci.REPOSITORY + "@" + digest)
                self.assertEqual([c.args for c in tool.call_args_list if c.args[0] == "push"], [("push", image)])
                with self.assertRaises(FileExistsError): ci.publish(directory)
                self.assertEqual(sum(c.args[0] == "push" for c in tool.call_args_list), 1)

    def test_changed_tag_never_reaches_push(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(ci.os.environ, self.environment()):
            directory = Path(tmp)
            ci.dump(directory / "candidate.json", {"status": "PASS_CPU_ONLY", "commit": COMMIT,
                    "image": "tag", "image_id": "old", "run_id": self.environment()["GITHUB_RUN_ID"],
                    "run_attempt": int(self.environment()["GITHUB_RUN_ATTEMPT"])})
            with mock.patch.object(ci, "inspect", return_value={"Id": "different"}), mock.patch.object(ci, "docker") as tool:
                with self.assertRaises(ValueError): ci.publish(directory)
                tool.assert_not_called()

    def test_workflows_route_only_one_build_without_dispatch(self):
        import yaml
        workflow = yaml.load((ROOT / ".github/workflows/omni032-h3-sr.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        self.assertEqual(set(workflow["on"]), {"push"})
        steps = workflow["jobs"]["reviewed-image"]["steps"]
        for step in steps:
            if "uses" in step:
                self.assertRegex(step["uses"], r"@[0-9a-f]{40}$")
        builds = [step for step in steps if step.get("uses", "").startswith("docker/build-push-action@")]
        self.assertEqual(len(builds), 1)
        self.assertEqual(builds[0]["with"]["push"], "false")
        self.assertEqual(builds[0]["with"]["load"], "true")
        self.assertNotIn("github.run_attempt == 1", workflow["jobs"]["reviewed-image"]["if"])
        legacy = yaml.load((ROOT / ".github/workflows/dingo-router-ci.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        push_jobs = [job for job in legacy["jobs"].values() if "github.event_name == 'push'" in job.get("if", "")]
        self.assertEqual(len(push_jobs), 6)
        for job in push_jobs:
            self.assertIn("github.event_name == 'push' && !contains(github.event.head_commit.message, '[omni032-h3-sr]')", job["if"])


if __name__ == "__main__": unittest.main()
