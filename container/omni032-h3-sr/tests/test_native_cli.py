# SPDX-License-Identifier: Apache-2.0
"""Source/configuration contracts only; real native parsing runs in the image."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import typing
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
FIXTURES = HERE / "tests/fixtures/upstream"
EXPECTED_SHARED_SOURCE_SHA256 = "3c6c91516856e952852311a08f5d177bdbe4681de31ae08da4401a5fc27f6643"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cli = load("_test_native_cli_gate", HERE / "native_cli_smoke.py")


def official_offload():
    # This official module has only stdlib imports. It is a real configuration
    # parser; executing it is neither an Omni CLI import nor a GPU acceptance.
    return load("_test_official_032_offload", FIXTURES / "vllm_omni/diffusion/offloader/config.py")


def official_stage_parser():
    path = FIXTURES / "vllm_omni/entrypoints/utils.py"
    function = next(node for node in ast.parse(path.read_bytes()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "parse_stage_overrides")
    scope = {"Any": typing.Any, "json": json}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "official-stage-parser-only", "exec"), scope)
    return scope["parse_stage_overrides"]


class NamespaceFixture(SimpleNamespace):
    def get_explicit_kwargs_dict(self):
        return vars(self)


class CacheFixture:
    @classmethod
    def from_dict(cls, value):
        return SimpleNamespace(**value)


class NativeCLIContracts(unittest.TestCase):
    def test_frozen_artifacts_are_lf_and_snapshot_identity_uses_actual_bytes(self):
        for path in (HERE / "native-cli-argv.json", HERE / "native-cli-k8s-source.json", Path(__file__)):
            with self.subTest(path=path.name):
                self.assertNotIn(b"\r", path.read_bytes())
        snapshot = (HERE / "native-cli-k8s-source.json").read_bytes()
        value = cli.contract()
        self.assertEqual(hashlib.sha256(snapshot).hexdigest(), EXPECTED_SHARED_SOURCE_SHA256)
        self.assertEqual(value["k8s_contract"]["shared_document_sha256"], EXPECTED_SHARED_SOURCE_SHA256)
        self.assertEqual(value["k8s_contract"]["shared_document_line_endings"], "LF")

    def test_crlf_export_cannot_reuse_lf_raw_source_identity(self):
        original = (HERE / "native-cli-k8s-source.json").read_bytes()
        crlf = original.replace(b"\n", b"\r\n")
        self.assertEqual(json.loads(crlf), json.loads(original))
        self.assertNotEqual(hashlib.sha256(crlf).hexdigest(), EXPECTED_SHARED_SOURCE_SHA256)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "native-cli-argv.json").write_bytes((HERE / "native-cli-argv.json").read_bytes())
            (root / "native-cli-k8s-source.json").write_bytes(crlf)
            with patch.object(cli, "HERE", root), patch.object(cli, "CONTRACT", root / "native-cli-argv.json"):
                with self.assertRaisesRegex(ValueError, "source snapshot changed"):
                    cli.contract()

    def test_lf_normalized_export_satisfies_the_unchanged_actual_byte_contract(self):
        original = (HERE / "native-cli-k8s-source.json").read_bytes()
        normalized = original.replace(b"\n", b"\r\n").replace(b"\r\n", b"\n")
        self.assertEqual(normalized, original)
        self.assertEqual(hashlib.sha256(normalized).hexdigest(), EXPECTED_SHARED_SOURCE_SHA256)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "native-cli-argv.json").write_text((HERE / "native-cli-argv.json").read_text(encoding="utf-8"),
                                                       encoding="utf-8", newline="\n")
            (root / "native-cli-k8s-source.json").write_bytes(normalized)
            with patch.object(cli, "HERE", root), patch.object(cli, "CONTRACT", root / "native-cli-argv.json"):
                value = cli.contract()
            self.assertEqual(value["k8s_contract"]["argv_fingerprint_sha256"],
                             "90e5283f161883ca330c9fad3fb07dc9ea784ea67b7054943e7ba116f148ad6e")

    def test_frozen_profiles_are_complete_and_probe_is_separate(self):
        value = cli.contract()
        self.assertEqual([item["name"] for item in value["profiles"]], list(cli.PROFILE_NAMES))
        h3, sr, probe = value["profiles"]
        self.assertEqual(probe["argv"], h3["argv"] + ["--profiler-config", "{}"])
        self.assertNotIn("--profiler-config", h3["argv"])
        self.assertEqual(probe["role"], "parser_only_extension_not_a_K8S_launch_change")
        for flag in ("--diffusion-offload-config", "--use-hsdp", "--no-ulysses-a2a-permute",
                     "--enable-diffusion-pipeline-profiler", "--cache-config", "--enable-cache-dit-summary"):
            self.assertIn(flag, h3["argv"])
        self.assertIn("--stage-overrides", sr["argv"])
        self.assertEqual(sr["expected"]["stage_overrides"]["0"], {
            "ulysses_degree": 8, "vae_patch_parallel_size": 8, "vae_parallel_mode": "spatial_shard_height"})

    def test_freeze_rejects_modified_argv_without_matching_hash(self):
        value = copy.deepcopy(cli.contract())
        value["profiles"][0]["argv"] += ["--unexpected"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "argv.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with patch.object(cli, "CONTRACT", path), self.assertRaisesRegex(ValueError, "argv changed"):
                cli.contract()

    def test_changed_argv_cannot_diverge_from_k8s_snapshot_even_with_new_inner_hash(self):
        value = copy.deepcopy(cli.contract())
        profile = value["profiles"][0]
        position = profile["argv"].index("--ulysses-degree") + 1
        profile["argv"][position] = "4"
        profile["argv_sha256"] = cli.canonical_hash(profile["argv"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "argv.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with patch.object(cli, "CONTRACT", path), self.assertRaisesRegex(ValueError, "shared K8S source"):
                cli.contract()

    def test_cli_sources_match_the_official_frozen_parser_contract(self):
        value = cli.contract()
        for name, expected in value["parser_sources_sha256"].items():
            path = FIXTURES / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
        serve = ast.parse((FIXTURES / "vllm_omni/entrypoints/cli/serve.py").read_bytes())
        command = next(node for node in serve.body if isinstance(node, ast.ClassDef) and node.name == "OmniServeCommand")
        init = next(node for node in command.body if isinstance(node, ast.FunctionDef) and node.name == "subparser_init")
        functions = {node.func.id for node in ast.walk(init) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertTrue({"_ensure_vllm_platform", "make_arg_parser"} <= functions)

    def test_gate_uses_complete_parse_args_without_server_dispatch(self):
        tree = ast.parse((HERE / "native_cli_smoke.py").read_bytes())
        attrs = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertIn("parse_args", attrs)
        self.assertNotIn("parse_known_args", attrs)
        self.assertNotIn("cmd", attrs)
        self.assertNotIn("validate", attrs)
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        text = ast.unparse(run)
        self.assertIn("serve._ensure_vllm_platform()", text)
        self.assertIn("command.subparser_init", text)
        self.assertIn("unknown-omni032-parser-probe", text)

    def test_public_offload_parser_uses_native_typed_contract(self):
        module = official_offload()
        raw = cli.contract()["profiles"][0]["expected"]["diffusion_offload_config"]
        parsed = module.parse_diffusion_offload_config(raw)
        self.assertIsInstance(parsed, module.ParsedDiffusionOffloadConfig)
        self.assertIs(parsed.mode, module.OffloadMode.MODULE)
        self.assertEqual(parsed.components, frozenset({"dit", "text_encoder"}))
        self.assertIs(parsed.pin_memory, True)
        for modified in ({**raw, "pin_memory": "true"}, {**raw, "components": ["vae"]}, {**raw, "unknown": 1}):
            with self.subTest(config=modified), self.assertRaises((TypeError, ValueError)):
                module.parse_diffusion_offload_config(modified)

    def test_native_stage_override_shape_rejects_bad_stage_ids(self):
        parse = official_stage_parser()
        expected = cli.contract()["profiles"][1]["expected"]["stage_overrides"]
        self.assertEqual(parse(json.dumps(expected)), expected)
        for raw in ([], {"foo": {}}, {"0": []}, {"-1": {}}):
            with self.subTest(config=raw), self.assertRaises(ValueError):
                parse(raw)

    def test_configuration_guard_rejects_changed_native_fields(self):
        profile = cli.contract()["profiles"][0]
        values = copy.deepcopy(profile["expected"])
        values["ulysses_degree"] = 4
        namespace = NamespaceFixture(**values)
        with self.assertRaisesRegex(ValueError, "frozen values"):
            cli.verify_profile(profile, namespace, stage_parser=official_stage_parser(),
                offload_module=official_offload(), cache_class=CacheFixture)

    def test_unrequested_engine_defaults_are_not_serialized_or_instantiated(self):
        namespace = NamespaceFixture(dtype="bfloat16", cache_config='{"x":1}', opaque_engine_default=object())
        self.assertEqual(cli.parsed_values(namespace, {"dtype", "cache_config"}), {"dtype": "bfloat16", "cache_config": {"x": 1}})


if __name__ == "__main__":
    unittest.main()
