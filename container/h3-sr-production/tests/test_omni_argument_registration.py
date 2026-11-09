"""Execute the real argument group and configuration helpers without engine imports."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import redirect_stderr
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
from typing import Any, Optional
import unittest
from unittest.mock import patch


def source_path(relative):
    repository = Path(__file__).resolve().parents[3]
    local = repository / relative
    if local.is_file():
        return local
    return Path(importlib.metadata.distribution("ai-dingo").locate_file(relative))


def argument_source():
    path = source_path("dingo/vllm/omni/args.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    group = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OmniArgGroup")
    method = next(node for node in group.body if isinstance(node, ast.FunctionDef) and node.name == "add_arguments")
    parse_json = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_parse_diffusion_quantization_config")
    helper_path = source_path("dingo/common/configuration/utils.py")
    spec = importlib.util.spec_from_file_location("_production_argument_helpers", helper_path)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    namespace = {
        "argparse": argparse,
        "json": json,
        "Any": Any,
        "Optional": Optional,
        "add_argument": helpers.add_argument,
        "add_negatable_bool_argument": helpers.add_negatable_bool_argument,
    }
    exec(compile(ast.Module(body=[parse_json, method], type_ignores=[]), str(path), "exec"), namespace)
    return tree, namespace["add_arguments"]


class OmniArgumentRegistration(unittest.TestCase):
    def setUp(self):
        clean = {name: value for name, value in os.environ.items() if not name.startswith("DYN_OMNI_")}
        environment = patch.dict(os.environ, clean, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def parser(self):
        _, register = argument_source()
        parser = argparse.ArgumentParser()
        register(None, parser)
        return parser

    def test_complete_real_group_uses_standard_conflict_checks(self):
        parser = self.parser()
        self.assertEqual(parser.conflict_handler, "error")
        flags = [flag for action in parser._actions for flag in action.option_strings]
        self.assertTrue(flags)
        self.assertEqual(len(flags), len(set(flags)))
        self.assertEqual(flags.count("--diffusion-offload-config"), 1)
        self.assertFalse(parser.parse_args(["--no-record-output-peak-memory"]).record_output_peak_memory)

    def test_offload_keeps_json_object_and_environment(self):
        offload = {"mode": "module", "components": ["dit", "text_encoder"], "pin_memory": True}
        overrides = {"0": {"worker_extension_cls": "service.worker_extension.SRIdleCacheWorkerExtension"}}
        with patch.dict(os.environ, {
            "DYN_OMNI_DIFFUSION_OFFLOAD_CONFIG": json.dumps(offload),
            "DYN_OMNI_STAGE_OVERRIDES": json.dumps(overrides),
        }):
            parser = self.parser()
            parsed = parser.parse_args([])
        self.assertEqual(parsed.diffusion_offload_config, offload)
        self.assertEqual(parsed.stage_overrides, overrides)

    def test_empty_offload_is_disabled_for_cli_and_environment(self):
        self.assertIsNone(self.parser().parse_args(["--diffusion-offload-config", ""]).diffusion_offload_config)
        with patch.dict(os.environ, {"DYN_OMNI_DIFFUSION_OFFLOAD_CONFIG": ""}):
            self.assertIsNone(self.parser().parse_args([]).diffusion_offload_config)

    def test_non_object_and_malformed_cli_offload_are_rejected(self):
        for value in ("[]", "1", "null", "true", '"string"', "{broken"):
            with self.subTest(value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                self.parser().parse_args(["--diffusion-offload-config", value])
            self.assertEqual(error.exception.code, 2)

    def test_non_object_and_malformed_environment_offload_are_rejected(self):
        for value in ("[]", "1", "null", "true", '"string"', "{broken"):
            with self.subTest(value=value), patch.dict(os.environ, {"DYN_OMNI_DIFFUSION_OFFLOAD_CONFIG": value}):
                with self.assertRaises(argparse.ArgumentTypeError):
                    self.parser()

    def test_explicit_offload_overrides_environment(self):
        with patch.dict(os.environ, {"DYN_OMNI_DIFFUSION_OFFLOAD_CONFIG": '{"mode":"env"}'}):
            parser = self.parser()
            parsed = parser.parse_args(["--diffusion-offload-config", '{"mode":"module"}'])
        self.assertEqual(parsed.diffusion_offload_config, {"mode": "module"})

    def test_stage_overrides_retains_json_object_validation(self):
        parser = self.parser()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            parser.parse_args(["--stage-overrides", "[]"])
        self.assertEqual(error.exception.code, 2)

    def test_configuration_fields_are_declared_once(self):
        tree, _ = argument_source()
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                fields = [field.target.id for field in node.body if isinstance(field, ast.AnnAssign)
                          and isinstance(field.target, ast.Name)]
                duplicates = {name: count for name, count in Counter(fields).items() if count > 1}
                self.assertEqual(duplicates, {}, node.name)


if __name__ == "__main__":
    unittest.main()
