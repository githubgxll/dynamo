"""Execute gateway compatibility paths with CPU-only, lightweight API fixtures.

Selected real methods are compiled without their GPU/native imports. Installed
module imports are separately checked by service_smoke.py inside the image.
"""

from __future__ import annotations

import ast
import asyncio
from contextlib import asynccontextmanager
import importlib.util
import logging
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[3]
OMNI = ROOT / "dingo/vllm/omni"


def definitions(filename, names, namespace=None, class_name=None):
    tree = ast.parse((OMNI / filename).read_text(encoding="utf-8"))
    body = tree.body
    if class_name:
        body = next(node.body for node in body if isinstance(node, ast.ClassDef) and node.name == class_name)
    selected = [node for node in body if getattr(node, "name", None) in names]
    if len(selected) != len(names):
        raise AssertionError(f"Missing source definitions: {names}")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + selected, type_ignores=[])
    scope = namespace if namespace is not None else {}
    exec(compile(ast.fix_missing_locations(module), str(OMNI / filename), "exec"), scope)
    return scope


def formatter_namespace():
    return definitions("output_formatter.py", {
        "get_text_request_output", "TextFormatter", "OutputFormatter", "_error_chunk", "_build_completion_usage"
    }, {"time": time, "normalize_finish_reason": lambda value: value})


def formatter(scope):
    result = scope["OutputFormatter"].__new__(scope["OutputFormatter"])
    result._formatters = {"text": scope["TextFormatter"]("h3")}
    return result


def text_output(text, *, legacy, finished=False):
    content = NS(outputs=[NS(text=text, token_ids=list(range(len(text))), finish_reason="stop" if finished else None)],
                 prompt_token_ids=[1, 2], num_cached_tokens=1)
    if legacy:
        return NS(final_output_type="text", request_output=content)
    content.final_output_type = "text"
    return content


class OutputCompatibilityTests(unittest.TestCase):
    def test_flat_and_nested_outputs_keep_delta_and_usage(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                scope = formatter_namespace()
                result = asyncio.run(formatter(scope).format(text_output("hello", legacy=legacy, finished=True), "r1", previous_text="hel"))
                self.assertEqual(result["choices"][0]["delta"]["content"], "lo")
                self.assertEqual(result["usage"], {"prompt_tokens": 2, "completion_tokens": 5, "total_tokens": 7, "prompt_tokens_details": {"cached_tokens": 1}})

    def test_empty_flat_output_returns_existing_error_response(self):
        scope = formatter_namespace()
        result = asyncio.run(formatter(scope).format(NS(final_output_type="text", outputs=[]), "r1"))
        self.assertEqual(result["choices"][0]["finish_reason"], "error")

    def test_legacy_content_is_not_displaced_by_outer_attributes(self):
        scope = formatter_namespace()
        nested = NS(outputs=["real completion"])
        outer = NS(request_output=nested, outputs=[])
        self.assertIs(scope["get_text_request_output"](outer), nested)

    def test_streaming_handler_updates_previous_text_for_both_layouts(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                scope = formatter_namespace()
                request_types = NS(CHAT_COMPLETION="chat", VIDEO_GENERATION="video")
                scope.update({"cast": lambda kind, value: value,
                              "NvCreateImageRequest": type("ImageRequest", (), {}),
                              "NvCreateVideoRequest": type("VideoRequest", (), {}),
                              "Dict": dict, "Any": object, "RequestType": request_types,
                              "parse_request_type": lambda request, modalities: (request, "chat"),
                              "EngineShutdown": type("EngineShutdown", (Exception,), {}),
                              "logger": logging.getLogger("test")})
                # typing.Union is evaluated at runtime inside the real method.
                from typing import Union, Dict, Any
                scope.update(Union=Union, Dict=Dict, Any=Any)
                definitions("omni_handler.py", {"_generate_openai_mode"}, scope, "OmniHandler")

                @asynccontextmanager
                async def monitor(*args):
                    yield

                async def inputs(*args, **kwargs):
                    return NS(prompt={"prompt": "x"}, sampling_params_list=None, request_type="chat", fps=0,
                              response_format=None, output_format=None, speed=1.0)

                async def generate(**kwargs):
                    yield text_output("hel", legacy=legacy)
                    yield text_output("hello", legacy=legacy, finished=True)

                handler = NS(config=NS(output_modalities=[]), request_adapter=None,
                             build_engine_inputs=inputs, engine_client=NS(generate=generate),
                             output_formatter=formatter(scope), _abort_monitor=monitor,
                             _error_chunk=lambda *args: {"unexpected_error": args})

                async def consume():
                    return [chunk async for chunk in scope["_generate_openai_mode"](handler, {}, None, "r1")]

                chunks = asyncio.run(consume())
                self.assertEqual([item["choices"][0]["delta"]["content"] for item in chunks], ["hel", "lo"])
                self.assertEqual(chunks[-1]["usage"]["total_tokens"], 7)


class VideoPromptTests(unittest.TestCase):
    def test_negative_prompt_is_mapping_field_and_preserves_parameters(self):
        from typing import TypedDict

        class Prompt(TypedDict):
            prompt: str
            negative_prompt: str

        scope = {"parse_size": lambda value: (768, 1280), "DEFAULT_VIDEO_FPS": 16,
                 "VideoNvExt": NS, "compute_num_frames": lambda **kwargs: 81,
                 "OmniTextPrompt": Prompt, "OmniDiffusionSamplingParams": NS,
                 "EngineInputs": NS, "RequestType": NS(VIDEO_GENERATION="video"),
                 "logger": logging.getLogger("test")}
        definitions("omni_handler.py", {"_engine_inputs_from_video"}, scope, "OmniHandler")
        ext = NS(negative_prompt="blur", num_frames=81, fps=16, num_inference_steps=8,
                 guidance_scale=1.0, seed=42, boundary_ratio=None, guidance_scale_2=None,
                 aspect_ratio="9:16", frame_indices=[0, 80])
        req = NS(prompt="test", nvext=ext, size="768x1280", seconds=5,
                 input_reference=None, input_references=None, response_format="url", output_format="mp4")
        handler = NS(_update_if_not_none=lambda obj, name, value: setattr(obj, name, value) if value is not None else None,
                     _build_sampling_params_list=lambda params: [params])
        result = scope["_engine_inputs_from_video"](handler, req)
        self.assertEqual(result.prompt, {"prompt": "test", "negative_prompt": "blur"})
        self.assertEqual(result.sampling_params_list[0].num_inference_steps, 8)
        self.assertEqual(result.sampling_params_list[0].extra_args, {"aspect_ratio": "9:16", "frame_indices": [0, 80]})


def load_stage_compat(legacy=None):
    utils = ModuleType("vllm_omni.entrypoints.utils")
    entrypoints = ModuleType("vllm_omni.entrypoints")
    entrypoints.utils = utils
    engine = ModuleType("vllm_omni.engine.omni_engine_base")
    if legacy is not None:
        utils.load_and_resolve_stage_configs = legacy

    def prepare(model, kwargs, *, trust_remote_code, snapshot_model=False):
        normalized = dict(kwargs)
        if "stage_configs_path" in normalized:
            raise ValueError("removed upstream argument")
        return NS(model=model, kwargs=normalized,
                  deploy_config_path=normalized.pop("deploy_config", None),
                  trust_remote_code=trust_remote_code, stage_overrides=None, strategy_config_path=None)

    def resolve(model, kwargs, *, trust_remote_code, deploy_config_path, stage_overrides, strategy_config_path):
        return NS(config_path=deploy_config_path, stage_configs=("stage0",))

    utils.prepare_stage_config_inputs = Mock(side_effect=prepare)
    engine.load_and_resolve_stage_configs = Mock(side_effect=resolve)
    modules = {"vllm_omni": ModuleType("vllm_omni"), "vllm_omni.entrypoints": entrypoints,
               "vllm_omni.entrypoints.utils": utils, "vllm_omni.engine": ModuleType("vllm_omni.engine"),
               "vllm_omni.engine.omni_engine_base": engine}
    spec = importlib.util.spec_from_file_location("stage_config_under_test", OMNI / "stage_config_compat.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module, utils, engine


class StageConfigCompatibilityTests(unittest.TestCase):
    def test_current_signature_and_typed_resolution_are_bridged(self):
        module, utils, engine = load_stage_compat()
        kwargs = {"trust_remote_code": False}
        self.assertEqual(module.load_and_resolve_stage_configs("h3", "new-deploy.yaml", kwargs=kwargs), ("new-deploy.yaml", ["stage0"]))
        self.assertEqual(kwargs, {"trust_remote_code": False})
        engine.load_and_resolve_stage_configs.assert_called_once_with(
            "h3", {"trust_remote_code": False}, trust_remote_code=False,
            deploy_config_path="new-deploy.yaml", stage_overrides=None, strategy_config_path=None)

    def test_legacy_signature_and_result_are_preserved(self):
        legacy = Mock(return_value=("old.yaml", ["stage0"]))
        module, utils, engine = load_stage_compat(legacy)
        self.assertEqual(module.stage_config_engine_kwargs("old.yaml"), {"stage_configs_path": "old.yaml"})
        self.assertEqual(module.load_and_resolve_stage_configs("h3", "old.yaml", kwargs={}), ("old.yaml", ["stage0"]))
        legacy.assert_called_once_with("h3", "old.yaml", kwargs={})
        engine.load_and_resolve_stage_configs.assert_not_called()

    def test_unsupported_old_yaml_fails_before_current_loader(self):
        module, utils, engine = load_stage_compat()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "legacy.yaml"
            path.write_text("stage_args:\n  - stage_id: 0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "removed the stage_args YAML schema"):
                module.load_and_resolve_stage_configs("h3", str(path), kwargs={})
        engine.load_and_resolve_stage_configs.assert_not_called()

    def test_native_deploy_yaml_passes_and_no_config_stays_empty(self):
        module, _, _ = load_stage_compat()
        self.assertEqual(module.stage_config_engine_kwargs(None), {})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "deploy.yaml"
            path.write_text("stages:\n  - stage_id: 0\n    devices: '0'\n", encoding="utf-8")
            self.assertEqual(module.stage_config_engine_kwargs(str(path)), {"deploy_config": str(path)})


if __name__ == "__main__":
    unittest.main()
