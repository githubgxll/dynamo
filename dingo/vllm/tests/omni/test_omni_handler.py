# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

try:
    from PIL import Image
    from vllm.sampling_params import SamplingParams
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    from dingo.common.protocols.audio_protocol import NvCreateAudioSpeechRequest
    from dingo.common.protocols.image_protocol import NvCreateImageRequest
    from dingo.common.protocols.video_protocol import NvCreateVideoRequest, VideoNvExt
    from dingo.common.utils.output_modalities import RequestType
    from dingo.vllm.omni.audio_handler import AudioGenerationHandler
    from dingo.vllm.omni.omni_handler import EngineInputs, OmniHandler
    from dingo.vllm.omni.utils import build_original_prompt, parse_omni_request
except ImportError:
    pytest.skip("vLLM omni dependencies not available", allow_module_level=True)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _make_handler(stage_types=("diffusion",)):
    with patch(
        "dingo.vllm.omni.omni_handler.BaseOmniHandler.__init__", return_value=None
    ):
        handler = OmniHandler.__new__(OmniHandler)

    config = MagicMock()
    config.model = "test-model"
    config.served_model_name = None
    config.output_modalities = ["text"]
    handler.config = config

    defaults = []
    for st in stage_types:
        if st == "diffusion":
            defaults.append(OmniDiffusionSamplingParams())
        else:
            llm_default = MagicMock(spec=SamplingParams)
            llm_default.clone.return_value = SamplingParams()
            defaults.append(llm_default)

    engine_client = MagicMock()
    engine_client.default_sampling_params_list = defaults
    engine_client.engine.get_stage_metadata.side_effect = lambda i: SimpleNamespace(
        stage_type=stage_types[i]
    )
    handler.engine_client = engine_client
    return handler


class TestEngineInputs:
    def test_defaults(self):
        """EngineInputs uses CHAT_COMPLETION, fps=0, and None optionals by default."""
        ei = EngineInputs(prompt={"prompt": "hello"})
        assert ei.request_type == RequestType.CHAT_COMPLETION
        assert ei.fps == 0
        assert ei.sampling_params_list is None
        assert ei.response_format is None
        assert ei.output_format is None


class TestRequestAdapterLifecycle:
    @pytest.mark.asyncio
    async def test_generate_wraps_stream_in_adapter_scope(self):
        handler = _make_handler()
        events = []

        class _Adapter:
            @asynccontextmanager
            async def request_scope(self, request_id, context=None):
                assert context is not None
                events.append(("enter", request_id))
                try:
                    yield "scope-token"
                finally:
                    events.append(("exit", request_id))

        handler.request_adapter = _Adapter()

        async def _generate(request, context, request_id, request_scope=None):
            assert request_scope == "scope-token"
            yield {"status": "completed"}

        handler._generate_openai_mode = _generate
        context = SimpleNamespace(id=lambda: "request-lifecycle")

        chunks = [chunk async for chunk in handler.generate({}, context)]

        assert chunks == [{"status": "completed"}]
        assert events == [
            ("enter", "request-lifecycle"),
            ("exit", "request-lifecycle"),
        ]

    @pytest.mark.asyncio
    async def test_adapter_scope_exits_when_consumer_closes_stream(self):
        handler = _make_handler()
        closed = asyncio.Event()

        class _Adapter:
            @asynccontextmanager
            async def request_scope(self, request_id, context=None):
                assert context is not None
                try:
                    yield object()
                finally:
                    closed.set()

        handler.request_adapter = _Adapter()

        async def _generate(request, context, request_id, request_scope=None):
            yield {"status": "in_progress"}
            await asyncio.Event().wait()

        handler._generate_openai_mode = _generate
        context = SimpleNamespace(id=lambda: "request-cancelled")
        stream = handler.generate({}, context)

        await anext(stream)
        await stream.aclose()

        assert closed.is_set()

    @pytest.mark.asyncio
    async def test_generate_without_adapter_keeps_noop_scope(self):
        handler = _make_handler()
        handler.request_adapter = None

        async def _generate(request, context, request_id, request_scope=None):
            assert request_scope is None
            yield {"status": "completed"}

        handler._generate_openai_mode = _generate
        context = SimpleNamespace(id=lambda: "ordinary-request")

        assert [chunk async for chunk in handler.generate({}, context)] == [
            {"status": "completed"}
        ]


class TestBuildEngineInputs:
    @pytest.mark.asyncio
    async def test_chat_completion(self):
        """Chat request extracts text prompt with no sampling params."""
        handler = _make_handler()
        raw = {"messages": [{"role": "user", "content": "hello"}]}
        inputs = await handler.build_engine_inputs(raw, RequestType.CHAT_COMPLETION)
        assert inputs.request_type == RequestType.CHAT_COMPLETION
        assert inputs.prompt["prompt"] == "hello"
        assert inputs.sampling_params_list is None

    @pytest.mark.asyncio
    async def test_image_generation(self):
        """Image request parses prompt, size, and creates diffusion sampling params."""
        handler = _make_handler()
        req = NvCreateImageRequest(prompt="a cat", size="512x512")
        inputs = await handler.build_engine_inputs(req, RequestType.IMAGE_GENERATION)
        assert inputs.request_type == RequestType.IMAGE_GENERATION
        assert inputs.prompt["prompt"] == "a cat"
        assert inputs.prompt["modalities"] == ["image"]
        assert inputs.prompt["mm_processor_kwargs"] == {
            "target_h": 512,
            "target_w": 512,
        }
        assert len(inputs.sampling_params_list) == 1
        sp = inputs.sampling_params_list[0]
        assert sp.height == 512
        assert sp.width == 512

    @pytest.mark.asyncio
    async def test_image_chat_completion_uses_multimodal_prompt(self):
        """Image chat requests must use vLLM-Omni multimodal preprocessing."""
        handler = _make_handler(stage_types=("llm", "diffusion"))
        handler.config.output_modalities = ["image"]
        raw = {
            "messages": [{"role": "user", "content": "a glass teapot"}],
            "extra_body": {"height": 768, "width": 512, "seed": 123},
        }

        inputs = await handler.build_engine_inputs(raw, RequestType.CHAT_COMPLETION)

        assert inputs.request_type == RequestType.CHAT_COMPLETION
        assert inputs.prompt["prompt"] == "a glass teapot"
        assert inputs.prompt["modalities"] == ["image"]
        assert inputs.prompt["mm_processor_kwargs"] == {
            "target_h": 768,
            "target_w": 512,
        }
        assert len(inputs.sampling_params_list) == 2
        sp = inputs.sampling_params_list[1]
        assert sp.height == 768
        assert sp.width == 512

    @pytest.mark.asyncio
    async def test_video_generation(self):
        """Video request preserves output delivery fields for the formatter."""
        handler = _make_handler()
        req = NvCreateVideoRequest(
            prompt="a drone",
            model="test",
            size="832x480",
            seconds=2,
            response_format="b64_json",
            output_format="mp4",
        )
        inputs = await handler.build_engine_inputs(req, RequestType.VIDEO_GENERATION)
        assert inputs.request_type == RequestType.VIDEO_GENERATION
        assert inputs.prompt["prompt"] == "a drone"
        assert inputs.fps > 0
        assert inputs.response_format == "b64_json"
        assert inputs.output_format == "mp4"

    @pytest.mark.asyncio
    async def test_audio_generation_delegates_toaudio(self):
        """Audio request delegates to audio."""
        handler = _make_handler()
        expected = EngineInputs(
            prompt={"prompt": "Hello world"},
            request_type=RequestType.AUDIO_GENERATION,
        )

        async def mock_engine_inputs(req):
            return expected

        handler.audio = MagicMock()
        handler.audio.build_engine_inputs = mock_engine_inputs
        inputs = await handler.build_engine_inputs(
            NvCreateAudioSpeechRequest(input="Hello world"),
            RequestType.AUDIO_GENERATION,
        )
        assert inputs.request_type == RequestType.AUDIO_GENERATION
        assert inputs.prompt["prompt"] == "Hello world"


class TestI2VEngineInputs:
    """Tests for image-to-video: multi_modal_data attachment, I2V nvext params, and protocol fields."""

    @pytest.mark.asyncio
    async def test_t2v_no_multi_modal_data_and_i2v_attaches_image(self):
        """T2V has no multi_modal_data; I2V attaches image to prompt."""
        handler = _make_handler()
        req = NvCreateVideoRequest(
            prompt="a drone", model="test", size="832x480", seconds=2
        )

        # T2V: no image
        t2v = await handler.build_engine_inputs(req, RequestType.VIDEO_GENERATION)
        assert "multi_modal_data" not in t2v.prompt

        # I2V: image attached
        img = Image.new("RGB", (64, 64), color="red")
        i2v = await handler.build_engine_inputs(
            req, RequestType.VIDEO_GENERATION, image=img
        )
        assert i2v.prompt["multi_modal_data"]["image"] is img

    @pytest.mark.asyncio
    async def test_i2v_nvext_params_on_sampling_params(self):
        """boundary_ratio and guidance_scale_2 are forwarded to sampling params."""
        handler = _make_handler()
        req = NvCreateVideoRequest(
            prompt="bear",
            model="test",
            size="832x480",
            nvext=VideoNvExt(
                boundary_ratio=0.875, guidance_scale_2=1.0, num_inference_steps=40
            ),
        )
        result = await handler.build_engine_inputs(req, RequestType.VIDEO_GENERATION)
        sp = result.sampling_params_list[0]
        assert sp.boundary_ratio == 0.875
        assert sp.guidance_scale_2 == 1.0
        assert sp.num_inference_steps == 40

    def test_i2v_protocol_roundtrip(self):
        """VideoNvExt and NvCreateVideoRequest serialize/deserialize I2V fields correctly."""
        req = NvCreateVideoRequest(
            prompt="bear playing",
            model="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            input_reference="/tmp/bear.png",
            size="832x480",
            nvext=VideoNvExt(boundary_ratio=0.9, guidance_scale_2=2.0, seed=42),
        )
        data = req.model_dump()
        assert data["input_reference"] == "/tmp/bear.png"
        assert data["nvext"]["boundary_ratio"] == 0.9
        assert data["nvext"]["guidance_scale_2"] == 2.0

        # Defaults are None
        empty = VideoNvExt()
        assert empty.boundary_ratio is None
        assert empty.guidance_scale_2 is None

    @pytest.mark.asyncio
    async def test_aspect_ratio_forwarded_to_sampling_params(self):
        """B1: nvext.aspect_ratio is forwarded to extra_args.aspect_ratio."""
        handler = _make_handler()
        req = NvCreateVideoRequest(
            prompt="a drone",
            model="test",
            size="1344x768",
            seconds=2,
            nvext=VideoNvExt(aspect_ratio="16:9"),
        )
        inputs = await handler.build_engine_inputs(req, RequestType.VIDEO_GENERATION)
        sp = inputs.sampling_params_list[0]
        assert sp.extra_args["aspect_ratio"] == "16:9"

    @pytest.mark.asyncio
    async def test_frame_indices_forwarded_to_sampling_params(self):
        """B4: nvext.frame_indices is forwarded to extra_args.frame_indices."""
        handler = _make_handler()
        req = NvCreateVideoRequest(
            prompt="a drone",
            model="test",
            size="1344x768",
            seconds=2,
            nvext=VideoNvExt(frame_indices=[0, -1]),
        )
        inputs = await handler.build_engine_inputs(req, RequestType.VIDEO_GENERATION)
        sp = inputs.sampling_params_list[0]
        assert sp.extra_args["frame_indices"] == [0, -1]

    @pytest.mark.asyncio
    async def test_input_references_attaches_image_list(self):
        """B4: list of images is attached to multi_modal_data.image in order."""
        handler = _make_handler()
        first = Image.new("RGB", (64, 64), color="red")
        last = Image.new("RGB", (64, 64), color="blue")
        inputs = await handler.build_engine_inputs(
            NvCreateVideoRequest(
                prompt="a drone",
                model="test",
                size="1344x768",
                seconds=2,
            ),
            RequestType.VIDEO_GENERATION,
            image=[first, last],
        )
        attached = inputs.prompt["multi_modal_data"]["image"]
        assert attached == [first, last]

    @pytest.mark.asyncio
    async def test_input_reference_and_input_references_mutually_exclusive(self):
        """B4: setting both input_reference and input_references raises."""
        handler = _make_handler()
        req = NvCreateVideoRequest(
            prompt="a drone",
            model="test",
            size="1344x768",
            seconds=2,
            input_reference="data:image/png;base64,AAA",
            input_references=["data:image/png;base64,BBB"],
        )
        with pytest.raises(ValueError, match="mutually exclusive"):
            await handler.build_engine_inputs(req, RequestType.VIDEO_GENERATION)

    def test_video_protocol_aspect_ratio_and_frame_indices_roundtrip(self):
        """B1+B4: VideoNvExt and NvCreateVideoRequest carry the new fields."""
        req = NvCreateVideoRequest(
            prompt="bear",
            model="test",
            size="1344x768",
            input_references=["data:image/png;base64,AAA", "data:image/png;base64,BBB"],
            nvext=VideoNvExt(aspect_ratio="16:9", frame_indices=[0, -1]),
        )
        data = req.model_dump()
        assert data["input_references"] == [
            "data:image/png;base64,AAA",
            "data:image/png;base64,BBB",
        ]
        assert data["nvext"]["aspect_ratio"] == "16:9"
        assert data["nvext"]["frame_indices"] == [0, -1]


class TestBuildSamplingParamsList:
    def test_single_diffusion_stage(self):
        handler = _make_handler(stage_types=("diffusion",))
        sp = OmniDiffusionSamplingParams(height=512, width=512)
        result = handler._build_sampling_params_list(sp)
        assert len(result) == 1
        assert result[0] is sp

    def test_llm_then_diffusion(self):
        handler = _make_handler(stage_types=("llm", "diffusion"))
        sp = OmniDiffusionSamplingParams(height=512, width=512)
        result = handler._build_sampling_params_list(sp)
        assert len(result) == 2
        assert isinstance(result[0], SamplingParams)
        assert result[1] is sp

    def test_fallback_when_defaults_empty(self):
        handler = _make_handler()
        handler.engine_client.default_sampling_params_list = []
        sp = OmniDiffusionSamplingParams(height=512, width=512)
        result = handler._build_sampling_params_list(sp)
        assert result == [sp]

    def test_llm_default_is_cloned(self):
        handler = _make_handler(stage_types=("llm", "diffusion"))
        sp = OmniDiffusionSamplingParams()
        handler._build_sampling_params_list(sp)
        handler.engine_client.default_sampling_params_list[0].clone.assert_called_once()


class TestBuildOriginalPrompt:
    """build_original_prompt only carries prompt/negative_prompt/multi_modal_data.

    height/width/num_inference_steps live in OmniDiffusionSamplingParams, not the prompt.
    """

    def test_basic_fields(self):
        result = build_original_prompt(
            {"prompt": "a cat"}, nvext={}, height=512, width=512
        )
        assert result["prompt"] == "a cat"
        assert result.get("negative_prompt") is None
        assert "height" not in result
        assert "width" not in result

    def test_negative_prompt_from_request(self):
        result = build_original_prompt(
            {"prompt": "a cat", "negative_prompt": "blurry"},
            nvext={"negative_prompt": "ignored"},
            height=1024,
            width=1024,
        )
        assert result["negative_prompt"] == "blurry"

    def test_multi_modal_data_forwarded(self):
        img = object()
        result = build_original_prompt(
            {"prompt": "x", "multi_modal_data": {"image": img}},
            nvext={},
            height=512,
            width=512,
        )
        assert result["multi_modal_data"]["image"] is img

    def test_no_inference_steps_or_guidance(self):
        result = build_original_prompt(
            {"prompt": "x"},
            nvext={"num_inference_steps": 50, "guidance_scale": 7.5},
            height=512,
            width=512,
        )
        assert "num_inference_steps" not in result
        assert "guidance_scale" not in result


class TestParseOmniRequest:
    """parse_omni_request: image geometry goes into sampling params and processor kwargs."""

    @pytest.mark.asyncio
    async def test_image_sampling_params_has_geometry(self):
        request = {
            "prompt": "a sunset",
            "size": "512x512",
            "output_modalities": ["image"],
        }
        result = await parse_omni_request(request, ["image"])
        sp = result["sampling_params_list"]
        assert sp["height"] == 512
        assert sp["width"] == 512

    @pytest.mark.asyncio
    async def test_image_prompt_uses_multimodal_preprocessor_kwargs(self):
        request = {
            "prompt": "a sunset",
            "size": "512x512",
            "output_modalities": ["image"],
        }
        result = await parse_omni_request(request, ["image"])
        prompt = result["engine_inputs"]
        assert prompt["prompt"] == "a sunset"
        assert prompt["modalities"] == ["image"]
        assert prompt["mm_processor_kwargs"] == {"target_h": 512, "target_w": 512}

        op = result["original_prompt"]
        assert op["prompt"] == "a sunset"
        assert "height" not in op
        assert "width" not in op
        assert op["modalities"] == ["image"]
        assert op["mm_processor_kwargs"] == {"target_h": 512, "target_w": 512}

    def test_image_request_uses_nvext_negative_prompt(self):
        request = {
            "prompt": "a red apple",
            "size": "1024x1024",
            "nvext": {"negative_prompt": "blurry, low quality"},
        }

        result = asyncio.run(parse_omni_request(request, ["image"]))

        assert result["engine_inputs"]["negative_prompt"] == "blurry, low quality"
        assert result["original_prompt"]["negative_prompt"] == "blurry, low quality"

    def test_image_request_uses_nvext_dimensions_consistently(self):
        request = {
            "prompt": "a red apple",
            "size": "512x512",
            "nvext": {"height": 640, "width": 768},
        }

        result = asyncio.run(parse_omni_request(request, ["image"]))

        assert result["sampling_params_list"]["height"] == 640
        assert result["sampling_params_list"]["width"] == 768
        assert result["engine_inputs"]["mm_processor_kwargs"] == {
            "target_h": 640,
            "target_w": 768,
        }
        assert result["original_prompt"]["mm_processor_kwargs"] == {
            "target_h": 640,
            "target_w": 768,
        }

    @pytest.mark.asyncio
    async def test_nvext_params_go_into_sampling_params_not_prompt(self):
        request = {
            "prompt": "x",
            "size": "512x512",
            "nvext": {"num_inference_steps": 30, "guidance_scale": 4.0},
        }
        result = await parse_omni_request(request, ["image"])
        sp = result["sampling_params_list"]
        assert sp["num_inference_steps"] == 30
        assert sp["guidance_scale"] == 4.0
        op = result["original_prompt"]
        assert "num_inference_steps" not in op
        assert "guidance_scale" not in op

    @pytest.mark.asyncio
    async def test_image_chat_request_uses_multimodal_preprocessor_kwargs(self):
        request = {
            "messages": [{"role": "user", "content": "a glass teapot"}],
            "extra_body": {"height": 768, "width": 512, "guidance_scale": 1.5},
        }

        result = await parse_omni_request(request, ["image"])

        prompt = result["engine_inputs"]
        assert prompt["prompt"] == "a glass teapot"
        assert prompt["modalities"] == ["image"]
        assert prompt["mm_processor_kwargs"] == {"target_h": 768, "target_w": 512}
        assert result["original_prompt"] == prompt
        assert result["sampling_params_list"] == {
            "height": 768,
            "width": 512,
            "guidance_scale": 1.5,
        }


# ---------------------------------------------------------------------------
# AudioGenerationHandler — data_source / response_format field mapping
# ---------------------------------------------------------------------------


def _make_audio_handler():
    config = MagicMock()
    config.tts_max_instructions_length = 200
    config.tts_max_new_tokens_min = 1
    config.tts_max_new_tokens_max = 4096
    config.tts_ref_audio_timeout = 10
    config.tts_ref_audio_max_bytes = 1024 * 1024
    engine_client = MagicMock()
    engine_client.model_config.hf_config.talker_config = None
    return AudioGenerationHandler(config, engine_client, None, None)


class TestAudioHandlerFieldMapping:
    """AudioGenerationHandler maps data_source→response_format and response_format→output_format."""

    @pytest.mark.asyncio
    async def test_generic_path_maps_data_source_to_response_format(self):
        handler = _make_audio_handler()
        handler._is_tts_model = MagicMock(return_value=False)

        req = NvCreateAudioSpeechRequest(
            input="hello", data_source="url", response_format="mp3"
        )
        result = await handler.build_engine_inputs(req)

        assert result.response_format == "url"  # data_source → response_format
        assert result.output_format == "mp3"  # response_format → output_format

    @pytest.mark.asyncio
    async def test_generic_path_maps_data_source_b64_json(self):
        handler = _make_audio_handler()
        handler._is_tts_model = MagicMock(return_value=False)

        req = NvCreateAudioSpeechRequest(
            input="hello", data_source="b64_json", response_format="opus"
        )
        result = await handler.build_engine_inputs(req)

        assert result.response_format == "b64_json"
        assert result.output_format == "opus"

    @pytest.mark.asyncio
    async def test_generic_path_no_data_source_passes_none(self):
        handler = _make_audio_handler()
        handler._is_tts_model = MagicMock(return_value=False)

        # No data_source → response_format in EngineInputs will be None
        req = NvCreateAudioSpeechRequest(input="hello", response_format="wav")
        result = await handler.build_engine_inputs(req)

        assert result.response_format is None
        assert result.output_format == "wav"

    @pytest.mark.asyncio
    async def test_tts_path_applies_same_field_mapping(self):
        handler = _make_audio_handler()
        handler._is_tts_model = MagicMock(return_value=True)
        handler._validate_tts_request = MagicMock()
        handler._estimate_tts_prompt_len = MagicMock(return_value=10)

        req = NvCreateAudioSpeechRequest(
            input="hi", data_source="url", response_format="flac"
        )
        result = await handler.build_engine_inputs(req)

        assert result.response_format == "url"
        assert result.output_format == "flac"

    @pytest.mark.asyncio
    async def test_request_type_is_audio_generation(self):
        handler = _make_audio_handler()
        handler._is_tts_model = MagicMock(return_value=False)

        result = await handler.build_engine_inputs(
            NvCreateAudioSpeechRequest(input="hi")
        )
        assert result.request_type == RequestType.AUDIO_GENERATION
