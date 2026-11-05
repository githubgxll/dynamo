"""Exact-source request peak-memory policy and public availability contract.

The production profiles disable observation explicitly. Upstream callers keep
the historical True default. No model, precision, allocation or profile budget
algorithm is changed by this patcher.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHANGES = (
    "vllm_omni/diffusion/data.py",
    "vllm_omni/config/omni_config.py",
    "vllm_omni/config/stage_config.py",
    "vllm_omni/engine/arg_utils.py",
    "vllm_omni/entrypoints/cli/serve.py",
    "vllm_omni/diffusion/worker/diffusion_model_runner.py",
    "vllm_omni/entrypoints/openai/api_server.py",
    "vllm_omni/entrypoints/openai/protocol/videos.py",
    "vllm_omni/entrypoints/openai/video/generation/helpers.py",
)

POLICY_HELPER = '''def _should_record_output_peak_memory(od_config, requested=True):
    """Keep capacity profiling's explicit False independent of serving policy."""
    policy = getattr(od_config, "record_output_peak_memory", True)
    if type(policy) is not bool or type(requested) is not bool:
        raise TypeError("record_output_peak_memory must be a boolean")
    return requested and policy


'''


def _observation_helper(name):
    return f'''def {name}(value):
    """Expose no numeric sample for disabled/unavailable internal sentinels."""
    if value is None:
        return {{"peak_memory_mb": None, "peak_memory_status": "not_collected"}}
    if isinstance(value, bool):
        return {{"peak_memory_mb": None, "peak_memory_status": "unavailable"}}
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return {{"peak_memory_mb": None, "peak_memory_status": "unavailable"}}
    if not math.isfinite(numeric) or numeric == 0:
        return {{"peak_memory_mb": None, "peak_memory_status": "unavailable"}}
    if numeric < 0:
        return {{"peak_memory_mb": None, "peak_memory_status": "not_collected"}}
    return {{"peak_memory_mb": numeric, "peak_memory_status": "collected"}}


'''


def _once(source, before, after, count=1):
    if source.count(before) != count:
        raise ValueError("Peak patch source anchor count differs: " + before[:100])
    return source.replace(before, after)


def planned_patch(relative, data):
    """Pure reviewed transformation; callers must enforce the before/after lock."""
    source = data.decode("utf-8")
    if relative == CHANGES[0]:
        marker = "    enable_diffusion_pipeline_profiler: bool = False\n"
        source = _once(source, marker, marker + "    # Serving request observation only; profile_run keeps its own budget context.\n    record_output_peak_memory: bool = True\n")
    elif relative == CHANGES[1]:
        marker = "    enable_diffusion_pipeline_profiler: bool = False\n"
        source = _once(source, marker, marker + "    record_output_peak_memory: bool = True\n")
    elif relative == CHANGES[2]:
        marker = "    enable_diffusion_pipeline_profiler: bool | None = None\n"
        source = _once(source, marker, marker + "    record_output_peak_memory: bool | None = None\n")
    elif relative == CHANGES[3]:
        for annotation in ("bool | None = None", "bool = False"):
            marker = "    enable_diffusion_pipeline_profiler: " + annotation + "\n"
            value = "bool | None = None" if "None" in annotation else "bool = True"
            source = _once(source, marker, marker + "    record_output_peak_memory: " + value + "\n")
        marker = '        "enable_diffusion_pipeline_profiler",\n'
        source = _once(source, marker, marker + '        "record_output_peak_memory",\n')
    elif relative == CHANGES[4]:
        marker = "        # Enable diffusion pipeline profiling\n"
        source = _once(source, marker, '''        omni_config_group.add_argument(
            "--record-output-peak-memory",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Record serving-request peak memory. Disable with --no-record-output-peak-memory; "
                 "does not disable profile_run capacity measurement.",
        )

''' + marker)
    elif relative == CHANGES[5]:
        source = _once(source, "class DiffusionModelRunner(DiffusionStagePayloadMixin):\n",
                       POLICY_HELPER + "class DiffusionModelRunner(DiffusionStagePayloadMixin):\n")
        marker = '''        assert self.pipeline is not None, "Model not loaded. Call load_model() first."
        if not reqs:
'''
        source = _once(source, marker, '''        assert self.pipeline is not None, "Model not loaded. Call load_model() first."
        record_output_peak_memory = _should_record_output_peak_memory(od_config, record_output_peak_memory)
        if not reqs:
''')
        marker = '''                for output in outputs:
                    output.peak_memory_mb = max(output.peak_memory_mb, batch_peak_memory_mb)
'''
        source = _once(source, marker, marker + '''            elif not record_output_peak_memory:
                # Preserve the internal float carrier while distinguishing disabled
                # observation from legacy zero/unavailable values at the API boundary.
                for output in outputs:
                    output.peak_memory_mb = -1.0
''')
        source = _once(source, "            record_output_peak_memory=True,\n",
                       "            record_output_peak_memory=_should_record_output_peak_memory(self.od_config),\n")
        marker = '''        """Run the denoise step after metadata admission and row installation."""

        use_hsdp = self.od_config.parallel_config.use_hsdp
'''
        source = _once(source, marker, '''        """Run the denoise step after metadata admission and row installation."""

        record_output_peak_memory = _should_record_output_peak_memory(self.od_config, record_output_peak_memory)
        use_hsdp = self.od_config.parallel_config.use_hsdp
''')
        marker = '''                terminal_request_ids = [
                    runner_output.request_id for runner_output in runner_output_list if runner_output.finished
                ]
'''
        source = _once(source, marker, '''                if not record_output_peak_memory:
                    for state in states:
                        state.peak_memory_mb = -1.0
                    for runner_output in runner_output_list:
                        if runner_output.result is not None:
                            runner_output.result.peak_memory_mb = -1.0

''' + marker)
    elif relative == CHANGES[6]:
        source = _once(source, "import json\n", "import json\nimport math\n")
        marker = "def _build_image_response_metrics(\n"
        headers = '''def _output_peak_memory_headers(value):
    observation = _peak_memory_observation(value)
    headers = {"X-Peak-Memory-Status": observation["peak_memory_status"]}
    if observation["peak_memory_mb"] is not None:
        headers["X-Peak-Memory-MB"] = f"{observation['peak_memory_mb']:.3f}"
    return headers


'''
        source = _once(source, marker, _observation_helper("_peak_memory_observation") + headers + marker)
        source = _once(source, '    metrics["peak_memory_mb"] = float(peak_memory_mb) if peak_memory_mb else None\n',
                       '    metrics.update(_peak_memory_observation(peak_memory_mb))\n')
        source = _once(source, '            "X-Peak-Memory-MB": f"{peak_memory_mb:.3f}",\n',
                       '            **_output_peak_memory_headers(peak_memory_mb),\n')
    elif relative == CHANGES[7]:
        source = _once(source, "import mimetypes\n", "import math\nimport mimetypes\n")
        marker = "class VideoGenerationResponse(BaseModel):\n"
        mixin = _observation_helper("_protocol_peak_memory_observation") + '''class _PeakMemoryAvailability(BaseModel):
    peak_memory_status: Literal["collected", "not_collected", "unavailable"] = "unavailable"

    @model_validator(mode="before")
    @classmethod
    def _normalize_peak_memory(cls, value):
        if isinstance(value, dict) and "peak_memory_mb" in value:
            value = dict(value)
            # Stored null+availability is already normalized. A GET/Pydantic
            # reconstruction must not turn an unavailable legacy sample into
            # an intentionally disabled measurement.
            if (value["peak_memory_mb"] is None
                    and value.get("peak_memory_status") in ("not_collected", "unavailable")):
                return value
            value.update(_protocol_peak_memory_observation(value["peak_memory_mb"]))
        return value


'''
        source = _once(source, marker, mixin + "class VideoGenerationResponse(_PeakMemoryAvailability):\n")
        source = _once(source, "class VideoResponse(BaseModel):\n", "class VideoResponse(_PeakMemoryAvailability):\n")
        source = _once(source, "    peak_memory_mb: float = Field(\n", "    peak_memory_mb: float | None = Field(\n", count=2)
    elif relative == CHANGES[8]:
        source = _once(source, "import json\n", "import json\nimport math\n")
        marker = "def _unpack_video_generation_result(\n"
        source = _once(source, marker, _observation_helper("_video_peak_memory_observation") + marker)
        marker = '''        updated_fields = {
            "status": VideoGenerationStatus.COMPLETED,
'''
        source = _once(source, marker, '''        peak_observation = _video_peak_memory_observation(peak_memory_mb)
        updated_fields = {
            "status": VideoGenerationStatus.COMPLETED,
''')
        source = _once(source, '            "peak_memory_mb": peak_memory_mb,\n',
                       '            **peak_observation,\n')
    else:
        raise ValueError("Unreviewed peak patch target: " + relative)
    output = source.encode("utf-8")
    compile(output, relative, "exec")
    return output


def policy():
    lock = json.loads((HERE / "peak-patches.lock.json").read_text(encoding="utf-8"))
    if lock.get("schema") != "h3-sr-production-peak-patches/1" or set(lock.get("changes", {})) != set(CHANGES):
        raise ValueError("Peak patch scope/lock differs")
    return lock


def transform(relative: str, data: bytes) -> bytes:
    entry = policy()["changes"].get(relative)
    if entry is None or hashlib.sha256(data).hexdigest() != entry["before_sha256"]:
        raise ValueError("Peak input differs from exact published c88 source: " + relative)
    output = planned_patch(relative, data)
    if hashlib.sha256(output).hexdigest() != entry["after_sha256"]:
        raise ValueError("Peak output differs from reviewed source lock: " + relative)
    return output


def validate_installed_runtime():
    """CPU validation against the actual installed API/Pydantic classes."""
    from vllm_omni.entrypoints.openai.api_server import _output_peak_memory_headers, _build_image_response_metrics
    from vllm_omni.entrypoints.openai.protocol.videos import VideoResponse, VideoGenerationResponse
    from vllm_omni.entrypoints.openai.video.generation.helpers import _video_peak_memory_observation

    checks = []
    for value, expected_status, expected_peak in ((-1.0, "not_collected", None), (0.0, "unavailable", None),
                                                   (123.5, "collected", 123.5)):
        headers = _output_peak_memory_headers(value)
        if headers.get("X-Peak-Memory-Status") != expected_status or ("X-Peak-Memory-MB" in headers) != (expected_peak is not None):
            raise ValueError("Installed sync API peak availability differs")
        metrics = _build_image_response_metrics(response_metrics={}, stage_durations={}, peak_memory_mb=value)
        if metrics.get("peak_memory_mb") != expected_peak or metrics.get("peak_memory_status") != expected_status:
            raise ValueError("Installed image API peak availability differs")
        objects = [VideoResponse(model="cpu-availability-check", prompt="no generation", peak_memory_mb=value),
                   VideoGenerationResponse(created=0, data=[], peak_memory_mb=value)]
        for item in objects:
            encoded = item.model_dump(mode="json")
            if encoded.get("peak_memory_mb") != expected_peak or encoded.get("peak_memory_status") != expected_status:
                raise ValueError("Installed Pydantic video peak availability differs")
            rebuilt = type(item).model_validate(encoded).model_dump(mode="json")
            if rebuilt.get("peak_memory_mb") != expected_peak or rebuilt.get("peak_memory_status") != expected_status:
                raise ValueError("Stored asynchronous peak availability changed on GET reconstruction")
        observation = _video_peak_memory_observation(value)
        if observation != {"peak_memory_mb": expected_peak, "peak_memory_status": expected_status}:
            raise ValueError("Installed async job peak availability differs")
        checks.append({"input": value, "status": expected_status, "public_value": expected_peak})
    return {"status": "PASS_INSTALLED_PEAK_AVAILABILITY_CPU_ONLY", "checks": checks,
            "scope": "Actual installed API helpers and Pydantic serialization; no model/GPU execution"}
