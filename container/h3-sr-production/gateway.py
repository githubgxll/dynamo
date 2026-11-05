"""Loopback bridge to supervisor-owned H3/SR models; never creates an engine.

The durable permit covers native generation, asynchronous SR completion,
content verification, original-audio remux and the caller's artifact writer.
No inference submission or output-unknown failure is automatically retried.
"""
from __future__ import annotations

import asyncio
import base64
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import time

from service.control import HTTPControlClient
from service.ownership import GatewayExecutor, TerminalResult
from service.profile import load as load_profile
from service.state import identifier, read_json

MAX_MEDIA_BYTES = 128 * 1024 * 1024
MAX_REFERENCES_BYTES = 256 * 1024 * 1024
SR_ROUTE = "/v1/seedvr2/restore-long"
MIMES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp",
         "video/mp4": ".mp4", "video/quicktime": ".mov", "audio/mpeg": ".mp3", "audio/wav": ".wav",
         "audio/x-wav": ".wav", "audio/flac": ".flac", "audio/ogg": ".ogg"}


def geometry(width, height, tier):
    """Exact-ratio even delivery and 16-aligned model canvas, without crop."""
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError("Invalid Base geometry")
    if tier not in ("1080", "1440"):
        raise ValueError("sr_tier must be 1080 or 1440")
    divisor = math.gcd(width, height)
    a, b = width // divisor, height // divisor
    needed = Fraction(int(tier), min(a, b))
    step2 = math.lcm(2 // math.gcd(a, 2), 2 // math.gcd(b, 2))
    k2 = math.floor(needed / step2) * step2
    step16 = math.lcm(16 // math.gcd(a, 16), 16 // math.gcd(b, 16))
    k16 = math.ceil(needed / step16) * step16
    if k2 <= 0 or min(a * k2, b * k2) != int(tier) or a * k16 < width or b * k16 < height:
        raise ValueError("SR tier cannot downscale the Base")
    return {"internal": (a * k16, b * k16), "delivery": (a * k2, b * k2),
            "window_frames": 33 if tier == "1080" else 65}


def admit_sr(width, height, tier, profile):
    plan = geometry(width, height, tier)
    env = profile.get("sr", {}).get("stage_environment", {})
    frame_limit = int(env.get("VLLM_OMNI_SEEDVR2_SHARDED_FRAME_PIXELS", 3791872))
    clip_limit = int(env.get("VLLM_OMNI_SEEDVR2_SHARDED_CLIP_PIXELS", 246471680))
    window_limit = int(env.get("VLLM_OMNI_SEEDVR2_LONG_MAX_WINDOW", 65))
    pixels = math.prod(plan["internal"])
    if min(frame_limit, clip_limit, window_limit) <= 0 or pixels > frame_limit or pixels * plan["window_frames"] > clip_limit or plan["window_frames"] > window_limit:
        raise ValueError("Requested SR canvas/window exceeds the frozen production admission budget")
    return plan


TIMING_SCOPE = "Nested stages are inclusive and not additive; SR summaries are maximum individual observations, not request totals."


def _write_receipt(path, value):
    data = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    with Path(path).open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(data).hexdigest()


def h3_stage_header(headers):
    raw = next((value for key, value in headers.items() if key.lower() == "x-stage-durations"), None)
    if not isinstance(raw, str) or len(raw) > 16384:
        raise RuntimeError("Stage mode requires the native X-Stage-Durations response")
    values = json.loads(raw)
    if not isinstance(values, dict) or not values or len(values) > 64:
        raise RuntimeError("Native H3 stage header is empty or invalid")
    for name, value in values.items():
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", name)
                or type(value) not in (int, float) or not math.isfinite(value) or value < 0
                or any(word in name.lower() for word in ("memory", "allocator", "nvml"))):
            raise RuntimeError("Native H3 stage duration is not a finite timing observation")
    return values


def sr_timing_summary(bundle, job, expected):
    if (not isinstance(bundle, dict) or bundle.get("schema") != "sr-timing-only/1"
            or bundle.get("measurement_mode") != "timing" or bundle.get("memory_observation") != "not_collected"
            or bundle.get("sr_request_id") != job or bundle.get("metrics_status") != "PASS_COMPLETE"
            or bundle.get("missing_rank_records") != [] or bundle.get("inconsistent_rank_records") != []):
        raise RuntimeError("SR timing is incomplete or has a different job/schema")
    files = bundle.get("files")
    request = files.get("request.json") if isinstance(files, dict) else None
    if (not isinstance(request, dict) or request.get("schema") != "sr-timing-only/1"
            or request.get("sr_request_id") != job or request.get("measurement_mode") != "timing"
            or any(request.get(key) != value for key, value in expected.items())):
        raise RuntimeError("SR timing request/context identity differs")
    def no_memory(value):
        if isinstance(value, dict):
            return not any(name.startswith("allocator") or "nvml" in name.lower() for name in value) and all(no_memory(v) for v in value.values())
        return all(no_memory(v) for v in value) if isinstance(value, list) else True
    if not no_memory(bundle):
        raise RuntimeError("Timing-only bundle contains memory observations")
    api, gpu = {}, {}
    for row in files.get("server-stages.jsonl", []):
        value, name = row.get("cpu_wall_ms"), row.get("stage")
        if type(value) in (int, float) and math.isfinite(value) and value >= 0 and isinstance(name, str):
            api[name] = max(api.get(name, 0), value)
    for relative, record in files.items():
        if re.fullmatch(r"windows/[0-9]+/rank-[0-7]\.json", relative) and isinstance(record, dict):
            for stage in record.get("stages", []):
                value, name = stage.get("cuda_stream_elapsed_ms"), stage.get("phase")
                if type(value) in (int, float) and math.isfinite(value) and value >= 0 and isinstance(name, str):
                    gpu[name] = max(gpu.get(name, 0), value)
    return {"job_id": job, "metrics_status": "PASS_COMPLETE", "unit": "milliseconds",
        "api_stage_max_row_ms": api, "gpu_stage_max_window_rank_ms": gpu}


def decode_references(request, workflow):
    """Decode Gateway data URLs/envelopes without following user-controlled URLs."""
    one, many = request.get("input_reference"), request.get("input_references")
    if one is not None and many is not None:
        raise ValueError("input_reference and input_references are mutually exclusive")
    raw = one if one is not None else many
    if isinstance(raw, str) and not raw.startswith("data:"):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError) as error:
            raise ValueError("Only embedded data URLs or Gateway envelopes are accepted") from error
    indices = (request.get("nvext") or {}).get("frame_indices")
    if isinstance(raw, dict):
        kind = raw.get("type")
        if kind == "fl2va_keyframes_v1" and workflow == "fl2va":
            if set(raw) != {"type", "images", "frame_indices"}:
                raise ValueError("Invalid FL keyframe envelope")
            indices, raw = raw["frame_indices"], raw["images"]
        elif kind == "ref2va_mixed_v1" and workflow == "ref2va":
            if set(raw) != {"type", "images", "videos", "audios"}:
                raise ValueError("Invalid Ref mixed envelope")
            values = []
            for key, prefix in (("images", "image/"), ("videos", "video/"), ("audios", "audio/")):
                if not isinstance(raw[key], list):
                    raise ValueError("Mixed reference categories must be lists")
                for value in raw[key]:
                    if not isinstance(value, str) or not value.startswith("data:" + prefix):
                        raise ValueError("Mixed reference MIME category differs")
                values.extend(raw[key])
            raw = values
        else:
            raise ValueError("Reference envelope does not match worker workflow")
    values = [] if raw is None else raw if isinstance(raw, list) else [raw]
    if len(values) > 12:
        raise ValueError("Too many reference files")
    result, total = [], 0
    for value in values:
        if not isinstance(value, str) or not value.startswith("data:") or ";base64," not in value:
            raise ValueError("Reference must be a base64 data URL")
        head, body = value.split(",", 1)
        mime = head[5:-7].lower()  # remove data: and ;base64
        if mime not in MIMES or len(body) > 4 * ((MAX_REFERENCES_BYTES + 2) // 3):
            raise ValueError("Unsupported or oversized reference")
        try:
            data = base64.b64decode(body, validate=True)
        except ValueError as error:
            raise ValueError("Invalid reference base64") from error
        total += len(data)
        limit = (30 if mime.startswith("image/") else 50 if mime.startswith("video/") else 15) * 1024 * 1024
        if not data or len(data) > limit or total > MAX_REFERENCES_BYTES:
            raise ValueError("Empty reference or aggregate media limit exceeded")
        result.append((mime, data))
    if workflow == "fl2va":
        if len(result) > 2 or any(not mime.startswith("image/") for mime, _ in result):
            raise ValueError("FL requires zero, one or two image keyframes")
        if indices is None:
            indices = [] if not result else [0] if len(result) == 1 else [0, -1]
        if indices not in ([], [0], [-1], [0, -1]) or len(indices) != len(result):
            raise ValueError("Invalid FL first/last frame indices")
    else:
        if not result:
            raise ValueError("Ref workflow requires reference media")
        counts = {kind: sum(mime.startswith(kind + "/") for mime, _ in result) for kind in ("image", "video", "audio")}
        if counts["image"] > 9 or counts["video"] > 3 or counts["audio"] > 3:
            raise ValueError("Ref media category count exceeds native limits")
        if indices is not None:
            raise ValueError("Ref workflow cannot set FL frame indices")
    return result, indices


def native_h3_request(request, workflow, *, stages=False):
    refs, indices = decode_references(request, workflow)
    ext = request.get("nvext") or {}
    size = request.get("size", "1280x768")
    if not isinstance(size, str) or not re.fullmatch(r"[1-9][0-9]{1,4}x[1-9][0-9]{1,4}", size):
        raise ValueError("Invalid H3 size")
    width, height = map(int, size.split("x"))
    seconds = request.get("seconds")
    if seconds is None:
        frames = ext.get("num_frames")
        if type(frames) is not int:
            raise ValueError("A duration or requested frame count is required")
        seconds = frames / 24
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 4 <= seconds <= 15:
        raise ValueError("H3 duration must be 4..15 seconds")
    steps = ext.get("num_inference_steps", 50)
    if steps != 50 or ext.get("guidance_scale", 1) != 1 or ext.get("fps", 24) != 24:
        raise ValueError("Production Base profile requires 50 steps, CFG1 and 24 FPS")
    if ext.get("negative_prompt") is not None:
        raise ValueError("Production profile has no negative-prompt override")
    seed = ext.get("seed", 666)
    if type(seed) is not int or seed < 0:
        raise ValueError("Invalid seed")
    prompt = request.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt is required")
    extra = {"task": "t2va" if workflow == "fl2va" and not refs else workflow,
             "audio_flow_shift": 3}
    if workflow == "fl2va" and refs:
        extra["frame_indices"] = indices
    fields = {"model": "minimax-h3-" + workflow, "prompt": prompt,
        "seconds": seconds, "size": size, "width": width, "height": height,
        "fps": 24, "aspect_ratio": "1:1" if width == height else "16:9" if width > height else "9:16",
        "num_inference_steps": 50, "num_outputs_per_prompt": 1,
        "guidance_scale": 1, "flow_shift": 12, "seed": seed,
        "extra_params": json.dumps(extra, separators=(",", ":"))}
    if stages:
        fields["return_stage_metrics"] = "true"
    return fields, refs, (width, height)


async def process(argv):
    child = await asyncio.create_subprocess_exec(*map(str, argv),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        output, errors = await child.communicate()
    except BaseException:
        if child.returncode is None:
            child.kill()
        await child.communicate()
        raise
    if child.returncode:
        raise RuntimeError(f"{argv[0]} failed: " + errors[-2000:].decode(errors="replace"))
    if len(output) > 8 * 1024 * 1024:
        raise RuntimeError("Media probe response exceeds limit")
    return output


async def probe(path):
    value = json.loads(await process(["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json", path]))
    videos = [s for s in value["streams"] if s.get("codec_type") == "video"]
    audios = [s for s in value["streams"] if s.get("codec_type") == "audio"]
    if len(videos) != 1 or len(audios) > 1:
        raise RuntimeError("Expected one video and at most one original audio stream")
    video = videos[0]
    frames = int(video.get("nb_read_frames", "0"))
    if frames <= 0 or Fraction(video.get("avg_frame_rate", "0")) != 24:
        raise RuntimeError("Backend video must have actual frames at 24 FPS")
    if video.get("codec_name") != "h264" or video.get("pix_fmt") != "yuv420p":
        raise RuntimeError("Backend must provide copy-compatible H264/yuv420p")
    return {"width": video["width"], "height": video["height"], "frames": frames, "audio": bool(audios)}


async def audio_identity(path):
    value = json.loads(await process(["ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_packets", "-show_data_hash", "sha256", "-show_entries",
        "packet=pts_time,dts_time,duration_time,size,data_hash", "-of", "json", path]))
    return value.get("packets", [])


def remux_command(raw, base, final, has_audio):
    argv = ["ffmpeg", "-nostdin", "-n", "-v", "error", "-xerror", "-copyts",
        "-noautorotate", "-i", str(raw), "-noautorotate", "-i", str(base), "-map", "0:v:0"]
    argv += ["-map", "1:a:0", "-c:a", "copy"] if has_audio else ["-an"]
    return argv + ["-c:v", "copy", "-fps_mode", "passthrough", "-avoid_negative_ts", "disabled",
        "-map_metadata", "1", "-movflags", "+faststart", str(final)]


class NativeHTTP:
    def __init__(self):
        import httpx
        self.client = httpx.AsyncClient(trust_env=False, follow_redirects=False,
            timeout=httpx.Timeout(10800, connect=10), limits=httpx.Limits(max_connections=4))

    async def close(self):
        await self.client.aclose()

    async def json(self, port, route, *, method="GET", fields=None, files=None, headers=None, status=200, max_bytes=1024 * 1024):
        async with self.client.stream(method, f"http://127.0.0.1:{port}{route}",
            data=None if fields is None else {k: str(v) for k, v in fields.items()}, files=files, headers=headers) as response:
            if response.status_code != status or response.headers.get("content-type", "").split(";", 1)[0] != "application/json":
                raise RuntimeError(f"Native {port} {route} did not return expected JSON/status {status}")
            chunks, count = [], 0
            async for chunk in response.aiter_bytes():
                count += len(chunk)
                if count > max_bytes:
                    raise RuntimeError("Native JSON response exceeds limit")
                chunks.append(chunk)
            return json.loads(b"".join(chunks))

    async def download(self, port, route, output, *, fields=None, files=None, headers=None):
        method = "GET" if fields is None else "POST"
        async with self.client.stream(method, f"http://127.0.0.1:{port}{route}",
            data=None if fields is None else {k: str(v) for k, v in fields.items()}, files=files, headers=headers) as response:
            if response.status_code != 200 or response.headers.get("content-type", "").split(";", 1)[0] != "video/mp4":
                raise RuntimeError("Native video response status/content type differs")
            count = 0
            with Path(output).open("xb") as stream:
                async for chunk in response.aiter_bytes():
                    count += len(chunk)
                    if count > MAX_MEDIA_BYTES:
                        raise RuntimeError("Native video exceeds artifact limit")
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            if count == 0:
                raise RuntimeError("Native video is empty")
            return dict(response.headers)

    async def health(self, port):
        response = await self.client.get(f"http://127.0.0.1:{port}/health", timeout=5)
        if response.status_code != 200:
            raise RuntimeError("Supervisor-owned native backend is not Ready")


class GatewayBridge:
    def __init__(self, profile, executor, *, http=None, probe_media=probe,
                 run_process=process, probe_audio=audio_identity, timing_context=None):
        self.profile, self.executor = profile, executor
        self.workflow = profile["profile"].removesuffix("-sr")
        self.http = http or NativeHTTP()
        self.probe, self.process, self.audio = probe_media, run_process, probe_audio
        self.stages = profile["monitoring"]["mode"] == "stages"
        self.timing_context = timing_context

    @classmethod
    def from_profile(cls, path):
        profile = load_profile(path)
        root = Path(os.environ.get("H3SR_CONTROL_ROOT", "/work/h3-sr/control"))
        state = read_json(root / "state.json")
        if state.get("profile_sha256") != hashlib.sha256(Path(path).read_bytes()).hexdigest():
            raise RuntimeError("Gateway profile differs from supervisor-owned profile")
        if state.get("profile") != profile["profile"] or state.get("service_ready") is not True:
            raise RuntimeError("Both supervisor-owned backends must be Ready before Gateway registration")
        epoch = state["service_epoch"]
        control = HTTPControlClient(root, epoch=epoch)
        executor = GatewayExecutor(root, profile=profile["profile"], service_epoch=epoch,
            handoff=control.release if profile["profile"] == "ref2va-sr" else None)
        return cls(profile, executor, timing_context=cls._timing_identity(profile) if profile["monitoring"]["mode"] == "stages" else None)

    @staticmethod
    def _timing_identity(profile):
        image = os.environ.get("H3SR_IMAGE_DIGEST", "")
        if not re.fullmatch(r"[^\s]{1,300}@sha256:[0-9a-f]{64}", image):
            raise ValueError("Stage timing requires the actual deployment H3SR_IMAGE_DIGEST")
        manifest = Path(profile["sr_weight_preparation"]["output_dir"]) / "conversion-manifest.json"
        dist = importlib.metadata.distribution("vllm-omni")
        names = {"sr_telemetry.py": "diffusion/models/seedvr2/sr_telemetry.py",
            "pipeline_seedvr2.py": "diffusion/models/seedvr2/pipeline_seedvr2.py",
            "ipc.py": "diffusion/ipc.py"}
        hashes = {name: hashlib.sha256(Path(dist.locate_file("vllm_omni/" + relative)).read_bytes()).hexdigest()
                  for name, relative in names.items()}
        return {"image_digest": image, "source_hashes": hashes,
            "model_manifest_hash": hashlib.sha256(manifest.read_bytes()).hexdigest(), "expected_world_size": 8}

    async def check_ready(self):
        self.executor.latch.assert_open()
        if not self.executor._file_ready():
            raise RuntimeError("Supervisor readiness or service epoch changed")
        await self.http.health(8123)
        await self.http.health(8124)

    async def close(self):
        await self.http.close()

    async def execute(self, request, action_id, output_writer):
        action_id = identifier(action_id)
        fields, refs, expected_size = native_h3_request(request, self.workflow, stages=self.stages)
        tier = request.get("sr_tier")
        if tier is not None and tier not in ("1080", "1440"):
            raise ValueError("Invalid SR tier")
        if tier is not None:
            admit_sr(*expected_size, tier, self.profile)
        folder = self.executor.root.parent / "requests" / action_id
        folder.mkdir(parents=True, exist_ok=False)
        base, raw, final = (folder / name for name in ("base.mp4", "sr.mp4", "final.mp4"))
        state = {}
        durations = {}
        timing = {"schema": "h3sr-production-timing/1", "scope": TIMING_SCOPE} if self.stages else None

        async def h3():
            tick = time.monotonic()
            files = [("input_references", (f"reference-{i}{MIMES[mime]}", data, mime)) for i, (mime, data) in enumerate(refs)]
            headers = await self.http.download(8123, "/v1/videos/sync", base, fields=fields,
                files=files, headers=self.executor.operation_headers())
            media = await self.probe(base)
            if (media["width"], media["height"]) != expected_size:
                raise RuntimeError("Base geometry differs from accepted request")
            requested_frames = round(float(fields["seconds"]) * 24)
            if media["frames"] != requested_frames + ((5 - requested_frames) % 17):
                raise RuntimeError("Actual Base frames differ from native H3 alignment")
            state.update(base=media, headers=headers)
            if self.stages:
                values = h3_stage_header(headers)
                receipt = {"schema": "h3-native-stage-header/1", "action_id": action_id,
                    "service_epoch": self.executor.epoch, "stage_durations": values,
                    "unit": "upstream_native", "scope": TIMING_SCOPE}
                timing["h3"] = {"sha256": _write_receipt(folder / "h3-native-stages.json", receipt),
                    "stage_durations": values, "unit": "upstream_native"}
                durations["h3_loopback_and_media_s"] = time.monotonic() - tick
            return TerminalResult(action_id, True, True, True, str(base))

        async def sr():
            tick = time.monotonic()
            media = state["base"]
            plan = admit_sr(media["width"], media["height"], tier, self.profile)
            state["plan"] = plan
            sr = {"prompt": " ", "size": "%dx%d" % plan["internal"],
                "num_frames": media["frames"], "loop_input": "false", "seed": 7723,
                "extra_params": '{"color_correction_method":"lab"}',
                "encoding_mode": "single_encode", "window_frames": plan["window_frames"],
                "delivery_size": "%dx%d" % plan["delivery"],
                "measurement_mode": "timing" if self.stages else "off"}
            if self.stages:
                if not self.timing_context:
                    raise RuntimeError("Missing stage timing identity")
                context = {**self.timing_context,
                    "run_id": self.executor.epoch, "batch_id": "gateway", "case_id": action_id,
                    "action_id": action_id, "phase": "measured"}
                sr["measurement_context"] = json.dumps(context, separators=(",", ":"))
            headers = self.executor.operation_headers()
            with base.open("rb") as stream:
                task = await self.http.json(8124, SR_ROUTE, method="POST", fields=sr,
                    files=[("input_references", ("base.mp4", stream, "video/mp4"))], headers=headers, status=202)
            job = task.get("id")
            if not isinstance(job, str) or not re.fullmatch(r"[0-9a-f]{32}", job) or task.get("status") != "queued":
                raise RuntimeError("SR submission identity is unknown")
            route = SR_ROUTE + "/" + job
            # Persist the job identity before polling; a transport failure never resubmits.
            (folder / "sr-job.json").write_text(json.dumps(task), encoding="utf-8")
            while True:
                status = await self.http.json(8124, route, headers=headers)
                if type(status.get("frames")) is not int or not 0 <= status["frames"] <= media["frames"]:
                    raise RuntimeError("Invalid SR frame completion state")
                if status.get("status") == "completed":
                    if status["frames"] != media["frames"]:
                        raise RuntimeError("SR completion lost Base frames")
                    break
                if status.get("status") not in ("queued", "running"):
                    raise RuntimeError("SR failed/cancelled or returned unknown state")
                await asyncio.sleep(1)
            manifest = await self.http.json(8124, route + "/manifest", headers=headers)
            if manifest.get("schema") != "seedvr2-long-optimization/1" or manifest.get("sr_request_id") != job:
                raise RuntimeError("SR output manifest identity differs")
            if manifest.get("encoding_mode") != "single_encode" or manifest.get("final_lossy_video_encode_count") != 1:
                raise RuntimeError("SR did not prove single final encoding")
            if (manifest.get("configured_window") != plan["window_frames"]
                    or manifest.get("selected_window") != plan["window_frames"]
                    or manifest.get("model_size") != dict(zip(("width", "height"), plan["internal"]))
                    or manifest.get("delivery_size") != dict(zip(("width", "height"), plan["delivery"]))):
                raise RuntimeError("SR silently changed the selected window/model/delivery canvas")
            await self.http.download(8124, route + "/content", raw, headers=headers)
            if hashlib.sha256(raw.read_bytes()).hexdigest() != manifest.get("sha256"):
                raise RuntimeError("SR content differs from terminal manifest")
            output = await self.probe(raw)
            if (output["width"], output["height"], output["frames"]) != (*plan["delivery"], media["frames"]):
                raise RuntimeError("SR delivery geometry or actual frame count differs")
            if self.stages:
                bundle = await self.http.json(8124, route + "/metrics", headers=headers, max_bytes=17 * 1024 * 1024)
                expected = {**context, "target_frames": media["frames"],
                    "model_width": plan["internal"][0], "model_height": plan["internal"][1],
                    "delivery_width": plan["delivery"][0], "delivery_height": plan["delivery"][1],
                    "configured_window": plan["window_frames"], "selected_window": plan["window_frames"]}
                summary = sr_timing_summary(bundle, job, expected)
                summary["sha256"] = _write_receipt(folder / "sr-timing-only.json", bundle)
                timing["sr"] = summary
                durations["sr_job_output_e2e_s"] = time.monotonic() - tick
            return TerminalResult(action_id, True, True, True, str(raw), job)

        async def finish():
            tick = time.monotonic()
            chosen = base
            if tier is not None:
                await self.process(remux_command(raw, base, final, state["base"]["audio"]))
                media = await self.probe(final)
                if (media["width"], media["height"], media["frames"], media["audio"]) != (*state["plan"]["delivery"], state["base"]["frames"], state["base"]["audio"]):
                    raise RuntimeError("Remux output differs from completed Base/SR contract")
                if state["base"]["audio"] and await self.audio(base) != await self.audio(final):
                    raise RuntimeError("Original Base audio packet/time identity changed")
                chosen = final
            # Full decode completes before releasing the business permit.
            await self.process(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", chosen, "-f", "null", "-"])
            payload = await output_writer(chosen, durations, timing=timing)
            if self.stages:
                durations["final_output_and_artifact_s"] = time.monotonic() - tick
            return TerminalResult(action_id, True, True, True, payload)

        async with asyncio.timeout(10800):
            result = await self.executor.execute_owned(action_id, h3_callable=h3,
                sr_callable=sr if tier is not None else None, remux_callable=finish)
        return result["results"]["remux"].payload
