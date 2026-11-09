"""CPU checks against installed HTTP/media libraries, never CUDA acceptance."""
from __future__ import annotations

import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
if not (REPO / "dingo/video_gateway/adapters/minimax_h3.py").is_file():
    REPO = Path(importlib.metadata.distribution("ai-dingo").locate_file(""))
if not (REPO / "dingo/video_gateway/adapters/minimax_h3.py").is_file():
    raise RuntimeError("Gateway checks require actual installed ai-dingo adapter source")
sys.path[:0] = [str(ROOT), str(REPO)]
import gateway
import sr_timing_patch


class TimingWireCompatibility(unittest.TestCase):
    def test_stages_wire_is_accepted_by_actual_transformed_helper(self):
        source = ROOT / "tests/fixtures/sr-timing/long_video_support.py"
        data = sr_timing_patch.transform("vllm_omni/diffusion/models/seedvr2/long_video_support.py", source.read_bytes())
        namespace = {"__name__": "gateway_support_fixture", "__file__": str(source)}
        exec(compile(data, str(source), "exec"), namespace)
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"SEEDVR2_METRICS_DIR": directory}):
            identity = {"run_id": "svc-test", "batch_id": "gateway", "case_id": "job", "action_id": "job",
                "phase": "measured", "expected_world_size": 8,
                "image_digest": "registry.test/repo@sha256:" + "a" * 64,
                "source_hashes": {name: "b" * 64 for name in ("sr_telemetry.py", "pipeline_seedvr2.py", "ipc.py")},
                "model_manifest_hash": "c" * 64}
            value = namespace["options"](encoding_mode="single_encode", delivery_size="1800x1080",
                window_frames=33, measurement_mode="timing", measurement_context=json.dumps(identity))
        self.assertEqual(value["measurement_context"], identity)

    def test_actual_timing_identity_has_required_source_keys_and_full_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "conversion-manifest.json").write_text('{"status":"VERIFIED"}', encoding="utf-8")
            for name in ("sr_telemetry.py", "pipeline_seedvr2.py", "ipc.py"):
                (root / name).write_text("# actual CPU fixture\n", encoding="utf-8")
            dist = SimpleNamespace(locate_file=lambda name: root / Path(name).name)
            profile = {"sr_weight_preparation": {"output_dir": str(root)}}
            with mock.patch.object(gateway.importlib.metadata, "distribution", return_value=dist), \
                    mock.patch.dict(os.environ, {"H3SR_IMAGE_DIGEST": "registry.test/repo@sha256:" + "a" * 64}):
                identity = gateway.GatewayBridge._timing_identity(profile)
            self.assertEqual(set(identity["source_hashes"]), {"sr_telemetry.py", "pipeline_seedvr2.py", "ipc.py"})
            self.assertEqual(identity["expected_world_size"], 8)
            with mock.patch.dict(os.environ, {"H3SR_IMAGE_DIGEST": "sha256:" + "a" * 64}):
                with self.assertRaises(ValueError): gateway.GatewayBridge._timing_identity(profile)


@unittest.skipUnless(importlib.util.find_spec("httpx"), "Requires actual httpx CPU test dependency")
class HTTPClientContract(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import httpx
        self.httpx = httpx
        self.http = gateway.NativeHTTP()
        await self.http.client.aclose()
        self.requests = []
    async def asyncTearDown(self): await self.http.close()
    def transport(self, callback):
        self.http.client = self.httpx.AsyncClient(transport=self.httpx.MockTransport(callback),
            trust_env=False, follow_redirects=False)

    async def test_actual_httpx_multipart_uses_repeated_native_plural_field(self):
        async def respond(req):
            self.requests.append((req, await req.aread()))
            return self.httpx.Response(202, json={"id": "a" * 32, "status": "queued"})
        self.transport(respond)
        response = await self.http.json(8124, gateway.SR_ROUTE, method="POST", fields={"num_frames": 158},
            files=[("input_references", ("one.png", b"ONE", "image/png")),
                   ("input_references", ("two.wav", b"TWO", "audio/wav"))], status=202)
        self.assertEqual(response["status"], "queued")
        req, body = self.requests[0]
        self.assertEqual(str(req.url), "http://127.0.0.1:8124" + gateway.SR_ROUTE)
        self.assertEqual(body.count(b'name="input_references"'), 2)

    async def test_redirect_is_not_followed_or_retried(self):
        def respond(req):
            self.requests.append(req)
            return self.httpx.Response(307, headers={"location": "https://example.invalid/"})
        self.transport(respond)
        with self.assertRaises(RuntimeError): await self.http.json(8124, "/job")
        self.assertEqual(len(self.requests), 1)

    async def test_streamed_json_limit_rejects_oversized_backend(self):
        self.transport(lambda req: self.httpx.Response(200, json={"value": "a" * (1024 * 1024)}))
        with self.assertRaisesRegex(RuntimeError, "exceeds limit"):
            await self.http.json(8124, "/job")

    async def test_video_download_is_actual_streamed_file_and_preserves_headers(self):
        self.transport(lambda req: self.httpx.Response(200, content=b"mp4-fixture",
            headers={"content-type": "video/mp4", "x-stage-durations": "{}"}))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.mp4"
            headers = await self.http.download(8123, "/v1/videos/sync", path, fields={"prompt": "test"})
            self.assertEqual(path.read_bytes(), b"mp4-fixture")
            self.assertEqual(headers["x-stage-durations"], "{}")


@unittest.skipUnless(importlib.util.find_spec("PIL") and importlib.util.find_spec("av"), "Requires actual Pillow/PyAV CPU test dependencies")
class PublicAdapterContract(unittest.TestCase):
    def adapter(self, enabled=True, validate=False):
        from dingo.video_gateway.adapters.minimax_h3 import MiniMaxH3VideoAdapter
        pool = SimpleNamespace(pool_id="public-fl", backend_model="worker-fl", sr_profile=None, adapter=SimpleNamespace(
            workflow="fl2va", options={"enable_sr": enabled, "validate_media": validate}))
        return MiniMaxH3VideoAdapter(pool)

    def fields(self, **changes):
        return {**{"prompt": ["A test scene"], "model": ["public-fl"], "seconds": ["6"],
            "size": ["1280x768"], "num_inference_steps": ["50"], "guidance_scale": ["1"], "seed": ["666"]}, **changes}

    def test_actual_adapter_sr_normalization_and_worker_payload(self):
        adapter = self.adapter()
        value = adapter.normalize_request(self.fields(sr_tier=["1080"]), [], "public-fl")
        self.assertEqual((value["width"], value["height"]), (1280, 768))
        self.assertEqual((value["final_width"], value["final_height"]), (1800, 1080))
        with tempfile.TemporaryDirectory() as directory:
            payload = adapter.build_worker_payload(value, [], Path(directory))
        self.assertEqual(payload["sr_tier"], "1080")
        self.assertEqual(payload["seconds"], 6)
        self.assertEqual(payload["nvext"]["frame_indices"], [])

    def test_sr_field_rejected_by_disabled_pool_and_invalid_value(self):
        from dingo.video_gateway.errors import GatewayError
        with self.assertRaises(GatewayError):
            self.adapter(False).normalize_request(self.fields(sr_tier=["1080"]), [], "public-fl")
        with self.assertRaises(GatewayError):
            self.adapter().normalize_request(self.fields(sr_tier=["2160"]), [], "public-fl")

    def test_valid_public_base_but_oversized_sr_is_rejected_before_dispatch(self):
        from dingo.video_gateway.errors import GatewayError
        with self.assertRaises(GatewayError) as error:
            self.adapter().normalize_request(self.fields(size=["1536x640"], sr_tier=["1440"]), [], "public-fl")
        self.assertEqual(error.exception.code, "sr_capacity_exceeded")

    def test_actual_pyav_validator_accepts_sr_delivery_size_instead_of_base(self):
        import av
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actual-cpu-encoded.mp4"
            with av.open(str(path), "w") as container:
                stream = container.add_stream("libx264", rate=24)
                stream.width, stream.height, stream.pix_fmt = 1800, 1080, "yuv420p"
                stream.options = {"preset": "ultrafast", "crf": "30"}
                for _ in range(5):
                    frame = av.VideoFrame(1800, 1080, "yuv420p")
                    for plane in frame.planes:
                        plane.update(bytes(plane.buffer_size))
                    for packet in stream.encode(frame): container.mux(packet)
                for packet in stream.encode(): container.mux(packet)
            adapter = self.adapter(validate=True)
            normalized = {"width": 1280, "height": 768, "final_width": 1800,
                "final_height": 1080, "num_frames": 5, "generate_sound": False}
            with av.open(str(path)) as container:
                proof = adapter._validate_open_artifact(container, normalized)
            self.assertEqual((proof["width"], proof["height"], proof["frames"]), (1800, 1080, 5))


class TimingDeliveryContract(unittest.TestCase):
    def timing(self):
        return {"schema": "h3sr-production-timing/1", "scope": gateway.TIMING_SCOPE,
            "h3": {"sha256": "a" * 64, "stage_durations": {"encode_prompt": 1.25},
                "unit": "upstream_native", "worker_local_path": "/work/private.json"},
            "sr": {"sha256": "b" * 64, "job_id": "c" * 32, "metrics_status": "PASS_COMPLETE",
                "unit": "milliseconds", "api_stage_max_row_ms": {"source_decode": 2.5},
                "gpu_stage_max_window_rank_ms": {"vae_encode": 3.5}, "url": "/does-not-exist"}}

    def response(self):
        return {"status": "completed", "data": [{"output_format": "mp4", "artifact": {
            "schema_version": 1, "filename": "worker-video-" + "d" * 32 + ".mp4",
            "bytes": 10, "sha256": "e" * 64}}], "h3_sr_timing": self.timing()}

    def test_actual_worker_inline_consumer_and_public_task_keep_summary_without_paths(self):
        from dingo.common.video_result_file import normalize_inline_result
        from dingo.common.video_timing_schemas import with_model_execution
        from dingo.video_gateway.adapters.minimax_h3 import _MiniMaxH3WorkerStreamConsumer
        from dingo.video_gateway.models import VideoTask, TaskStatus
        inline = normalize_inline_result(self.response())
        consumer = _MiniMaxH3WorkerStreamConsumer()
        consumer.consume(inline)
        result = consumer.finish()
        self.assertEqual(result.h3_sr_timing["sr"]["metrics_status"], "PASS_COMPLETE")
        normalized = with_model_execution({}, None, h3_sr_timing=result.h3_sr_timing)
        task = VideoTask(schema_version=1, id="job", deployment_id="deployment", pool_id="pool", model="model",
            backend_model="native", backend_target="dyn://ns.backend.generate", configuration_revision="revision",
            delivery_mode="async", status=TaskStatus.COMPLETED, request_digest="a" * 64,
            request_path="request", input_manifest_path="manifest", created_at_ms=1, queued_at_ms=1,
            expires_at_ms=100, normalized_request=normalized)
        public = task.public_dict()
        self.assertEqual(public["metrics"]["h3_sr_timing"], result.h3_sr_timing)
        text = json.dumps(public["metrics"]["h3_sr_timing"])
        self.assertNotIn("/work", text)
        self.assertNotIn("url", text)
        self.assertIn("not additive", text)

    def test_default_off_omits_optional_timing_through_inline_consumer(self):
        from dingo.common.video_result_file import normalize_inline_result
        from dingo.video_gateway.adapters.minimax_h3 import _MiniMaxH3WorkerStreamConsumer
        response = self.response()
        response.pop("h3_sr_timing")
        inline = normalize_inline_result(response)
        self.assertNotIn("h3_sr_timing", inline)
        consumer = _MiniMaxH3WorkerStreamConsumer()
        consumer.consume(inline)
        self.assertIsNone(consumer.finish().h3_sr_timing)

    def test_nonfinite_or_memory_stage_not_exposed_as_timing(self):
        from dingo.common.video_timing_schemas import normalize_h3_sr_timing
        value = self.timing()
        value["h3"]["stage_durations"] = {"allocator_memory": 5}
        self.assertIsNone(normalize_h3_sr_timing(value))
        value["h3"]["stage_durations"] = {"diffuse": float("nan")}
        self.assertIsNone(normalize_h3_sr_timing(value))

    def test_actual_public_task_distinguishes_base_target_and_verified_final_size(self):
        from dingo.video_gateway.models import VideoTask, TaskStatus
        task = VideoTask(schema_version=1, id="job", deployment_id="deployment", pool_id="pool", model="model",
            backend_model="native", backend_target="dyn://ns.backend.generate", configuration_revision="revision",
            delivery_mode="async", status=TaskStatus.IN_PROGRESS, request_digest="a" * 64,
            request_path="request", input_manifest_path="manifest", created_at_ms=1, queued_at_ms=1,
            expires_at_ms=100, normalized_request={"width": 1280, "height": 768, "sr_tier": "1080",
                "final_width": 1800, "final_height": 1080})
        pending = task.public_dict()
        self.assertEqual(pending["size"], "1280x768")
        self.assertEqual(pending["base_size"], "1280x768")
        self.assertEqual(pending["target_size"], "1800x1080")
        self.assertEqual(pending["sr_tier"], "1080")
        self.assertNotIn("final_size", pending)
        task.status = TaskStatus.COMPLETED
        self.assertNotIn("final_size", task.public_dict())
        # Final dimensions come from validated media, independently of the plan.
        task.normalized_request["_result_media"] = {"width": 1800, "height": 1080}
        task.normalized_request["final_width"] = 1840
        task.normalized_request["final_height"] = 1104
        completed = task.public_dict()
        self.assertEqual(completed["final_size"], "1800x1080")
        self.assertEqual(completed["target_size"], "1840x1104")
        self.assertEqual(completed["size"], "1280x768")
        task.normalized_request.pop("sr_tier")
        base_only = task.public_dict()
        for field in ("base_size", "sr_tier", "target_size", "final_size"):
            self.assertNotIn(field, base_only)


if __name__ == "__main__": unittest.main()
