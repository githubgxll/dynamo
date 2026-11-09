"""CPU/async simulations only; actual Gateway/GPU/media acceptance is separate."""
from __future__ import annotations

import asyncio
import ast
import base64
import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
if not (REPO / "dingo/vllm/omni/args.py").is_file():
    REPO = Path(importlib.metadata.distribution("ai-dingo").locate_file(""))
if not (REPO / "dingo/vllm/omni/args.py").is_file():
    raise RuntimeError("Gateway checks require the actual installed ai-dingo source")
sys.path.insert(0, str(ROOT))
import gateway
from service import ownership, state
import test_service_runtime as control_fixtures


def data(mime="image/png", body=b"reference"):
    return "data:" + mime + ";base64," + base64.b64encode(body).decode()


def request(**changes):
    value = {"prompt": "A stable test", "seconds": 6, "size": "1280x768",
        "response_format": "b64_json", "nvext": {"fps": 24, "num_frames": 144,
        "num_inference_steps": 50, "guidance_scale": 1, "seed": 666}}
    return {**value, **changes}


class WireContract(unittest.TestCase):
    def test_fl_tail_and_pair_indices_survive_gateway_envelopes(self):
        tail = json.dumps({"type": "fl2va_keyframes_v1", "images": [data()], "frame_indices": [-1]})
        fields, refs, _ = gateway.native_h3_request(request(input_reference=tail), "fl2va")
        self.assertEqual(json.loads(fields["extra_params"])["frame_indices"], [-1])
        self.assertEqual(refs, [("image/png", b"reference")])
        pair = json.dumps([data(), data()])
        fields, _, _ = gateway.native_h3_request(request(input_reference=pair), "fl2va")
        self.assertEqual(json.loads(fields["extra_params"])["frame_indices"], [0, -1])

    def test_mixed_ref_media_preserve_categories_and_native_task(self):
        envelope = {"type": "ref2va_mixed_v1", "images": [data()],
            "videos": [data("video/mp4")], "audios": [data("audio/wav")]}
        fields, refs, _ = gateway.native_h3_request(request(input_reference=json.dumps(envelope)), "ref2va")
        self.assertEqual([v[0] for v in refs], ["image/png", "video/mp4", "audio/wav"])
        self.assertEqual(json.loads(fields["extra_params"]), {"task": "ref2va", "audio_flow_shift": 3})
        self.assertEqual(fields["flow_shift"], 12)

    def test_remote_url_or_cross_workflow_envelope_rejected(self):
        for value in ["https://example.com/image.png", '{"type":"ref2va_mixed_v1","images":[],"videos":[],"audios":[]}']:
            with self.assertRaises(ValueError):
                gateway.native_h3_request(request(input_reference=value), "fl2va")

    def test_extreme_nine_images_three_videos_is_accepted_without_total_nine_cap(self):
        value = {"type": "ref2va_mixed_v1", "images": [data()] * 9,
            "videos": [data("video/mp4")] * 3, "audios": []}
        refs, _ = gateway.decode_references(request(input_reference=json.dumps(value)), "ref2va")
        self.assertEqual(len(refs), 12)
        value["images"].append(data())
        with self.assertRaises(ValueError):
            gateway.decode_references(request(input_reference=json.dumps(value)), "ref2va")

    def test_each_ref_category_limit_is_separate(self):
        value = {"type": "ref2va_mixed_v1", "images": [data()], "videos": [data("video/mp4")] * 4, "audios": []}
        with self.assertRaises(ValueError):
            gateway.decode_references(request(input_reference=json.dumps(value)), "ref2va")

    def test_fl_no_reference_is_t2va(self):
        fields, refs, _ = gateway.native_h3_request(request(), "fl2va")
        self.assertEqual(json.loads(fields["extra_params"])["task"], "t2va")
        self.assertFalse(refs)
        self.assertNotIn("return_stage_metrics", fields)

    def test_base_configuration_cannot_silently_change(self):
        for ext in [{"fps": 30}, {"num_inference_steps": 8}, {"guidance_scale": 2}]:
            with self.assertRaises(ValueError):
                gateway.native_h3_request(request(nvext=ext), "fl2va")

    def test_exact_geometry_matches_frozen_five_to_three(self):
        self.assertEqual(gateway.geometry(1280, 768, "1080"),
            {"internal": (1840, 1104), "delivery": (1800, 1080), "window_frames": 33})
        self.assertEqual(gateway.geometry(1280, 768, "1440"),
            {"internal": (2400, 1440), "delivery": (2400, 1440), "window_frames": 65})

    def test_invalid_sr_tier_and_reference_base64_rejected(self):
        with self.assertRaises(ValueError): gateway.geometry(1280, 768, "2160")
        with self.assertRaises(ValueError): gateway.decode_references(request(input_reference="data:image/png;base64,?"), "fl2va")

    def test_remux_stream_copies_original_audio_and_never_encodes_video(self):
        argv = gateway.remux_command("sr.mp4", "base.mp4", "final.mp4", True)
        self.assertEqual(argv[argv.index("-c:v") + 1], "copy")
        self.assertEqual(argv[argv.index("-c:a") + 1], "copy")
        self.assertIn("1:a:0", argv)
        self.assertNotIn("-shortest", argv)
        self.assertIn("-an", gateway.remux_command("sr", "base", "final", False))


class HTTPFixture:
    def __init__(self, executor):
        self.executor, self.calls = executor, []
        self.waiting, self.release = asyncio.Event(), asyncio.Event()
        self.pause_status = False
        self.bad_manifest = False
        self.submits = 0
        self.latest_fields = None
    def owned(self):
        assert self.executor.owned is not None
        assert (self.executor.root / "active.json").exists()
    async def download(self, port, route, output, **kwargs):
        self.owned()
        self.calls.append(("download", port, route, kwargs))
        Path(output).write_bytes(b"BASE" if port == 8123 else b"SR")
        return {"x-stage-durations": '{"encode_prompt":1.25,"diffuse":2.5}'}
    async def json(self, port, route, **kwargs):
        self.owned()
        self.calls.append(("json", port, route, kwargs))
        if kwargs.get("method") == "POST":
            self.submits += 1
            self.latest_fields = kwargs["fields"]
            return {"id": "a" * 32, "status": "queued"}
        if route.endswith("/metrics"):
            context = json.loads(self.latest_fields["measurement_context"])
            identity = {**context, "schema": "sr-timing-only/1", "measurement_mode": "timing",
                "sr_request_id": "a" * 32, "target_frames": 158,
                "model_width": 1840, "model_height": 1104, "delivery_width": 1800,
                "delivery_height": 1080, "configured_window": 33, "selected_window": 33}
            return {"schema": "sr-timing-only/1", "measurement_mode": "timing", "memory_observation": "not_collected",
                "sr_request_id": "a" * 32, "metrics_status": "PASS_COMPLETE",
                "missing_rank_records": [], "inconsistent_rank_records": [],
                "files": {"request.json": identity, "server-stages.jsonl": [{"stage": "source_decode", "cpu_wall_ms": 2.0}],
                    "windows/0/rank-0.json": {"stages": [{"phase": "vae_encode", "cuda_stream_elapsed_ms": 3.0}]}}}
        if route.endswith("/manifest"):
            model_width, model_height = map(int, self.latest_fields["size"].split("x"))
            delivery_width, delivery_height = map(int, self.latest_fields["delivery_size"].split("x"))
            return {"schema": "seedvr2-long-optimization/1", "sr_request_id": "a" * 32,
                "encoding_mode": "single_encode", "final_lossy_video_encode_count": 1,
                "configured_window": self.latest_fields["window_frames"], "selected_window": self.latest_fields["window_frames"],
                "model_size": {"width": model_width, "height": model_height},
                "delivery_size": {"width": delivery_width, "height": delivery_height},
                "sha256": "0" * 64 if self.bad_manifest else hashlib.sha256(b"SR").hexdigest()}
        if self.pause_status:
            self.waiting.set()
            await self.release.wait()
        return {"status": "completed", "frames": 158}
    async def health(self, port): self.calls.append(("health", port))
    async def close(self): pass


class OwnedBridge(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "control"
        self.root.mkdir()
        self.executor = ownership.GatewayExecutor(self.root, profile="fl2va-sr",
            service_epoch=control_fixtures.EPOCH, ready_check=lambda: True,
            birth_reader=lambda: control_fixtures.BIRTH)
        self.http = HTTPFixture(self.executor)
        self.process_calls = []
        self.bridge = self.make_bridge()
    async def asyncTearDown(self): self.tmp.cleanup()
    def make_bridge(self, *, ref=False, stages=False):
        async def probe(path):
            self.http.owned()
            return {"width": 1280 if Path(path).name == "base.mp4" else 1800,
                "height": 768 if Path(path).name == "base.mp4" else 1080,
                "frames": 158, "audio": True}
        async def run(argv):
            self.http.owned()
            self.process_calls.append(argv)
            if str(argv[-1]).endswith("final.mp4"):
                Path(argv[-1]).write_bytes(b"FINAL")
        async def audio(path):
            self.http.owned()
            return [{"data_hash": "original", "pts_time": "0"}]
        return gateway.GatewayBridge({"profile": "ref2va-sr" if ref else "fl2va-sr",
            "monitoring": {"mode": "stages" if stages else "off"}}, self.executor,
            http=self.http, probe_media=probe, run_process=run, probe_audio=audio,
            timing_context={"image_digest": "registry.test/repo@sha256:" + "f" * 64, "source_hashes": {"source": "f" * 64},
                "model_manifest_hash": "b" * 64} if stages else None)
    async def output(self, path, durations, timing=None):
        self.http.owned()
        self.assertTrue(path.is_file())
        return {"path": path.name, "durations": durations, "timing": timing}

    async def test_full_chain_uses_actual_base_frames_and_native_multipart(self):
        result = await self.bridge.execute(request(sr_tier="1080", input_reference=data()), "job", self.output)
        self.assertEqual(result["path"], "final.mp4")
        self.assertFalse((self.root / "active.json").exists())
        self.assertFalse((self.root / "blocked.json").exists())
        post = next(v for v in self.http.calls if v[0] == "json" and v[3].get("method") == "POST")
        self.assertEqual(post[3]["fields"]["num_frames"], 158)
        self.assertEqual(post[3]["fields"]["measurement_mode"], "off")
        self.assertNotIn("measurement_context", post[3]["fields"])
        self.assertEqual(post[3]["files"][0][0], "input_references")
        self.assertEqual(self.http.submits, 1)
        self.assertEqual(result["durations"], {})
        self.assertFalse(any(v[0] == "json" and v[2].endswith("/metrics") for v in self.http.calls))
        self.assertIsNone(result["timing"])

    async def test_pure_base_never_calls_sr_or_ref_handoff(self):
        result = await self.bridge.execute(request(), "base-only", self.output)
        self.assertEqual(result["path"], "base.mp4")
        self.assertEqual(self.http.submits, 0)

    async def test_http202_retains_permit_through_poll_and_artifact_writer(self):
        self.http.pause_status = True
        task = asyncio.create_task(self.bridge.execute(request(sr_tier="1080"), "job", self.output))
        await self.http.waiting.wait()
        acquired = asyncio.Event()
        async def contender():
            async with ownership.AsyncFileLock(self.root / "operation.lock"):
                acquired.set()
        waiter = asyncio.create_task(contender())
        await asyncio.sleep(.05)
        self.assertFalse(acquired.is_set())
        self.http.release.set()
        await task
        await waiter
        self.assertTrue(acquired.is_set())

    async def test_cancelled_sr_job_is_latched_before_unlock_never_resubmitted(self):
        self.http.pause_status = True
        task = asyncio.create_task(self.bridge.execute(request(sr_tier="1080"), "job", self.output))
        await self.http.waiting.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task
        self.assertTrue((self.root / "blocked.json").exists())
        self.assertTrue((self.root / "active.json").exists())
        with self.assertRaises(RuntimeError):
            await self.bridge.execute(request(), "next-job", self.output)
        self.assertEqual(self.http.submits, 1)

    async def test_wrong_output_hash_blocks_even_after_completed_status(self):
        self.http.bad_manifest = True
        with self.assertRaisesRegex(RuntimeError, "content differs"):
            await self.bridge.execute(request(sr_tier="1080"), "job", self.output)
        self.assertTrue((self.root / "blocked.json").exists())

    async def test_artifact_write_failure_is_inside_permit_and_blocks(self):
        async def fail(path, durations, timing=None):
            self.http.owned()
            raise OSError("artifact output incomplete")
        with self.assertRaises(OSError): await self.bridge.execute(request(), "job", fail)
        self.assertTrue((self.root / "blocked.json").exists())

    async def test_ref_handoff_runs_inside_owned_permit_before_h3(self):
        control_fixtures.ready(self.root)
        seen = []
        async def release(reason, lease):
            self.http.owned()
            seen.append((reason, lease["action_id"], len(self.http.calls)))
            return control_fixtures.release_report("job", reason)
        self.executor = ownership.GatewayExecutor(self.root, profile="ref2va-sr",
            service_epoch=control_fixtures.EPOCH, handoff=release,
            birth_reader=lambda: control_fixtures.BIRTH)
        self.http.executor = self.executor
        self.bridge = self.make_bridge(ref=True)
        await self.bridge.execute(request(input_reference=data()), "job", self.output)
        self.assertEqual(seen, [("after-sr-load-before-first-h3", "job", 0)])

    async def test_stage_context_is_only_sent_in_explicit_timing_mode(self):
        self.bridge = self.make_bridge(stages=True)
        result = await self.bridge.execute(request(sr_tier="1080"), "job", self.output)
        post = next(v for v in self.http.calls if v[0] == "json" and v[3].get("method") == "POST")
        self.assertEqual(post[3]["fields"]["measurement_mode"], "timing")
        self.assertEqual(json.loads(post[3]["fields"]["measurement_context"])["action_id"], "job")
        self.assertIn("h3_loopback_and_media_s", result["durations"])
        self.assertEqual(result["timing"]["h3"]["stage_durations"], {"encode_prompt": 1.25, "diffuse": 2.5})
        self.assertEqual(result["timing"]["sr"]["gpu_stage_max_window_rank_ms"], {"vae_encode": 3.0})
        self.assertTrue((self.root.parent / "requests/job/sr-timing-only.json").is_file())
        self.assertTrue((self.root.parent / "requests/job/h3-native-stages.json").is_file())

    async def test_extreme_sr_geometry_is_rejected_before_permit_or_h3(self):
        self.executor.execute_owned = mock.AsyncMock(side_effect=AssertionError("must not own GPU"))
        with self.assertRaisesRegex(ValueError, "admission budget"):
            await self.bridge.execute(request(size="1536x640", sr_tier="1440"), "job", self.output)
        self.executor.execute_owned.assert_not_called()
        self.assertEqual(self.http.calls, [])
        self.assertFalse((self.root / "active.json").exists())

    async def test_wrong_timing_context_or_incomplete_metrics_blocks(self):
        self.bridge = self.make_bridge(stages=True)
        original = self.http.json
        async def wrong(port, route, **kwargs):
            result = await original(port, route, **kwargs)
            if route.endswith("/metrics"):
                result["files"]["request.json"]["action_id"] = "another-job"
            return result
        self.http.json = wrong
        with self.assertRaisesRegex(RuntimeError, "context identity differs"):
            await self.bridge.execute(request(sr_tier="1080"), "job", self.output)
        self.assertTrue((self.root / "blocked.json").exists())


class EntrySelection(unittest.TestCase):
    def test_production_handler_does_not_inherit_model_constructing_base(self):
        tree = ast.parse((REPO / "dingo/vllm/omni/production_handler.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ProductionOmniHandler")
        self.assertEqual(cls.bases, [])
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        self.assertNotIn("AsyncOmni", [n.func.id for n in calls])

    def test_new_diffusion_flags_have_upstream_defaults(self):
        tree = ast.parse((REPO / "dingo/vllm/omni/args.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OmniDiffusionKwargs")
        fields = {n.target.id: ast.literal_eval(n.value) for n in cls.body if isinstance(n, ast.AnnAssign)}
        self.assertIs(fields["record_output_peak_memory"], True)
        self.assertIsNone(fields["diffusion_offload_config"])
        self.assertIsNone(fields["stage_overrides"])


if __name__ == "__main__":
    unittest.main()
