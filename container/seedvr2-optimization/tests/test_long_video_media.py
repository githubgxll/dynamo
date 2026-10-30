# SPDX-License-Identifier: Apache-2.0
"""Synthetic CPU media tests; actual installed module is used in image smoke."""
import hashlib
import asyncio
import importlib
import importlib.util
import io
import json
import os
import subprocess
from pathlib import Path
from fractions import Fraction
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch
from contextlib import contextmanager

HERE = Path(__file__).resolve().parents[1]


def file_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parent_fixture():
    c14 = HERE.parent / "seedvr2-c14"
    if not c14.is_dir():
        c14 = Path("/opt/seedvr2-c14")
    relative = "vllm_omni/diffusion/models/seedvr2/long_video.py"
    original = (c14 / "vendor/original" / relative).read_bytes()
    parent_patch = file_module("_sr_long_parent_patch", c14 / "source_patches.py")
    return parent_patch.transform(relative, original)


def tested_module():
    if os.environ.get("SR_OPT_REQUIRE_RUNTIME") == "1":
        return importlib.import_module("vllm_omni.diffusion.models.seedvr2.long_video")
    support = file_module("_sr_media_support", HERE / "long_video_support.py")
    transform = file_module("_sr_media_patch", HERE / "long_video_patch.py")
    c14_path = HERE.parent / "seedvr2-c14/c14_capture.py"
    c14 = file_module("_sr_media_c14", c14_path)
    modules = {}
    for name in ("vllm_omni", "vllm_omni.diffusion", "vllm_omni.diffusion.models", "vllm_omni.diffusion.models.seedvr2"):
        modules[name] = types.ModuleType(name)
        modules[name].__path__ = []
    prefix = "vllm_omni.diffusion.models.seedvr2"
    config = types.ModuleType(prefix + ".config")
    config.COLOR_CORRECTION_METHODS = ("lab", "wavelet", "none")
    config.DEFAULT_COLOR_CORRECTION_METHOD = "lab"
    config.VLLM_OMNI_SEEDVR2_LONG_MAX_UPLOAD_BYTES = 64 * 1024 * 1024
    modules[prefix + ".config"] = config
    modules[prefix + ".c14_capture"] = c14
    modules[prefix + ".long_video_support"] = support
    pipeline = types.ModuleType(prefix + ".pipeline_seedvr2")
    pipeline.max_frames = lambda: 65
    pipeline.sharded_budget = lambda: (2400 * 1440, 2400 * 1440 * 65)
    modules[prefix + ".pipeline_seedvr2"] = pipeline
    module = types.ModuleType("_sr_longvideo_cpu_fixture")
    sys.modules[module.__name__] = module
    with patch.dict(sys.modules, modules):
        exec(compile(transform.transform(parent_fixture()), "long_video.py", "exec"), module.__dict__)
    return module


class LongVideoPatch(unittest.TestCase):
    def test_parent_exact_source_transform_and_repeat_rejection(self):
        transform = file_module("_sr_long_compile_patch", HERE / "long_video_patch.py")
        parent = parent_fixture()
        source = transform.transform(parent)
        compile(source, "long_video.py", "exec")
        for token in (b"_c14.fused_frame", b"_c14.long_encoder", b"_c14.run_mux", b"_c14.window_context",
                      b'"seedvr2_measurement"', b'Form("legacy")', b'Form("off")'):
            self.assertIn(token, source)
        with self.assertRaises(ValueError):
            transform.transform(source)
        with self.assertRaises(ValueError):
            transform.transform(parent.replace(b"c14_context=None", b"c14_context=1"))


class LongVideoMedia(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import av
        import numpy
        cls.av, cls.np, cls.module = av, numpy, tested_module()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.job = self.root / ("c" * 32)
        self.job.mkdir()
        self.scratch = self.root / "scratch"
        self.scratch.mkdir()
        self.profile = patch.dict(os.environ, {"SEEDVR2_C14_PROFILE": ""})
        self.profile.start()

    def tearDown(self):
        self.profile.stop()
        self.temporary.cleanup()

    def source(self, count=20):
        with self.av.open(str(self.job / "input.mp4"), "w") as container:
            video = container.add_stream("libx264", rate=24)
            video.width, video.height, video.pix_fmt = 80, 48, "yuv420p"
            video.options = {"preset": "ultrafast", "qp": "0", "bf": "0"}
            for index in range(count):
                array = self.np.zeros((48, 80, 3), dtype=self.np.uint8)
                array[:, :, 0] = index * 7 % 256
                array[:, :, 1] = self.np.arange(80, dtype=self.np.uint8)[None, :] * 3
                frame = self.av.VideoFrame.from_ndarray(array, format="rgb24")
                frame.pts, frame.time_base = index, Fraction(1, 24)
                for packet in video.encode(frame):
                    container.mux(packet)
            for packet in video.encode():
                container.mux(packet)

    def fake_restore(self, frames, width, height, seed, method, port, authorization, c14, measurement, metrics):
        return [self.np.full((height, width, 3), (measurement["window_start"] + i) * 7 % 256,
                            dtype=self.np.uint8) for i in range(len(frames))]

    def invoke(self, config, restore=None, count=20, fused=None):
        self.source(count)
        with patch.object(self.module, "_restore", side_effect=restore or self.fake_restore), \
             patch.object(self.module._c14, "fused_frame", side_effect=fused):
            self.module._restore_to(self.job, self.scratch, 17, 80, 48, count, False, 7723, "lab", 8000, "", config)

    def test_real_cpu_single_writer_pts_size_and_internal_fused_identity(self):
        fused = []
        config = self.module._support.options("single_encode", "1800x1080", 17)
        self.invoke(config, fused=lambda job, index, array: fused.append((index, hashlib.sha256(array.tobytes()).hexdigest())))
        result = json.loads((self.job / "result.json").read_text())
        self.assertEqual(result["window_count"], 2)
        self.assertEqual((result["width"], result["height"]), (1800, 1080))
        self.assertEqual(result["final_lossy_video_encode_count"], 1)
        self.assertEqual(result["window_input_lossless_encode_count"], 2)
        self.assertEqual(result["writer"]["options"]["preset"], "medium")
        self.assertEqual(Fraction(result["writer"]["effective_output"]["sample_aspect_ratio"]), 1)
        self.assertEqual(result["writer"]["effective_output"]["pixel_format"], "yuv420p")
        self.assertEqual(result["writer"]["effective_output"]["codec"], "h264")
        self.assertNotIn("bf", result["writer"]["options"])
        self.assertNotIn("tune", result["writer"]["options"])
        self.assertEqual([row[0] for row in fused], list(range(20)))
        with self.av.open(str(self.job / "output.mp4")) as output:
            stream = output.streams.video[0]
            frames = list(output.decode(video=0))
            self.assertEqual((stream.width, stream.height, len(frames)), (1800, 1080, 20))
            self.assertEqual([frame.pts * frame.time_base for frame in frames], [Fraction(i, 24) for i in range(20)])
        # The fused RGB is captured before resizing. It retains the exact
        # original blending arithmetic and global frame order.
        expected = [(i, hashlib.sha256(self.np.full((48, 80, 3), i * 7 % 256, dtype=self.np.uint8).tobytes()).hexdigest())
                    for i in range(20)]
        self.assertEqual(fused, expected)

    def test_real_cpu_legacy_preserves_internal_geometry_and_preset(self):
        self.invoke(self.module._support.options(window_frames=17))
        result = json.loads((self.job / "result.json").read_text())
        self.assertEqual((result["width"], result["height"]), (80, 48))
        self.assertEqual(result["writer"]["options"]["preset"], "veryfast")
        self.assertIsNone(result["final_lossy_video_encode_count"])
        self.assertTrue(result["external_lossy_encode_required"])

    def test_single_default_bframe_writer_preserves_inherited_audio_mux(self):
        self.source(20)
        source = self.job / "input.mp4"
        audio_source = self.root / "audio-source.mp4"
        subprocess.run([self.module.imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-y", "-v", "error",
            "-i", str(source), "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=32000",
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-t", str(20 / 24),
            str(audio_source)], check=True, capture_output=True)
        audio_source.replace(source)
        config = self.module._support.options("single_encode", "1800x1080", 17)
        with patch.object(self.module, "_restore", side_effect=self.fake_restore):
            self.module._restore_to(self.job, self.scratch, 17, 80, 48, 20, False, 7723, "lab", 8000, "", config)
        with self.av.open(str(self.job / "output.mp4")) as output:
            self.assertEqual(len(output.streams.audio), 1)
            frames = list(output.decode(video=0))
            self.assertEqual(len(frames), 20)
            self.assertEqual([frame.pts * frame.time_base for frame in frames], [Fraction(i, 24) for i in range(20)])

    def test_encoder_failure_closes_context_and_does_not_publish(self):
        original_open = self.module.av.open
        closed = []
        class BrokenContainer:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                closed.append(True)
            def add_stream(self, *args, **kwargs):
                def fail(frame):
                    raise RuntimeError("synthetic encoder failure")
                return types.SimpleNamespace(encode=fail, codec_context=types.SimpleNamespace())
        def opened(path, *args, **kwargs):
            if str(path) == str(self.scratch / "video.mp4"):
                return BrokenContainer()
            return original_open(path, *args, **kwargs)
        with patch.object(self.module.av, "open", side_effect=opened), self.assertRaisesRegex(RuntimeError, "synthetic encoder failure"):
            self.invoke(self.module._support.options())
        self.assertTrue(closed)
        self.assertFalse((self.job / "result.json").exists())

    def test_single_rejects_non_h264_output_before_publication(self):
        original_open = self.module.av.open
        class Stream:
            def __init__(self, stream):
                self.original = stream
                self.codec_context = types.SimpleNamespace(name="hevc", profile="synthetic",
                    has_b_frames=stream.codec_context.has_b_frames, format=stream.codec_context.format)
            def __getattr__(self, name):
                return getattr(self.original, name)
        class Container:
            def __init__(self, value):
                self.original = value
                self.streams = types.SimpleNamespace(video=[Stream(value.streams.video[0])], audio=value.streams.audio)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.original.close()
            def decode(self, **kwargs):
                return self.original.decode(**kwargs)
        def opened(path, *args, **kwargs):
            value = original_open(path, *args, **kwargs)
            return Container(value) if str(path) == str(self.scratch / "output.mp4") else value
        with patch.object(self.module.av, "open", side_effect=opened), self.assertRaisesRegex(self.module.JobError, "h264/yuv420p"):
            self.invoke(self.module._support.options("single_encode", "1800x1080", 17))
        self.assertFalse((self.job / "result.json").exists())
        self.assertFalse((self.job / "output.mp4").exists())

    def test_cancel_waits_queued_futures_and_does_not_publish_completed(self):
        finished = threading.Event()
        def restoration(*args):
            (self.job / "cancel").touch()
            time.sleep(.05)
            result = self.fake_restore(*args)
            finished.set()
            return result
        with self.assertRaises(self.module.JobCancelledError):
            self.invoke(self.module._support.options(), restore=restoration)
        self.assertTrue(finished.is_set())
        self.assertFalse((self.job / "result.json").exists())
        self.assertFalse((self.job / "output.mp4").exists())

    def cancel_request(self):
        with patch.object(self.module, "_job_dir", return_value=self.job):
            return asyncio.run(self.module.cancel_long_video(self.job.name))

    def assert_cancelled_not_published(self):
        self.assertFalse((self.job / "result.json").exists())
        self.assertFalse((self.job / "output.mp4").exists())
        self.assertEqual(list(self.job.glob(".output-*.tmp")), [])
        self.assertNotEqual(json.loads((self.job / "status.json").read_text())["status"], "completed")
        with patch.object(self.module, "_job_dir", return_value=self.job):
            for api in (self.module.get_long_video_manifest, self.module.download_long_video):
                with self.assertRaises(self.module.HTTPException) as failure:
                    asyncio.run(api(self.job.name))
                self.assertEqual(failure.exception.status_code, 404)

    def test_delete_during_media_validation_prevents_terminal_publication(self):
        original = self.module._support.ServerMetrics.stage
        replies = []
        @contextmanager
        def observed(metrics, name, **extra):
            with original(metrics, name, **extra):
                if name == "media_validate":
                    replies.append(self.cancel_request())
                yield
        with patch.object(self.module._support.ServerMetrics, "stage", observed), self.assertRaises(self.module.JobCancelledError):
            self.invoke(self.module._support.options())
        self.assertEqual(replies, [{"id": self.job.name, "status": "cancelling"}])
        self.assert_cancelled_not_published()

    def test_delete_during_hash_prevents_terminal_publication(self):
        original = self.module.hashlib.sha256
        replies = []
        def hasher(*args, **kwargs):
            replies.append(self.cancel_request())
            return original(*args, **kwargs)
        with patch.object(self.module.hashlib, "sha256", side_effect=hasher), self.assertRaises(self.module.JobCancelledError):
            self.invoke(self.module._support.options())
        self.assertEqual(replies, [{"id": self.job.name, "status": "cancelling"}])
        self.assert_cancelled_not_published()

    def test_delete_during_large_copy_is_not_blocked_and_cleans_staging(self):
        original = self.module.shutil.move
        replies = []
        def copied(*args, **kwargs):
            result = original(*args, **kwargs)
            replies.append(self.cancel_request())
            return result
        with patch.object(self.module.shutil, "move", side_effect=copied), self.assertRaises(self.module.JobCancelledError):
            self.invoke(self.module._support.options())
        self.assertEqual(replies, [{"id": self.job.name, "status": "cancelling"}])
        self.assert_cancelled_not_published()

    def test_delete_during_audio_mux_prevents_terminal_publication(self):
        self.source(20)
        source = self.job / "input.mp4"
        audio_source = self.root / "audio-source.mp4"
        subprocess.run([self.module.imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-y", "-v", "error",
            "-i", str(source), "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=32000",
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-t", str(20 / 24),
            str(audio_source)], check=True, capture_output=True)
        audio_source.replace(source)
        original = self.module._c14.run_mux
        replies = []
        def muxed(*args, **kwargs):
            result = original(*args, **kwargs)
            replies.append(self.cancel_request())
            return result
        with patch.object(self.module, "_restore", side_effect=self.fake_restore), \
             patch.object(self.module._c14, "run_mux", side_effect=muxed), self.assertRaises(self.module.JobCancelledError):
            self.module._restore_to(self.job, self.scratch, 17, 80, 48, 20, False, 7723, "lab", 8000, "", self.module._support.options())
        self.assertEqual(replies, [{"id": self.job.name, "status": "cancelling"}])
        self.assert_cancelled_not_published()

    def test_commit_wins_race_delete_rejects_and_manifest_waits(self):
        self.module._status(self.job, "running", 20)
        staged = self.job / ".output-race.tmp"
        staged.write_bytes(b"synthetic completed media")
        entered, release, cancel_started, manifest_started = (threading.Event() for _ in range(4))
        original = self.module._support.atomic_json
        errors, cancelled, manifests = [], [], []
        def atomic(path, value):
            if Path(path) == self.job / "result.json":
                entered.set()
                if not release.wait(3):
                    raise RuntimeError("test did not release commit")
            return original(path, value)
        def commit():
            try:
                self.module._commit_result(self.job, staged, {"frames": 20}, 20)
            except BaseException as error:
                errors.append(error)
        def cancel():
            cancel_started.set()
            try:
                self.cancel_request()
            except self.module.HTTPException as error:
                cancelled.append(error.status_code)
        def manifest():
            manifest_started.set()
            try:
                manifests.append(asyncio.run(self.module.get_long_video_manifest(self.job.name)))
            except BaseException as error:
                errors.append(error)
        with patch.object(self.module._support, "atomic_json", side_effect=atomic), \
             patch.object(self.module, "_job_dir", return_value=self.job):
            publisher = threading.Thread(target=commit)
            publisher.start()
            self.assertTrue(entered.wait(3))
            deleter, reader = threading.Thread(target=cancel), threading.Thread(target=manifest)
            deleter.start()
            reader.start()
            self.assertTrue(cancel_started.wait(3) and manifest_started.wait(3))
            try:
                time.sleep(.03)
                self.assertTrue(deleter.is_alive() and reader.is_alive())
                self.assertFalse((self.job / "status.json").read_text().find('"completed"') >= 0)
            finally:
                release.set()
                for thread in (publisher, deleter, reader):
                    thread.join(3)
                    self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(cancelled, [409])
        self.assertEqual(manifests, [{"frames": 20}])
        self.assertFalse((self.job / "cancel").exists())
        self.assertEqual(json.loads((self.job / "status.json").read_text())["status"], "completed")

    def test_failed_commit_rolls_back_owned_files_and_remnants_stay_private(self):
        self.module._status(self.job, "running", 20)
        staged = self.job / ".output-failed.tmp"
        staged.write_bytes(b"synthetic media")
        with patch.object(self.module._support, "atomic_json", side_effect=OSError("synthetic metadata failure")), \
             self.assertRaisesRegex(OSError, "synthetic metadata failure"):
            self.module._commit_result(self.job, staged, {}, 20)
        self.assertFalse(staged.exists())
        self.assert_cancelled_not_published()
        # Even stale files left by an interrupted old process are not public
        # merely because their names exist.
        (self.job / "result.json").write_text('{"frames":20}')
        (self.job / "output.mp4").write_bytes(b"private remnant")
        self.module._status(self.job, "cancelled", 20)
        with patch.object(self.module, "_job_dir", return_value=self.job):
            for api in (self.module.get_long_video_manifest, self.module.download_long_video):
                with self.assertRaises(self.module.HTTPException):
                    asyncio.run(api(self.job.name))

    def test_model_failure_does_not_publish_media(self):
        def failed(*args):
            raise RuntimeError("synthetic model failure")
        with self.assertRaisesRegex(RuntimeError, "synthetic model failure"):
            self.invoke(self.module._support.options(), restore=failed)
        self.assertFalse((self.job / "result.json").exists())

    def test_media_validation_failure_does_not_publish(self):
        # Too many model frames must fail the final frame-count contract.
        def wrong(*args):
            return self.fake_restore(*args) + [self.np.zeros((48, 80, 3), dtype=self.np.uint8)]
        with self.assertRaises(self.module.JobError):
            self.invoke(self.module._support.options(), restore=wrong)
        self.assertFalse((self.job / "result.json").exists())

    def test_lossless_http_window_identity_and_response_close(self):
        self.source(5)
        with self.av.open(str(self.job / "input.mp4")) as source:
            frames = list(source.decode(video=0))
        config = self.module._support.options()
        metrics = self.module._support.ServerMetrics(self.job, config, 80, 48, 5, 17)
        measurement = metrics.window(0, 0, 5)
        measurement["measurement_mode"] = "basic"
        response = types.SimpleNamespace(content=(self.job / "input.mp4").read_bytes(),
                                         raise_for_status=lambda: None, close=lambda: None)
        with patch.object(self.module.requests, "post", return_value=response) as post, \
             patch.object(response, "close") as close:
            restored = self.module._restore(frames, 80, 48, 7723, "lab", 8000, "", None, measurement, metrics)
            self.assertEqual(len(restored), 5)
            sent = json.loads(post.call_args.kwargs["data"]["extra_params"])
            self.assertEqual(sent["seedvr2_measurement"]["model_request_id"], measurement["model_request_id"])
            self.assertEqual(sent["video_codec_options"], {"preset": "ultrafast", "qp": "0"})
            close.assert_called_once()

    def api_call(self, **overrides):
        arguments = {"raw_request": types.SimpleNamespace(
            app=types.SimpleNamespace(state=types.SimpleNamespace(api_server_count=1, seedvr2_long_port=8000)), headers={}),
            "input_references": None, "num_frames": 20, "size": "80x48", "loop_input": False,
            "prompt": " ", "seed": 7723, "extra_params": '{"color_correction_method":"lab"}',
            "encoding_mode": "single_encode", "delivery_size": "1800x1080", "window_frames": 17,
            "measurement_mode": "off", "measurement_context": '{"case_id":"C08"}'}
        arguments.update(overrides)
        capture = []
        async def action():
            result = await self.module.create_long_video(**arguments)
            await self.module.active_task
            return result
        with patch.object(self.module, "_window", return_value=65), \
             patch.object(self.module, "_jobs_root", return_value=self.root / "jobs"), \
             patch.object(self.module, "_sweep_expired_jobs", return_value=None), \
             patch.object(self.module, "_background", side_effect=lambda *args: capture.append(args)):
            response = asyncio.run(action())
        return response, capture

    def test_api_form_threads_options_into_background_job(self):
        self.source(20)
        upload = self.module.UploadFile(io.BytesIO((self.job / "input.mp4").read_bytes()), filename="source.mp4")
        response, capture = self.api_call(input_references=upload, window_frames=65)
        self.assertEqual(len(capture), 1)
        passed = capture[0][-1]
        self.assertEqual(passed["window_frames"], 65)
        self.assertEqual(passed["encoding_mode"], "single_encode")
        self.assertEqual(passed["delivery_size"], (1800, 1080))
        self.assertEqual(passed["measurement_context"]["case_id"], "C08")
        self.assertEqual(response["sr_request_id"], response["id"])
        saved = json.loads((self.root / "jobs" / response["id"] / "optimization.json").read_text())
        self.assertEqual(saved["window_frames"], 65)

    def test_api_rejects_source_ratio_without_starting_model(self):
        self.source(20)
        upload = self.module.UploadFile(io.BytesIO((self.job / "input.mp4").read_bytes()), filename="source.mp4")
        with self.assertRaises(self.module.HTTPException) as failure:
            self.api_call(input_references=upload, delivery_size="1920x1080")
        self.assertEqual(failure.exception.status_code, 400)
        self.assertEqual(list((self.root / "jobs").iterdir()), [])

    def test_api_rejects_insufficient_explicit_window_before_upload(self):
        arguments = {"raw_request": types.SimpleNamespace(app=types.SimpleNamespace(
            state=types.SimpleNamespace(api_server_count=1))), "input_references": None,
            "num_frames": 20, "size": "80x48", "loop_input": False, "prompt": " ", "seed": 7723,
            "extra_params": None, "encoding_mode": "legacy", "delivery_size": None,
            "window_frames": 65, "measurement_mode": "off", "measurement_context": None}
        with patch.object(self.module, "_window", return_value=17), self.assertRaises(self.module.HTTPException) as failure:
            asyncio.run(self.module.create_long_video(**arguments))
        self.assertEqual(failure.exception.status_code, 400)
        self.assertIn("no automatic fallback", str(failure.exception.detail))

    def test_metrics_missing_stage_and_d2h_cannot_pass_complete(self):
        with patch.dict(os.environ, {"SEEDVR2_METRICS_DIR": str(self.root / "metrics")}):
            support = self.module._support
            config = support.options(window_frames=17, measurement_mode="basic")
            metrics = support.ServerMetrics(self.job, config, 80, 48, 5, 17)
            item = metrics.window(0, 0, 5)
            for rank in range(8):
                support.atomic_json(metrics.root / "windows/0" / f"rank-{rank}.json",
                    {**item, "rank": rank, "world_size": 8, "status": "PASS"})
            metrics.finish("completed")
            bundle = support.metrics_bundle(self.job.name)
            self.assertEqual(bundle["metrics_status"], "METRICS_INCOMPLETE")
            self.assertEqual(len(bundle["inconsistent_rank_records"]), 9)


if __name__ == "__main__":
    unittest.main()
