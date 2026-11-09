"""SR-only request admission and the shared production execution path on CPU."""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import gateway
from service import ownership
from test_gateway import HTTPFixture, data, request
import test_service_runtime as fixtures


def sr_request(**changes):
    value = {"operation": "sr_only", "sr_tier": "1080",
        "input_reference": data("video/mp4", b"UPLOADED"),
        "nvext": {"fps": 24, "num_frames": 158}}
    return {**value, **changes}


class SourceAdmission(unittest.TestCase):
    def test_actual_frame_range_is_independent_of_h3_duration(self):
        for count in (1, 158, 362):
            decoded, frames = gateway.decode_sr_source(sr_request(nvext={"fps": 24, "num_frames": count}))
            self.assertEqual(decoded, b"UPLOADED")
            self.assertEqual(frames, count)

    def test_one_embedded_mp4_and_actual_frames_are_required(self):
        invalid = [
            {"input_reference": "https://example.com/base.mp4"},
            {"input_reference": data("image/png")},
            {"input_reference": [data("video/mp4")]},
            {"input_references": [data("video/mp4")]},
            {"input_reference": "data:video/mp4;base64,?"},
            {"input_reference": "data:video/mp4;base64,"},
            {"nvext": {"fps": 30, "num_frames": 158}},
            {"nvext": {"fps": 24, "num_frames": True}},
            {"nvext": {"fps": 24, "num_frames": 0}},
            {"nvext": {"fps": 24, "num_frames": 363}},
            {"nvext": None},
        ]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                gateway.decode_sr_source(sr_request(**changes))

    def test_oversized_source_is_rejected_before_decoding(self):
        with mock.patch.object(gateway, "MAX_MEDIA_BYTES", 4), \
                mock.patch.object(gateway.base64, "b64decode") as decode:
            with self.assertRaisesRegex(ValueError, "128 MiB"):
                gateway.decode_sr_source(sr_request())
            decode.assert_not_called()

    def test_native_sr_admission_uses_actual_installed_options_and_source(self):
        distribution = mock.Mock()
        distribution.locate_file.return_value = ROOT / "tests/fixtures/sr-timing/long_video_support.py"
        with mock.patch.object(gateway.importlib.metadata, "distribution", return_value=distribution):
            plan = gateway.geometry(1280, 768, "1080")
            gateway.native_sr_admission(1280, 768, plan)
            with self.assertRaisesRegex(ValueError, "model size must preserve"):
                gateway.native_sr_admission(1280, 768, {**plan, "internal": (1856, 1104)})
            with self.assertRaisesRegex(ValueError, "short edge 1080 or 1440"):
                gateway.native_sr_admission(1280, 768, {**plan, "delivery": (2016, 1152)})


class SourceProbe(unittest.IsolatedAsyncioTestCase):
    def streams(self, **video_changes):
        video = {"codec_type": "video", "codec_name": "h264", "pix_fmt": "yuv420p",
            "width": 1280, "height": 768, "nb_read_frames": "158", "avg_frame_rate": "24/1"}
        return {"streams": [{**video, **video_changes}, {"codec_type": "audio", "codec_name": "aac"}],
            "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "tags": {"major_brand": "isom"}}}

    async def test_source_probe_preserves_audio_codec_for_copy_compatibility_admission(self):
        with mock.patch.object(gateway, "process", return_value=json.dumps(self.streams()).encode()) as process:
            result = await gateway.probe("source.mp4")
        self.assertEqual(result["audio_codec"], "aac")
        self.assertEqual(result["major_brand"], "isom")
        self.assertIn("mp4", result["format_name"].split(","))
        self.assertIn("-show_format", process.call_args.args[0])

    async def test_source_probe_rejects_other_video_codec_pixel_format_or_fps(self):
        for change in ({"codec_name": "hevc"}, {"pix_fmt": "yuv444p"}, {"avg_frame_rate": "30/1"}):
            with self.subTest(change=change), \
                    mock.patch.object(gateway, "process", return_value=json.dumps(self.streams(**change)).encode()):
                with self.assertRaises(RuntimeError):
                    await gateway.probe("source.mp4")


class SROnlyBridge(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "control"
        self.root.mkdir()
        self.executor = ownership.GatewayExecutor(self.root, profile="fl2va-sr",
            service_epoch=fixtures.EPOCH, ready_check=lambda: True, birth_reader=lambda: fixtures.BIRTH)
        self.http = HTTPFixture(self.executor)
        self.source_media = {"width": 1280, "height": 768, "frames": 158, "audio": True, "audio_codec": "aac",
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2", "major_brand": "isom"}
        self.process_calls = []
        self.admission_patch = mock.patch.object(gateway, "native_sr_admission")
        self.admit = self.admission_patch.start()
        self.addCleanup(self.admission_patch.stop)
        self.bridge = self.make_bridge()

    async def asyncTearDown(self):
        self.tmp.cleanup()

    def make_bridge(self, *, ref=False, stages=False):
        async def probe(path):
            if Path(path).name == "base.mp4":
                return dict(self.source_media)
            self.http.owned()
            width, height = map(int, self.http.latest_fields["delivery_size"].split("x"))
            return {**self.source_media, "width": width, "height": height}

        async def run(argv):
            if self.executor.owned is None:
                self.assertEqual(Path(argv[argv.index("-i") + 1]).name, "base.mp4")
                self.assertEqual(argv[-3:], ["-f", "null", "-"])
            self.process_calls.append(argv)
            if str(argv[-1]).endswith("final.mp4"):
                self.http.owned()
                Path(argv[-1]).write_bytes(b"FINAL")

        async def audio(path):
            self.http.owned()
            return [{"data_hash": "original", "pts_time": "0"}]

        context = {"image_digest": "registry.test/repo@sha256:" + "f" * 64,
            "source_hashes": {"source": "f" * 64}, "model_manifest_hash": "b" * 64}
        return gateway.GatewayBridge({"profile": "ref2va-sr" if ref else "fl2va-sr",
            "monitoring": {"mode": "stages" if stages else "off"}}, self.executor, http=self.http,
            probe_media=probe, run_process=run, probe_audio=audio, timing_context=context if stages else None)

    async def output(self, path, durations, timing=None):
        self.http.owned()
        return {"path": path.name, "durations": dict(durations), "timing": timing}

    async def test_ref_sr_only_skips_h3_and_handoff_and_records_completed_sr(self):
        handoff = mock.AsyncMock(side_effect=AssertionError("SR-only must not run H3 handoff"))
        self.executor = ownership.GatewayExecutor(self.root, profile="ref2va-sr",
            service_epoch=fixtures.EPOCH, handoff=handoff, ready_check=lambda: True,
            birth_reader=lambda: fixtures.BIRTH)
        self.http.executor = self.executor
        self.bridge = self.make_bridge(ref=True)
        with mock.patch.object(gateway, "native_h3_request", side_effect=AssertionError("H3 parser must be skipped")):
            result = await self.bridge.execute(sr_request(), "sr-old-base", self.output)
        self.assertEqual(result["path"], "final.mp4")
        self.assertFalse(any(call[1] == 8123 for call in self.http.calls))
        handoff.assert_not_called()
        self.assertEqual(json.loads((self.root / "history.json").read_text()), {"h3_completed": 0, "last_business": "sr"})
        receipt = json.loads((self.root / "actions/sr-old-base.json").read_text())
        self.assertEqual(receipt["phases"], ["sr", "remux"])
        folder = self.root.parent / "requests/sr-old-base"
        source = json.loads((folder / "source.json").read_text())
        self.assertEqual(source["sha256"], hashlib.sha256(b"UPLOADED").hexdigest())
        self.assertTrue(source["full_decode"])
        self.assertTrue((folder / "sr-job.json").is_file())
        self.assertTrue((folder / "sr-manifest.json").is_file())
        self.assertEqual(self.http.submits, 1)
        self.admit.assert_called_once_with(1280, 768, gateway.geometry(1280, 768, "1080"))

    async def test_no_audio_source_remuxes_without_an_audio_track(self):
        self.source_media.update(audio=False, audio_codec=None)
        await self.bridge.execute(sr_request(), "silent", self.output)
        remux = next(argv for argv in self.process_calls if str(argv[-1]).endswith("final.mp4"))
        self.assertIn("-an", remux)
        self.assertNotIn("-c:a", remux)

    async def test_sr_only_1440_uses_its_own_canvas_and_window(self):
        await self.bridge.execute(sr_request(sr_tier="1440"), "sr-1440", self.output)
        self.assertEqual(self.http.latest_fields["size"], "2400x1440")
        self.assertEqual(self.http.latest_fields["delivery_size"], "2400x1440")
        self.assertEqual(self.http.latest_fields["window_frames"], 65)
        self.assertEqual(self.http.latest_fields["num_frames"], 158)

    async def test_source_mismatch_is_rejected_before_gpu_permit(self):
        for changes in ({"height": 720}, {"frames": 159}, {"audio_codec": "mp3"}):
            self.source_media.update(width=1280, height=768, frames=158, audio=True, audio_codec="aac")
            self.source_media.update(changes)
            with mock.patch.object(self.executor, "execute_owned") as execute:
                with self.assertRaisesRegex(ValueError, "768-pixel short edge"):
                    await self.bridge.execute(sr_request(), "reject-" + next(iter(changes)), self.output)
                execute.assert_not_called()
        self.assertEqual(self.http.calls, [])
        self.assertFalse((self.root / "blocked.json").exists())

    async def test_other_containers_disguised_as_mp4_are_rejected_before_permit(self):
        for index, changes in enumerate(({"format_name": "mpegts"}, {"format_name": "matroska,webm"},
                {"major_brand": "qt"}, {"major_brand": "3gp4"}, {"major_brand": ""})):
            self.source_media.update(format_name="mov,mp4,m4a,3gp,3g2,mj2", major_brand="isom")
            self.source_media.update(changes)
            with mock.patch.object(self.executor, "execute_owned") as execute:
                with self.assertRaisesRegex(ValueError, "ISO MP4 container"):
                    await self.bridge.execute(sr_request(), "other-container-" + str(index), self.output)
                execute.assert_not_called()
        self.assertEqual(self.http.calls, [])

    async def test_native_rejection_precedes_gpu_submission(self):
        self.admit.side_effect = ValueError("native source rejection")
        with mock.patch.object(self.executor, "execute_owned") as execute:
            with self.assertRaisesRegex(ValueError, "native source rejection"):
                await self.bridge.execute(sr_request(), "native-rejected", self.output)
            execute.assert_not_called()
        self.assertEqual(self.http.calls, [])

    async def test_corrupt_source_decode_is_rejected_before_gpu_permit(self):
        self.bridge.process = mock.AsyncMock(side_effect=RuntimeError("source decode failed"))
        with mock.patch.object(self.executor, "execute_owned") as execute:
            with self.assertRaisesRegex(RuntimeError, "source decode failed"):
                await self.bridge.execute(sr_request(), "corrupt-source", self.output)
            execute.assert_not_called()
        self.assertEqual(self.http.calls, [])

    async def test_unknown_operation_and_missing_tier_fail_before_h3_or_gpu(self):
        for index, payload in enumerate((request(operation="unknown"), sr_request(sr_tier=None))):
            with mock.patch.object(gateway, "native_h3_request") as h3, \
                    mock.patch.object(self.executor, "execute_owned") as execute:
                with self.assertRaises(ValueError):
                    await self.bridge.execute(payload, "bad-operation-" + str(index), self.output)
                h3.assert_not_called()
                execute.assert_not_called()

    async def test_explicit_generate_preserves_ordinary_h3_behavior(self):
        result = await self.bridge.execute(request(operation="generate"), "explicit-generate", self.output)
        self.assertEqual(result["path"], "base.mp4")
        self.assertTrue(any(call[1] == 8123 for call in self.http.calls))
        self.assertEqual(self.http.submits, 0)

    async def test_sr_only_keeps_permit_until_async_output_and_writer_finish(self):
        self.http.pause_status = True
        task = asyncio.create_task(self.bridge.execute(sr_request(), "async-sr", self.output))
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

    async def test_sr_only_output_failure_retains_existing_failure_latch(self):
        self.http.bad_manifest = True
        with self.assertRaisesRegex(RuntimeError, "content differs"):
            await self.bridge.execute(sr_request(), "bad-output", self.output)
        self.assertTrue((self.root / "blocked.json").exists())
        self.assertEqual(self.http.submits, 1)

    async def test_sr_only_stage_mode_records_sr_without_h3_timing(self):
        self.bridge = self.make_bridge(stages=True)
        result = await self.bridge.execute(sr_request(), "timed-sr", self.output)
        self.assertIn("sr", result["timing"])
        self.assertNotIn("h3", result["timing"])
        self.assertIn("sr_job_output_e2e_s", result["durations"])
        self.assertNotIn("h3_loopback_and_media_s", result["durations"])


if __name__ == "__main__":
    unittest.main()
