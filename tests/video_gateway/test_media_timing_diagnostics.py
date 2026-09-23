# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic media timing must never change artifact inspection results."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dingo.video_gateway.adapters import minimax_h3


class _FakeAdapter:
    options = {"validate_media": True}

    def validate_artifact(self, path, normalized, diagnostic_timings=None):
        return {"container": "mp4"}

    def _validate_open_artifact(self, container, normalized, diagnostic_timings=None):
        return {"container": "mp4"}


class _FakeContainer:
    def __init__(self, *, with_audio):
        video = SimpleNamespace(
            width=256,
            height=256,
            average_rate=24,
            frames=124,
            codec_context=SimpleNamespace(
                name="h264", extradata=b"x", options={}, skip_frame="ALL"
            ),
        )
        audio = SimpleNamespace(
            duration=124,
            time_base=1 / 24,
            codec_context=SimpleNamespace(
                name="aac", extradata=b"x", options={}, skip_frame="ALL"
            ),
        )
        items = [video, audio] if with_audio else [video]
        self.streams = _FakeStreams(items)

    def close(self):
        pass


class _FakeStreams:
    def __init__(self, items):
        self.items = items
        self.video = items[:1]
        self.audio = items[1:]

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)


class MediaTimingDiagnosticTests(unittest.TestCase):
    def test_sound_request_with_validation_disabled_never_reads_media(self):
        adapter = _FakeAdapter()
        adapter.options = {"validate_media": False}
        path = Path("/tmp/video-01M36AFXE5GZQPEVRFQANDH7R9/result.mp4")
        with (
            patch.object(minimax_h3, "_MEDIA_TIMING_ENABLED", True),
            patch.object(adapter, "validate_artifact") as validate,
            patch.object(minimax_h3._LOGGER, "info") as logged,
        ):
            result = minimax_h3.MiniMaxH3VideoAdapter.inspect_artifact_for_publication(
                adapter, path, {"generate_sound": True}
            )
        self.assertEqual(result, (False, {"container": "mp4"}))
        validate.assert_not_called()
        payload = json.loads(logged.call_args.args[1])
        self.assertFalse(payload["validate_media"])
        self.assertNotIn("header_read_s", payload)

    def test_direct_validation_disabled_never_opens_media(self):
        adapter = _FakeAdapter()
        adapter.options = {"validate_media": False}
        path = Path("/tmp/video-01M36AFXE5GZQPEVRFQANDH7R9/result.mp4")
        with patch.object(Path, "open", side_effect=AssertionError("media read")):
            result = minimax_h3.MiniMaxH3VideoAdapter.validate_artifact(
                adapter, path, {"generate_sound": True}
            )
        self.assertEqual(result, {"container": "mp4"})

    def test_sound_request_preserves_result_when_logging_fails(self):
        adapter = _FakeAdapter()
        path = Path("/tmp/video-01M36AFXE5GZQPEVRFQANDH7R9/result.mp4")
        with (
            patch.object(minimax_h3, "_MEDIA_TIMING_ENABLED", True),
            patch.object(minimax_h3.json, "dumps", side_effect=RuntimeError("log failed")),
        ):
            result = minimax_h3.MiniMaxH3VideoAdapter.inspect_artifact_for_publication(
                adapter, path, {"generate_sound": True}
            )
        self.assertEqual(result, (False, {"container": "mp4"}))

    def test_silent_request_logs_fast_open_and_processing_decision(self):
        adapter = _FakeAdapter()
        fake_av = SimpleNamespace(
            open=lambda *_args, **_kwargs: _FakeContainer(with_audio=True),
            error=SimpleNamespace(FFmpegError=RuntimeError),
        )
        with tempfile.TemporaryDirectory() as directory:
            task_root = Path(directory) / "video-01M36AFXE5GZQPEVRFQANDH7R9"
            task_root.mkdir()
            path = task_root / "result.mp4"
            path.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 20)
            with (
                patch.dict(sys.modules, {"av": fake_av}),
                patch.object(minimax_h3, "_MEDIA_TIMING_ENABLED", True),
                patch.object(minimax_h3._LOGGER, "info") as logged,
            ):
                result = (
                    minimax_h3.MiniMaxH3VideoAdapter.inspect_artifact_for_publication(
                        adapter, path, {"generate_sound": False}
                    )
                )
        self.assertEqual(result, (True, None))
        payload = json.loads(logged.call_args.args[1])
        self.assertEqual(payload["task_id"], task_root.name)
        self.assertFalse(payload["generate_sound"])
        self.assertTrue(payload["needs_processing"])
        self.assertTrue(payload["indexed_fast_path"])
        self.assertIn("indexed_open_s", payload)
        self.assertIn("indexed_metadata_s", payload)
        self.assertIn("stream_topology_s", payload)


if __name__ == "__main__":
    unittest.main()
