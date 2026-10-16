from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dingo.common.video_encoding import VideoEncoder, frame_conversion_workers


@pytest.mark.parametrize("value", ["0", "17", "-1", "1.5", "true", ""])
def test_invalid_deployment_thread_count(monkeypatch, value):
    monkeypatch.setenv("DINGO_VIDEO_FRAME_CONVERSION_WORKERS", value)
    with pytest.raises(ValueError):
        frame_conversion_workers()


def test_default_and_valid_deployment_thread_counts(monkeypatch):
    monkeypatch.delenv("DINGO_VIDEO_FRAME_CONVERSION_WORKERS", raising=False)
    assert frame_conversion_workers() == 8
    monkeypatch.setenv("DINGO_VIDEO_FRAME_CONVERSION_WORKERS", "2")
    assert frame_conversion_workers() == 2


@pytest.mark.parametrize("workers", [True, 0, 17, 2.5])
def test_no_implicit_thread_count_coercion(workers):
    with pytest.raises(ValueError):
        VideoEncoder(workers)


@pytest.mark.parametrize("enabled", [False, True])
def test_default_and_old_encoder_preserve_call(enabled):
    def encoder(video, fps, audio):
        assert video == "frames" and fps == 24 and audio == "sound"
        return b"mp4"

    factory = Mock(side_effect=AssertionError("old encoder must not create converter"))
    api = SimpleNamespace(_encode_video_bytes=encoder, _PlanarFrameConverter=factory)
    shared = VideoEncoder(2 if enabled else 1)
    assert shared.encode(api, "frames", fps=24, audio="sound") == b"mp4"
    shared.close()
    factory.assert_not_called()


def test_missing_converter_keeps_legacy_encoder():
    encoder = Mock(return_value=b"mp4")
    api = SimpleNamespace(_encode_video_bytes=encoder)
    shared = VideoEncoder(2)
    assert shared.encode(api, "frames", fps=24) == b"mp4"
    shared.close()
    encoder.assert_called_once_with("frames", fps=24)


@pytest.mark.parametrize("failure", [False, True])
def test_bounded_converter_is_closed_on_success_and_error(failure):
    converter = SimpleNamespace(shutdown=Mock())
    factory = Mock(return_value=converter)
    calls = []

    def encoder(video, fps, audio, video_codec_options, frame_converter=None):
        calls.append(video)
        assert frame_converter is converter and fps == 24 and audio == "sound"
        assert video_codec_options == {"preset": "ultrafast", "threads": "0"}
        if failure:
            raise RuntimeError("encoder failed")
        return b"mp4"

    api = SimpleNamespace(_encode_video_bytes=encoder, _PlanarFrameConverter=factory)
    shared = VideoEncoder(2)

    def run():
        return shared.encode(
            api,
            "frames",
            fps=24,
            audio="sound",
            video_codec_options={"preset": "ultrafast", "threads": "0"},
        )

    if failure:
        with pytest.raises(RuntimeError, match="encoder failed"):
            run()
    else:
        assert run() == b"mp4"
    factory.assert_called_once_with(max_workers=2)
    converter.shutdown.assert_not_called()
    shared.close()
    converter.shutdown.assert_called_once_with()
    assert calls == ["frames"]


def test_native_converter_is_reused_until_shutdown():
    converter = SimpleNamespace(shutdown=Mock())
    factory = Mock(return_value=converter)

    def encode(video, *, frame_converter=None):
        assert frame_converter is converter
        return video

    api = SimpleNamespace(_encode_video_bytes=encode, _PlanarFrameConverter=factory)
    encoder = VideoEncoder(8)
    assert encoder.encode(api, b"first") == b"first"
    assert encoder.encode(api, b"second") == b"second"
    factory.assert_called_once_with(max_workers=8)
    converter.shutdown.assert_not_called()
    encoder.close()
    encoder.close()
    converter.shutdown.assert_called_once_with()
    with pytest.raises(RuntimeError, match="closed"):
        encoder.encode(api, b"third")


def test_native_encoder_failure_does_not_discard_pool():
    converter = SimpleNamespace(shutdown=Mock())
    factory = Mock(return_value=converter)

    def encode(video, *, frame_converter=None):
        if video == b"bad":
            raise ValueError("bad frame")
        return video

    api = SimpleNamespace(_encode_video_bytes=encode, _PlanarFrameConverter=factory)
    encoder = VideoEncoder(8)
    with pytest.raises(ValueError, match="bad frame"):
        encoder.encode(api, b"bad")
    assert encoder.encode(api, b"good") == b"good"
    factory.assert_called_once()
    converter.shutdown.assert_not_called()
    encoder.close()
    converter.shutdown.assert_called_once()


def test_native_shutdown_waits_for_inflight_encode():
    import threading
    from concurrent.futures import ThreadPoolExecutor

    started = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    converter = SimpleNamespace(shutdown=Mock())

    def encode(video, *, frame_converter=None):
        started.set()
        assert release.wait(5)
        return video

    api = SimpleNamespace(
        _encode_video_bytes=encode, _PlanarFrameConverter=Mock(return_value=converter)
    )
    encoder = VideoEncoder(8)

    def close():
        encoder.close()
        closed.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        result = pool.submit(encoder.encode, api, b"video")
        try:
            assert started.wait(2)
            closing = pool.submit(close)
            assert not closed.wait(0.05)
            converter.shutdown.assert_not_called()
        finally:
            release.set()
        assert result.result(timeout=2) == b"video"
        closing.result(timeout=2)
    converter.shutdown.assert_called_once()
