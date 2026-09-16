"""Fast media probing keeps validation and the full-decode fallback usable."""

from types import SimpleNamespace

import pytest

from dingo.video_gateway.adapters import create_adapter
from dingo.video_gateway.adapters.minimax_h3 import _open_artifact_metadata
from tests.video_gateway.test_minimax_h3_adapter import _write_h264_aac_mp4


@pytest.mark.parametrize("sound", [False, True])
def test_fast_probe_matches_original_and_restores_decoding(
    tmp_path, make_gateway_config, sound
):
    import av

    adapter = create_adapter(make_gateway_config().pools[0])
    adapter.options["validate_media"] = True
    path = tmp_path / "sample.mp4"
    _write_h264_aac_mp4(path, frames=124, width=256, height=256)
    normalized = dict(width=256, height=256, num_frames=124, generate_sound=sound)
    adapter.prepare_artifact(path, normalized)
    with av.open(str(path)) as original:
        expected = adapter._validate_open_artifact(original, normalized)
    assert adapter.validate_artifact(path, normalized) == expected
    with _open_artifact_metadata(path) as optimized:
        assert adapter._validate_open_artifact(optimized, normalized) == expected
        assert optimized.streams.video[0].codec_context.skip_frame == "DEFAULT"
        # The probe option must not leak into subsequent decoder initialization.
        assert sum(1 for _ in optimized.decode(video=0)) == 124


@pytest.mark.parametrize("missing", ["frames", "extradata", "rate", "width"])
def test_incomplete_probe_reopens_with_original_defaults(
    tmp_path, make_gateway_config, monkeypatch, missing
):
    import av

    adapter = create_adapter(make_gateway_config().pools[0])
    adapter.options["validate_media"] = True
    path = tmp_path / "sample.mp4"
    _write_h264_aac_mp4(path, frames=124, width=256, height=256)
    normalized = dict(width=256, height=256, num_frames=124, generate_sound=True)
    expected = adapter.validate_artifact(path, normalized)
    original_open = av.open
    calls = []

    def opened(*args, **kwargs):
        calls.append(kwargs)
        container = original_open(*args, **kwargs)
        if not kwargs.get("options"):
            return container
        video = container.streams.video[0]
        proxy = SimpleNamespace(
            width=video.width,
            height=video.height,
            frames=video.frames,
            average_rate=video.average_rate,
            codec_context=SimpleNamespace(
                name=video.codec_context.name, extradata=video.codec_context.extradata
            ),
        )
        if missing == "extradata":
            proxy.codec_context.extradata = b""
        else:
            setattr(proxy, {"rate": "average_rate"}.get(missing, missing), 0)

        class Streams:
            video = [proxy]
            audio = container.streams.audio

            def __len__(self):
                return len(container.streams)

        return SimpleNamespace(streams=Streams(), close=container.close)

    monkeypatch.setattr(av, "open", opened)
    assert adapter.validate_artifact(path, normalized) == expected
    assert calls == [{"options": {"skip_frame": "all"}}, {}]


def test_truncated_mp4_not_accepted_by_fast_probe(tmp_path, make_gateway_config):
    adapter = create_adapter(make_gateway_config().pools[0])
    adapter.options["validate_media"] = True
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isommp42")
    with pytest.raises(Exception):
        adapter.validate_artifact(
            path, dict(width=256, height=256, num_frames=124, generate_sound=False)
        )
