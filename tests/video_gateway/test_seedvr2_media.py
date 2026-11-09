import copy
import json
from pathlib import Path

import pytest

from dingo.vllm.omni import seedvr2_media as media


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({}, True),
        ({"sample_aspect_ratio": "1:1"}, True),
        ({"sample_aspect_ratio": "0:1"}, True),
        ({"sample_aspect_ratio": "16:15"}, False),
        ({"codec_name": "hevc"}, False),
        ({"pix_fmt": "yuv420p10le"}, False),
        ({"side_data_list": [{"rotation": 90}]}, False),
        ({"tags": {"rotate": "180"}}, False),
    ],
)
async def test_probe_copy_compatibility_guard(monkeypatch, changes, expected):
    video = {
        "codec_type": "video",
        "codec_name": "h264",
        "pix_fmt": "yuv420p",
        "width": 1920,
        "height": 1088,
        "nb_read_frames": "124",
        "avg_frame_rate": "24/1",
    }
    video.update(changes)

    async def command(args):
        return json.dumps({"streams": [video]}).encode()

    monkeypatch.setattr(media, "command", command)
    monkeypatch.setattr(media, "file_sha256", lambda path: "digest")
    assert (await media.probe(Path("raw")))["copy_compatible"] is expected


@pytest.mark.parametrize(
    "compatible,same_size,audio",
    [
        (True, True, True),
        (True, True, False),
        (False, True, True),
        (True, False, True),
    ],
)
async def test_copy_only_for_compatible_exact_size(
    monkeypatch, compatible, same_size, audio
):
    original = {
        "width": 960,
        "height": 544,
        "frames": 124,
        "start_time": "0",
        "audio": {"codec": "aac", "packets": []} if audio else None,
    }
    raw = dict(
        original,
        width=1920 if same_size else 3840,
        height=1088 if same_size else 2176,
        copy_compatible=compatible,
    )
    result = dict(original, width=1920, height=1088)
    inputs = {"base": original, "raw": raw, "final": result}
    commands = []

    async def probe(path):
        return copy.deepcopy(inputs[str(path)])

    async def command(args):
        commands.append(args)
        return b""

    monkeypatch.setattr(media, "probe", probe)
    monkeypatch.setattr(media, "command", command)
    value = await media.finalize(
        Path("raw"), Path("base"), Path("final"), width=1920, height=1088
    )
    args = commands[0]
    expected = compatible and same_size
    assert args[args.index("-c:v") + 1] == ("copy" if expected else "libx264")
    assert ("-vf" not in args) == expected
    assert ("-an" not in args) == audio
    if audio:
        assert args[args.index("-c:a") + 1] == "copy"
    assert value["postprocess_mode"] == ("stream_copy" if expected else "encode")
    assert all(t >= 0 for t in value["postprocess_timings"].values())


@pytest.mark.parametrize(
    "field,value", [("frames", 123), ("audio", None), ("start_time", "1")]
)
async def test_copy_still_validates_final_output(monkeypatch, field, value):
    info = {
        "width": 1920,
        "height": 1088,
        "frames": 124,
        "audio": {"codec": "aac"},
        "start_time": "0",
        "copy_compatible": True,
    }

    async def probe(path):
        result = dict(info)
        if str(path) == "final":
            result[field] = value
        return result

    async def command(args):
        return b""

    monkeypatch.setattr(media, "probe", probe)
    monkeypatch.setattr(media, "command", command)
    with pytest.raises(ValueError):
        await media.finalize(
            Path("raw"), Path("base"), Path("final"), width=1920, height=1088
        )
