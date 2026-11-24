"""Validate delivered timelines, full decode, audio endpoints and x264 settings."""
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_new(path, document):
    with Path(path).open("x", encoding="utf-8") as handle:
        if isinstance(document, str):
            handle.write(document)
        else:
            json.dump(document, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def finite_number(value, label, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0),
            label + " must be a finite " + ("positive" if positive else "nonnegative") + " number")
    return float(value)


def run_tool(argv, label, directory, timeout):
    write_new(directory / (label + ".argv.json"), argv)
    try:
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                encoding="utf-8", errors="replace", timeout=timeout, check=False)
    except subprocess.TimeoutExpired as error:
        for name, value in (("stdout", error.stdout), ("stderr", error.stderr)):
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            write_new(directory / (label + "." + name + ".txt"), value or "")
        raise RuntimeError(label + " timed out; preserve intent, no automatic retry") from error
    write_new(directory / (label + ".stdout.txt"), result.stdout)
    write_new(directory / (label + ".stderr.txt"), result.stderr)
    require(result.returncode == 0, label + " failed with exit " + str(result.returncode))
    return result.stdout


def media_probe(video, selector, spec, directory, *, expected_size=(2560, 1440)):
    ffprobe = str(Path(spec["tools"]["ffprobe"]["path"]).resolve())
    ffmpeg = str(Path(spec["tools"]["ffmpeg"]["path"]).resolve())
    timeout = spec["command_timeout_seconds"]
    meta = json.loads(run_tool([ffprobe, "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(video)],
                               "ffprobe-streams", directory, timeout))
    videos = [stream for stream in meta.get("streams", []) if stream.get("codec_type") == "video"]
    audios = [stream for stream in meta.get("streams", []) if stream.get("codec_type") == "audio"]
    require(len(videos) == 1, "Expected exactly one video stream")
    stream = videos[0]
    require((stream.get("width"), stream.get("height")) == expected_size,
            f"Expected decoded media at {expected_size[0]}x{expected_size[1]}")
    for name in ("avg_frame_rate", "r_frame_rate"):
        try:
            rate = Fraction(stream.get(name, "0/1"))
        except (ValueError, TypeError, ZeroDivisionError) as error:
            raise ValueError("Actual media frame rate is unknown") from error
        require(rate == 24, "Actual media frame rate differs from24fps")
    require(int(stream.get("nb_read_frames", -1)) == selector["num_frames"], "Actual decoded frame count differs")
    frames = json.loads(run_tool([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_frames", "-show_entries",
                                  "frame=best_effort_timestamp_time,pkt_duration_time,duration_time", "-of", "json", str(video)],
                                 "ffprobe-video-timeline", directory, timeout)).get("frames", [])
    count = selector["num_frames"]
    require(len(frames) == count, "Video frame timestamp inventory incomplete")
    tolerance = spec["tolerances"]["timestamp_seconds"]
    pts = []
    for index, frame in enumerate(frames):
        value = float(frame.get("best_effort_timestamp_time", "nan"))
        require(math.isfinite(value) and abs(value - index / 24) <= tolerance, "Video frame timestamp/tail differs")
        pts.append(value)
    tail_duration = frames[-1].get("duration_time", frames[-1].get("pkt_duration_time"))
    require(tail_duration is not None and abs(finite_number(float(tail_duration), "Tail frame duration", positive=True) - 1 / 24) <= tolerance,
            "Actual final frame duration is absent or differs")
    video_end = pts[-1] + float(tail_duration)
    require(abs(video_end - count / 24) <= tolerance, "Final video endpoint differs")
    policy = selector["audio_policy"]
    if policy == "required":
        require(len(audios) >= 1, "Required audio stream missing")
    if policy == "none":
        require(not audios, "Unexpected audio stream under no-audio contract")
    audio_checks = []
    for audio_index, audio in enumerate(audios):
        packets = json.loads(run_tool([ffprobe, "-v", "error", "-select_streams", "a:" + str(audio_index), "-show_packets", "-show_entries",
                                       "packet=pts_time,duration_time", "-of", "json", str(video)],
                                      "ffprobe-audio-" + str(audio_index), directory, timeout)).get("packets", [])
        require(packets, "Audio stream has no real packets")
        ends = []
        for packet in packets:
            pts_value = float(packet.get("pts_time", "nan"))
            duration = float(packet.get("duration_time", "nan"))
            require(math.isfinite(pts_value) and math.isfinite(duration) and duration > 0, "Audio packet endpoint is unknown")
            ends.append(pts_value + duration)
        audio_end = max(ends)
        require(abs(audio_end - video_end) <= spec["tolerances"]["audio_end_seconds"], "Audio/video endpoints differ")
        audio_checks.append({"stream_index": audio.get("index"), "packet_count": len(packets),
                             "end_seconds": audio_end, "difference_from_video_seconds": audio_end - video_end})
    tail_path = directory / "decoded-tail.framemd5"
    # Output1 decodes every selected video/audio frame to null; output2 records
    # decoded pixels of the last3 frames from that same input. -n forbids overwrite.
    run_tool([ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-xerror", "-n", "-hwaccel", "none", "-i", str(video),
              "-map", "0:v:0", "-map", "0:a?", "-f", "null", "-",
              "-map", "0:v:0", "-vf", "select=gte(n\\," + str(count - 3) + ")", "-an", "-fps_mode", "passthrough",
              "-f", "framemd5", str(tail_path)], "ffmpeg-full-decode-and-tail", directory, timeout)
    require(tail_path.is_file(), "Decoded tail checksum file missing")
    tail_rows = [line for line in tail_path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    require(len(tail_rows) == 3 and all(re.fullmatch("[0-9a-f]{32}", line.rsplit(",", 1)[-1].strip()) for line in tail_rows),
            "Decoded final three frames are incomplete")
    return {"status": "PASS_SPEC_FULL_DECODE_AND_TAIL", "num_frames": count, "width": stream["width"], "height": stream["height"],
            "fps": 24, "video_end_seconds": video_end, "tail_pts_seconds": pts[-3:],
            "decoded_tail_md5": [line.rsplit(",", 1)[-1].strip() for line in tail_rows],
            "audio": audio_checks, "audio_status": "PASS_ENDPOINT" if audios else ("NOT_APPLICABLE" if policy == "none" else "PENDING_NO_AUDIO_PRESENT"),
            "quality_status": "UNREVIEWED"}


_X264_UUID = bytes.fromhex("dc45e9bde6d948b7962cd820d923eeef")
_SLOW_SETTINGS = {"ref": "5", "subme": "8", "trellis": "2", "rc_lookahead": "50"}


def _sei_number(rbsp, offset):
    end = offset
    while end < len(rbsp) and rbsp[end] == 255:
        end += 1
    if end >= len(rbsp):
        raise ValueError("Truncated H.264 SEI header")
    return 255 * (end - offset) + rbsp[end], end + 1


def _x264_messages(annex_b):
    for nal in re.split(b"\x00\x00\x00?\x01", annex_b):
        if not nal or nal[0] != 6:
            continue
        rbsp = re.sub(b"\x00\x00\x03(?=[\x00-\x03])", b"\x00\x00", nal[1:])
        offset = 0
        while offset < len(rbsp) and rbsp[offset:] != b"\x80":
            kind, offset = _sei_number(rbsp, offset)
            size, offset = _sei_number(rbsp, offset)
            if size > len(rbsp) - offset:
                raise ValueError("Truncated H.264 SEI payload")
            payload = rbsp[offset:offset + size]
            offset += size
            if kind == 5 and payload.startswith(_X264_UUID):
                yield payload[len(_X264_UUID):].rstrip(b"\x00").decode("ascii")


def verify_delivery_video(path: Path) -> dict:
    """Verify observed CRF and slow settings; raise ValueError on any mismatch."""
    result = subprocess.run(
        [shutil.which('ffmpeg') or 'ffmpeg', "-nostdin", "-v", "error", "-f", "mov", "-i", str(Path(path).resolve(strict=True)),
         "-map", "0:v:0", "-frames:v", "1", "-c:v", "copy", "-bsf:v",
         "h264_mp4toannexb,filter_units=pass_types=6", "-f", "h264", "pipe:1"],
        capture_output=True, timeout=120, check=False,
    )
    if result.returncode:
        raise ValueError("Could not demux H.264 SEI: " + result.stderr.decode(errors="replace").strip()[-800:])
    messages = list(_x264_messages(result.stdout))
    if len(messages) != 1:
        raise ValueError(f"Expected one x264 user-data SEI on the first video access unit, found {len(messages)}")
    match = re.fullmatch(r"x264 - core (\d+)[^\r\n]*? - options: (.+)", messages[0])
    if not match:
        raise ValueError("Missing x264 core/options in H.264 user-data SEI")
    options = dict(item.split("=", 1) for item in match[2].split() if "=" in item)
    crf = float(options.get("crf", "nan"))
    observed = {name: options.get(name) for name in _SLOW_SETTINGS}
    if options.get("rc") != "crf" or crf != 12 or observed != _SLOW_SETTINGS:
        raise ValueError(f"Expected x264 CRF 12 and slow settings, saw rc={options.get('rc')}, "
                         f"crf={crf}, settings={observed}")
    return {
        "status": "PASS_H264_CRF12_SLOW_SETTINGS",
        "requested": {"video_codec": "libx264", "crf": 12, "preset": "slow"},
        "observed": {"codec": "h264", "encoder": "x264", "encoder_core": int(match[1]),
                     "rc": "crf", "crf": crf, "preset_evidence": observed},
    }
