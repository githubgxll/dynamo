# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only bounded media verification and exact-ratio finalization."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import signal
import time
from fractions import Fraction
from pathlib import Path

MAX_BYTES = 128 * 1024**2
PROBE_VIDEO_THREADS = int(os.environ.get("DINGO_VIDEO_SR_PROBE_THREADS", "16"))
if not 1 <= PROBE_VIDEO_THREADS <= 32:
    raise ValueError("DINGO_VIDEO_SR_PROBE_THREADS must be between 1 and 32")


def file_sha256(path: Path) -> str:
    if (
        path.is_symlink()
        or not path.is_file()
        or not 0 < path.stat().st_size <= MAX_BYTES
    ):
        raise ValueError("invalid or oversized SR media file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def command(argv: list[str], timeout_s: float = 300) -> bytes:
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        async with asyncio.timeout(timeout_s):
            stdout, stderr = await proc.communicate()
        if proc.returncode:
            raise RuntimeError(
                f"media command failed: {stderr[-2048:].decode(errors='replace')}"
            )
        if len(stdout) > 8 * 1024**2:
            raise RuntimeError("media metadata exceeds limit")
        return stdout
    finally:
        if proc.returncode is None:
            # This group contains only the child launched above; never kill by
            # name, namespace, shared port, or unrelated process enumeration.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await proc.wait()


async def probe(path: Path) -> dict:
    mark = time.perf_counter()
    digest = await asyncio.to_thread(file_sha256, path)
    timings = {"hash_s": time.perf_counter() - mark}
    mark = time.perf_counter()
    raw = await command(
        [
            "ffprobe",
            "-v",
            "error",
            "-threads:v:0",
            str(PROBE_VIDEO_THREADS),
            "-count_frames",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ]
    )
    streams = json.loads(raw)["streams"]
    timings["video_scan_s"] = time.perf_counter() - mark
    videos = [s for s in streams if s["codec_type"] == "video"]
    audios = [s for s in streams if s["codec_type"] == "audio"]
    if len(videos) != 1 or len(audios) > 1:
        raise ValueError("SR media requires one video and at most one audio stream")
    video = videos[0]
    frames = int(video["nb_read_frames"])
    if not 0 < frames <= 7200 or Fraction(video["avg_frame_rate"]) != 24:
        raise ValueError("SeedVR2 long-video profile requires 24 FPS")
    # Hash compressed audio packets, not a re-encoded waveform. Capture timing
    # too, so a mux that shifts sound is not mistaken for preservation.
    audio = None
    mark = time.perf_counter()
    if audios:
        packets = json.loads(
            await command(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_packets",
                    "-show_data_hash",
                    "sha256",
                    "-show_entries",
                    "packet=pts_time,dts_time,duration_time,data_hash",
                    "-of",
                    "json",
                    str(path),
                ]
            )
        )["packets"]
        audio = {"codec": audios[0]["codec_name"], "packets": packets}
    timings["audio_scan_s"] = time.perf_counter() - mark if audios else 0.0
    return {
        "width": video["width"],
        "height": video["height"],
        "frames": frames,
        "sha256": digest,
        "audio": audio,
        "start_time": video.get("start_time", "0"),
        "probe_timings": timings,
        "copy_compatible": (
            video.get("codec_name") == "h264"
            and video.get("pix_fmt") == "yuv420p"
            and video.get("sample_aspect_ratio") in {None, "0:1", "1:1"}
            and str(video.get("tags", {}).get("rotate", "0")) == "0"
            and not any(
                s.get("rotation", 0) != 0 for s in video.get("side_data_list", [])
            )
        ),
    }


async def _verified_probe(path: Path, cached: dict | None) -> dict:
    """Reuse an in-process full probe only while the file's bytes still match."""
    if cached is None:
        return await probe(path)
    started = time.perf_counter()
    digest = await asyncio.to_thread(file_sha256, path)
    if digest != cached.get("sha256"):
        raise ValueError("SR media changed after its complete probe")
    return dict(cached, probe_timings={
        "hash_s": time.perf_counter() - started,
        "video_scan_s": 0.0, "audio_scan_s": 0.0,
    })


async def finalize(
    raw: Path, base: Path, output: Path, *, width: int, height: int,
    require_stream_copy: bool = False,
    expected_base_size: tuple[int, int] | None = None,
    base_info: dict | None = None, raw_info: dict | None = None,
) -> dict:
    started = time.perf_counter()
    if base_info is None and raw_info is None:
        original, restored = await probe(base), await probe(raw)
    else:
        original, restored = await asyncio.gather(
            _verified_probe(base, base_info), _verified_probe(raw, raw_info)
        )
    timings = {"input_probe_s": time.perf_counter() - started}
    if (
        restored["frames"] != original["frames"]
        or restored["width"] * height != restored["height"] * width
        or ((original["width"],original["height"]) != expected_base_size if expected_base_size is not None else original["width"] * height != original["height"] * width)
    ):
        raise ValueError("SR changed frame count or exact aspect ratio")
    args = [
        "ffmpeg",
        "-nostdin",
        "-n",
        "-v",
        "error",
        "-xerror",
        "-copyts",
        "-noautorotate",
        "-i",
        str(raw),
        "-noautorotate",
        "-i",
        str(base),
        "-map",
        "0:v:0",
    ]
    args += ["-map", "1:a:0", "-c:a", "copy"] if original["audio"] else ["-an"]
    copy_video = (restored["width"], restored["height"]) == (
        width,
        height,
    ) and restored.get("copy_compatible") is True
    if require_stream_copy and not copy_video:
        raise ValueError("single_encode SR output must support stream copy at delivery size")
    if copy_video:
        args += ["-c:v", "copy"]
    else:
        args += [
            "-vf",
            f"scale={width}:{height}:flags=lanczos,setsar=1",
            "-c:v",
            "libx264",
            "-threads",
            "4",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
        ]
    args += [
        "-fps_mode",
        "passthrough",
        "-avoid_negative_ts",
        "disabled",
        "-map_metadata",
        "1",
        "-movflags",
        "+faststart",
        str(output),
    ]
    mark = time.perf_counter()
    await command(args)
    timings["mux_or_encode_s"] = time.perf_counter() - mark
    if base_info is not None or raw_info is not None:
        mark = time.perf_counter()
        base_digest, raw_digest = await asyncio.gather(
            asyncio.to_thread(file_sha256, base),
            asyncio.to_thread(file_sha256, raw),
        )
        if base_digest != original["sha256"] or raw_digest != restored["sha256"]:
            raise ValueError("SR input media changed during finalization")
        timings["input_recheck_s"] = time.perf_counter() - mark
    mark = time.perf_counter()
    result = await probe(output)
    timings["output_probe_s"] = time.perf_counter() - mark
    if (result["width"], result["height"], result["frames"]) != (
        width,
        height,
        original["frames"],
    ):
        raise ValueError("final video geometry/frame count mismatch")
    if (
        result["audio"] != original["audio"]
        or result["start_time"] != original["start_time"]
    ):
        raise ValueError("final video changed original audio or start time")
    result["postprocess_mode"] = "stream_copy" if copy_video else "encode"
    for label, info in (("base", original), ("raw", restored), ("final", result)):
        for key, value in info.get("probe_timings", {}).items():
            timings[f"{label}_{key}"] = value
    timings["total_s"] = time.perf_counter() - started
    result["postprocess_timings"] = timings
    return result
