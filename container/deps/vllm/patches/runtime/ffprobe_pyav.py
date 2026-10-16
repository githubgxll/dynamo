#!/usr/bin/env python3
"""Minimal ffprobe JSON compatibility for MiniMax-H3 reference media.

The pinned Dingo image contains PyAV but not the ffprobe executable.  This
implements only the stream/format fields requested by vLLM-Omni's
``reference_video.py`` and intentionally is not a general ffprobe replacement.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import av


def _seconds(value: Any, time_base: Any) -> float | None:
    if value is None or time_base is None:
        return None
    return float(value * time_base)


def probe_document(path: str) -> dict[str, Any]:
    """Return the supported ffprobe-style document for one media file."""
    with av.open(path) as container:
        streams: list[dict[str, Any]] = []
        for stream in container.streams:
            item: dict[str, Any] = {
                "codec_type": stream.type,
                "codec_name": str(stream.codec_context.name or "").lower(),
            }
            duration = _seconds(stream.duration, stream.time_base)
            if duration is not None:
                item["duration"] = f"{duration:.9f}"
            if stream.type == "video":
                rate = stream.average_rate or stream.base_rate or stream.guessed_rate
                if rate is not None:
                    numerator = getattr(rate, "numerator", None)
                    denominator = getattr(rate, "denominator", None)
                    if numerator is None or denominator is None:
                        item["r_frame_rate"] = f"{int(rate)}/1"
                    else:
                        item["r_frame_rate"] = f"{numerator}/{denominator}"
                item["width"] = int(stream.width)
                item["height"] = int(stream.height)
                item["sample_aspect_ratio"] = str(
                    stream.sample_aspect_ratio or "1:1"
                )
                # Prefer the container-provided frame count.  Prepared
                # MiniMax-H3 reference videos are emitted by FFmpeg with a
                # reliable ``nb_frames`` value, so decoding the complete
                # high-bitrate lossless RGB stream just to count it is pure
                # overhead.  Retain a decode fallback for unusual inputs
                # whose container omits the count.
                frame_count = int(stream.frames or 0)
                if frame_count <= 0:
                    frame_count = sum(1 for _ in container.decode(stream))
                    item["nb_read_frames"] = str(frame_count)
                item["nb_frames"] = str(frame_count)
            streams.append(item)

        container_duration = (
            float(container.duration) / 1_000_000.0
            if container.duration is not None
            else None
        )
        format_info: dict[str, Any] = {
            "format_name": str(container.format.name or ""),
            "size": str(os.path.getsize(path)),
        }
        if container_duration is not None:
            format_info["duration"] = f"{container_duration:.9f}"
        return {"streams": streams, "format": format_info}


def probe_video_metadata(path: str) -> dict[str, Any]:
    """Return vLLM-Omni's internal MiniMax-H3 video metadata contract."""
    probe = probe_document(path)
    streams = probe["streams"]
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    if not videos:
        raise ValueError(f"media has no video stream: {path}")
    stream = videos[0]
    numerator, denominator = str(stream["r_frame_rate"]).split("/", 1)
    fps = float(numerator) / float(denominator)
    raw_duration = stream.get("duration") or probe["format"].get("duration")
    frame_count = int(stream.get("nb_read_frames") or stream["nb_frames"])
    duration = float(raw_duration) if raw_duration is not None else frame_count / fps
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "format_names": tuple(probe["format"]["format_name"].split(",")),
        "video_codec": str(stream.get("codec_name", "")).lower(),
        "audio_codecs": tuple(
            str(item.get("codec_name", "")).lower()
            for item in streams
            if item.get("codec_type") == "audio" and item.get("codec_name")
        ),
        "file_size": int(probe["format"]["size"]),
    }


def probe_audio_metadata(path: str) -> dict[str, Any]:
    """Return vLLM-Omni's internal MiniMax-H3 audio metadata contract."""
    probe = probe_document(path)
    streams = [
        stream for stream in probe["streams"] if stream.get("codec_type") == "audio"
    ]
    if not streams:
        raise ValueError(f"media has no audio stream: {path}")
    stream = streams[0]
    raw_duration = stream.get("duration") or probe["format"].get("duration")
    if raw_duration is None:
        raise ValueError(f"cannot determine audio duration: {path}")
    return {
        "duration": float(raw_duration),
        "format_names": tuple(probe["format"]["format_name"].split(",")),
        "codec": str(stream.get("codec_name", "")).lower(),
        "file_size": int(probe["format"]["size"]),
    }


def _main() -> int:
    if len(sys.argv) < 2:
        print("ffprobe_pyav: missing input path", file=sys.stderr)
        return 2
    json.dump(probe_document(sys.argv[-1]), sys.stdout)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
