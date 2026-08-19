"""CUDA-neutral backport of vLLM-Omni PR #6064 for MiniMax-H3 Ref2VA.

The pinned image predates the upstream optimization and has no system
``ffprobe``.  This module keeps the existing PyAV metadata compatibility but
avoids complete lossless-RGB scans when container metadata is available, and
uses one FFmpeg raw-video pipe for selective or full frame decoding.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any

import numpy as np

from ffprobe_pyav import probe_audio_metadata, probe_video_metadata


def _ffmpeg_executable() -> str:
    configured = os.getenv("IMAGEIO_FFMPEG_EXE", "").strip()
    if configured:
        if not os.path.isfile(configured) or not os.access(configured, os.X_OK):
            raise RuntimeError(
                "IMAGEIO_FFMPEG_EXE is not executable: " f"{configured!r}"
            )
        return configured
    return "ffmpeg"


def decode_video_frames_ffmpeg(
    path: str,
    *,
    frame_count: int,
    width: int,
    height: int,
    indices: list[int] | None = None,
) -> np.ndarray:
    """Decode exact RGB24 frames through one FFmpeg process."""
    from vllm_omni.errors import OmniClientError

    frame_count = int(frame_count)
    width = int(width)
    height = int(height)
    if frame_count <= 0:
        raise OmniClientError(f"video has no frames: {path}")
    if width <= 0 or height <= 0:
        raise OmniClientError(
            f"video has invalid dimensions {width}x{height}: {path}"
        )
    if indices is not None and (
        not indices
        or any(index < 0 or index >= frame_count for index in indices)
    ):
        raise OmniClientError(f"invalid frame indices for {path}: {indices}")

    output_frame_count = frame_count if indices is None else len(indices)
    command = [
        _ffmpeg_executable(),
        "-loglevel",
        "error",
        "-threads",
        "0",
        "-i",
        path,
        "-map",
        "0:v:0",
        "-an",
    ]
    if indices is not None:
        # H3 samples prepared 24 FPS references at 2 FPS, so the normal
        # sequence is 0,12,24,... .  The static FFmpeg bundled in the pinned
        # image evaluates a long sum of eq() expressions surprisingly slowly;
        # the equivalent modulo predicate avoids that filter regression.
        step = indices[1] - indices[0] if len(indices) > 1 else 0
        arithmetic = (
            indices[0] == 0
            and step > 0
            and indices == list(range(0, indices[-1] + 1, step))
        )
        if arithmetic:
            select = f"not(mod(n\\,{step}))"
        else:
            select = "+".join(f"eq(n\\,{index})" for index in indices)
        command.extend(["-vf", f"select={select}"])
    command.extend(
        [
            "-vsync",
            "0",
            "-frames:v",
            str(output_frame_count),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
    )
    result = subprocess.run(command, check=True, capture_output=True)
    expected_size = output_frame_count * width * height * 3
    if len(result.stdout) != expected_size:
        raise OmniClientError(
            f"decoded {len(result.stdout)} bytes from {path}, "
            f"expected {expected_size}"
        )
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(
        output_frame_count,
        height,
        width,
        3,
    )


def install() -> None:
    """Install the process-local MiniMax-H3 reference decode fast path."""
    from vllm_omni.diffusion.models.minimax_h3 import reference_video
    from vllm_omni.errors import OmniClientError

    def load_video_frames(path: str) -> np.ndarray:
        metadata = probe_video_metadata(path)
        return decode_video_frames_ffmpeg(
            path,
            frame_count=int(metadata["frame_count"]),
            width=int(metadata["width"]),
            height=int(metadata["height"]),
        )

    def sample_reference_video_frames(prepared_path: str) -> dict[str, Any]:
        metadata = probe_video_metadata(prepared_path)
        ratio = (
            reference_video.MINIMAX_H3_FPS
            / reference_video.MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS
        )
        indices: list[int] = []
        cursor = 0.0
        while True:
            frame_index = int(round(cursor))
            if frame_index >= int(metadata["frame_count"]):
                break
            if not indices or frame_index > indices[-1]:
                indices.append(frame_index)
            cursor += ratio
        if not indices:
            raise OmniClientError(f"no frames sampled from {prepared_path}")

        # The pinned static FFmpeg's select filter is much slower than its
        # raw full-stream path on H100 nodes (about 12 s versus 1.1 s for the
        # validated 124-frame reference).  Decode the prepared stream once
        # through the fast raw path and select in NumPy.  This remains far
        # faster than the image's PyAV full decode (~8.2 s) and is pixel exact.
        decoded = decode_video_frames_ffmpeg(
            prepared_path,
            frame_count=int(metadata["frame_count"]),
            width=int(metadata["width"]),
            height=int(metadata["height"]),
        )
        frames = [np.asarray(decoded[index]) for index in indices]
        timestamps = [
            index / reference_video.MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS
            for index in range(len(indices))
        ]
        timestamps += [timestamps[-1]] * (
            (-len(timestamps)) % reference_video.MINIMAX_H3_QWEN_TEMPORAL_PATCH
        )
        patch = reference_video.MINIMAX_H3_QWEN_TEMPORAL_PATCH
        block_timestamps = [
            (timestamps[index] + timestamps[index + patch - 1]) / 2
            for index in range(0, len(timestamps), patch)
        ]
        return {"frames": frames, "block_timestamps": block_timestamps}

    reference_video._probe_video = probe_video_metadata
    reference_video._probe_audio = probe_audio_metadata
    reference_video._decode_video_frames_ffmpeg = decode_video_frames_ffmpeg
    reference_video.load_video_frames = load_video_frames
    reference_video.sample_reference_video_frames = sample_reference_video_frames
    print(
        "[compat-child] enabled Ref2VA selective/full FFmpeg RGB decode "
        f"for pid={os.getpid()}",
        flush=True,
    )
