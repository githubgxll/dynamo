#!/usr/bin/env python3
"""CPU-only, offline native decoder check using a disposable synthetic MP4."""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile


def decode_probe() -> dict:
    import numpy as np
    import torch
    import decord
    import cv2

    for module in (np, torch, decord, cv2):
        if not Path(module.__file__).resolve().is_relative_to(Path("/opt/venv")):
            raise RuntimeError("Media smoke imported outside the image virtual environment")
    for name in ("ffmpeg", "ffprobe"):
        if not shutil.which(name):
            raise RuntimeError(name + " is missing")
    count, height, width, rate = 8, 64, 96, 8
    indices = [0, 3, 7, 3]
    frames = np.empty((count, height, width, 3), dtype=np.uint8)
    for number in range(count):
        frames[number, :, :, :] = (20 + number * 25, 180 - number * 15, 30 + number * 10)
    with tempfile.TemporaryDirectory(prefix="vbench-decoder-smoke-") as directory:
        video = Path(directory) / "synthetic.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{width}x{height}", "-r", str(rate),
            "-i", "pipe:0", "-frames:v", str(count), "-an", "-c:v", "libx264", "-threads", "1",
            "-crf", "18", "-pix_fmt", "yuv420p", str(video)], input=frames.tobytes(), check=True)
        probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,nb_frames,avg_frame_rate", "-of", "json", str(video)], text=True))
        stream = probe["streams"][0]
        if (stream["codec_name"], stream["width"], stream["height"]) != ("h264", width, height):
            raise RuntimeError("Synthetic MP4 has unexpected codec or geometry")
        reference_bytes = subprocess.check_output(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-i", str(video), "-threads", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
        reference = np.frombuffer(reference_bytes, dtype=np.uint8).reshape(count, height, width, 3)
        capture = cv2.VideoCapture(str(video))
        cv_frames = []
        try:
            if not capture.isOpened():
                raise RuntimeError("OpenCV could not open the synthetic MP4")
            for _ in range(count + 1):
                ok, frame = capture.read()
                if not ok:
                    break
                cv_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
        if len(cv_frames) != count:
            raise RuntimeError("OpenCV decoded an unexpected frame count")
        cv_decoded = np.stack(cv_frames)
        if cv_decoded.shape != reference.shape:
            raise RuntimeError("OpenCV decoded unexpected frame geometry")
        cv_difference = int(np.max(np.abs(cv_decoded.astype(np.int16) - reference.astype(np.int16))))
        if cv_difference > 3:
            raise RuntimeError("OpenCV RGB values disagree with FFmpeg")
        decord.bridge.set_bridge("native")
        reader = decord.VideoReader(str(video), ctx=decord.cpu(0), num_threads=1)
        if len(reader) != count or abs(reader.get_avg_fps() - rate) > 0.01:
            raise RuntimeError("Decord reported an unexpected frame count or frame rate")
        native = reader.get_batch(indices).asnumpy()
        if native.shape != (len(indices), height, width, 3) or native.dtype != np.uint8:
            raise RuntimeError("Decord returned invalid RGB frames")
        difference = int(np.max(np.abs(native.astype(np.int16) - reference[indices].astype(np.int16))))
        # Different bundled FFmpeg versions may round YUV/RGB conversion slightly.
        if difference > 3 or np.array_equal(native[0], native[2]):
            raise RuntimeError("Decord frame order or RGB values disagree with FFmpeg")
        decord.bridge.set_bridge("torch")
        try:
            tensor = reader.get_batch(indices)
            if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.dtype != torch.uint8:
                raise RuntimeError("Decord Torch bridge returned an unexpected tensor")
            if not np.array_equal(tensor.numpy(), native):
                raise RuntimeError("Decord Torch bridge changed decoded frames")
        finally:
            decord.bridge.set_bridge("native")
        del reader
    return {"status": "PASS_CPU_MEDIA_DECODE_ONLY", "decord": metadata.version("decord"),
            "opencv": metadata.version("opencv-python-headless"),
            "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
            "generated_only": True, "user_media_used": False, "codec": "h264", "pixel_format": "yuv420p",
            "width": width, "height": height, "frame_count": count, "batch_indices": indices,
            "max_rgb_difference_from_ffmpeg": difference, "torch_bridge_exact_match": True,
            "opencv_max_rgb_difference_from_ffmpeg": cv_difference,
            "boundary": "Synthetic CPU decode only; actual H3 MP4 scoring on H100 remains required."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "linux" or platform.machine() != "x86_64" or sys.version_info[:2] != (3, 10):
        parser.error("Media smoke must run in the Linux amd64 Python 3.10 image")
    audit = args.output.with_suffix(".network-audit.jsonl")
    if args.output.exists() or audit.exists():
        parser.error("Use a fresh output path; existing evidence will not be overwritten")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from offline_guard import install, assert_no_attempts
    install(audit)
    result = {"status": "FAILED", "scope": "CPU synthetic media decode"}
    try:
        result = decode_probe()
        assert_no_attempts(audit)
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
