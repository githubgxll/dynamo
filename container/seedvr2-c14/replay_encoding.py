"""A3: reproduce captured single-window native encoding, then compare lossless RGB."""
from __future__ import annotations

import argparse
import io
from fractions import Fraction
from pathlib import Path

from replay_common import (begin_output, contained, load_tensor, load_window, read_json,
                           require_hash, save_failure, sha256, verify_sources, write_json)


def decode_rgb(path_or_bytes, max_frames=17):
    import av
    import numpy as np
    source = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else str(path_or_bytes)
    with av.open(source) as container:
        stream = container.streams.video[0]
        codec = stream.codec_context
        metadata = {"codec": codec.name, "width": stream.width, "height": stream.height,
                    "pixel_format": codec.pix_fmt, "fps": str(stream.average_rate),
                    "color_range": str(codec.color_range), "colorspace": str(codec.colorspace),
                    "color_primaries": str(codec.color_primaries), "color_trc": str(codec.color_trc)}
        frames, timestamps = [], []
        for frame in container.decode(stream):
            if len(frames) >= max_frames:
                raise ValueError("First-batch A3 supports one captured window, not unbounded media")
            frames.append(frame.to_ndarray(format="rgb24"))
            timestamps.append(str(frame.pts * frame.time_base))
    if not frames:
        raise ValueError("Encoded video contains no frames")
    return np.stack(frames), {**metadata, "frames": len(frames), "timestamps": timestamps}


def pixel_compare(actual, expected) -> dict:
    import numpy as np
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("Encoding changed frame geometry/count/dtype")
    difference = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    return {"pass": bool(np.array_equal(actual, expected)), "mismatched_values": int(np.count_nonzero(difference)),
            "max_abs": int(difference.max()), "mean_abs": float(difference.mean()),
            "per_frame_max_abs": difference.max(axis=(1, 2, 3)).tolist()}


def write_lossless_rgb(path: Path, frames, fps: int = 24, container_format: str = "matroska") -> dict:
    """FFV1 bgr0 is verified by readback; codec name alone is not a lossless claim."""
    import av
    if container_format not in ("matroska", "nut"):
        raise ValueError("Unsupported lossless container")
    if path.exists():
        raise FileExistsError(path)
    iterator = iter(frames)
    try:
        first = next(iterator)
    except StopIteration as error:
        raise ValueError("Empty RGB sequence") from error
    from itertools import chain
    with av.open(str(path), "w", format=container_format) as container:
        stream = container.add_stream("ffv1", rate=fps)
        stream.width, stream.height, stream.pix_fmt = first.shape[1], first.shape[0], "bgr0"
        stream.time_base = Fraction(1, fps)
        stream.options = {"level": "3", "coder": "1", "context": "1", "slicecrc": "1"}
        count = 0
        for index, array in enumerate(chain((first,), iterator)):
            if array.dtype.name != "uint8" or array.shape != first.shape:
                raise ValueError("Lossless RGB sequence changed format/geometry")
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
            count += 1
        for packet in stream.encode():
            container.mux(packet)
    return {"path": path.name, "sha256": sha256(path), "frames": count, "fps": fps,
            "codec": "ffv1", "pixel_format": "bgr0", "container": container_format, "pixel_readback": "caller must verify"}


def export_color_videos(directory: Path, frame_records: list[dict]) -> dict:
    """Uniform 24/12 FPS full-frame and ROI viewing copies, after full-frame color."""
    from PIL import Image
    import numpy as np
    exports = []
    for scope, key in (("full", "full"), ("roi", "roi")):
        def frames():
            for record in frame_records:
                with Image.open(contained(directory, record["display"][key])) as picture:
                    yield np.asarray(picture.convert("RGB"))
        for fps in (24, 12):
            path = directory / f"display-{scope}-{fps}fps.mkv"
            saved = write_lossless_rgb(path, frames(), fps=fps)
            decoded, media = decode_rgb(path)
            source = np.stack(list(frames()))
            check = pixel_compare(decoded, source)
            if not check["pass"]:
                raise RuntimeError("Viewing export was not RGB lossless")
            exports.append({**saved, "readback": check, "media": media})
    return {"files": exports, "display_only": True, "audio": "No audio in diagnostic window viewing copies"}


def replay(window_dir: Path, output: Path) -> None:
    import av
    import numpy as np
    import torch
    from vllm_omni.entrypoints.openai import video_api_utils

    from sr_only import check_run, identity
    window_dir = window_dir.resolve(strict=True)
    run = window_dir.parents[3]
    if window_dir != run / "capture" / run.name / "windows" / window_dir.name:
        raise ValueError("Use a window under the complete A0 run/capture/run-id/windows directory")
    manifest = check_run(run, model=False)
    if identity(run / "release.json") != manifest["identity"]:
        raise ValueError("Runtime image/source changed since A0 import")
    if read_json(run / "state.json")["status"] != "A0_CAPTURE_COMPLETE_QUALITY_UNDETERMINED":
        raise ValueError("Encoding replay requires settled successful A0")
    if output.resolve().is_relative_to(run):
        raise ValueError("Replay output must be outside the frozen A0 run directory")
    window = load_window(window_dir)
    evidence = read_json(contained(window_dir, "encoding.json"))
    required = {"video_codec_options", "enable_borrowed_frames", "frame_conversion_workers", "selected_path"}
    if not required.issubset(evidence) or evidence.get("status") != "completed":
        raise ValueError("Actual generic encoder invocation was not captured; A3 baseline is blocked")
    if type(evidence["enable_borrowed_frames"]) is not bool:
        raise ValueError("Invalid captured encoding transport")
    if evidence.get("muxer", {}).get("pyav_version") != av.__version__:
        raise ValueError("Missing actual muxer evidence or PyAV version differs from capture")
    workers = evidence["frame_conversion_workers"]
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError("Unsupported captured encoder worker count")
    if evidence.get("audio_present", False):
        raise ValueError("Single-window A3 requires the native audio-free MP4 output route")
    sources = verify_sources(window, ("vllm_omni.entrypoints.openai.video_api_utils",
                                     "vllm_omni.diffusion.utils.media_utils"))
    original_path = contained(window_dir, "window-output.mp4")
    original_meta = read_json(contained(window_dir, "window-output.mp4.json"))
    original_hash = require_hash(original_path, original_meta.get("sha256"))
    pixels, pixel_meta = load_tensor(window_dir, "quantized_rgb")
    if pixel_meta.get("layout") != "BTHWC" or pixels.dtype != torch.uint8 or tuple(pixels.shape) != (1, 17, 1104, 1840, 3):
        raise ValueError("A3 requires the full original quantized BTHWC uint8 window")
    source = pixels[0].contiguous().numpy()
    begin_output(output, "A3-replay-single-window-encoding", window, {"source_hashes": sources,
                 "quantized_source_sha256": pixel_meta["sha256"], "original_video_sha256": original_hash,
                 "encoding": evidence, "scope": "Single-window generic encoder only; fusion/final publish are separate stages"})
    try:
        original, original_probe = decode_rgb(original_path)
        observed = []
        previous_logger = video_api_utils._log_video_encoding_path
        from vllm_omni.diffusion.models.seedvr2 import c14_capture
        previous_muxer = c14_capture.muxer_settings
        observed_muxer = []
        def capture_muxer(stream, codec, fps, options):
            context = stream.codec_context
            observed_muxer.append({"codec": codec, "pixel_format": stream.pix_fmt,
                "width": stream.width, "height": stream.height, "fps": fps, "rate": str(stream.average_rate),
                "options": dict(options), **{key: int(getattr(context, key)) for key in
                ("colorspace", "color_range", "color_primaries", "color_trc")}})
        def capture_path(**kwargs):
            observed.append({key: value for key, value in kwargs.items()
                             if key in ("selected_path", "effective_frame_conversion_workers", "reason")})
            previous_logger(**kwargs)
        converter = video_api_utils._PlanarFrameConverter(max_workers=workers)
        try:
            video_api_utils._log_video_encoding_path = capture_path
            c14_capture.muxer_settings = capture_muxer
            content = video_api_utils._encode_video_bytes(source, fps=24, video_codec_options=evidence["video_codec_options"],
                       frame_converter=converter, enable_borrowed_frames=evidence["enable_borrowed_frames"])
        finally:
            video_api_utils._log_video_encoding_path = previous_logger
            c14_capture.muxer_settings = previous_muxer
            converter.shutdown()
        with (output / "native-replay.mp4").open("xb") as stream:
            stream.write(content)
        native, native_probe = decode_rgb(content)
        check = pixel_compare(native, original)
        same_route = (len(observed) == 1 and observed[0]["selected_path"] == evidence["selected_path"]
                      and observed[0]["effective_frame_conversion_workers"] == evidence["effective_frame_conversion_workers"])
        same_times = native_probe["timestamps"] == original_probe["timestamps"]
        same_muxer = (len(observed_muxer) == 1 and all(
            evidence["muxer"].get(key) == value for key, value in observed_muxer[0].items()))
        media_keys = ("codec", "width", "height", "pixel_format", "fps", "color_range", "colorspace", "color_primaries", "color_trc")
        same_media = all(native_probe[k] == original_probe[k] for k in media_keys)
        write_json(output / "baseline.json", {"pixels": check, "same_route": same_route, "same_timestamps": same_times,
                   "same_color_and_media_tags": same_media,
                   "same_muxer_arguments": same_muxer, "observed_muxer": observed_muxer,
                   "captured_media": original_probe, "replayed_media": native_probe, "observed_encoder": observed})
        if not check["pass"] or not same_route or not same_times or not same_media or not same_muxer:
            raise RuntimeError("Original native window encoding did not reproduce; lossless comparison was not run")
        lossless_path = output / "same-pixels-rgb-lossless.mkv"
        saved = write_lossless_rgb(lossless_path, source)
        lossless, lossless_probe = decode_rgb(lossless_path)
        lossless_check = pixel_compare(lossless, source)
        if not lossless_check["pass"]:
            raise RuntimeError("FFV1 route failed RGB pixel identity check")
        write_json(output / "COMPLETE.json", {"status": "PASS_BASELINE_AND_ENCODING_COMPARISON_ONLY",
                   "native_encoding_difference_from_exact_input": pixel_compare(native, source),
                   "lossless": {**saved, "readback": lossless_check, "media": lossless_probe},
                   "scope": "Single window. Does not reconstruct whole-shot overlap fusion or final audio/resize publish chain.",
                   "quality_acceptance": "Requires same-frame/continuous-video review; no automatic artifact diagnosis"})
    except BaseException as error:
        save_failure(output, error)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.window, args.output)
    print("PASS A3 single-window encoder baseline and RGB-lossless readback; full publish-chain acceptance pending")


if __name__ == "__main__":
    main()
