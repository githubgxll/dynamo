"""A3 full-shot replay from captured fused pixels; no model construction or inference."""
from __future__ import annotations
import argparse
from fractions import Fraction
from itertools import zip_longest
from pathlib import Path

from replay_common import (begin_output, contained, load_window, read_json, require_hash, roi_box,
                           save_failure, sha256, verify_sources, write_json)
from replay_encoding import decode_rgb, pixel_compare, write_lossless_rgb


def fused_frame(root, index):
    import numpy as np
    from safetensors.numpy import load_file
    path = contained(root, f"fusion/frame-{index:06d}.safetensors")
    meta = read_json(path.with_suffix(".json"))
    if (meta.get("schema"), meta.get("global_frame"), meta.get("shape"), meta.get("dtype"), meta.get("layout")) != (
        "seedvr2-c14-fusion/1", index, [1104, 1840, 3], "uint8", "HWC"
    ):
        raise ValueError("Fused frame geometry/identity changed")
    require_hash(path, meta.get("sha256"))
    tensors = load_file(str(path))
    if set(tensors) != {"tensor"} or tensors["tensor"].shape != (1104, 1840, 3) or tensors["tensor"].dtype != np.uint8:
        raise ValueError("Fused tensor does not match metadata")
    return tensors["tensor"]


def fusion_baseline(root):
    """Replicate the native rint blend on selected windows; don't infer unseen borders."""
    import numpy as np
    rows, previous = [], None
    for start in (130, 143, 156, 169, 182):
        directory = root / "windows" / f"{start:06d}"
        load_window(directory)
        path = contained(directory, "window-output.mp4")
        require_hash(path, read_json(path.with_suffix(".mp4.json"))["sha256"])
        current, _ = decode_rgb(path)
        if current.shape != (17, 1104, 1840, 3):
            raise ValueError("Unexpected selected-window decoded geometry")
        for local in range(4 if previous is None else 0, 13):
            pixel = current[local]
            if local < 4:
                weight = (local + 1) / 5
                pixel = np.rint(previous[13 + local] * (1 - weight) + pixel * weight).astype(np.uint8)
            rows.append({"global_frame": start + local,
                         **pixel_compare(pixel[None], fused_frame(root, start + local)[None])})
        previous = current
    return {"pass": all(row["pass"] for row in rows), "frames": rows,
            "scope": "Global134..194; selected windows' unavailable outer neighbours are excluded"}


def encode_native(path, frames, settings):
    import av
    if path.exists():
        raise FileExistsError(path)
    if settings.get("pyav_version") != av.__version__:
        raise ValueError("Native PyAV version differs from capture")
    if (settings.get("codec"), settings.get("pixel_format"), settings.get("width"), settings.get("height"),
        settings.get("rate"), settings.get("options")) != (
        "libx264", "yuv420p", 1840, 1104, "24",
        {"preset": "veryfast", "crf": "18", "tune": "zerolatency", "bf": "0"}
    ):
        raise ValueError("Native long encoder settings differ from frozen source")
    with av.open(str(path), "w", format="mp4") as container:
        stream = container.add_stream(settings["codec"], rate=24)
        stream.width, stream.height, stream.pix_fmt = 1840, 1104, settings["pixel_format"]
        stream.options = settings["options"]
        for key in ("colorspace", "color_range", "color_primaries", "color_trc"):
            if int(getattr(stream.codec_context, key)) != settings[key]:
                raise ValueError("Native long encoder color defaults differ")
        count = 0
        for index, array in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 24)
            for packet in stream.encode(frame):
                container.mux(packet)
            count += 1
        if count != 362:
            raise ValueError("Full replay requires exactly362 fused frames")
        for packet in stream.encode():
            container.mux(packet)
    return {"sha256": sha256(path), "frames": count, "audio": "video-only; final uses original Base audio"}


def frames_of(path, max_frames=362):
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        count = 0
        for frame in container.decode(stream):
            if count >= max_frames:
                raise ValueError("Decoded media exceeds bounded replay frame count")
            yield frame.to_ndarray(format="rgb24"), frame.pts * frame.time_base
            count += 1


def compare_video(left, right, count=362, exact_times=True):
    import av
    def media_info(path):
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            codec = stream.codec_context
            return {"codec": codec.name, "width": stream.width, "height": stream.height,
                    "fps": str(stream.average_rate), "pixel_format": codec.pix_fmt,
                    **{key: int(getattr(codec, key)) for key in
                    ("colorspace", "color_range", "color_primaries", "color_trc")}}
    left_media, right_media = media_info(left), media_info(right)
    rows = []
    for index, pair in enumerate(zip_longest(frames_of(left, count), frames_of(right, count))):
        a, b = pair
        if a is None or b is None:
            raise ValueError("Replay video frame counts differ")
        row = {"global_frame": index, **pixel_compare(a[0][None], b[0][None]),
               "same_timestamp": a[1] == b[1], "left_timestamp": str(a[1]), "right_timestamp": str(b[1])}
        x0, y0, x1, y1 = roi_box(a[0].shape[0], a[0].shape[1])
        row["roi"] = pixel_compare(a[0][None, y0:y1, x0:x1], b[0][None, y0:y1, x0:x1])
        rows.append(row)
    if len(rows) != count:
        raise ValueError("Replay video did not contain complete expected timeline")
    return {"pass": left_media == right_media and all(row["pass"] and (row["same_timestamp"] or not exact_times) for row in rows),
            "frames": rows, "same_media_tags": left_media == right_media, "left_media": left_media, "right_media": right_media}


def lossless_readback(path, frames, count=362, exact_timeline=False):
    rows = []
    for index, pair in enumerate(zip_longest(frames_of(path, count), frames)):
        actual, expected = pair
        if actual is None or expected is None:
            raise ValueError("Lossless route changed frame count")
        if exact_timeline and actual[1] != Fraction(index, 24):
            raise ValueError("Lossless intermediary must preserve exact zero-origin24FPS timestamps")
        rows.append({"global_frame": index, **pixel_compare(actual[0][None], expected[None])})
    if len(rows) != count or not all(row["pass"] for row in rows):
        raise ValueError("Lossless RGB readback differs")
    return {"pass": True, "frames": rows, "exact_timeline_checked": exact_timeline}


def replay(run, output):
    from media import command, compare_audio, probe_media
    from sr_only import check_run, identity, mux_command
    from vllm_omni.diffusion.models.seedvr2.c14_capture import verify_run
    run = run.resolve(strict=True)
    if output.resolve().is_relative_to(run):
        raise ValueError("Replay output must be outside the frozen A0 run directory")
    manifest = check_run(run, model=False)
    if identity(run / "release.json") != manifest["identity"]:
        raise ValueError("Runtime image/source changed since A0 import")
    capture = contained(run, "capture/" + run.name)
    attempt = contained(run, "attempt")
    result = read_json(contained(attempt, "result.json"))
    if result.get("status") != "PASS_CAPTURE_AND_MEDIA_ONLY":
        raise ValueError("Full-chain replay requires successful complete A0 evidence")
    verify_run(capture)
    window = load_window(capture / "windows/000156")
    sources = verify_sources(window, ("vllm_omni.diffusion.models.seedvr2.long_video",))
    for name, role in (("raw1104.mp4", "raw"), ("final1080.mp4", "final")):
        require_hash(contained(attempt, name), result[role]["sha256"])
    base = contained(run, "base.mp4")
    require_hash(base, window["parent_base_sha256"])
    recorded = read_json(contained(attempt, "final-encoding-command.json"))
    if recorded["argv"] != mux_command(attempt / "raw1104.mp4", base, attempt / "final1080.mp4"):
        raise ValueError("Final mux/scale command differs from the current pinned implementation")
    settings = read_json(contained(capture, "long-encoding.json"))
    begin_output(output, "A3-fusion-and-full-encoding-chain", window,
                 {"run": run.name, "source_hashes": sources, "long_encoder": settings,
                  "parent_result_sha256": sha256(attempt / "result.json"),
                  "unique_candidate_change": "lossless RGB intermediate replaces raw H264/YUV420; final command unchanged"})
    try:
        fusion = fusion_baseline(capture)
        write_json(output / "fusion-baseline.json", fusion)
        if not fusion["pass"]:
            raise RuntimeError("Original selected-window fusion did not reproduce")
        def frames():
            for index in range(362):
                yield fused_frame(capture, index)
        native = output / "native-raw1104.mp4"
        encode_native(native, frames(), settings)
        raw_check = compare_video(native, attempt / "raw1104.mp4")
        write_json(output / "raw-baseline.json", raw_check)
        if not raw_check["pass"]:
            raise RuntimeError("Original complete raw video did not reproduce; no candidate computed")
        baseline_final = output / "native-final1080.mp4"
        argv = mux_command(native, base, baseline_final)
        write_json(output / "native-final-command.json", {"argv": argv})
        command(argv)
        final_check = compare_video(baseline_final, attempt / "final1080.mp4")
        write_json(output / "final-baseline.json", final_check)
        if not final_check["pass"]:
            raise RuntimeError("Original final video did not reproduce; no candidate computed")
        baseline_probe = probe_media(baseline_final, output, "native-final")
        compare_audio(result["final"]["audio"], baseline_probe["audio"])
        # NUT carries exact rational timestamps; Matroska's millisecond ticks
        # would perturb final passthrough timing and invalidate this comparison.
        lossless = output / "lossless-raw1104.nut"
        write_lossless_rgb(lossless, frames(), container_format="nut")
        write_json(output / "lossless-readback.json", lossless_readback(lossless, frames(), exact_timeline=True))
        candidate = output / "lossless-intermediate-final1080.mp4"
        argv = mux_command(lossless, base, candidate)
        write_json(output / "candidate-final-command.json", {"argv": argv})
        command(argv)
        candidate_probe = probe_media(candidate, output, "candidate-final")
        compare_audio(result["final"]["audio"], candidate_probe["audio"])
        write_json(output / "candidate-vs-baseline.json", compare_video(candidate, baseline_final))
        write_json(output / "COMPLETE.json", {"status": "PASS_BASELINE_AND_REPLAY_ONLY", "fusion_frames": "134..194",
                   "raw_final_frames": 362, "candidate_sha256": sha256(candidate),
                   "scope": "Video pixels across fused/raw/final stages plus original final audio. Raw intermediate audio reencode is not repeated.",
                   "quality_acceptance": "Review n144..192 and all362 frames; changes are not automatically improvements"})
    except BaseException as error:
        save_failure(output, error)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Settled A0 SR-only run directory")
    parser.add_argument("--output", type=Path, required=True, help="New exclusive output directory")
    args = parser.parse_args()
    replay(args.run, args.output)
    print("PASS A3 full-shot baseline and same-pixel encoding comparison; quality review pending")


if __name__ == "__main__":
    main()
