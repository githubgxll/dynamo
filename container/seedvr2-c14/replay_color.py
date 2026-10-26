"""A1: replay the captured full-frame LAB baseline before none/wavelet comparisons."""
from __future__ import annotations

import argparse
from pathlib import Path

from replay_common import (begin_output, compare, configure_numeric_context, export_rgb_frame, load_tensor, load_window,
                           register_guard_process, require_guard, save_failure, save_tensor, tolerance, validate_decode_context, verify_sources, write_json)


def replay(window_dir: Path, output: Path, device: str, policy_path: Path | None = None) -> None:
    register_guard_process(require_guard(window_dir, output, "color"))
    import torch
    from vllm_omni.diffusion.models.seedvr2.pipeline_seedvr2 import correct_video_color

    window = load_window(window_dir)
    actual_context = validate_decode_context(window)
    policy = tolerance(window, policy_path)
    if not device.startswith("cuda:") or not torch.cuda.is_available():
        raise ValueError("A1 baseline must use the captured CUDA backend; CPU is only a unit-test backend")
    provenance = verify_sources(window, ("vllm_omni.diffusion.models.seedvr2.pipeline_seedvr2",))
    autocast = configure_numeric_context(actual_context)
    begin_output(output, "A1-replay-color", window, {"device": device, "tolerance": policy,
                 "methods": ["lab-baseline", "none", "wavelet", "lab"], "source_hashes": provenance,
                 "numeric_context": actual_context})
    try:
        baseline = []
        # Complete baseline pass precedes ALL candidate calculation or output.
        with torch.inference_mode(), torch.autocast("cuda", **autocast):
            for index in range(window["valid_frames"]):
                source, source_meta = load_tensor(window_dir, f"color_input_f{index:03d}", device, restore_stride=True)
                reference, reference_meta = load_tensor(window_dir, f"color_reference_f{index:03d}", device, restore_stride=True)
                expected, expected_meta = load_tensor(window_dir, f"color_output_f{index:03d}", device)
                for value, metadata in ((source, source_meta), (reference, reference_meta), (expected, expected_meta)):
                    if metadata.get("layout") != "BCTHW" or tuple(value.shape) != (1, 3, 1, 1104, 1840):
                        raise ValueError("Color replay requires full-frame BCTHW tensors")
                    if not str(metadata.get("device", "")).startswith("cuda"):
                        raise ValueError("Capture color backend is not CUDA")
                    if value.min() < 0 or value.max() > 1:
                        raise ValueError("Captured color RGB is outside native [0,1] range")
                actual = correct_video_color(source, reference, method="lab", luminance_weight=window["luminance_weight"])
                check = compare(actual, expected, policy)
                baseline.append({"local_frame": index, "global_frame": window["start"] + index, **check,
                                 "source_sha256": source_meta["sha256"], "reference_sha256": reference_meta["sha256"],
                                 "expected_sha256": expected_meta["sha256"]})
                del actual, source, reference, expected
            write_json(output / "baseline.json", {"status": "PASS" if all(c["pass"] for c in baseline) else "FAIL",
                       "tolerance": policy, "frames": baseline})
            if not all(check["pass"] for check in baseline):
                raise RuntimeError("Original LAB replay differs; candidates were not calculated")
            for method in ("none", "wavelet", "lab"):
                target = output / method
                target.mkdir()
                frames = []
                for index in range(window["valid_frames"]):
                    source, source_meta = load_tensor(window_dir, f"color_input_f{index:03d}", device, restore_stride=True)
                    reference, reference_meta = load_tensor(window_dir, f"color_reference_f{index:03d}", device, restore_stride=True)
                    expected, _ = load_tensor(window_dir, f"color_output_f{index:03d}", device)
                    corrected = correct_video_color(source, reference, method=method, luminance_weight=window["luminance_weight"])
                    saved = save_tensor(target / f"rgb-f{index:03d}.safetensors", corrected)
                    display = export_rgb_frame(corrected, target, "display", window["start"] + index)
                    frames.append({"local_frame": index, "source_sha256": source_meta["sha256"],
                                   "reference_sha256": reference_meta["sha256"], "output": saved, "display": display,
                                   "difference_from_captured_lab": compare(corrected, expected, policy)})
                    del corrected, source, reference, expected
                write_json(target / "manifest.json", {"method": method, "frames": frames,
                           "processing": "Complete frame first; ROI display is derived afterwards"})
                from replay_encoding import export_color_videos
                write_json(target / "viewing-copies.json", export_color_videos(target, frames))
        write_json(output / "COMPLETE.json", {"status": "PASS_BASELINE_AND_REPLAY_ONLY",
                   "quality_acceptance": "Requires continuous-window and complete-shot human review"})
    except BaseException as error:
        save_failure(output, error)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory; never overwrite prior replay")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tolerance-policy", type=Path, help="Only accepted if its SHA was sealed in capture profile")
    args = parser.parse_args()
    replay(args.window, args.output, args.device, args.tolerance_policy)
    print("PASS A1 LAB baseline and three full-frame replay branches; quality review pending")


if __name__ == "__main__":
    main()
