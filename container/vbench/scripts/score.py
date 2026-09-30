#!/usr/bin/env python3
"""Six independent VBench dimensions; original MP4s, unique evidence directory."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import queue
import random
import subprocess
import sys
import time
from pathlib import Path

from verify_runtime import DIMENSIONS, SOURCE_COMMIT, digest


def video_inventory(path: Path) -> list[dict]:
    path = path.resolve(strict=True)
    if path.is_file():
        videos = [path]
    else:
        videos = sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() == ".mp4")
        if any(item.is_file() and item.suffix.lower() == ".gif" for item in path.iterdir()):
            raise ValueError("This protocol accepts original MP4s only; remove GIFs from the input directory")
    if not videos or any(item.suffix.lower() != ".mp4" for item in videos):
        raise ValueError("Input must be one MP4 or a flat directory containing MP4s")
    if any(item.suffix != ".mp4" for item in videos):
        raise ValueError("Upstream requires lowercase .mp4 filenames; rename the extension without transcoding")
    return [{"path": str(item), "bytes": item.stat().st_size, "sha256": digest(item)} for item in videos]


def worker_command(videos: Path, output: Path, dimension: str) -> list[str]:
    if dimension not in DIMENSIONS:
        raise ValueError("Unsupported dimension: " + dimension)
    return [sys.executable, str(Path(__file__).resolve()), "--videos", str(videos), "--output", str(output),
            "--worker-dimension", dimension]


def normalized_results(raw: dict, expected_paths: set[str]) -> list[dict]:
    rows = []
    for dimension in DIMENSIONS:
        aggregate, items = raw[dimension]
        if not math.isfinite(float(aggregate)):
            raise RuntimeError("Non-finite aggregate for " + dimension)
        observed = set()
        for item in items:
            path = str(Path(item["video_path"]).resolve())
            if path not in expected_paths or path in observed:
                raise RuntimeError("Unexpected/duplicate result video: " + path)
            observed.add(path)
            value = float(item["video_results"])
            if not math.isfinite(value):
                raise RuntimeError("Non-finite video score")
            # Upstream normalizes MUSIQ's aggregate but NOT its per-video list.
            score = value / 100.0 if dimension == "imaging_quality" else value
            rows.append({"video_path": path, "dimension": dimension, "raw_video_score": value,
                         "score": score, "normalization": "divide_100_once" if dimension == "imaging_quality" else "none"})
        if observed != expected_paths:
            raise RuntimeError("Incomplete video results for " + dimension)
    return rows


def worker(args: argparse.Namespace) -> None:
    from offline_guard import install, assert_no_attempts
    from verify_runtime import environment_evidence
    output = args.output
    output.mkdir(parents=False, exist_ok=False)
    audit = output / "network-audit.jsonl"
    install(audit)
    report = {"status": "FAILED", "dimension": args.worker_dimension,
              "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        report["environment"] = environment_evidence()
        import torch
        from vbench import VBench
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Each metric worker must see exactly one usable CUDA GPU")
        report["gpu"] = torch.cuda.get_device_name(0)
        import numpy as np
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        report["evaluation_seed"] = 0
        torch.cuda.set_device(0)
        probe = torch.ones(8, device="cuda") * 2
        if probe.sum().item() != 16:
            raise RuntimeError("CUDA tensor smoke check failed")
        torch.cuda.synchronize()
        evaluator = VBench("cuda", "/opt/VBench/vbench/VBench_full_info.json", str(output))
        evaluator.evaluate(videos_path=str(args.videos), name="score", dimension_list=[args.worker_dimension],
                           local=True, read_frame=False, mode="custom_input",
                           imaging_quality_preprocessing_mode="longer")
        torch.cuda.synchronize()
        assert_no_attempts(audit)
        report["status"] = "PASS_GPU_DIMENSION"
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (output / "worker-evidence.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def parent(args: argparse.Namespace) -> None:
    if not args.gpu_devices or len(set(args.gpu_devices)) != len(args.gpu_devices):
        raise ValueError("Specify one or more distinct GPU device identifiers")
    if any(not item.isdigit() for item in args.gpu_devices):
        raise ValueError("GPU identifiers must be local numeric device indices exposed to this Pod")
    if len(args.gpu_devices) > 4:
        raise ValueError("This image protocol supports at most four independent metric workers")
    inventory = video_inventory(args.videos)
    args.videos = args.videos.resolve()
    args.output = args.output.resolve()
    if args.output == args.videos or (args.videos.is_dir() and args.output.is_relative_to(args.videos)):
        raise ValueError("Output must be outside the input directory")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "RUNNING", "started_epoch": time.time(), "vbench_commit": SOURCE_COMMIT,
                "protocol": {"mode": "custom_input", "dimensions": DIMENSIONS, "local": True,
                             "read_frame": False, "imaging_quality_preprocessing_mode": "longer",
                             "evaluation_seed": 0,
                             "external_transcoding": False, "gpu_assignment": "one independent dimension per process"},
                "inputs": inventory, "gpu_devices": args.gpu_devices,
                "scoring_entry_sha256": digest(Path(__file__)),
                "image_reference": os.environ.get("VBENCH_IMAGE_REFERENCE", "not supplied; record deployed digest separately"),
                "offline_boundary": "Python socket/process audit; does not establish full Pod network isolation"}
    manifest_path = args.output / "run-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    available: queue.Queue[str] = queue.Queue()
    for gpu in args.gpu_devices:
        available.put(gpu)

    def run_dimension(dimension: str) -> tuple[str, int]:
        gpu = available.get()
        try:
            env = dict(os.environ)
            # These are independent workers, not torchrun/distributed ranks.
            for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
                env.pop(name, None)
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env.setdefault("OMP_NUM_THREADS", "4")
            with (args.output / (dimension + ".log")).open("w", encoding="utf-8") as log:
                completed = subprocess.run(worker_command(args.videos, args.output / dimension, dimension),
                                           env=env, stdout=log, stderr=subprocess.STDOUT)
            return dimension, completed.returncode
        finally:
            available.put(gpu)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpu_devices)) as pool:
            outcomes = dict(pool.map(run_dimension, DIMENSIONS))
        manifest["worker_exit_codes"] = outcomes
        if any(code != 0 for code in outcomes.values()):
            raise RuntimeError("One or more metrics failed; inspect per-dimension logs and retain this run directory")
        if video_inventory(args.videos) != inventory:
            raise RuntimeError("Input videos changed during scoring")
        raw = {}
        for dimension in DIMENSIONS:
            raw.update(json.loads((args.output / dimension / "score_eval_results.json").read_text(encoding="utf-8")))
        rows = normalized_results(raw, {item["path"] for item in inventory})
        (args.output / "scores-raw.json").write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
        with (args.output / "scores-per-video.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        manifest["status"] = "PASS_GPU_ALL_SIX_DIMENSIONS"
        manifest["per_video_score_count"] = len(rows)
        manifest["interpretation"] = "No automatic pairing or synthetic overall quality score. Dynamic degree is not monotonically better."
    except BaseException as error:
        manifest["status"] = "FAILED"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["finished_epoch"] = time.time()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-devices", nargs="+", default=["0"])
    parser.add_argument("--worker-dimension", choices=DIMENSIONS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_dimension:
        worker(args)
    else:
        parent(args)


if __name__ == "__main__":
    main()
