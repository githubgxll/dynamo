#!/usr/bin/env python3
"""Verify pinned assets/imports and load all selected models with downloads blocked."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

DIMENSIONS = ["subject_consistency", "background_consistency", "motion_smoothness",
              "dynamic_degree", "aesthetic_quality", "imaging_quality"]
SOURCE_COMMIT = "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"
EXPECTED = {
    "torch": "2.3.1+cu121", "torchvision": "0.18.1+cu121", "numpy": "1.26.4",
    "pyiqa": "0.1.13", "timm": "1.0.12", "transformers": "4.37.2",
    "accelerate": "0.28.0", "bitsandbytes": "0.43.1", "huggingface-hub": "0.23.5",
    "tokenizers": "0.15.2", "safetensors": "0.4.3", "opencv-python-headless": "4.10.0.84",
    "facexlib": "0.3.0+vbench1", "Pillow": "10.4.0", "decord": "0.6.0",
    "openai-clip": "1.0.1", "omegaconf": "2.3.0", "easydict": "1.13", "imageio": "2.35.1",
}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify_dependencies() -> dict:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    distributions = {}
    for dist in metadata.distributions():
        name = canonicalize_name(dist.metadata["Name"])
        if name in distributions:
            raise RuntimeError("duplicate installed metadata: " + name)
        distributions[name] = dist
    errors = []
    for name, version in EXPECTED.items():
        actual = distributions.get(canonicalize_name(name))
        if actual is None or actual.version != version:
            errors.append(f"{name}: expected {version}; got {actual.version if actual else 'missing'}")
    for unwanted in ("opencv-python", "opencv-contrib-python", "opencv-contrib-python-headless"):
        if unwanted in distributions:
            errors.append("multiple cv2 providers forbidden: " + unwanted)
    for name, dist in distributions.items():
        for spec in dist.requires or []:
            req = Requirement(spec)
            if req.marker and not req.marker.evaluate({"extra": ""}):
                continue
            installed = distributions.get(canonicalize_name(req.name))
            if installed is None or (req.specifier and not req.specifier.contains(installed.version, prereleases=True)):
                errors.append(f"{name} requires {req}; installed {installed.version if installed else 'missing'}")
    if errors:
        raise RuntimeError("Runtime dependencies do not match:\n" + "\n".join(errors))
    return {name: dist.version for name, dist in sorted(distributions.items())}


def verify_assets() -> dict:
    from prepare_assets import verify_manifest
    manifest = Path("/opt/vbench-build-info/assets.lock.json")
    checked = verify_manifest(manifest, Path("/opt/vbench-assets"), Path("/opt/VBench"))
    if checked.get("revisions", {}).get("vbench-source") != SOURCE_COMMIT:
        raise RuntimeError("Asset lock does not reference the selected VBench commit")
    return {"manifest": str(manifest), "sha256": digest(manifest), "verification": checked}


def verify_paths() -> dict:
    expected_env = {"VBENCH_CACHE_DIR": "/opt/vbench-assets/vbench", "TORCH_HOME": "/opt/vbench-assets/torch"}
    for name, value in expected_env.items():
        if os.environ.get(name) != value:
            raise RuntimeError(f"{name} must be {value}; overriding the image cache can cause lazy downloads")
    result = {}
    for name in ("torch", "torchvision", "cv2", "pyiqa", "clip"):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(Path("/opt/venv")):
            raise RuntimeError(f"{name} imported outside image venv: {path}")
        result[name] = str(path)
    vbench = importlib.import_module("vbench")
    if not Path(vbench.__file__).resolve().is_relative_to(Path("/opt/VBench")):
        raise RuntimeError("VBench import does not use the pinned image source")
    result["vbench"] = str(Path(vbench.__file__).resolve())
    import pyiqa
    import clip
    resources = [Path(pyiqa.__file__).parent / "archs/class_mapping.json",
                 Path(clip.__file__).parent / "bpe_simple_vocab_16e6.txt.gz"]
    for path in resources:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError("required package resource missing: " + str(path))
        result[str(path)] = digest(path)
    for executable in ("ffmpeg", "ffprobe"):
        path = shutil.which(executable)
        if not path:
            raise RuntimeError(executable + " is missing")
        result[executable] = subprocess.check_output([path, "-version"], text=True).splitlines()[0]
    return result


def environment_evidence() -> dict:
    import torch
    if torch.version.cuda != "12.1":
        raise RuntimeError("Unexpected PyTorch CUDA build: " + str(torch.version.cuda))
    return {"python": sys.version, "vbench_commit": SOURCE_COMMIT,
            "dependencies": verify_dependencies(), "paths": verify_paths(),
            "torch_cuda_build": torch.version.cuda,
            "requirements_lock_sha256": digest(Path("/opt/vbench-build/locks/requirements.lock")),
            "assets": verify_assets()}


def load_models(device: str) -> list[dict]:
    import torch
    import clip
    from vbench.utils import init_submodules
    from vbench.motion_smoothness import MotionSmoothness
    from vbench.dynamic_degree import DynamicDegree
    from vbench.aesthetic_quality import get_aesthetic_model
    from pyiqa.archs.musiq_arch import MUSIQ
    from easydict import EasyDict

    modules = init_submodules(DIMENSIONS, local=True, read_frame=False)
    results = []
    for dimension in DIMENSIONS:
        spec = modules[dimension]
        if dimension == "subject_consistency":
            loaded = torch.hub.load(**spec).to(device)
        elif dimension == "background_consistency":
            loaded, _ = clip.load(spec[0], device=device)
        elif dimension == "motion_smoothness":
            loaded = MotionSmoothness(spec["config"], spec["ckpt"], device)
        elif dimension == "dynamic_degree":
            loaded = DynamicDegree(EasyDict(model=spec["model"], small=False, mixed_precision=False,
                                             alternate_corr=False), device)
        elif dimension == "aesthetic_quality":
            loaded = (get_aesthetic_model(spec[1]).to(device), clip.load(spec[0], device=device)[0])
        elif dimension == "imaging_quality":
            loaded = MUSIQ(pretrained_model_path=spec["model_path"]).to(device)
        results.append({"dimension": dimension, "weights_loaded": True, "device": device})
        del loaded
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from offline_guard import install, assert_no_attempts
    audit = args.output.with_suffix(".network-audit.jsonl")
    if args.output.exists() or audit.exists():
        parser.error("Output or network audit already exists; preserve it and choose a fresh --output path")
    install(audit)
    result = {"status": "FAILED", "device": args.device,
              "boundary": "Model loading only. No video scoring or GPU kernel proof from CPU checks."}
    try:
        result["environment"] = environment_evidence()
        if args.device == "cuda":
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            result["gpu"] = torch.cuda.get_device_name(0)
        result["models"] = load_models(args.device)
        assert_no_attempts(audit)
        result["status"] = "PASS_MODEL_LOAD_ONLY"
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
