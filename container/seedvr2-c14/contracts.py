# SPDX-License-Identifier: Apache-2.0
"""Fixed C14 input/model contract. Read-only checks; no inference or downloads."""
from __future__ import annotations
import json
import re
from pathlib import Path
from media import require, sha, read_json

PARENT = "registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm@sha256:9382949d2789b23cdfe4ea71d68c7973592de28f209082bd7907c1d09fe008c9"
REVISION = "37255ff8cccfb01071b87f635a5948ca8d53117c"
BASE = "5a4631d554d14fee83d2e836aa9307432bf4446cba9a0ffea06f9e6659a2dc88"
AUDIO = "f2cfe982b172dc1add9509118fc2ee9c4f72f64514ffb0ce8d1db0ebbe247671"
SOURCE_LOCK = {
    "seedvr2_ema_3b.pth": {"bytes": 13566090228, "sha256": "6bcc5ac59447e97b100477480aebb01be2ec724c8340bb83faae21f64848604b"},
    "ema_vae.pth": {"bytes": 1002691902, "sha256": "c7df8a67e68b7f9aca3d5d2153d2ce8ab4373687741a0f9ce87cb356ace51cac"},
    "pos_emb.pt": {"bytes": 595100, "sha256": "fa07a14844314772266b66c3b95deb0027696d8fe7065721263db5176f45d799"},
}
FROZEN = {
    "base768.mp4": BASE,
    "old-raw1104.mp4": "73e4233da65311a584ab66acd5bef9e137469c6e9ba0c7196ae7df08c75b2b31",
    "old-final1080.mp4": "89440ed8240f255b64be445a71de73579f5df5e740a0ee276327d9c8dc6ec09d",
    "parent-request.json": "b1cdcf0dee803a2482af061083a7d0f0270adb80c5c4a9c3b8f6a28a3d6bfa8e",
    "parent-base-manifest.json": "868701123b2ea903eb5fe8e9f408fc84a1583d5ccba83bec5d24ba17e6893d7b",
    "sr1080-wire-request.json": "a835f7f50e0a77feb63905ef2c5b428a65b6ee18510ab0167b64be032da1a598",
}
PROFILE = {"model_revision": REVISION, "dtype": "float16", "gpus": 8, "sp": 8,
    "input_frames": 362, "fps": 24, "input_size": [1280, 768], "sr_size": [1840, 1104],
    "final_size": [1800, 1080], "seed": 7723, "window": 17, "overlap": 4, "stride": 13,
    "euler_steps": 1, "cfg": 1, "color_correction_method": "lab", "lab_luminance_weight": 0.8,
    "vae_use_tiling": True, "vae_parallel_mode": "spatial_shard_height", "decode_chunk": 2,
    "eager": True, "cache_backend": "none", "h3_resident": False, "capture_window_starts": [130, 143, 156, 169, 182]}


def plain(path, *, directory=False):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        require(not item.is_symlink(), "Symlink inputs are not accepted: " + str(item))
        if item.exists():
            require(not getattr(item.lstat(), "st_file_attributes", 0) & 0x400, "Reparse path refused")
    require(path.is_dir() if directory else path.is_file(), "Missing input: " + str(path))
    return path.resolve()


def validate_baseline(directory):
    directory = plain(directory, directory=True)
    manifest = read_json(plain(directory / "manifest.json"))
    require(manifest.get("schema") == "seedvr2-c14-frozen-baseline/1" and
            manifest.get("status") == "LOCAL_FROZEN_VERIFIED", "Use the frozen P0 baseline")
    require(read_json(plain(directory / "COPY-STATE.json")).get("status") == "LOCAL_FROZEN_VERIFIED",
            "P0 copy is incomplete")
    require(manifest.get("image") == PARENT and manifest.get("parent_base_sha256") == BASE,
            "Wrong parent image or Base")
    require((manifest.get("parent_run_id"), manifest.get("parent_job_id")) ==
            ("h3sr8-ref7-fix-a", "ref7-rest-1022-a-c14"), "Wrong parent run/job")
    rows = manifest.get("files", [])
    require(len(rows) == 13 and len({x["path"] for x in rows}) == 13, "Expected 13 distinct frozen artifacts")
    mapping = {x["path"]: x for x in rows}
    require(set(FROZEN) <= set(mapping), "Missing fixed baseline inputs")
    for name, row in mapping.items():
        require(re.fullmatch(r"[a-zA-Z0-9_.-]+", name) is not None, "Unsafe baseline member")
        path = plain(directory / name)
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "Frozen input changed: " + name)
        if name in FROZEN:
            require(row["sha256"] == FROZEN[name], "Incorrect original baseline identity: " + name)
    archived = manifest.get("archived_media", {})
    require([archived.get(x) for x in ("width", "height", "frames", "fps", "original_audio_sequence_sha256")] ==
            [1280, 768, 362, 24, AUDIO], "Archived media differs")
    return manifest


def validate_model(directory):
    """Reuse original verified FP16 weights; never convert, download or mutate them."""
    directory = plain(directory, directory=True)
    manifest = json.loads(plain(directory / "conversion-manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("schema") == "h3-seedvr2-fp16-conversion/1" and
            manifest.get("contract_version") == 1 and manifest.get("status") == "VERIFIED", "Unverified FP16 conversion")
    require(manifest.get("source") == {"repo": "ByteDance-Seed/SeedVR2-3B", "revision": REVISION, "files": SOURCE_LOCK},
            "Model revision/source identity changed")
    names = {"seedvr2_ema_3b_fp16.safetensors", "ema_vae_fp16.safetensors", "pos_emb.pt", "model_index.json"}
    require(set(manifest.get("outputs", {})) == names and
            {p.name for p in directory.iterdir()} == names | {"conversion-manifest.json"}, "Unexpected model inventory")
    for name, expected in manifest["outputs"].items():
        path = plain(directory / name)
        require({"bytes": path.stat().st_size, "sha256": sha(path)} == expected, "FP16 weight changed: " + name)
    require(read_json(directory / "model_index.json") == {"_class_name": "SeedVR2Pipeline"}, "Wrong SR pipeline")
    require(manifest["outputs"]["pos_emb.pt"] == SOURCE_LOCK["pos_emb.pt"], "Original embedding changed")
    return manifest


def validate_media(proof, size):
    require([proof.get(k) for k in ("width", "height", "frames", "fps")] == [*size, 362, 24],
            "C14 geometry/time differs; never crop to 360 frames")
    require(proof.get("full_decode") is True, "Full media decode is required")


def validate_release(release, source_commit):
    require(release.get("schema") == "seedvr2-c14-publication/1" and
            release.get("status") == "PASS_PUBLISHED_AND_PULLED_CPU_VALIDATED", "Use this recipe's published CI image.json")
    require(release.get("source_commit") == source_commit and release.get("parent_image") == PARENT,
            "Release and running code markers differ")
    require(re.fullmatch(re.escape(PARENT.split("@")[0]) + r"@sha256:[0-9a-f]{64}", release.get("deployment_image", "")),
            "Missing immutable C14 deployment image")
    return release["deployment_image"]
