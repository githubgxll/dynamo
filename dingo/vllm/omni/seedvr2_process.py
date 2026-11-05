# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in supervisor for one owned, loopback-only SeedVR2 engine."""

from __future__ import annotations

import asyncio
import json
import hashlib
import re
import os
import signal
import socket
import uuid
from pathlib import Path

import aiohttp

from dingo.common.video_super_resolution import H3_SR_PROFILE, H3_SR_PROFILE_V2, H3_SR_PROFILE_V3, H3_SR_PROFILES
from dingo.vllm.omni.seedvr2_client import SeedVR2Client
from dingo.vllm.omni.seedvr2_execution import SeedVR2Execution
from dingo.vllm.omni.seedvr2_handler import SeedVR2Handler


def measurement_source_hashes(package_root, *, include_geometry=False):
    # Native /2 validation compares each rank's actual pipeline, telemetry and
    # IPC hashes against the declaration, in addition to request correlation.
    sources = ["vllm_omni/diffusion/models/seedvr2/pipeline_seedvr2.py",
               "vllm_omni/diffusion/models/seedvr2/sr_telemetry.py",
               "vllm_omni/diffusion/ipc.py",
               "vllm_omni/diffusion/models/seedvr2/long_video.py",
               "dingo/vllm/omni/seedvr2_handler.py"]
    if include_geometry:
        sources += ["vllm_omni/diffusion/models/seedvr2/long_video_support.py",
                    "dingo/video_gateway/adapters/h3_output_policy.py",
                    "dingo/common/video_super_resolution.py",
                    "dingo/vllm/omni/seedvr2_media.py"]
    return {name: hashlib.sha256((package_root / name).read_bytes()).hexdigest() for name in sources}


class SeedVR2Process:
    def __init__(self, config: dict):
        if not {"model", "work_root", "gpus", "engine_identity"} <= set(config) or set(config) - {"model", "work_root", "gpus", "engine_identity", "profile", "image_digest"}:
            raise ValueError("invalid owned SeedVR2 process configuration")
        # Keep explicit profiles; the launcher derives both sequence and VAE
        # parallelism from this count and inherits the Pod's GPU visibility.
        if type(config["gpus"]) is not int or config["gpus"] not in (1, 2, 4, 8):
            raise ValueError(
                "composite SR profile requires one, two, four or eight allocated GPUs"
            )
        self.profile = config.get("profile", H3_SR_PROFILE)
        if self.profile not in H3_SR_PROFILES:
            raise ValueError("unsupported SR profile")
        if self.profile in {H3_SR_PROFILE_V2,H3_SR_PROFILE_V3}:
            if config["gpus"] not in (4, 8) or os.getenv("DINGO_VIDEO_SEEDVR2_CUDA_VISIBLE_DEVICES"):
                raise ValueError("SR v2/v3 requires four or eight shared GPUs")
            if not re.fullmatch(r"[^\s]{1,300}@sha256:[0-9a-f]{64}", config.get("image_digest", "")):
                raise ValueError("SR v2 requires the pinned image digest")
        self.cuda_visible_devices = os.environ.get("DINGO_VIDEO_SEEDVR2_CUDA_VISIBLE_DEVICES")
        if self.cuda_visible_devices is not None:
            devices = self.cuda_visible_devices.split(",")
            base_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if (len(devices) != config["gpus"] or len(set(devices)) != len(devices)
                    or not all(d.startswith("GPU-") for d in devices)
                    or not all(d.startswith("GPU-") for d in base_devices)
                    or set(devices) & set(base_devices)):
                raise ValueError("dedicated SR GPUs require disjoint explicit GPU UUID lists")
        elif config["gpus"] == 1:
            raise ValueError("single-GPU SR requires a dedicated allocated GPU")
        for key in ("model", "work_root"):
            if not isinstance(config[key], str) or not Path(config[key]).is_absolute():
                raise ValueError(f"SR {key} must be an absolute local path")
        if (
            not isinstance(config["engine_identity"], str)
            or not config["engine_identity"]
        ):
            raise ValueError("SR requires an engine/model revision identity")
        for filename in ("seedvr2_ema_3b_fp16.safetensors", "ema_vae_fp16.safetensors"):
            if not (Path(config["model"]) / filename).is_file():
                raise ValueError(f"missing converted FP16 SR weight: {filename}")
        self.config = config
        self.incarnation = uuid.uuid4().hex
        self.process = None
        self.log = None
        self.ready = False
        self.root = Path(config["work_root"]) / self.incarnation
        self.client = SeedVR2Client()

    def healthy(self):
        return (
            self.ready and self.process is not None and self.process.returncode is None
        )

    async def start(self, handler):
        self.root.mkdir(parents=True, exist_ok=False)
        # Refuse to attach to any pre-existing listener we do not own.
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 8124))
        n = str(self.config["gpus"])
        argv = [
            "vllm-omni",
            "serve",
            self.config["model"],
            "--omni",
            "--model-class-name",
            "SeedVR2Pipeline",
            "--dtype",
            "float16",
            "--enforce-eager",
            "--cache-backend",
            "none",
            "--num-gpus",
            n,
            "--distributed-executor-backend",
            "mp",
            "--vae-use-tiling",
            "--stage-overrides",
            json.dumps(
                {
                    "0": {
                        "ulysses_degree": int(n),
                        "vae_patch_parallel_size": int(n),
                        "vae_parallel_mode": "tile" if int(n) == 1 else "spatial_shard_height",
                    }
                }
            ),
            "--api-server-count",
            "1",
            "--tensor-parallel-size",
            "1",
            "--ring-degree",
            "1",
            "--cfg-parallel-size",
            "1",
            "--allgather-degree",
            "1",
            "--text-encoder-tp-size",
            "1",
            "--ulysses-mode",
            "strict",
            "--no-ulysses-a2a-permute",
            "--max-num-seqs",
            "1",
            "--request-batch-max-wait-ms",
            "0",
            "--host",
            "127.0.0.1",
            "--port",
            "8124",
            "--init-timeout",
            "10800",
            "--stage-init-timeout",
            "10800",
        ]
        env = dict(os.environ)
        if self.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        for key in list(env):
            if key.startswith(
                ("H3_", "DIFFUSION_CACHE_", "VLLM_OMNI_SEEDVR2_")
            ) or key in {
                "MASTER_PORT",
                "MASTER_ADDR",
                "RANK",
                "WORLD_SIZE",
                "LOCAL_RANK",
                "PYTHONPATH",
                "PYTHONHOME",
            }:
                env.pop(key, None)
        env.update(
            {
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "29610",
                "DINGO_ENABLE_MINIMAX_H3_PATCHES": "0",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                "VLLM_OMNI_SEEDVR2_SHARDED_FRAME_PIXELS": "3791872",
                "VLLM_OMNI_SEEDVR2_SHARDED_CLIP_PIXELS": "64461824",
                "VLLM_OMNI_SEEDVR2_MAX_FRAMES": "17",
                "VLLM_OMNI_SEEDVR2_LONG_MAX_WINDOW": "17",
                "VLLM_OMNI_SEEDVR2_LONG_OUTPUT_DIR": str(self.root / "jobs"),
                "VLLM_OMNI_SEEDVR2_LONG_JOB_TTL_SECONDS": "86400",
            }
        )
        measurement_identity = None
        if self.profile in {H3_SR_PROFILE_V2,H3_SR_PROFILE_V3}:
            metrics_dir = self.root / "metrics"
            metrics_dir.mkdir()
            env.update({"VLLM_OMNI_SEEDVR2_SHARDED_CLIP_PIXELS": "246471680",
                        "VLLM_OMNI_SEEDVR2_MAX_FRAMES": "65",
                        "VLLM_OMNI_SEEDVR2_LONG_MAX_WINDOW": "65",
                        "VLLM_OMNI_SEEDVR2_LONG_JOB_TTL_SECONDS": "604800",
                        "SEEDVR2_METRICS_DIR": str(metrics_dir)})
            package_root = Path(__file__).resolve().parents[3]
            measurement_identity = {
                "image_digest": self.config["image_digest"], "expected_world_size": self.config["gpus"],
                "source_hashes": measurement_source_hashes(package_root, include_geometry=self.profile == H3_SR_PROFILE_V3),
                "model_manifest_hash": hashlib.sha256((Path(self.config["model"]) / "conversion-manifest.json").read_bytes()).hexdigest(),
            }
        if self.profile == H3_SR_PROFILE_V3:
            env.update({"VLLM_OMNI_SEEDVR2_SHARDED_FRAME_PIXELS":"3850240",
                        "VLLM_OMNI_SEEDVR2_SHARDED_CLIP_PIXELS":"250265600"})
        if int(n) == 1:
            env["DINGO_SEEDVR2_DEDICATED_SINGLE_GPU"] = "1"
        self.log = (self.root / "service.log").open("xb")
        try:
            self.process = await asyncio.create_subprocess_exec(
                *argv,
                env=env,
                stdout=self.log,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            async with asyncio.timeout(10800):
                async with aiohttp.ClientSession(
                    trust_env=False, timeout=aiohttp.ClientTimeout(total=5)
                ) as session:
                    while True:
                        if self.process.returncode is not None:
                            raise RuntimeError(
                                "owned SeedVR2 process exited before readiness"
                            )
                        try:
                            async with session.get(
                                "http://127.0.0.1:8124/health", allow_redirects=False
                            ) as response:
                                if response.status == 200:
                                    self.ready = True
                                    break
                        except (aiohttp.ClientError, TimeoutError):
                            pass
                        await asyncio.sleep(2)
            execution = SeedVR2Execution(
                self.client, owner=self.incarnation, quarantine=self.quarantine
            )
            return SeedVR2Handler(
                handler,
                execution,
                engine_identity=self.config["engine_identity"],
                healthy=self.healthy,
                profile=self.profile,
                measurement_identity=measurement_identity,
            )
        except BaseException:
            await self.close()
            raise

    def quarantine(self, reason):
        self.ready = False

    async def close(self):
        self.ready = False
        if self.process is not None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self.process.wait(), timeout=30)
            except TimeoutError:
                pass
            # Also clean surviving ranks if the API parent exited first.
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await self.process.wait()
        if self.log is not None:
            self.log.close()
