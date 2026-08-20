#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Apply Dingo's MiniMax-H3 vLLM-Omni patches to the installed vllm-omni site-packages.
#
# Two kinds of patches are shipped:
#   1. Source overlays: replacement .py files copied verbatim into site-packages.
#      These implement upstream PRs (#5990 fused Q/K RMSNorm + RoPE, #6173 strict
#      Ulysses rank-local boundary) that are not in vllm-omni 0.27.0rc1.
#   2. Runtime overlay modules: monkey-patch scripts copied to a fixed location
#      (/opt/dynamo/vllm-omni-patches) and loaded on every Python startup by the
#      generated dingo_vllm_omni_patches.py bootstrap (triggered via a .pth file,
#      not sitecustomize.py — the base image's apport sitecustomize shadows any
#      dist-packages/sitecustomize.py).  Each overlay is gated by an environment
#      flag so a deployment can opt in/out without rebuilding the image.
#
# The script fails fast on any checksum mismatch so a vllm-omni version bump is
# caught at image build time instead of producing a broken worker.

set -euo pipefail

: "${PYTHON_SITE_PACKAGES:?PYTHON_SITE_PACKAGES must be set (e.g. /usr/local/lib/python3.12/dist-packages)}"
: "${PATCHES_DIR:=/opt/dynamo/vllm-omni-patches}"
: "${SRC_DIR:=/tmp/vllm_omni_patches}"

VLLM_OMNI_DIR="${PYTHON_SITE_PACKAGES}/vllm_omni"
PATCHED_VLLM_VERSION="0.27.1"
PATCHED_VLLM_OMNI_VERSION="0.27.0rc1"

if [ ! -d "${VLLM_OMNI_DIR}" ]; then
    echo "ERROR: vllm_omni not found at ${VLLM_OMNI_DIR}" >&2
    exit 1
fi

installed_vllm_version="$(
    python3 -c 'from importlib.metadata import version; print(version("vllm"))'
)"
installed_vllm_omni_version="$(
    python3 -c 'from importlib.metadata import version; print(version("vllm-omni"))'
)"
if [ "${installed_vllm_version}" != "${PATCHED_VLLM_VERSION}" ]; then
    echo "ERROR: MiniMax-H3 patches are validated only for vllm ${PATCHED_VLLM_VERSION}." >&2
    echo "  installed ${installed_vllm_version}" >&2
    echo "  Re-evaluate every overlay and runtime hook before changing the pinned version." >&2
    exit 1
fi
if [ "${installed_vllm_omni_version}" != "${PATCHED_VLLM_OMNI_VERSION}" ]; then
    echo "ERROR: MiniMax-H3 patches are validated only for vllm-omni ${PATCHED_VLLM_OMNI_VERSION}." >&2
    echo "  installed ${installed_vllm_omni_version}" >&2
    echo "  Re-evaluate every overlay and runtime hook before changing the pinned version." >&2
    exit 1
fi
echo "OK: vllm ${installed_vllm_version} + vllm-omni ${installed_vllm_omni_version} are explicitly supported"

echo "=== [1/3] Verifying pinned vllm-omni base files ==="
# The checksums below pin the exact vllm-omni 0.27.0rc1 files that the source
# overlays replace.  A mismatch means vllm-omni was upgraded and the overlays
# must be re-validated.
declare -A PINNED_BASES
# Source-overlay targets (replaced in step 2).
PINNED_BASES["${VLLM_OMNI_DIR}/diffusion/models/minimax_h3/minimax_h3_transformer.py"]="90befde4d0e4c2e11a4bb38275e6b945d6f0a08e79fe0975e021c29ddc0b6c9b"
PINNED_BASES["${VLLM_OMNI_DIR}/diffusion/cache/teacache/extractors.py"]="6db51d5e3c858dbb831fd4a509726ed2500ca861ade7661ddfa8e1893ee1dae6"
PINNED_BASES["${VLLM_OMNI_DIR}/diffusion/models/minimax_h3/denoise_loop.py"]="a378e1b953fd4b7e71cf3cf14f999ad1b778371828baac1448fca9472fefc26c"
# Runtime-overlay API surface (checked but not overwritten). DiffusionFormatter
# imports _encode_video_bytes / _normalize_video_outputs from these modules at
# request time; a vllm-omni upgrade that changes their signatures must be
# caught at build time rather than producing broken MP4 output at runtime.
PINNED_BASES["${VLLM_OMNI_DIR}/entrypoints/openai/video_api_utils.py"]="1ee9a9292a59a8a2cbe94de5331bbfdef0a7fc94877bceb186d6d3cc67813131"
PINNED_BASES["${VLLM_OMNI_DIR}/entrypoints/openai/serving_video.py"]="c7ad58967db6e0860f939eb3603fa167619ae8d253f5cc8780b873fb0bc613c0"

for target in "${!PINNED_BASES[@]}"; do
    if [ ! -f "${target}" ]; then
        echo "ERROR: pinned base file missing: ${target}" >&2
        exit 1
    fi
    actual="$(sha256sum "${target}" | awk '{print $1}')"
    expected="${PINNED_BASES[$target]}"
    if [ "${actual}" != "${expected}" ]; then
        echo "ERROR: checksum mismatch for ${target}" >&2
        echo "  expected ${expected}" >&2
        echo "  got      ${actual}" >&2
        echo "  vllm-omni may have been upgraded; re-validate the overlay." >&2
        exit 1
    fi
done
echo "OK: pinned base files verified"

echo "=== [2/3] Installing source overlays (PR #5990 + #6173) ==="
# Source overlays replace the pinned base files in-place.
install -m 0644 "${SRC_DIR}/vllm_omni/diffusion/layers/fused_qk_norm_rope.py" \
    "${VLLM_OMNI_DIR}/diffusion/layers/fused_qk_norm_rope.py"
install -m 0644 "${SRC_DIR}/vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py" \
    "${VLLM_OMNI_DIR}/diffusion/models/minimax_h3/minimax_h3_transformer.py"
install -m 0644 "${SRC_DIR}/vllm_omni/diffusion/cache/teacache/extractors.py" \
    "${VLLM_OMNI_DIR}/diffusion/cache/teacache/extractors.py"
echo "OK: source overlays installed"

echo "=== [3/3] Installing runtime overlay modules ==="
mkdir -p "${PATCHES_DIR}"
install -m 0755 "${SRC_DIR}/runtime/ffprobe_pyav.py" "${PATCHES_DIR}/ffprobe_pyav.py"
install -m 0755 "${SRC_DIR}/runtime/vae_compile_optimization.py" "${PATCHES_DIR}/vae_compile_optimization.py"
install -m 0755 "${SRC_DIR}/runtime/ref2va_decode_optimization.py" "${PATCHES_DIR}/ref2va_decode_optimization.py"

# Install a bootstrap module into site-packages that loads the runtime overlays
# on every Python startup (including multiprocessing spawn children).  A .pth
# file is used instead of sitecustomize.py because the base image's
# /usr/lib/python3.12/sitecustomize.py (apport hook) shadows any
# dist-packages/sitecustomize.py — the generated sitecustomize.py was silently
# never loaded, so the FP8+HSDP compatibility patch did not apply in spawned
# workers (FSDP non-contiguous parameter error at startup).
#
# Each overlay is gated by an environment flag (default reflects the validated
# deployment state).  All MiniMax-H3 runtime hooks are opt-in so importing this
# image for unrelated models (for example GLM or DeepSeek) does not import or
# mutate vLLM-Omni's MiniMax-H3 modules.  FP8+HSDP is additionally gated by the
# presence of H3_DIFFUSION_QUANTIZATION_CONFIG.
cat > "${PYTHON_SITE_PACKAGES}/dingo_vllm_omni_patches.py" <<'BOOTSTRAP'
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Auto-generated by install_vllm_omni_patches.sh. Loads Dingo MiniMax-H3
vLLM-Omni runtime overlays on every Python process (including spawn children).

Triggered by a .pth file so it runs even when the base image's system
sitecustomize.py shadows dist-packages/sitecustomize.py."""

from __future__ import annotations

import os
import sys
from importlib.metadata import version

_PATCHES_DIR = "/opt/dynamo/vllm-omni-patches"
_PATCHED_VLLM_VERSION = "0.27.1"
_PATCHED_VLLM_OMNI_VERSION = "0.27.0rc1"
if _PATCHES_DIR not in sys.path:
    sys.path.insert(0, _PATCHES_DIR)

_LOADED = False


def _env_enabled(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _enable_online_fp8_hsdp() -> None:
    if not os.getenv("H3_DIFFUSION_QUANTIZATION_CONFIG", "").strip():
        return
    from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
    from vllm.model_executor.layers.quantization.online.fp8 import (
        Fp8PerTensorOnlineLinearMethod,
    )
    from vllm_omni.diffusion.quantization import hsdp_fp8

    supported_types = hsdp_fp8.Fp8LinearMethod
    if isinstance(supported_types, tuple):
        if (
            Fp8LinearMethod in supported_types
            and Fp8PerTensorOnlineLinearMethod in supported_types
        ):
            return
        raise RuntimeError("Unexpected pre-existing FP8 HSDP type tuple")
    if supported_types is not Fp8LinearMethod:
        raise RuntimeError("Unexpected pre-existing FP8 HSDP compatibility patch")

    hsdp_fp8.Fp8LinearMethod = (
        Fp8LinearMethod,
        Fp8PerTensorOnlineLinearMethod,
    )
    print(
        "[vllm-omni-patch] enabled online FP8 HSDP handling "
        f"for pid={os.getpid()}",
        flush=True,
    )


def _enable_ref2va_media_probe() -> None:
    from ffprobe_pyav import probe_audio_metadata, probe_video_metadata
    from vllm_omni.diffusion.models.minimax_h3 import reference_video

    reference_video._probe_video = probe_video_metadata
    reference_video._probe_audio = probe_audio_metadata
    print(
        f"[vllm-omni-patch] enabled Ref2VA PyAV media probe for pid={os.getpid()}",
        flush=True,
    )


def _enable_vae_regional_compile() -> None:
    from vae_compile_optimization import install

    install()


def _enable_ref2va_decode_optimization() -> None:
    from ref2va_decode_optimization import install

    install()


def _skip(name: str, exc: BaseException) -> None:
    print(
        f"[vllm-omni-patch] skipped {name} for pid={os.getpid()}: "
        f"{type(exc).__name__}: {exc}",
        flush=True,
    )


def load() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True

    installed_vllm_version = version("vllm")
    installed_vllm_omni_version = version("vllm-omni")
    if (
        installed_vllm_version != _PATCHED_VLLM_VERSION
        or installed_vllm_omni_version != _PATCHED_VLLM_OMNI_VERSION
    ):
        raise RuntimeError(
            "Refusing to load MiniMax-H3 runtime patches validated for "
            f"vllm {_PATCHED_VLLM_VERSION} + vllm-omni "
            f"{_PATCHED_VLLM_OMNI_VERSION} into vllm {installed_vllm_version} "
            f"+ vllm-omni {installed_vllm_omni_version}. "
            "Re-evaluate the patches for the installed version first."
        )

    # FP8+HSDP is required for the validated online FP8 configuration.
    # Guarded so a non-MiniMax-H3 vLLM worker that shares this image is unaffected.
    if os.getenv("H3_DIFFUSION_QUANTIZATION_CONFIG", "").strip():
        try:
            _enable_online_fp8_hsdp()
        except ImportError as exc:
            _skip("online FP8 HSDP", exc)
    # Ref2VA media probe replaces the missing ffprobe binary with PyAV.
    if _env_enabled("H3_ENABLE_REF2VA_MEDIA_PROBE", False):
        try:
            _enable_ref2va_media_probe()
        except ImportError as exc:
            _skip("Ref2VA media probe", exc)
    # VAE regional compile (PR #5979) is explicitly enabled by H3 deployments.
    if _env_enabled("H3_ENABLE_VIDEO_VAE_REGIONAL_COMPILE", False):
        try:
            _enable_vae_regional_compile()
        except ImportError as exc:
            _skip("VAE regional compile", exc)
    # Ref2VA selective/full FFmpeg RGB decode (PR #6064) is opt-in.
    if _env_enabled("H3_ENABLE_REF2VA_REFERENCE_DECODE_OPTIMIZATION", False):
        try:
            _enable_ref2va_decode_optimization()
        except ImportError as exc:
            _skip("Ref2VA decode optimization", exc)

    # VAE uint8 output path (PR #5937 CUDA-safe subset) is opt-in.
    if _env_enabled("H3_ENABLE_VAE_UINT8_OPTIMIZATION", False):
        try:
            import inspect

            import numpy as np
            import torch
            from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3
            from vllm_omni.entrypoints.openai import video_api_utils
            decode_parameters = inspect.signature(
                pipeline_minimax_h3.MiniMaxH3Pipeline.decode
            ).parameters
            if set(decode_parameters) != {
                "self",
                "video_latent",
                "audio_latent",
                "height",
                "width",
            }:
                raise RuntimeError("Unexpected MiniMaxH3Pipeline.decode signature")

            def _decode_uint8(
                self,
                video_latent: torch.Tensor,
                audio_latent: torch.Tensor,
                *,
                height: int,
                width: int,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                with self._component_on_device(self.video_vae):
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.float16,
                        enabled=True,
                        cache_enabled=False,
                    ):
                        video = self.video_vae.decode_latent(video_latent)
                video = video[..., :height, :width]
                video = video.clamp(0, 1).mul(255).round().to(torch.uint8)
                with self._component_on_device(self.audio_vae):
                    audio = self.audio_vae.decode_latent(audio_latent)
                return video, audio

            def _minimax_h3_post_process_uint8(output, output_type: str = "np"):
                if not isinstance(output, tuple) or len(output) != 2:
                    return output
                video, audio = output
                if output_type == "latent":
                    return output
                if output_type == "np":
                    video = video.detach().cpu()
                    if video.is_floating_point():
                        video = (
                            video.float()
                            .clamp(0, 1)
                            .permute(0, 2, 3, 4, 1)
                            .numpy()
                        )
                        video = [sample for sample in video]
                    else:
                        video = (
                            video.permute(0, 2, 3, 4, 1)
                            .squeeze(0)
                            .contiguous()
                            .numpy()
                        )
                        if video.ndim != 4 or video.shape[-1] != 3:
                            raise ValueError(
                                "MiniMax-H3 uint8 output must have shape (T,H,W,3), "
                                f"got {video.shape}"
                            )
                        video = [video[index] for index in range(video.shape[0])]
                    audio = audio.detach().float().cpu().numpy()

                from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3

                return {
                    "video": video,
                    "audio": audio,
                    "audio_sample_rate": pipeline_minimax_h3.MINIMAX_H3_AUDIO_SAMPLE_RATE,
                    "fps": pipeline_minimax_h3.MINIMAX_H3_FPS,
                }

            original_coerce = video_api_utils._coerce_video_to_uint8_frames

            def _coerce_validated_rgb_uint8(video):
                if (
                    isinstance(video, np.ndarray)
                    and video.dtype == np.uint8
                    and video.ndim == 4
                    and video.shape[-1] == 3
                ):
                    return np.ascontiguousarray(video)
                if (
                    isinstance(video, list)
                    and video
                    and all(
                        isinstance(frame, np.ndarray)
                        and frame.dtype == np.uint8
                        and frame.ndim == 3
                        and frame.shape[-1] == 3
                        and frame.shape == video[0].shape
                        for frame in video
                    )
                ):
                    return np.ascontiguousarray(np.stack(video, axis=0))
                return original_coerce(video)

            pipeline_minimax_h3.MiniMaxH3Pipeline.decode = _decode_uint8
            pipeline_minimax_h3._minimax_h3_post_process = (
                _minimax_h3_post_process_uint8
            )
            video_api_utils._coerce_video_to_uint8_frames = _coerce_validated_rgb_uint8
            print(
                "[vllm-omni-patch] enabled MiniMax-H3 lean VAE + validated RGB uint8 path "
                f"for pid={os.getpid()}",
                flush=True,
            )
        except ImportError as exc:
            _skip("MiniMax-H3 VAE uint8", exc)


load()
BOOTSTRAP

chmod 0644 "${PYTHON_SITE_PACKAGES}/dingo_vllm_omni_patches.py"

# Create a .pth file that triggers the bootstrap only for an explicitly opted-in
# MiniMax-H3 process. Unrelated GLM/DeepSeek workers do not import the bootstrap
# module at all. The same explicit flag is inherited by multiprocessing spawn
# children, so H3 still receives the patches in every diffusion rank.
echo 'import os; os.getenv("DINGO_ENABLE_MINIMAX_H3_PATCHES") == "1" and __import__("dingo_vllm_omni_patches")' > "${PYTHON_SITE_PACKAGES}/dingo_vllm_omni_patches.pth"
chmod 0644 "${PYTHON_SITE_PACKAGES}/dingo_vllm_omni_patches.pth"

# Prove that a clean non-H3 Python startup does not even import the bootstrap or
# any model runtime, then prove that explicit H3 opt-in loads the bootstrap while
# individual feature flags remain off. This catches both cross-model regressions
# and future site-packages/bootstrap failures at image build time.
env \
    -u DINGO_ENABLE_MINIMAX_H3_PATCHES \
    -u H3_DIFFUSION_QUANTIZATION_CONFIG \
    -u H3_ENABLE_REF2VA_MEDIA_PROBE \
    -u H3_ENABLE_VIDEO_VAE_REGIONAL_COMPILE \
    -u H3_ENABLE_REF2VA_REFERENCE_DECODE_OPTIMIZATION \
    -u H3_ENABLE_VAE_UINT8_OPTIMIZATION \
    python3 -c 'import sys; assert "dingo_vllm_omni_patches" not in sys.modules; assert not any(name == "vllm" or name.startswith("vllm.") or name == "vllm_omni" or name.startswith("vllm_omni.") or name == "torch" or name.startswith("torch.") or name == "numpy" or name.startswith("numpy.") for name in sys.modules)'
env \
    -u H3_DIFFUSION_QUANTIZATION_CONFIG \
    -u H3_ENABLE_REF2VA_MEDIA_PROBE \
    -u H3_ENABLE_VIDEO_VAE_REGIONAL_COMPILE \
    -u H3_ENABLE_REF2VA_REFERENCE_DECODE_OPTIMIZATION \
    -u H3_ENABLE_VAE_UINT8_OPTIMIZATION \
    DINGO_ENABLE_MINIMAX_H3_PATCHES=1 \
    python3 -c 'import sys; assert "dingo_vllm_omni_patches" in sys.modules; assert not any(name == "vllm" or name.startswith("vllm.") or name == "vllm_omni" or name.startswith("vllm_omni.") or name == "torch" or name.startswith("torch.") or name == "numpy" or name.startswith("numpy.") for name in sys.modules)'

echo "OK: runtime overlay modules installed to ${PATCHES_DIR}"
echo "=== vLLM-Omni patches installed ==="
