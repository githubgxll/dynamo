{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/vllm_runtime.Dockerfile ===
##################################
########## Runtime Image #########
##################################

{% if platform == "multi" %}
FROM --platform=linux/amd64 ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS vllm_runtime_amd64
FROM --platform=linux/arm64 ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS vllm_runtime_arm64
FROM vllm_runtime_${TARGETARCH} AS pre_runtime
{% else %}
FROM ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS pre_runtime
{% endif %}

ARG PYTHON_VERSION
ARG ENABLE_KVBM
ARG ENABLE_GPU_MEMORY_SERVICE
ARG VLLM_OMNI_REF
ARG TRANSFORMERS_VERSION
ARG TOKENIZERS_VERSION
ARG NIXL_REF
{% if device == "cuda" %}
ARG CUDA_MAJOR
{% endif %}
ARG MODELEXPRESS_VERSION

WORKDIR /workspace

ENV DYNAMO_HOME=/opt/dynamo
ENV HOME=/home/dynamo
{% if device != "cuda" %}
ENV PATH=/usr/local/ucx/bin:/usr/local/bin/etcd:${PATH}
{% else %}
ENV PATH=/usr/local/bin/etcd:${PATH}
{% endif %}

{% if device != "cuda" %}
ARG SITE_PACKAGES=/usr/local/lib/python${PYTHON_VERSION}/dist-packages
ENV TORCH_LIB_DIR=${SITE_PACKAGES}/torch/lib
{% if device == "xpu" %}
ENV NIXL_PREFIX=/opt/intel/intel_nixl
ENV NIXL_LIB_DIR=${NIXL_PREFIX}/lib/x86_64-linux-gnu
# vLLM 0.27.1's XPU image installs the oneAPI runtime and SYCL headers in
# /opt/venv through the intel-sycl-rt wheel. Do not set ONEAPI_ROOT to the
# removed /opt/intel/oneapi tree: Triton gives that variable priority over its
# wheel-metadata fallback and would search a nonexistent compiler include path.
{% elif device == "cpu" %}
ENV NIXL_PREFIX=/opt/nvidia/nvda_nixl
ENV NIXL_LIB_DIR=${NIXL_PREFIX}/lib/x86_64-linux-gnu
{% endif %}
ENV NIXL_PLUGIN_DIR=${NIXL_LIB_DIR}/plugins
ENV LD_LIBRARY_PATH=${NIXL_LIB_DIR}:${NIXL_PLUGIN_DIR}:/usr/local/ucx/lib:/usr/local/ucx/lib/ucx:${TORCH_LIB_DIR}:${LD_LIBRARY_PATH:-}
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
{% else %}
# Expose libnixl.so from the upstream nixl-cu${CUDA_MAJOR} PyPI wheel through a
# stable prefix so non-Python consumers use the same NIXL copy that Python imports.
# This keeps Rust nixl-sys dlopen("libnixl.so") from falling into stub mode in
# processes that do not import the nixl Python package first.
ARG SITE_PACKAGES=/usr/local/lib/python${PYTHON_VERSION}/dist-packages
ENV NIXL_PREFIX=/opt/dynamo/nixl \
    NIXL_LIB_DIR=/opt/dynamo/nixl \
    NIXL_PLUGIN_DIR=/opt/dynamo/nixl/plugins
COPY --chmod=755 container/deps/vllm/install_nixl_from_wheel.sh /usr/local/bin/install_nixl_from_wheel
RUN install_nixl_from_wheel \
    --cuda-major "${CUDA_MAJOR}" \
    --site-packages "${SITE_PACKAGES}" \
    --prefix "${NIXL_PREFIX}" \
    --skip-headers
ENV LD_LIBRARY_PATH=${NIXL_LIB_DIR}:${NIXL_PLUGIN_DIR}:${LD_LIBRARY_PATH:-}
{% endif %}

# Install NATS and ETCD
COPY --from=dynamo_base /usr/bin/nats-server /usr/bin/nats-server
COPY --from=dynamo_base /usr/local/bin/etcd/ /usr/local/bin/etcd/
COPY --from=dynamo_base /opt/uv/bin/uv /opt/uv/bin/uvx /opt/uv/bin/
ENV PATH=/opt/uv/bin:${PATH}

{% if device == "cuda" %}
# Bring base-image OS packages up to the current patch releases published in
# the distro archives. --only-upgrade skips anything not already installed, so
# no new packages are added; versions are left unpinned so a cache-busted
# rebuild picks up the newest patch level (BuildKit reuses this layer otherwise).
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends --only-upgrade \
        dirmngr \
        gnupg \
        gnupg-utils \
        gnupg2 \
        gpg \
        gpg-agent \
        gpgconf \
        gpgsm \
        gpgv \
        keyboxd \
        libssl3t64 \
        openssl && \
    rm -rf /var/lib/apt/lists/*
{% endif %}

# Create dynamo user with group 0 for OpenShift compatibility.
# Pin -u 1000 explicitly: the vllm/vllm-openai >=0.22 image ships a `vllm` user at
# UID 2000, so after freeing 1000 (ubuntu) useradd would otherwise auto-assign the
# next-highest UID (2001) and fail the `id -u dynamo` == 1000 assertion below.
RUN userdel -r ubuntu > /dev/null 2>&1 || true \
    && useradd -u 1000 -m -s /bin/bash -g 0 dynamo \
    && [ `id -u dynamo` -eq 1000 ] \
    && mkdir -p /home/dynamo/.cache/vllm /opt/dynamo \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && chown dynamo:0 /home/dynamo /home/dynamo/.cache /home/dynamo/.cache/vllm /opt/dynamo /workspace \
    # Arbitrary OpenShift UIDs need to create the vLLM and Triton caches under $HOME.
    && chmod g+rwx /home/dynamo /home/dynamo/.cache /home/dynamo/.cache/vllm \
    && mkdir -p /etc/profile.d \
    && echo 'umask 002' > /etc/profile.d/00-umask.sh

# FlashInfer creates package-local cubin symlinks at runtime. Grant group 0
# write access so arbitrary OpenShift UIDs can initialize the cubin cache.
RUN SITE_PACKAGES="$(python3 -c 'import site; print(site.getsitepackages()[0])')" && \
    CUBINS_DIR="$SITE_PACKAGES/flashinfer_cubin/cubins" && \
    if [ -d "$CUBINS_DIR" ]; then \
        find "$CUBINS_DIR" -type d -exec chmod g+rwx {} + ; \
    fi

{% if device != "cuda" %}
# Copy UCX and NIXL from wheel_builder for CPU/XPU devices
# (CUDA devices use NIXL from upstream vLLM wheels)
COPY --from=wheel_builder /usr/local/ucx /usr/local/ucx
COPY --chown=dynamo:0 --from=wheel_builder ${NIXL_PREFIX} ${NIXL_PREFIX}
{% if device == "xpu" %}
# XPU NIXL uses lib/x86_64-linux-gnu; copy to NIXL_LIB_DIR to ensure lib dir is populated
COPY --chown=dynamo:0 --from=wheel_builder /opt/intel/intel_nixl/lib/x86_64-linux-gnu/. ${NIXL_LIB_DIR}/
{% endif %}
# Copy NIXL Python wheels
COPY --chown=dynamo:0 --from=wheel_builder /opt/dynamo/dist/nixl/ /opt/dynamo/wheelhouse/nixl/
COPY --chown=dynamo:0 --from=wheel_builder /workspace/nixl/build/src/bindings/python/nixl-meta/nixl-*.whl /opt/dynamo/wheelhouse/nixl/

# Install RDMA libraries required for UCX to find RDMA devices
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libibverbs1 \
        rdma-core \
        ibverbs-utils \
        libibumad3 \
        libnuma1 \
        librdmacm1 \
        ibverbs-providers && \
    rm -rf /var/lib/apt/lists/*
{% endif %}

{% if device == "xpu" %}
ADD --checksum=sha256:f60e802b6f41350393e34b24793db888a8be514054769bd17e7a6e9c0c058b87 \
    https://github.com/intel/xpumanager/releases/download/v1.3.6/xpu-smi_1.3.6_20260206.143628.1004f6cb.u24.04_amd64.deb \
    /tmp/xpu-smi.deb

# Install xpu-smi in the runtime stage so dev/local-dev inherit it, without
# explicitly changing the Intel compute runtime stack.
RUN apt-get update && \
    if command -v xpu-smi >/dev/null 2>&1; then \
        echo "xpu-smi already present in base image, skipping install"; \
    else \
        if apt-cache show intel-gsc >/dev/null 2>&1; then \
            apt-get install -y --no-install-recommends /tmp/xpu-smi.deb; \
        else \
            echo "WARNING: intel-gsc is not available from configured apt sources; skipping xpu-smi install"; \
        fi; \
    fi && \
    rm -f /tmp/xpu-smi.deb && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
{% endif %}

# Copy attribution files and wheels
COPY --chmod=664 --chown=dynamo:0 LICENSE /workspace/
COPY --chmod=775 --chown=dynamo:0 --from=wheel_builder /opt/dynamo/dist/*.whl /opt/dynamo/wheelhouse/

{% set pip_target = "--system" if device == "cuda" else "--python /opt/venv/bin/python" %}
{% set python_executable = "python3" if device == "cuda" else "/opt/venv/bin/python" %}
{# cuda installs into the system interpreter (/usr/local/bin); xpu and cpu run out
   of ${VIRTUAL_ENV} and prepend ${VIRTUAL_ENV}/bin to PATH. #}
{% set vllm_rs_link = "/usr/local/bin/vllm-rs" if device == "cuda" else "${VIRTUAL_ENV}/bin/vllm-rs" %}
{# Inline expression, not a block tag: render.py leaves trim_blocks off, so a tag
   on its own line inside the RUN breaks the backslash continuation. #}
{% set vllm_rs_required = "1" if device == "cuda" else "0" %}
{# TODO: Remove this workaround once bundled vllm-rs accepts extra output fields. #}
{% set vllm_rs_allowlist = "1" if target not in ("dev", "local-dev") else "0" %}
{% set vllm_rs_plugins = "modelexpress" if context.vllm.enable_modelexpress == "true" else "" %}

# Align Transformers and tokenizers before freezing Omni's protected dependencies.
RUN --mount=type=cache,id=uv-root-{{ context.dynamo.uv_version }},target=/root/.cache/uv,sharing=locked \
    export UV_CACHE_DIR=/root/.cache/uv && \
    uv pip install {{ pip_target }} --no-deps \
        "transformers==${TRANSFORMERS_VERSION}" "tokenizers==${TOKENIZERS_VERSION}"

{% if device != "cuda" %}
# NIXL meta package always tries to find a cuda-backend
# https://github.com/ai-dynamo/nixl/blob/v1.1.0/src/bindings/python/nixl-meta/nixl/__init__.py
#
# We therefore install nixl-cu* packages, and use LD_LIBRARY_PATH settings to point to our installation of nixl
# v1.1.0 nixl-cu13 has in-built RPATH point to conflicting built-in libs with symbols unsupported in non-cuda builds.
# we therefore avoid installing nixl-cu13

RUN --mount=type=cache,id=uv-root-{{ context.dynamo.uv_version }},target=/root/.cache/uv,sharing=locked \
    set -eu; \
    export UV_CACHE_DIR=/root/.cache/uv; \
    NIXL_VERSION="${NIXL_REF#v}"; \
    uv pip install \
        {{ pip_target }} --force-reinstall --no-deps \
        "nixl==${NIXL_VERSION}" \
        "nixl-cu12==${NIXL_VERSION}"
{% endif %}

# Install device-specific NIXL wheels for non-CUDA devices.
# These are custom-built in wheel_builder and required for dev builds to link against NIXL libraries.
{% if device != "cuda" %}
RUN --mount=type=cache,id=uv-root-{{ context.dynamo.uv_version }},target=/root/.cache/uv,sharing=locked \
    export UV_CACHE_DIR=/root/.cache/uv && \
    uv pip install {{ pip_target }} --no-deps /opt/dynamo/wheelhouse/nixl/nixl*.whl
{% endif %}

{% if target not in ("dev", "local-dev") %}
# Keep the upstream Python solve intact: install only Dynamo-owned wheels and
# suppress transitive dependency resolution unless a later validation proves a
# missing package must be added explicitly.

# Install Dynamo runtime wheels and optional KVBM/GMS wheels.
# Use --no-deps to prevent dependency conflicts (e.g., KVBM downgrading nixl).
RUN --mount=type=cache,id=uv-root-{{ context.dynamo.uv_version }},target=/root/.cache/uv,sharing=locked \
    export UV_CACHE_DIR=/root/.cache/uv && \
    uv pip install {{ pip_target }} --no-deps /opt/dynamo/wheelhouse/ai_dingo_runtime*.whl && \
    uv pip install {{ pip_target }} --no-deps /opt/dynamo/wheelhouse/ai_dingo*any.whl && \
    uv pip install {{ pip_target }} "redis>=6.2.0,<9.0.0" && \
    uv pip install {{ pip_target }} --no-deps /opt/dynamo/wheelhouse/aisimulate*.whl && \
    if [ "${ENABLE_KVBM}" = "true" ]; then \
        KVBM_WHEEL=$(ls /opt/dynamo/wheelhouse/kvbm*.whl 2>/dev/null | head -1); \
        if [ -n "$KVBM_WHEEL" ]; then uv pip install {{ pip_target }} --no-deps "$KVBM_WHEEL"; fi; \
    fi && \
    if [ "${ENABLE_GPU_MEMORY_SERVICE}" = "true" ]; then \
        GMS_WHEEL=$(ls /opt/dynamo/wheelhouse/gpu_memory_service*.whl 2>/dev/null | head -1); \
        if [ -n "$GMS_WHEEL" ]; then uv pip install {{ pip_target }} --no-deps "$GMS_WHEEL"; fi; \
    fi

# Launch-script examples use jq for readable curl output like the upstream omni
# image. SoX is intentionally NOT installed: vLLM-Omni replaced its sox audio path
# with a pure-numpy peak_normalize() (vllm_omni/utils/audio.py), pysox isn't
# installed, and nothing shells out to the sox binary — so `sox`/`libsox-fmt-all`
# were dead weight that only dragged in a GPL-2.0+ codec cluster (sox, libsox*,
# libao*, libmad0, libid3tag0, libltdl7) we'd then be redistributing. SoX is
# inherently GPL (no LGPL replacement), so the compliant fix is to not ship it.
# (sglang_runtime.Dockerfile is the reference codec-compliance pattern.)
# libjemalloc2 lets Dynamo processes opt into jemalloc via
# LD_PRELOAD or DYN_FRONTEND_JEMALLOC; it is not preloaded by default.
RUN set -eux; \
    printf '%s\n' \
        'deb http://mirrors.aliyun.com/ubuntu noble main' \
        'deb http://mirrors.aliyun.com/ubuntu noble-updates main' \
        'deb http://mirrors.aliyun.com/ubuntu noble-security main' \
        > /tmp/dingo-ubuntu.list; \
    apt-get \
        -o Dir::Etc::sourcelist=/tmp/dingo-ubuntu.list \
        -o Dir::Etc::sourceparts=- \
        -o Acquire::Retries=5 \
        update; \
    DEBIAN_FRONTEND=noninteractive apt-get \
        -o Dir::Etc::sourcelist=/tmp/dingo-ubuntu.list \
        -o Dir::Etc::sourceparts=- \
        -o Acquire::Retries=5 \
        install -y --no-install-recommends \
        jq libturbojpeg libjemalloc2; \
    rm -f /tmp/dingo-ubuntu.list; \
    ldconfig; \
    ldconfig -p | grep -q 'libturbojpeg.so.0'; \
    rm -rf /var/lib/apt/lists/*

# Layer the released vLLM-Omni package matching the pinned upstream ref while
# constraining packages already solved in the upstream vLLM image.
RUN --mount=type=bind,source=./container/deps/vllm/protected_packages.txt,target=/tmp/vllm_omni_protected_packages.txt \
    --mount=type=bind,source=./container/deps/vllm/install_vllm_omni.sh,target=/tmp/install_vllm_omni.sh \
    --mount=type=cache,id=uv-root-{{ context.dynamo.uv_version }},target=/root/.cache/uv,sharing=locked \
    set -eux; \
    export UV_CACHE_DIR=/root/.cache/uv; \
    export VLLM_OMNI_TARGET_DEVICE={{ device }}; \
    bash /tmp/install_vllm_omni.sh

# Apply Dingo's MiniMax-H3 vLLM-Omni patches (source overlays + runtime overlay
# modules) to the installed vllm-omni site-packages. The install script applies
# them only to the validated vLLM + vLLM-Omni version pair;
# other version pairs are skipped, while pinned-file mismatches within the
# supported pair still fail the build. The generated .pth-triggered bootstrap
# (dingo_vllm_omni_patches.py)
# loads runtime overlays only when DINGO_ENABLE_MINIMAX_H3_PATCHES=1. Individual
# hooks are additionally opt-in through H3_* flags, so unrelated GLM/DeepSeek
# model processes do not import the H3 bootstrap or mutate vLLM-Omni modules.
#
# Intentionally limited to CUDA builds: the source overlays replace H3
# transformer/denoise_loop files from CUDA-specific upstream PRs (#5990/#6173),
# and the runtime overlays (VAE regional compile, FP8+HSDP, Ref2VA FFmpeg
# decode) depend on CUDA-only code paths.  CPU/XPU workers that later need H3
# support must re-evaluate these overlays against their device backend before
# enabling them.
{% if device == "cuda" %}
RUN --mount=type=bind,source=./container/deps/vllm/install_vllm_omni_patches.sh,target=/tmp/install_vllm_omni_patches.sh \
    --mount=type=bind,source=./container/deps/vllm/patches,target=/tmp/vllm_omni_patches,readonly \
    set -eux; \
    export PYTHON_SITE_PACKAGES="${SITE_PACKAGES}"; \
    bash /tmp/install_vllm_omni_patches.sh
{% endif %}

{% if device == "xpu" %}
# Remove conflicting standard triton package for XPU and reinstall triton-xpu
# This must be done after vLLM-Omni installation to ensure no dependencies re-install triton
# Reinstalling triton-xpu ensures the triton namespace is properly configured
RUN uv pip uninstall triton && \
    uv pip install --force-reinstall --no-deps triton-xpu

# Resolve the same include directories Triton's XPU driver will use for its
# first-request JIT, and fail the image build if the SYCL development headers
# are not discoverable there.
RUN /opt/venv/bin/python <<'PY'
from pathlib import Path

from triton.backends.intel.driver import COMPILATION_HELPER

roots = COMPILATION_HELPER.include_dir
if not any((Path(root) / "sycl/sycl.hpp").is_file() for root in roots):
    raise RuntimeError(f"SYCL headers not found in Triton include paths: {roots}")
PY
{% endif %}

{% if context.vllm.enable_modelexpress == "true" %}
# Install only the ModelExpress client package. --no-deps preserves the upstream
# vLLM runtime dependency stack. google-crc32c is imported eagerly by the MX
# vLLM plugin (>=0.5.0) and is not in the XPU base image, so install it
# alongside; the plugin-load guard later in this stage fails the build on any
# remaining --no-deps gap.
RUN --mount=type=cache,id=uv-root-{{ context.dynamo.uv_version }},target=/root/.cache/uv,sharing=locked \
    set -eux; \
    export UV_CACHE_DIR=/root/.cache/uv; \
    uv pip install {{ pip_target }} --no-deps \
        "modelexpress==${MODELEXPRESS_VERSION}"; \
    uv pip install {{ pip_target }} "google-crc32c>=1.5.0"
{% endif %}

{% endif %}

{% if device == "xpu" %}
RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    #ffmpeg \
    libsndfile1 \
    libsm6 \
    libxext6 \
    libgl1 \
    lsb-release \
    numactl \
    wget \
    vim \
    linux-libc-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
{% endif %}

{% if device == "cuda" %}
# Preserve the upstream vllm/vllm-openai FFmpeg stack.  vLLM-Omni's Ref2VA
# preprocessing requires its software libx264rgb encoder and rawvideo support;
# replacing it with the reduced in-tree FFmpeg silently breaks video-reference
# requests.  The runtime probe below exercises the same lossless RGB path.

# TorchInductor/Triton JIT shells out to a host C/C++ compiler at runtime.
# Reproduce that compile path at build time so a missing compiler aborts the
# build instead of surfacing on the first production request.
RUN --mount=type=bind,source=./container/deps/vllm/validate_torch_compile_smoke.py,target=/tmp/validate_torch_compile_smoke.py,readonly \
    python3 /tmp/validate_torch_compile_smoke.py

# Guard the upstream media tools and the exact Ref2VA codec path.
RUN --mount=type=bind,source=./container/deps/vllm/validate_media_probe.py,target=/tmp/validate_media_probe.py,readonly \
    python3 /tmp/validate_media_probe.py
{% endif %}

# Remove the vLLM source tree shipped in the base image to avoid pytest
# collection conflicts (duplicate conftest plugin registration) and stale
# tool scripts referencing files not present in Dynamo's build context.
RUN rm -rf /workspace/vllm

{% if device == "cuda" %}
ENV NVIDIA_DRIVER_CAPABILITIES=video,compute,utility
{% endif %}

{% if target not in ("dev", "local-dev") and context.vllm.enable_modelexpress == "true" %}
# Regression guard for the --no-deps ModelExpress install above: resolve and
# invoke the vllm.general_plugins entry points exactly as vLLM does at every
# startup, so a missing transitive dependency fails the build here instead of
# at pod startup. Runs after every package/library install in this stage
# (including the XPU apt step and the cuda codec purge above) so the check is
# order-independent and sees the final image state.
RUN python3 -c "from importlib.metadata import entry_points; \
eps = [ep for ep in entry_points(group='vllm.general_plugins') if ep.name == 'modelexpress']; \
assert eps, 'modelexpress vllm.general_plugins entry point not found'; \
[ep.load()() for ep in eps]"
{% endif %}

# Check that later package layers preserve the Omni-compatible versions.
RUN {{ python_executable }} - "${TRANSFORMERS_VERSION}" "${TOKENIZERS_VERSION}" <<'PY'
import importlib.metadata as md
import sys

for package, expected in zip(("transformers", "tokenizers"), sys.argv[1:]):
    actual = md.version(package)
    if actual != expected:
        raise RuntimeError(f"expected {package} {expected}, found {actual}")
PY

# Use the packaged binary to match the installed vLLM version.
RUN set -eu; \
    pkg="$({{ python_executable }} -c 'import os, vllm; print(os.path.dirname(vllm.__file__))')"; \
    if [ -f "${pkg}/vllm-rs" ] && [ -x "${pkg}/vllm-rs" ]; then \
        if [ "{{ vllm_rs_allowlist }}" = "1" ]; then \
            printf '%s\n' \
                '#!/bin/sh' \
                '# Keep Omni from changing the EngineCore output schema.' \
                'VLLM_PLUGINS="${VLLM_PLUGINS-{{ vllm_rs_plugins }}}"' \
                'export VLLM_PLUGINS' \
                "exec \"${pkg}/vllm-rs\" \"\$@\"" \
                > {{ vllm_rs_link }}; \
            chmod 755 {{ vllm_rs_link }}; \
        else \
            ln -sf "${pkg}/vllm-rs" {{ vllm_rs_link }}; \
        fi; \
        vllm-rs --help >/dev/null; \
    elif [ "{{ vllm_rs_required }}" = "1" ]; then \
        echo "ERROR: installed vllm package (${pkg}) ships no executable vllm-rs" >&2; \
        exit 1; \
    else \
        echo "WARNING: installed vllm package (${pkg}) ships no executable vllm-rs; not putting it onto PATH" >&2; \
    fi

USER dynamo

# Copy the workspace surface needed by the current vLLM pre-merge test image.
# Keep optional framework trees like planner out of /workspace so the upstream
# runtime does not look like a fully-expanded generic image.
COPY --chmod=775 --chown=dynamo:0 tests /workspace/tests
COPY --chmod=775 --chown=dynamo:0 examples /workspace/examples
COPY --chmod=775 --chown=dynamo:0 dev /workspace/dev
COPY --chmod=775 --chown=dynamo:0 dingo/common /workspace/dingo/common
COPY --chmod=775 --chown=dynamo:0 dingo/frontend /workspace/dingo/frontend
COPY --chmod=775 --chown=dynamo:0 dingo/vllm /workspace/dingo/vllm
COPY --chown=dynamo:0 lib /workspace/lib

# Setup launch banner in common directory accessible to all users
USER root
RUN --mount=type=bind,source=./container/launch_message/runtime.txt,target=/opt/dynamo/launch_message.txt \
    sed '/^#\s/d' /opt/dynamo/launch_message.txt > /opt/dynamo/.launch_screen && \
    chmod 755 /opt/dynamo/.launch_screen && \
    echo 'cat /opt/dynamo/.launch_screen' >> /etc/bash.bashrc

USER dynamo

ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=${DYNAMO_COMMIT_SHA}

# Reset the upstream "vllm serve" entrypoint so the derived runtime behaves
# like other Dynamo images and can execute arbitrary commands directly.
ENTRYPOINT []


{# Compliance is skipped for dev/local-dev: those images are not shipped (release
   ships runtime/frontend/operator/planner/snapshot-agent), compliance-extract
   already skips them, and their pre_runtime carries no dynamo venv to scan.
   cpu likewise: unshipped, and no baseline_sbom to subtract. #}
{% if target not in ("dev", "local-dev") and device != "cpu" %}
{% include "templates/compliance.Dockerfile" %}
{% endif %}


FROM pre_runtime AS runtime
{% if target not in ("dev", "local-dev") and device != "cpu" %}
COPY --from=licenses /legal /legal
{% endif %}
