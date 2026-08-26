{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/router.Dockerfile ===
#######################################################
######## Minimal round-robin Dingo frontend image #####
#######################################################

# This builder intentionally does not inherit dynamo_base or the reusable CUDA
# Builder. It builds a text-only ai-dingo-runtime wheel with the Rust crate's
# default features disabled, which excludes KVBM, CUDA, NIXL and UCX.
FROM rust:1.93.1-slim-bookworm AS router_wheel_builder

ARG CARGO_BUILD_JOBS

ENV CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-16} \
    CARGO_TARGET_DIR=/opt/dynamo/target \
    UV_CACHE_DIR=/root/.cache/uv \
    UV_HTTP_TIMEOUT=300 \
    UV_HTTP_RETRIES=5

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    rm -rf /var/cache/apt/archives/partial/* && \
    apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        clang \
        cmake \
        git \
        libclang-dev \
        patchelf \
        pkg-config \
        protobuf-compiler \
        python3-dev \
        python3-venv && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.10.7 /uv /uvx /usr/local/bin/

WORKDIR /opt/dynamo

# Keep build-tool installation before application sources so it remains a
# reusable image layer when only Dingo Python/Rust source changes.
COPY .cargo/ /opt/dynamo/.cargo/
COPY Cargo.toml Cargo.lock rust-toolchain.toml pyproject.toml README.md LICENSE hatch_build.py /opt/dynamo/

RUN --mount=type=cache,target=/root/.cache/uv,sharing=shared \
    uv venv /opt/dynamo/build-venv --python python3 && \
    uv pip install --python /opt/dynamo/build-venv/bin/python 'maturin>=1.0,<2.0'

COPY lib/ /opt/dynamo/lib/
COPY dingo/ /opt/dynamo/dingo/

RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=shared \
    --mount=type=cache,target=/usr/local/cargo/git,sharing=shared \
    --mount=type=cache,target=/opt/dynamo/target,sharing=shared \
    --mount=type=cache,target=/root/.cache/uv,sharing=shared \
    uv build --wheel --out-dir /opt/dynamo/dist && \
    cd /opt/dynamo/lib/bindings/python && \
    /opt/dynamo/build-venv/bin/maturin build \
        --release \
        --locked \
        --no-default-features \
        --out /opt/dynamo/dist


FROM nvcr.io/nvidia/base/ubuntu:noble-20250619 AS router

ARG PYTHON_VERSION

USER root

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    rm -rf /var/cache/apt/archives/partial/* && \
    apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        libgcc-s1 \
        libstdc++6 \
        python${PYTHON_VERSION}-venv && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3

# UID 1000 and group 0 work with the common Kubernetes/OpenShift security
# context while keeping the runtime non-root.
RUN userdel -r ubuntu >/dev/null 2>&1 || true && \
    useradd -m -s /bin/bash -g 0 dynamo && \
    [ "$(id -u dynamo)" -eq 1000 ] && \
    mkdir -p /home/dynamo/.cache/huggingface /home/dynamo/.cache/uv /opt/dynamo /workspace && \
    chown -R dynamo:0 /home/dynamo /opt/dynamo /workspace && \
    chmod -R g+w /home/dynamo /opt/dynamo /workspace

ENV HOME=/home/dynamo \
    HF_HOME=/home/dynamo/.cache/huggingface \
    VIRTUAL_ENV=/opt/dynamo/venv \
    PATH=/opt/dynamo/venv/bin:/usr/local/bin:${PATH} \
    PYTHONUNBUFFERED=1

USER dynamo
WORKDIR /workspace

# The exact command uses the Rust-native Dynamo chat processor. Its Python
# import boundary is pydantic + uvloop + typing_extensions; vLLM, SGLang,
# transformers, Kubernetes client, pyzmq and NIXL wheels are intentionally not
# installed.
RUN --mount=type=bind,from=ghcr.io/astral-sh/uv:0.10.7,source=/uv,target=/usr/local/bin/uv,ro \
    --mount=type=bind,from=router_wheel_builder,source=/opt/dynamo/dist,target=/tmp/wheelhouse,ro \
    --mount=type=cache,target=/home/dynamo/.cache/uv,uid=1000,gid=0,mode=0775,sharing=shared \
    export UV_CACHE_DIR=/home/dynamo/.cache/uv UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 && \
    uv venv "${VIRTUAL_ENV}" --python "${PYTHON_VERSION}" && \
    uv pip install \
        'pydantic>=2.10.6,<=2.13' \
        'typing_extensions>=4.10.0' \
        'uvloop>=0.21.0' && \
    uv pip install --no-deps \
        /tmp/wheelhouse/ai_dingo_runtime*.whl \
        /tmp/wheelhouse/ai_dingo-*.whl && \
    python3 -c "import dingo.frontend.__main__; from dynamo.llm import EntrypointArgs, RouterConfig, RouterMode" && \
    python3 -m dingo.frontend --help >/dev/null && \
    core_so="$(python3 -c 'import dynamo._core; print(dynamo._core.__file__)')" && \
    if ldd "${core_so}" | grep -Eiq 'cuda|nixl|ucx|ibverbs|rdmacm'; then \
        echo "text-only frontend wheel unexpectedly links a GPU/RDMA library" >&2; \
        ldd "${core_so}" >&2; \
        exit 1; \
    fi

ENTRYPOINT ["python3", "-m", "dingo.frontend"]
CMD ["--namespace-prefix", "glm51-mixed-", "--discovery-backend", "etcd", "--request-plane", "tcp", "--event-plane", "nats", "--http-host", "0.0.0.0", "--http-port", "8000", "--dyn-chat-processor", "dynamo", "--router-mode", "round-robin", "--router-min-initial-workers", "1", "--trust-remote-code", "--metrics-prefix", "sg_glm51_mixed_dingo"]
