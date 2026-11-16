{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/dynamo_base.Dockerfile ===
##################################
########## Base Image ############
##################################

# Rebuild bundled Go services so crypto/x509 includes the intermediate-key blocklist.
FROM --platform=$BUILDPLATFORM golang:{{ context.dynamo.go_version }} AS go_tools_builder
ARG TARGETARCH
ARG NATS_VERSION
ARG NATS_COMMIT
ARG ETCD_VERSION
ARG ETCD_COMMIT
ENV CGO_ENABLED=0 GOOS=linux GOARCH=${TARGETARCH} GOTOOLCHAIN=local
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

{% include "templates/verify_ca_certificates.Dockerfile" %}
RUN for attempt in 1 2 3; do \
        git clone --depth 1 --branch "${NATS_VERSION}" https://github.com/nats-io/nats-server.git /src/nats && break; \
        rm -rf /src/nats; \
        [ "$attempt" -lt 3 ] || exit 1; \
        sleep 5; \
    done && \
    test "$(git -C /src/nats rev-parse HEAD)" = "${NATS_COMMIT}" && \
    cd /src/nats && go build -trimpath -o /out/nats-server .
RUN for attempt in 1 2 3; do \
        git clone --depth 1 --branch "${ETCD_VERSION}" https://github.com/etcd-io/etcd.git /src/etcd && break; \
        rm -rf /src/etcd; \
        [ "$attempt" -lt 3 ] || exit 1; \
        sleep 5; \
    done && \
    test "$(git -C /src/etcd rev-parse HEAD)" = "${ETCD_COMMIT}" && \
    cd /src/etcd && ./build.sh && \
    mkdir -p /out/etcd && cp bin/etcd bin/etcdctl bin/etcdutl /out/etcd/

ARG ENABLE_SOURCE_ARCHIVAL=false
RUN mkdir -p /out/native-sources && \
    if [ "$ENABLE_SOURCE_ARCHIVAL" = "true" ]; then \
        git -C /src/nats archive --format=tar.gz --prefix="nats-server-${NATS_VERSION}/" \
            -o "/out/native-sources/nats-server-${NATS_VERSION}.tar.gz" HEAD && \
        git -C /src/etcd archive --format=tar.gz --prefix="etcd-${ETCD_VERSION}/" \
            -o "/out/native-sources/etcd-${ETCD_VERSION}.tar.gz" HEAD && \
        tar --exclude=./cache -czf /out/native-sources/go-service-modules.tar.gz -C /go/pkg/mod .; \
    fi

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} AS dynamo_base

ARG TARGETARCH

USER root
WORKDIR /opt/dynamo

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

{% include "templates/verify_ca_certificates.Dockerfile" %}

{% if device == "cpu" %}
RUN apt clean && apt-get update -y && \
    apt-get install -y --no-install-recommends --fix-missing \
    curl ca-certificates zip unzip git lsb-release numactl wget vim
{% endif %}

# Install sccache into the base image so downstream stages can COPY it
# instead of downloading from GitHub (avoids 502 errors under parallel builds)
ARG SCCACHE_VERSION
RUN ARCH_ALT=$([ "${TARGETARCH}" = "amd64" ] && echo "x86_64" || echo "aarch64") && \
    wget --tries=3 --waitretry=5 \
        "https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/sccache-${SCCACHE_VERSION}-${ARCH_ALT}-unknown-linux-musl.tar.gz" && \
    tar -xzf "sccache-${SCCACHE_VERSION}-${ARCH_ALT}-unknown-linux-musl.tar.gz" && \
    mv "sccache-${SCCACHE_VERSION}-${ARCH_ALT}-unknown-linux-musl/sccache" /usr/local/bin/ && \
    rm -rf sccache*

# Install uv package manager.
# Download the uv release tarball directly from GitHub instead of pulling the
# ghcr.io/astral-sh/uv image: ghcr.io is unreachable (TLS handshake timeout) from
# some CI/build networks, and pinning the version keeps builds reproducible.
ARG UV_VERSION={{ context.dynamo.uv_version }}
RUN mkdir -p /opt/uv/bin && ARCH_ALT=$([ "${TARGETARCH}" = "amd64" ] && echo "x86_64" || echo "aarch64") && \
    wget --tries=3 --waitretry=5 \
        "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-${ARCH_ALT}-unknown-linux-gnu.tar.gz" && \
    tar -xzf "uv-${ARCH_ALT}-unknown-linux-gnu.tar.gz" -C /opt/uv/bin --strip-components=1 && \
    rm -f "uv-${ARCH_ALT}-unknown-linux-gnu.tar.gz"
ENV PATH=/opt/uv/bin:${PATH}

COPY --from=go_tools_builder /out/nats-server /usr/bin/nats-server
COPY --from=go_tools_builder /out/etcd/ /usr/local/bin/etcd/
RUN ln -s /usr/bin/nats-server /usr/local/bin/nats-server
ENV PATH=/usr/local/bin/etcd/:$PATH

# Rust Setup
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH \
    RUST_VERSION=1.96.1

# Install Rust — ARCH_ALT (x86_64/aarch64) is derived from TARGETARCH at build time
RUN ARCH_ALT=$([ "${TARGETARCH}" = "amd64" ] && echo "x86_64" || echo "aarch64") && \
    RUSTARCH="${ARCH_ALT}-unknown-linux-gnu" && \
    wget --tries=3 --waitretry=5 "https://static.rust-lang.org/rustup/archive/1.29.2/${RUSTARCH}/rustup-init" && \
    chmod +x rustup-init && \
    ./rustup-init -y --no-modify-path --profile minimal --default-toolchain $RUST_VERSION --default-host ${RUSTARCH} && \
    rm rustup-init && \
    chmod -R a+w $RUSTUP_HOME $CARGO_HOME
