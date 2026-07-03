{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/aws.Dockerfile ===
#############################
########## AWS EFA ##########
#############################
#
# This stage extends the runtime/dev stage with AWS EFA installer
# which includes: libfabric and aws-ofi-nccl plugin
#
# Use this stage when deploying on AWS infrastructure with EFA support

FROM ${EFA_BASE_IMAGE} AS aws

ARG EFA_VERSION

{% if target == "runtime" %}
USER root
{% endif %}

# Install AWS EFA installer with bundled libfabric and aws-ofi-nccl
# Flags explanation:
#   --skip-kmod: Skip kernel module installation (handled by host)
#   --skip-limit-conf: Skip ulimit configuration (handled by container runtime)
#   --no-verify: Skip GPG verification (optional, can be removed if verification is needed)
# Cache apt downloads; sharing=locked avoids apt/dpkg races with concurrent builds.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    mkdir -p /tmp/efa && \
    cd /tmp/efa && \
    curl --retry 3 --retry-delay 2 -fsSL -o aws-efa-installer-${EFA_VERSION}.tar.gz \
        https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_VERSION}.tar.gz && \
    tar -xf aws-efa-installer-${EFA_VERSION}.tar.gz && \
    cd aws-efa-installer && \
    apt-get update && \
    ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify && \
    rm -rf /tmp/efa && \
    rm -rf /opt/amazon/aws-ofi-nccl /etc/ld.so.conf.d/aws-ofi-nccl.conf && \
    ldconfig

ENV EFA_VERSION="${EFA_VERSION}"

ARG NIXL_LIBFABRIC_REF

# Copy the wheel_builder-built libfabric and register it with the dynamic linker
# ONLY if the EFA-bundled libfabric is older than NIXL_LIBFABRIC_REF.
# When a future EFA installer ships libfabric >= the version we build, the
# version comparison evaluates to false and this becomes a no-op automatically.
RUN --mount=from=wheel_builder,source=/usr/local/libfabric,target=/tmp/libfabric_build \
    EFA_PC=$(find /opt/amazon/efa -path '*/pkgconfig/libfabric.pc' 2>/dev/null | head -n1) && \
    EFA_LIBFABRIC_RAW=$(cat "$EFA_PC" 2>/dev/null | grep '^Version:' | awk '{print $2}') && \
    EFA_LIBFABRIC_VER=$(echo "$EFA_LIBFABRIC_RAW" | grep -oE '^[0-9]+\.[0-9]+(\.[0-9]+)?') && \
    REF_VER=$(echo "${NIXL_LIBFABRIC_REF}" | sed 's/^v//') && \
    if [ -n "$EFA_LIBFABRIC_VER" ] && [ -n "$REF_VER" ] && \
       [ "$(printf '%s\n' "$EFA_LIBFABRIC_VER" "$REF_VER" | sort -V | head -n1)" = "$EFA_LIBFABRIC_VER" ] && \
       [ "$EFA_LIBFABRIC_VER" != "$REF_VER" ]; then \
        rm -rf /opt/amazon/efa && \
        cp -Pfr /tmp/libfabric_build /opt/amazon/efa && \
        sed -i 's|^prefix=.*|prefix=/opt/amazon/efa|' /opt/amazon/efa/lib/pkgconfig/libfabric.pc && \
        echo "/opt/amazon/efa/lib" > /etc/ld.so.conf.d/000_efa.conf && \
        rm -f /etc/ld.so.conf.d/efa.conf && \
        ldconfig && \
        echo "[aws] libfabric overlay: ${REF_VER} (overwrites EFA stock ${EFA_LIBFABRIC_RAW})"; \
    else \
        echo "[aws] libfabric overlay: skipped (EFA stock ${EFA_LIBFABRIC_RAW:-unknown} >= ${REF_VER})"; \
    fi

{% if target == "runtime" %}
USER dynamo
{% endif %}
