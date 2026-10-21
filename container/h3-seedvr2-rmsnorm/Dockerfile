# Use the installed BuildKit frontend; no docker/dockerfile pull from Docker Hub.
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS runtime
ARG BASE_IMAGE
ARG H3_EXPERIMENTAL_SOURCE_COMMIT
USER root
COPY container/h3-seedvr2-rmsnorm/ /opt/h3-seedvr2-rmsnorm/
COPY delivery/h3-seedvr2-rmsnorm/source-lock.json delivery/h3-seedvr2-rmsnorm/input-seal.json /opt/h3-seedvr2-rmsnorm/inputs/
RUN --network=none python3 -B /opt/h3-seedvr2-rmsnorm/preserve_runtime.py capture && \
    python3 -B /opt/h3-seedvr2-rmsnorm/apply_patch.py --apply-installed \
      --report /opt/h3-seedvr2-rmsnorm/build-info/patch-application.json && \
    chown -R dynamo:0 /opt/h3-seedvr2-rmsnorm/build-info
ENV H3_EXPERIMENTAL_SOURCE_COMMIT=${H3_EXPERIMENTAL_SOURCE_COMMIT} \
    H3_RMSNORM_PARENT_IMAGE=${BASE_IMAGE} \
    H3_RMSNORM_UPSTREAM_COMMIT=0708f05e2e27f16fca8b48e7c5a085277d1cafda
LABEL org.opencontainers.image.title="MiniMax-H3 Omni 0.31rc1 with SeedVR2 PR8767 RMSNorm fix" \
      org.opencontainers.image.revision="${H3_EXPERIMENTAL_SOURCE_COMMIT}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}"
USER dynamo
RUN --network=none python3 -B /opt/h3-seedvr2-rmsnorm/preserve_runtime.py verify && \
    python3 -B /opt/h3-seedvr2-rmsnorm/validate_runtime.py \
      --output-dir /opt/h3-seedvr2-rmsnorm/build-info --device cpu && \
    python3 -B /opt/h3-seedvr2-rmsnorm/check_retained.py \
      --output-dir /opt/h3-seedvr2-rmsnorm/build-info/retained && \
    python3 -B /opt/h3-seedvr2-rmsnorm/preserve_runtime.py verify
# ENTRYPOINT, CMD, working directory, framework packages, weights and original
# build evidence are inherited. No pip/uv installation or network RUN is used.
