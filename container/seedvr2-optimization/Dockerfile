# Use the built-in frontend; preserve the successful runtime and all weights external.
ARG BASE_IMAGE=registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm@sha256:45b3ec519faec85ac4a40ff8766e033704bc92623f97fda16610efdc85e71e0f
FROM ${BASE_IMAGE} AS runtime
ARG BASE_IMAGE
ARG SEEDVR2_OPT_SOURCE_COMMIT
USER root
COPY container/seedvr2-optimization/ /opt/seedvr2-optimization/
COPY delivery/seedvr2-optimization/source-lock.json delivery/seedvr2-optimization/input-seal.json /opt/seedvr2-optimization/inputs/
ENV SEEDVR2_OPT_PARENT_IMAGE=${BASE_IMAGE} \
    SEEDVR2_OPT_SOURCE_COMMIT=${SEEDVR2_OPT_SOURCE_COMMIT}
RUN --network=none python3 -B /opt/seedvr2-optimization/runtime.py apply --output-dir /opt/seedvr2-optimization/build-info && \
    chown -R dynamo:0 /opt/seedvr2-optimization/build-info
LABEL org.opencontainers.image.title="Omni Gateway SeedVR2 optimization candidate GPU_NOT_RUN" \
      org.opencontainers.image.revision="${SEEDVR2_OPT_SOURCE_COMMIT}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}"
USER dynamo
RUN --network=none python3 -B /opt/seedvr2-optimization/runtime.py validate --output-dir /opt/seedvr2-optimization/build-info
# Original entrypoint/CMD/workdir and runtime controls remain inherited.
