# Built-in frontend and immutable successful parent; no dependency changes.
ARG BASE_IMAGE=registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm@sha256:9547a5e924383dcaba4cdb8cc4fc9c5b5b3b80c24a5290edbd64e4cf96992d69
FROM ${BASE_IMAGE} AS runtime
ARG BASE_IMAGE
ARG SEEDVR2_METRICS_REPAIR_SOURCE_COMMIT
USER root
COPY container/seedvr2-metrics-repair/ /opt/seedvr2-metrics-repair/
COPY delivery/seedvr2-metrics-repair/source-lock.json delivery/seedvr2-metrics-repair/input-seal.json delivery/seedvr2-metrics-repair/parent-source-manifest.json /opt/seedvr2-metrics-repair/inputs/
ENV SEEDVR2_METRICS_REPAIR_PARENT_IMAGE=${BASE_IMAGE} \
    SEEDVR2_METRICS_REPAIR_SOURCE_COMMIT=${SEEDVR2_METRICS_REPAIR_SOURCE_COMMIT}
RUN --network=none python3 -B /opt/seedvr2-metrics-repair/runtime.py apply --output-dir /opt/seedvr2-metrics-repair/build-info && \
    chown -R dynamo:0 /opt/seedvr2-metrics-repair/build-info
LABEL org.opencontainers.image.title="Omni Gateway SeedVR2 CPU and D2H metrics repair GPU_NOT_RUN" \
      org.opencontainers.image.revision="${SEEDVR2_METRICS_REPAIR_SOURCE_COMMIT}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}"
USER dynamo
RUN --network=none python3 -B /opt/seedvr2-metrics-repair/runtime.py validate --output-dir /opt/seedvr2-metrics-repair/build-info
# Preserve the original entrypoint, CMD, workdir and runtime user.
