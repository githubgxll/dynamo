# Use BuildKit's built-in frontend; do not fetch docker/dockerfile from Docker Hub.
ARG BASE_IMAGE=registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm@sha256:b3150072a8b917e643ceae5fd2b278e736df09fc76397b749b3ca4531cbe8bf3
FROM ${BASE_IMAGE} AS runtime
ARG BASE_IMAGE
ARG OMNI032_H3_SR_SOURCE_COMMIT
USER root
COPY container/omni032-h3-sr/ /opt/omni032-h3-sr/
COPY container/h3-experimental/environment/ /opt/omni032-h3-sr/legacy-environment/
COPY container/compliance/__init__.py container/compliance/overrides.py container/compliance/license_overrides.yaml /opt/omni032-h3-sr/compliance/
COPY container/compliance/generators/ /opt/omni032-h3-sr/compliance/generators/
COPY container/compliance/policy/ /opt/omni032-h3-sr/compliance/policy/
COPY container/compliance/spdx_licenses/ /opt/omni032-h3-sr/compliance/spdx_licenses/
COPY delivery/omni032-h3-sr/source-lock.json delivery/omni032-h3-sr/parent-source-manifest.json /opt/omni032-h3-sr/inputs/
ENV OMNI032_H3_SR_PARENT_IMAGE=${BASE_IMAGE} \
    OMNI032_H3_SR_SOURCE_COMMIT=${OMNI032_H3_SR_SOURCE_COMMIT}
RUN --network=none python3 -B /opt/omni032-h3-sr/runtime.py parent
RUN --mount=type=secret,id=pip-index-url \
    --mount=type=secret,id=uv-default-index \
    --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    UV_LINK_MODE=copy UV_CACHE_DIR=/root/.cache/uv \
    bash /opt/omni032-h3-sr/install_environment.sh /opt/omni032-h3-sr/build-info/environment
RUN --network=none python3 -B /opt/omni032-h3-sr/runtime.py apply && \
    python3 -B /opt/omni032-h3-sr/legal.py --output-dir /opt/omni032-h3-sr/build-info && \
    chown -R dynamo:0 /opt/omni032-h3-sr/build-info
LABEL org.opencontainers.image.title="Omni 0.32 H3 and SeedVR2 optimized Gateway GPU_NOT_RUN" \
      org.opencontainers.image.revision="${OMNI032_H3_SR_SOURCE_COMMIT}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}"
USER dynamo
RUN --network=none python3 -B /opt/omni032-h3-sr/runtime.py validate --output-dir /opt/omni032-h3-sr/build-info/validation
# Inherit entrypoint, CMD, workdir, UID and Gateway/monitoring/model configuration.
