# Exact two-source derivative; no dependency installation or old recipe replay.
FROM registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm@sha256:26d3767f090fa2fecb2d5a94cf5b326657dd2fef891a13ded41476056f99fe18
ARG SEEDVR2_BF16_SOURCE_COMMIT
USER root
COPY container/seedvr2-bf16-production/ /opt/seedvr2-bf16/
COPY delivery/seedvr2-bf16-production/release-lock.json delivery/seedvr2-bf16-production/parent-source-manifest.json delivery/seedvr2-bf16-production/expected-source-manifest.json /opt/seedvr2-bf16/inputs/
# apply first verifies the actual parent under its inherited H3/SR identity.
RUN --network=none python3 -B /opt/seedvr2-bf16/runtime.py apply --source-commit "${SEEDVR2_BF16_SOURCE_COMMIT}" && \
    chown -R dynamo:0 /opt/seedvr2-bf16/build-info
ENV SEEDVR2_BF16_SOURCE_COMMIT=${SEEDVR2_BF16_SOURCE_COMMIT} \
    SEEDVR2_BF16_PARENT_IMAGE=registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm@sha256:26d3767f090fa2fecb2d5a94cf5b326657dd2fef891a13ded41476056f99fe18 \
    SEEDVR2_DIT_BF16=1 \
    H3_SR_PRODUCTION_SOURCE_COMMIT=${SEEDVR2_BF16_SOURCE_COMMIT} \
    H3_SR_PRODUCTION_PARENT_IMAGE=registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm@sha256:26d3767f090fa2fecb2d5a94cf5b326657dd2fef891a13ded41476056f99fe18
LABEL org.opencontainers.image.revision="${SEEDVR2_BF16_SOURCE_COMMIT}" \
      org.opencontainers.image.base.name="registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm@sha256:26d3767f090fa2fecb2d5a94cf5b326657dd2fef891a13ded41476056f99fe18"
USER dynamo
RUN --network=none python3 -B /opt/seedvr2-bf16/runtime.py validate --output-dir /opt/seedvr2-bf16/build-info/validation
# ENTRYPOINT, CMD, WORKDIR, native libraries and the startup bundle are inherited.
