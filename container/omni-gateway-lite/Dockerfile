# Built-in Dockerfile frontend; no extra Docker Hub frontend pull.
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS runtime
ARG BASE_IMAGE
ARG H3_EXPERIMENTAL_SOURCE_COMMIT
USER root
COPY container/omni-gateway-lite/ /opt/omni-gateway-lite/
COPY container/h3-seedvr2-rmsnorm/patchlib.py container/h3-seedvr2-rmsnorm/apply_patch.py container/h3-seedvr2-rmsnorm/validate_runtime.py /opt/omni-gateway-lite/rmsnorm/
COPY container/h3-seedvr2-rmsnorm/upstream/ /opt/omni-gateway-lite/rmsnorm/upstream/
COPY container/omni-gateway-lite/rmsnorm-patch.lock.json /opt/omni-gateway-lite/rmsnorm/patch.lock.json
COPY container/h3-experimental/environment/ /opt/omni-gateway-lite/environment/
COPY container/omni-gateway-lite/environment-policy.json /opt/omni-gateway-lite/environment/gateway-dependency-policy.json
COPY delivery/omni-gateway-lite/source-lock.json delivery/omni-gateway-lite/input-seal.json /opt/omni-gateway-lite/inputs/
ENV H3_LITE_PARENT_IMAGE=${BASE_IMAGE}
RUN --network=none python3 -B /opt/omni-gateway-lite/runtime.py apply --output-dir /opt/omni-gateway-lite/build-info && \
    chown -R dynamo:0 /opt/omni-gateway-lite/build-info
ENV DYNAMO_GATEWAY_SOURCE_SHA=9ccee2f56d0122be94d7436ae2f5fec96d15ea9b \
    H3_EXPERIMENTAL_SOURCE_COMMIT=${H3_EXPERIMENTAL_SOURCE_COMMIT} \
    H3_RMSNORM_UPSTREAM_COMMIT=0708f05e2e27f16fca8b48e7c5a085277d1cafda
LABEL org.opencontainers.image.title="Complete Omni 0.31rc1 Gateway lite with SeedVR2 RMSNorm repair" \
      org.opencontainers.image.revision="${H3_EXPERIMENTAL_SOURCE_COMMIT}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}"
USER dynamo
RUN --network=none python3 -B /opt/omni-gateway-lite/runtime.py validate --output-dir /opt/omni-gateway-lite/build-info
# Keep inherited entrypoint, command, working directory and all runtime packages.
# Framework parent precedes the 5.64 GB adapter layer; no weights are removed or fetched.
