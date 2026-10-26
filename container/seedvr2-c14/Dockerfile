# Built-in Dockerfile frontend: no extra Docker Hub dependency.
ARG BASE_IMAGE=registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm@sha256:9382949d2789b23cdfe4ea71d68c7973592de28f209082bd7907c1d09fe008c9
FROM ${BASE_IMAGE} AS runtime
ARG BASE_IMAGE
ARG H3_C14_SOURCE_COMMIT
USER root
COPY container/seedvr2-c14/ /opt/seedvr2-c14/
COPY delivery/seedvr2-c14/source-lock.json delivery/seedvr2-c14/input-seal.json /opt/seedvr2-c14/inputs/
ENV H3_C14_PARENT_IMAGE=${BASE_IMAGE} \
    H3_C14_SOURCE_COMMIT=${H3_C14_SOURCE_COMMIT}
# Verify the inherited lite image with its original, strict verifier before any
# instrumentation. The original receipts and verifier remain unchanged.
USER dynamo
RUN --network=none python3 -B /opt/omni-gateway-lite/runtime.py validate --output-dir /tmp/c14-parent-validation
USER root
RUN --network=none python3 -B /opt/seedvr2-c14/runtime.py apply --output-dir /opt/seedvr2-c14/build-info && \
    cp -a /tmp/c14-parent-validation /opt/seedvr2-c14/build-info/parent-validation && \
    chown -R dynamo:0 /opt/seedvr2-c14/build-info
LABEL org.opencontainers.image.title="Omni Gateway lite SeedVR2 C14 controlled diagnostics" \
      org.opencontainers.image.revision="${H3_C14_SOURCE_COMMIT}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}"
USER dynamo
RUN --network=none python3 -B /opt/seedvr2-c14/runtime.py validate --output-dir /opt/seedvr2-c14/build-info
# Inherit original entrypoint/CMD/workdir/packages. Diagnostics remain off unless
# the isolated SR-only runner supplies a validated local capture profile.
