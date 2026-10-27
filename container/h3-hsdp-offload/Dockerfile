# Built-in frontend; no Docker Hub syntax image or dependency reinstall.
ARG BASE_IMAGE=registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm@sha256:87365073cbdf95fa17ca352a24a93cb74fde144a47c96135e91011d1a03df72e
FROM ${BASE_IMAGE} AS runtime
ARG BASE_IMAGE
ARG H3_HSDP_SOURCE_COMMIT
USER root
COPY container/h3-hsdp-offload/ /opt/h3-hsdp-offload/
COPY delivery/h3-hsdp-offload/source-lock.json delivery/h3-hsdp-offload/input-seal.json /opt/h3-hsdp-offload/inputs/
ENV H3_HSDP_PARENT_IMAGE=${BASE_IMAGE} \
    H3_HSDP_SOURCE_COMMIT=${H3_HSDP_SOURCE_COMMIT}
USER dynamo
RUN --network=none python3 -B /opt/seedvr2-c14/runtime.py validate --output-dir /tmp/hsdp-parent-validation
USER root
RUN --network=none python3 -B /opt/h3-hsdp-offload/runtime.py apply --output-dir /opt/h3-hsdp-offload/build-info && \
    cp -a /tmp/hsdp-parent-validation /opt/h3-hsdp-offload/build-info/parent-validation && \
    chown -R dynamo:0 /opt/h3-hsdp-offload/build-info
LABEL org.opencontainers.image.title="Omni Gateway HSDP offload candidate GPU_NOT_RUN" \
      org.opencontainers.image.revision="${H3_HSDP_SOURCE_COMMIT}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}"
USER dynamo
RUN --network=none python3 -B /opt/h3-hsdp-offload/runtime.py validate --output-dir /opt/h3-hsdp-offload/build-info
# Inherit all runtime settings and packages. GPU acceptance is a separate gate.
