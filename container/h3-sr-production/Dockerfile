# Copy-only derivative of the accepted H3/SR runtime; built-in BuildKit frontend.
ARG BASE_IMAGE=registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm@sha256:c88bef388ed25359c78c48ae841736049100d7447c572678992699ae6291e25d
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
ARG H3_SR_PRODUCTION_SOURCE_COMMIT
USER root
COPY container/h3-sr-production/ /opt/h3-sr/
COPY container/h3-sr-startup/ /opt/h3-sr-startup/
COPY delivery/h3-sr-production/source-lock.json delivery/h3-sr-production/parent-source-manifest.json delivery/h3-sr-production/expected-source-manifest.json /opt/h3-sr/inputs/
COPY delivery/h3-sr-production/dingo-overlay/ /opt/h3-sr/dingo-overlay/
COPY delivery/h3-sr-production/convert_seedvr2.py /opt/h3-weight-init/convert_seedvr2.py
ENV H3_SR_PRODUCTION_PARENT_IMAGE=${BASE_IMAGE} \
    H3_SR_PRODUCTION_SOURCE_COMMIT=${H3_SR_PRODUCTION_SOURCE_COMMIT} \
    PYTHONSAFEPATH=1
RUN --network=none python3 -c 'from pathlib import Path; Path("/usr/local/lib/python3.12/dist-packages/h3_sr_production.pth").write_text("/opt/h3-sr\n")' && \
    chmod 0755 /opt/h3-sr-startup/start-fl2va.sh /opt/h3-sr-startup/start-ref2va.sh /opt/h3-sr-startup/prepare-sr-weights.sh && \
    sh -n /opt/h3-sr-startup/start-fl2va.sh && \
    sh -n /opt/h3-sr-startup/start-ref2va.sh && \
    sh -n /opt/h3-sr-startup/prepare-sr-weights.sh && \
    mkdir -p /work/h3 /work/sr /work/h3-sr && \
    chown dynamo:0 /work/h3 /work/sr /work/h3-sr && \
    python3 -B /opt/h3-sr/runtime.py apply && \
    chown -R dynamo:0 /opt/h3-sr/build-info
LABEL org.opencontainers.image.title="H3 SR production profiles, default observation off" \
      org.opencontainers.image.revision="${H3_SR_PRODUCTION_SOURCE_COMMIT}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}"
USER dynamo
RUN --network=none python3 -B /opt/h3-sr/runtime.py validate --output-dir /opt/h3-sr/build-info/validation
# Inherit Gateway-compatible ENTRYPOINT/CMD/WORKDIR. Worker deployment runs
# /opt/h3-sr-startup/start-fl2va.sh or /opt/h3-sr-startup/start-ref2va.sh.
