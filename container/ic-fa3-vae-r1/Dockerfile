FROM registry.hd-04.alayanew.com:8443/openclaw/vllm-omni@sha256:9387d41fca398d0061cc36be37a2ef4527e56cbbb679727e7494b93add9f9658 AS parent

FROM parent AS payload
USER root
COPY kernel/stage_kernel.py kernel/download.py kernel/lock.json /opt/ic-r1/kernel/
RUN /usr/bin/python3 -B /opt/ic-r1/kernel/stage_kernel.py install --cache /opt/ic-fa3-cache
COPY verify_image.py /opt/ic-r1/
COPY runtime/release-manifest.json /opt/ic-r1/
COPY runtime/worker/ /opt/ic-r1/worker/
COPY runtime/source.zip /tmp/ic-source.zip
RUN --network=none /usr/bin/python3 -B /opt/ic-r1/verify_image.py --unpack /tmp/ic-source.zip

FROM parent
COPY --from=payload /opt/ic-fa3-cache/ /opt/ic-fa3-cache/
COPY --from=payload /opt/ic-r1/ /opt/ic-r1/
ENV PYTHONPATH=/opt/ic-r1/source \
    KERNELS_CACHE=/opt/ic-fa3-cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONDONTWRITEBYTECODE=1
LABEL org.opencontainers.image.title="Open H3 Regenerate 2K IC FA3 VAE R1" \
      org.opencontainers.image.base.name="registry.hd-04.alayanew.com:8443/openclaw/vllm-omni@sha256:9387d41fca398d0061cc36be37a2ef4527e56cbbb679727e7494b93add9f9658"
RUN --network=none --mount=type=tmpfs,target=/tmp \
    HOME=/tmp XDG_CACHE_HOME=/tmp/cache /usr/bin/python3 -B /opt/ic-r1/verify_image.py
