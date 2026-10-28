# Separate official-reference stack. CUDA extensions compile without a GPU.
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS runtime
ARG BASE_IMAGE
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=5 PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/venv/bin:$PATH CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SR_NATIVE_BACKEND=seedvr2
RUN apt-get -o Acquire::Retries=5 update && apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
      python3.10 python3.10-dev python3.10-venv build-essential git ca-certificates \
      ffmpeg libgl1 libglib2.0-0 && \
    python3.10 -m venv /opt/venv && rm -rf /var/lib/apt/lists/*
COPY container/sr-native/shared/download.py container/sr-native/shared/fetch_source.py /opt/sr-native/shared/
COPY container/sr-native/seedvr2/sources.lock.json /opt/sr-native/seedvr2/
RUN --mount=type=cache,id=sr-native-seedvr2-sources-v1,target=/var/cache/sr-native-sources,sharing=locked \
    python /opt/sr-native/shared/fetch_source.py \
      --lock /opt/sr-native/seedvr2/sources.lock.json --output-root /opt/sources
COPY container/sr-native/seedvr2/bootstrap.lock container/sr-native/seedvr2/requirements.lock container/sr-native/seedvr2/install.sh /opt/sr-native/seedvr2/
RUN --mount=type=cache,id=sr-native-seedvr2-pip-v1,target=/root/.cache/pip,sharing=locked \
    --mount=type=secret,id=pip-index-url \
    if [[ -s /run/secrets/pip-index-url ]]; then export PIP_INDEX_URL="$(cat /run/secrets/pip-index-url)"; fi; \
    bash /opt/sr-native/seedvr2/install.sh
COPY container/sr-native/seedvr2 /opt/sr-native/seedvr2
ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=${DYNAMO_COMMIT_SHA}
RUN mkdir -p /opt/sr-native/build-info /legal && \
    cp /opt/sources/SeedVR/LICENSE /legal/SeedVR-LICENSE && \
    cp /opt/sources/apex/LICENSE /legal/Apex-LICENSE && \
    cp /opt/sources/flash-attn/LICENSE /legal/FlashAttention-LICENSE && \
    python -m pip freeze --all > /opt/sr-native/build-info/requirements.installed.txt && \
    python -m pip inspect > /opt/sr-native/build-info/pip-inspect.json && \
    dpkg-query -W > /opt/sr-native/build-info/dpkg.txt && \
    useradd --uid 1000 --create-home --shell /bin/bash sr-native && \
    chmod -R a+rX /opt/sources /opt/sr-native /legal
LABEL org.opencontainers.image.title="SeedVR2 native 3B and standard 7B reference" \
      org.opencontainers.image.revision="${DYNAMO_COMMIT_SHA}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}"
WORKDIR /opt/sources/SeedVR
USER 1000
RUN --network=none python /opt/sr-native/seedvr2/audit.py --output-dir /tmp/sr-native-build-check
ENTRYPOINT ["python", "/opt/sr-native/seedvr2/run.py"]
CMD ["--help"]
