# Built-in BuildKit frontend: avoid an extra docker/dockerfile registry fetch.
ARG BASE_IMAGE=nvidia/cuda@sha256:cb239b67719dfa32ec6b525b54c1b78559bebd51a47249dd702f6c5429372154
FROM ${BASE_IMAGE} AS runtime
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=8 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:${PATH} \
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg \
    SR_NATIVE_BACKEND=flashvsr-plus SR_NATIVE_UPSTREAM=b548a2cc2c929d3d4676f2973abe27876ec4dee7
COPY container/sr-native/shared/install_system_packages.sh /opt/sr-native/shared/
RUN bash /opt/sr-native/shared/install_system_packages.sh \
      python3.10 python3.10-venv python3.10-dev build-essential git ca-certificates ffmpeg \
      libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/* && \
    python3.10 -m venv /opt/venv && useradd --uid 1000 --create-home --shell /bin/bash sr-native
COPY container/sr-native/shared/fetch_source.py container/sr-native/shared/download.py /opt/sr-native/shared/
COPY container/sr-native/flashvsr-plus/source.lock.json container/sr-native/flashvsr-plus/requirements.lock /opt/sr-native/flashvsr-plus/
RUN --mount=type=cache,id=sr-native-sources-v1,target=/var/cache/sr-native-sources,sharing=locked \
    python3 /opt/sr-native/shared/fetch_source.py --lock /opt/sr-native/flashvsr-plus/source.lock.json --output-root /opt/sources
RUN --mount=type=cache,id=sr-native-flashvsr-pip-v1,target=/root/.cache/pip,sharing=locked \
    --mount=type=secret,id=pip-index-url \
    if [[ -s /run/secrets/pip-index-url ]]; then export PIP_INDEX_URL="$(cat /run/secrets/pip-index-url)"; fi; \
    python3 -m pip install --require-hashes --only-binary=:all: \
      --extra-index-url https://download.pytorch.org/whl/cu126 \
      -r /opt/sr-native/flashvsr-plus/requirements.lock
# BSA is compiled against the exact installed Torch; never download a guessed wheel.
# CUTLASS is separately pinned to the repository's gitlink, already materialized.
RUN --network=none BLOCK_SPARSE_ATTN_FORCE_BUILD=TRUE BLOCK_SPARSE_ATTN_CUDA_ARCHS=90 \
      MAX_JOBS=2 NVCC_THREADS=2 python3 -m pip install --no-deps --no-build-isolation \
      /opt/sources/block-sparse-attention && \
    python3 -m pip install --no-deps --no-build-isolation -e /opt/sources/flashvsr-plus && \
    python3 -m pip check && mkdir -p /opt/sr-native/build-info && \
    python3 -m pip freeze --all > /opt/sr-native/build-info/installed-requirements.txt && \
    python3 -m pip inspect > /opt/sr-native/build-info/pip-inspect.json && \
    dpkg-query -W > /opt/sr-native/build-info/dpkg-packages.txt
COPY container/sr-native/flashvsr-plus/ /opt/sr-native/flashvsr-plus/
RUN --network=none python3 /opt/sr-native/flashvsr-plus/audit.py --capture-native-build /opt/sr-native/build-info/native-extension.json
ARG BASE_IMAGE
ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=${DYNAMO_COMMIT_SHA}
LABEL org.opencontainers.image.revision="${DYNAMO_COMMIT_SHA}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}" \
      org.opencontainers.image.title="FlashVSR+ native Base TC4 reference" \
      org.opencontainers.image.description="Independent H100 reference environment; GPU and quality qualification required"
USER 1000
WORKDIR /home/sr-native
RUN --network=none python3 /opt/sr-native/flashvsr-plus/audit.py --output-dir /tmp/sr-native-build-audit
ENTRYPOINT ["python3", "/opt/sr-native/flashvsr-plus/run.py"]
CMD ["--help"]
