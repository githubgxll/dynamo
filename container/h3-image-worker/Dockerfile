# Built-in BuildKit frontend, fixed official H3-Image runtime.
ARG BASE_IMAGE=docker.io/vllm/vllm-omni@sha256:77c3f76764346918576ba4183daf169240070b659c919c7fff45df31bbf333c9
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
ARG H3_IMAGE_WORKER_SOURCE_COMMIT
USER root
COPY container/h3-image-worker/ /opt/h3-image-worker/
COPY delivery/h3-image-worker/release-lock.json delivery/h3-image-worker/parent-source-manifest.json /opt/h3-image-worker/inputs/
RUN --network=none python3 -B /opt/h3-image-worker/runtime.py snapshot && \
    python3 -B /opt/h3-image-worker/runtime.py apply && \
    mkdir -p /work/tmp /work/cache/hf /work/cache/torch /work/cache/triton /work/cache/vllm /work/h3-image-worker && \
    chown -R 1000:1000 /work /opt/h3-image-worker/build-info
ENV H3_IMAGE_WORKER_PARENT_IMAGE=${BASE_IMAGE} \
    H3_IMAGE_WORKER_SOURCE_COMMIT=${H3_IMAGE_WORKER_SOURCE_COMMIT} \
    HOME=/work \
    TMPDIR=/work/tmp \
    HF_HOME=/work/cache/hf \
    TORCH_HOME=/work/cache/torch \
    XDG_CACHE_HOME=/work/cache \
    TRITON_CACHE_DIR=/work/cache/triton \
    VLLM_CACHE_ROOT=/work/cache/vllm \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONSAFEPATH=1
LABEL org.opencontainers.image.title="Native H3-Image Worker with local encoder and FP32 RoPE repairs" \
      org.opencontainers.image.revision="${H3_IMAGE_WORKER_SOURCE_COMMIT}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}"
WORKDIR /work
USER 1000:1000
RUN --network=none python3 -P -B /opt/h3-image-worker/runtime.py validate --output-dir /opt/h3-image-worker/build-info/validation
ENTRYPOINT ["python3", "-P", "/opt/h3-image-worker/worker.py"]
CMD ["--shared-root", "/source-models", "--profile", "h100-4-tp4"]
