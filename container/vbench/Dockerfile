# syntax=docker/dockerfile:1.7
# CPU-only construction; GPU qualification is performed separately on H100.
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS foundation
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# Keep the team's Ubuntu mirror convention; do not bake proxy addresses into ENV.
RUN sed -i -e 's|http://archive.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' \
           -e 's|http://security.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list && \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      python3.10 python3.10-venv python3-pip ca-certificates curl git unzip \
      ffmpeg libglib2.0-0 libgomp1 build-essential python3.10-dev && \
    rm -rf /var/lib/apt/lists/* && python3.10 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/VBench:/opt/h3-eval:/opt \
    VBENCH_CACHE_DIR=/opt/vbench-assets/vbench TORCH_HOME=/opt/vbench-assets/torch \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
# One version-controlled source selection covers bootstrap and later resolution.
# Keep it after apt so this fix can reuse the expensive system-package layer.
COPY container/vbench/package-sources.json /opt/vbench-build/package-sources.json
COPY container/vbench/scripts/package_sources.py /opt/vbench-build/scripts/package_sources.py
RUN python /opt/vbench-build/scripts/package_sources.py bootstrap --evidence /opt/vbench-build-info

# Keep resolver and asset layers independent of scoring code and Git commit IDs.
FROM foundation AS environment
ARG VBENCH_PHASE=prepare
COPY container/vbench/requirements.in /opt/vbench-build/requirements.in
COPY container/vbench/scripts/prepare_environment.py /opt/vbench-build/scripts/prepare_environment.py
COPY container/vbench/locks/ /opt/vbench-build/locks/
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    python /opt/vbench-build/scripts/prepare_environment.py --phase "${VBENCH_PHASE}" \
      --inputs /opt/vbench-build/requirements.in --lock /opt/vbench-build/locks/requirements.lock \
      --evidence /opt/vbench-build-info

FROM foundation AS assets
ARG VBENCH_PHASE=prepare
COPY container/vbench/assets-catalog.json /opt/vbench-build/assets-catalog.json
COPY container/vbench/scripts/prepare_assets.py /opt/vbench-build/scripts/prepare_assets.py
COPY container/vbench/locks/ /opt/vbench-build/locks/
RUN python /opt/vbench-build/scripts/prepare_assets.py --phase "${VBENCH_PHASE}" \
      --catalog /opt/vbench-build/assets-catalog.json --lock /opt/vbench-build/locks/assets.lock.json \
      --output /opt/vbench-assets --vbench-source /opt/VBench --evidence /opt/vbench-build-info

FROM environment AS prepared
ARG BASE_IMAGE
ARG VBENCH_PHASE=prepare
ARG DYNAMO_COMMIT_SHA
COPY --from=assets /opt/VBench/ /opt/VBench/
COPY --from=assets /opt/vbench-assets/ /opt/vbench-assets/
COPY --from=assets /opt/vbench-build-info/ /opt/vbench-build-info/
COPY container/vbench/ /opt/vbench-build/
COPY container/compliance/ /opt/compliance/
# In prepare mode an imported old lock may be in the context. Keep the newly
# resolved lock in the runtime location used by environment verification.
RUN if [ "${VBENCH_PHASE}" = prepare ]; then cp /opt/vbench-build-info/requirements.lock /opt/vbench-build/locks/requirements.lock; fi && \
    test "${BASE_IMAGE#*@sha256:}" != "${BASE_IMAGE}" && \
    mkdir -p /legal /data && \
    python /opt/vbench-build/scripts/fingerprint.py --phase "${VBENCH_PHASE}" \
      --vbench-root /opt/vbench-build --compliance-root /opt/compliance \
      --evidence /opt/vbench-build-info && \
    python /opt/vbench-build/scripts/build_identity.py \
      --phase "${VBENCH_PHASE}" --base-image "${BASE_IMAGE}" \
      --source-commit "${DYNAMO_COMMIT_SHA}" --output /opt/vbench-build-info/build-identity.json
RUN cp -a /opt/vbench-build/scripts /opt/h3-eval && \
    find /opt/VBench /opt/vbench-assets -type d -exec chmod 755 {} + && \
    find /opt/VBench /opt/vbench-assets -type f -exec chmod 444 {} +
RUN --network=none python -c "from prepare_assets import verify_manifest; print(verify_manifest('/opt/vbench-build-info/assets.lock.json','/opt/vbench-assets','/opt/VBench'))" && \
    python /opt/h3-eval/verify_runtime.py --device cpu --output /opt/vbench-build-info/cpu-check.json
RUN --network=none python /opt/h3-eval/license_report.py \
      --policy /opt/compliance/policy/licenses.toml --output /legal && \
    dpkg-query -W -f='${Package}\t${Version}\n' > /opt/vbench-build-info/dpkg-packages.tsv

# Preparation may report denied/unknown licenses, but exports only evidence.
# It cannot be used as a publishable target in the matrix.
FROM scratch AS preparation_artifact
COPY --from=prepared /opt/vbench-build-info/ /
COPY --from=prepared /legal/ /legal/

FROM prepared AS runtime
ARG VBENCH_PHASE=prepare
RUN test "${VBENCH_PHASE}" = publish && \
    python /opt/h3-eval/license_report.py --require-pass /legal/license-status.json
LABEL org.opencontainers.image.title="VBench six visual dimensions" \
      org.opencontainers.image.version="0.1.5-fd18b3d-cu121-r1" \
      org.opencontainers.image.source="https://github.com/githubgxll/dynamo" \
      org.opencontainers.image.description="Offline custom_input evaluator; GPU qualification required"
WORKDIR /data
CMD ["sleep", "infinity"]

FROM scratch AS image_evidence_artifact
COPY --from=runtime /opt/vbench-build-info/ /build-info/
COPY --from=runtime /legal/ /legal/
