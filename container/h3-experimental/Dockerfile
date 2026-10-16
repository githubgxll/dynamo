# Built-in Dockerfile frontend: avoid a separate docker/dockerfile registry pull.
# BASE_IMAGE is the just-built, digest-pinned runtime from the reviewed main.
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS environment
ARG BASE_IMAGE
ARG H3_EXPERIMENTAL_SOURCE_COMMIT
USER root
ENV DYNAMO_MAIN_SOURCE_SHA=049eece3f32f8d38dc1d8c52d4b793cb30e394bb \
    DYNAMO_VLLM_BASE_IMAGE=vllm/vllm-openai:v0.31.0-ubuntu2404@sha256:3fb023cc987cc7c2bca7ca14c1e0aff5ae06bd9431bf07773731ce7e8a682460 \
    H3_EXPERIMENTAL_SOURCE_COMMIT=${H3_EXPERIMENTAL_SOURCE_COMMIT} \
    H3_EXPERIMENTAL_BASE_IMAGE=${BASE_IMAGE}
COPY environment/ /opt/h3-experimental/environment/
# Restore Omni's CLI software mux/decode path without replacing main's
# separately suffixed FFmpeg 9 _dynamo shared libraries used by native Dynamo.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg && \
    ln -sfn /usr/bin/ffmpeg /usr/local/bin/ffmpeg && \
    ln -sfn /usr/bin/ffprobe /usr/local/bin/ffprobe && \
    rm -rf /var/lib/apt/lists/*
RUN --mount=type=secret,id=pip-index-url \
    --mount=type=secret,id=uv-default-index \
    python3 -c 'import os,re; assert re.fullmatch(r"[A-Za-z0-9._:/-]+@sha256:[0-9a-f]{64}",os.environ["H3_EXPERIMENTAL_BASE_IMAGE"]); assert re.fullmatch(r"[0-9a-f]{40}",os.environ["H3_EXPERIMENTAL_SOURCE_COMMIT"])' && \
    bash /opt/h3-experimental/environment/install_environment.sh

FROM ${BASE_IMAGE} AS assets
USER root
COPY assets/ /opt/h3-experimental/assets/
RUN --mount=type=cache,id=dingo-h3-distilled-assets-v1,target=/var/cache/h3-distilled,sharing=locked \
    python3 -B /opt/h3-experimental/assets/fetch_assets.py \
      --lock /opt/h3-experimental/assets/assets.lock.json \
      --cache-dir /var/cache/h3-distilled --output /opt/h3-adapters && \
    chmod -R a=rX /opt/h3-adapters

FROM environment AS runtime
ARG BASE_IMAGE
ARG H3_EXPERIMENTAL_SOURCE_COMMIT
LABEL org.opencontainers.image.title="MiniMax-H3 Omni 0.31rc1 Base and four distilled profiles" \
      org.opencontainers.image.revision="${H3_EXPERIMENTAL_SOURCE_COMMIT}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}"
USER root
COPY --from=assets /opt/h3-adapters/ /opt/h3-adapters/
COPY assets/assets.lock.json assets/verify_assets.py /opt/h3-experimental/assets/
COPY ref8/ /opt/h3-experimental/ref8/
COPY profiles/ /opt/h3-experimental/profiles/
RUN --network=none test "$(id -u dynamo)" = 1000 && \
    test -s /legal/osrb-deps.csv && test -s /legal/osrb.cdx.json && \
    mkdir -p /opt/dynamo/build-info /opt/h3-experimental/build-info && \
    python3 -B /opt/h3-experimental/ref8/apply_patch.py --apply-installed --report /opt/dynamo/build-info/ref8-patch.json && \
    chown -R dynamo:0 /opt/dynamo/build-info /opt/h3-experimental/build-info && \
    chmod -R a=rX /opt/h3-experimental/ref8 /opt/h3-experimental/profiles
USER dynamo
RUN --network=none python3 -B /opt/h3-experimental/assets/verify_assets.py \
      --lock /opt/h3-experimental/assets/assets.lock.json --root /opt/h3-adapters \
      --output /opt/h3-experimental/build-info/assets-report.json && \
    python3 -B /opt/h3-experimental/ref8/check_contract.py --installed \
      --report /opt/dynamo/build-info/ref8-contract.json && \
    python3 -B /opt/h3-experimental/profiles/check_profiles.py --installed \
      --output /opt/h3-experimental/build-info/profiles-check.json && \
    python3 -B /opt/h3-experimental/environment/audit_environment.py --output-dir /opt/h3-experimental/build-info && \
    python3 -B /opt/h3-experimental/environment/media_smoke.py --output-dir /opt/h3-experimental/build-info
# Keep the maintained parent's entrypoint/command. Pods explicitly select one
# profile via python3 /opt/h3-experimental/profiles/launch.py --profile ... --start.
# Base checkpoints remain outside the image on a preexisting read-only mount.

FROM scratch AS evidence
COPY --from=runtime /opt/h3-experimental/build-info/ /experimental/
COPY --from=runtime /opt/dynamo/build-info/ /framework/
COPY --from=runtime /legal/ /legal/
