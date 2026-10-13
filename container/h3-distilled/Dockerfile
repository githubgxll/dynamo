# Use the Runner's built-in Dockerfile frontend; no external syntax image pull.
# This image carries model adapters, without replacing the installed framework.
ARG BASE_IMAGE=registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm@sha256:aace31a2806c2c18535cb0bc6a81cc940e0901b4f02f380d2d880c1f34f965c2

FROM ${BASE_IMAGE} AS assets
USER root
COPY assets.lock.json fetch_assets.py verify_assets.py runtime_snapshot.py /opt/h3-distilled/
COPY legal/ /opt/h3-distilled/legal/
RUN mkdir -p /opt/h3-distilled/build-info && \
    python3 -B /opt/h3-distilled/runtime_snapshot.py /opt/h3-distilled/build-info/base-environment.json
RUN --mount=type=cache,id=dingo-h3-distilled-assets-v1,target=/var/cache/h3-distilled,sharing=locked \
    python3 -B /opt/h3-distilled/fetch_assets.py \
      --lock /opt/h3-distilled/assets.lock.json \
      --cache-dir /var/cache/h3-distilled --output /opt/h3-adapters && \
    chmod -R a=rX /opt/h3-adapters

FROM ${BASE_IMAGE} AS runtime
ARG H3_DISTILLED_SOURCE_COMMIT
ENV H3_DISTILLED_SOURCE_COMMIT=${H3_DISTILLED_SOURCE_COMMIT}
LABEL org.opencontainers.image.title="MiniMax-H3 Omni 0.30 distilled adapters" \
      org.opencontainers.image.revision="${H3_DISTILLED_SOURCE_COMMIT}" \
      org.opencontainers.image.base.name="registry.hd-04.alayanew.com:8443/openclaw/ai-dingo-vllm@sha256:aace31a2806c2c18535cb0bc6a81cc940e0901b4f02f380d2d880c1f34f965c2"
USER root
COPY --from=assets /opt/h3-adapters/ /opt/h3-adapters/
COPY --from=assets /opt/h3-distilled/build-info/base-environment.json /opt/h3-distilled/build-info/base-environment.json
COPY image.lock.json assets.lock.json image_reuse_check.py runtime_snapshot.py verify_assets.py validate_bundle.py profiles.json /opt/h3-distilled/
RUN chown -R dynamo:0 /opt/h3-distilled/build-info
USER dynamo
RUN --network=none python3 -B /opt/h3-distilled/validate_bundle.py --source-commit "${H3_DISTILLED_SOURCE_COMMIT}"

FROM scratch AS evidence
COPY --from=runtime /opt/h3-distilled/build-info/ /
