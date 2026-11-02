#!/bin/sh
set +x
set -eu
[ "${CI_DEBUG_TRACE:-false}" = false ] || { printf 'Disable CI_DEBUG_TRACE.\n' >&2; exit 1; }
: "${CI_IMAGE_REGISTRY_USER_XB01:?Set CI_IMAGE_REGISTRY_USER_XB01}"
: "${CI_IMAGE_REGISTRY_PASSWORD_XB01:?Set CI_IMAGE_REGISTRY_PASSWORD_XB01}"
umask 077
DOCKER_CONFIG="$(mktemp -d /tmp/dingo-build-auth.XXXXXX)"
export DOCKER_CONFIG
trap 'rm -rf "$DOCKER_CONFIG"' 0
trap 'exit 1' HUP INT TERM
registry_auth="$(printf '%s:%s' "$CI_IMAGE_REGISTRY_USER_XB01" "$CI_IMAGE_REGISTRY_PASSWORD_XB01" | base64 | tr -d '\n')"
printf '{"auths":{"%s":{"auth":"%s"}}}\n' "$CI_IMAGE_REGISTRY_HOST" "$registry_auth" > "$DOCKER_CONFIG/config.json"
unset registry_auth CI_IMAGE_REGISTRY_USER_XB01 CI_IMAGE_REGISTRY_PASSWORD_XB01
export HTTP_PROXY="$CI_BUILD_PROXY_URL" HTTPS_PROXY="$CI_BUILD_PROXY_URL" NO_PROXY="$CI_BUILD_NO_PROXY"
export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY" no_proxy="$NO_PROXY"
# render.py mounts indexes as secrets; build args alone are shadowed.
export PIP_INDEX_URL="$CI_BUILD_PIP_INDEX" UV_DEFAULT_INDEX="$CI_BUILD_PIP_INDEX"
mkdir -p .gitlab-published
: > .gitlab-published/images.tsv
while IFS="$(printf '\t')" read -r image dockerfile target platform frontend; do
    [ -n "$image" ] || continue
    metadata=".gitlab-published/${dockerfile}.json"
    buildctl --addr unix:///run/buildkit/buildkitd.sock build \
        --progress plain --frontend dockerfile.v0 \
        --local context=. --local dockerfile=.gitlab-render \
        --opt "filename=$dockerfile" --opt "target=$target" --opt "platform=$platform" \
        --opt "build-arg:BUILDKIT_SYNTAX=$frontend" \
        --source-policy-file .gitlab-render/source-policy.json \
        --opt "build-arg:CARGO_BUILD_JOBS=2" --opt "build-arg:MAX_JOBS=2" \
        --opt "build-arg:USE_SCCACHE=false" --opt "build-arg:DYNAMO_COMMIT_SHA=$CI_COMMIT_SHA" \
        --opt "build-arg:HTTP_PROXY=$HTTP_PROXY" --opt "build-arg:HTTPS_PROXY=$HTTPS_PROXY" \
        --opt "build-arg:NO_PROXY=$NO_PROXY" --opt "build-arg:http_proxy=$http_proxy" \
        --opt "build-arg:https_proxy=$https_proxy" --opt "build-arg:no_proxy=$no_proxy" \
        --secret id=pip-index-url,env=PIP_INDEX_URL \
        --secret id=uv-default-index,env=UV_DEFAULT_INDEX \
        --metadata-file "$metadata" --output "type=image,name=$image,push=true"
    # BuildKit emits a top-level string for containerimage.digest, not a tag lookup.
    digest="$(grep '"containerimage.digest":' "$metadata" | cut -d '"' -f 4)"
    printf '%s\n' "$digest" | grep -Eq '^sha256:[a-f0-9]{64}$' || exit 1
    printf '%s\t%s\n' "$image" "$digest" >> .gitlab-published/images.tsv
    printf 'Published %s@%s\n' "${image%:*}" "$digest"
done < .gitlab-render/images.tsv
