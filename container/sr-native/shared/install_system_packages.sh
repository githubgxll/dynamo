#!/usr/bin/env bash
# Build-time only: preserve signed official repositories and fail on partial indexes.
set -euo pipefail

configure_ubuntu_https() {
    local apt_dir="${1:-/etc/apt}"
    local ca_bundle="${2:-/etc/ssl/certs/ca-certificates.crt}"
    local source
    # Both pinned CUDA parents already include ca-certificates. Do not bootstrap
    # a changed parent by disabling TLS verification or falling back to HTTP.
    if [[ ! -s "$ca_bundle" ]]; then
        echo 'APT bootstrap failed: pinned CUDA parent has no CA bundle.' >&2
        return 1
    fi
    for source in "$apt_dir/sources.list" "$apt_dir/sources.list.d/"*.list "$apt_dir/sources.list.d/"*.sources; do
        [[ -f "$source" ]] || continue
        # Support both legacy and deb822 sources; keep suites, components,
        # Signed-By options, NVIDIA and unrelated repository URLs unchanged.
        sed -i -E 's#http://(archive|security)\.ubuntu\.com/ubuntu([/[:space:]]|$)#https://\1.ubuntu.com/ubuntu\2#g' "$source"
    done
    echo 'APT: official Ubuntu archive/security URLs use HTTPS; signature verification remains enabled.'
}

retry_apt_download() {
    local phase="$1"
    shift
    local attempt status=1
    for attempt in 1 2 3; do
        echo "APT ${phase}: attempt ${attempt}/3"
        if apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=60 \
                -o Acquire::https::Timeout=60 "$@"; then
            return 0
        else
            status=$?
        fi
        if [[ "$attempt" -lt 3 ]]; then
            sleep "$((attempt * 5))"
        fi
    done
    echo "APT ${phase} failed after 3 attempts (exit ${status}); check Runner/build-stage egress and proxy. No repository or TLS bypass was used." >&2
    return "$status"
}

install_system_packages() {
    if [[ "$#" -eq 0 ]]; then
        echo 'APT bootstrap requires an explicit package list.' >&2
        return 2
    fi
    # Error-Mode=any also rejects transient failures that otherwise permit stale
    # indexes. Retry downloads only; never loop over failing maintainer scripts.
    retry_apt_download indexes -o APT::Update::Error-Mode=any update || return "$?"
    retry_apt_download packages --download-only install -y --no-install-recommends "$@" || return "$?"
    apt-get --no-download install -y --no-install-recommends "$@"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    configure_ubuntu_https
    install_system_packages "$@"
fi
