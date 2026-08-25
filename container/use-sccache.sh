#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# sccache management script
# This script handles sccache installation, environment setup, and statistics display

SCCACHE_VERSION="v0.14.0"


usage() {
    cat << EOF
Usage: $0 [COMMAND] [OPTIONS]

Commands:
    install         Install sccache binary (architecture auto-detected via uname -m)
    setup-env       Print export statements to configure sccache for compilation
    show-stats      Display sccache statistics with optional build name
    help            Show this help message

setup-env modes:
    setup-env           Wraps CC/CXX with sccache + sets RUSTC_WRAPPER.
                        Works for autotools (make) and Meson builds.
    setup-env cmake     Also sets CMAKE_C/CXX/CUDA_COMPILER_LAUNCHER.
                        Use for CMake-based builds.

    Usage: eval \$($0 setup-env [cmake])

Environment variables:
    USE_SCCACHE             Set to 'true' to enable sccache
    SCCACHE_BUCKET          S3 bucket name (fallback if not passed as parameter)
    SCCACHE_REGION          S3 region (fallback if not passed as parameter)
    SCCACHE_GHA_ENABLED     Set to 'on' to use the GitHub Actions cache backend
    SCCACHE_GHA_VERSION     Cache namespace/version used for manual invalidation
    ACTIONS_RESULTS_URL     GitHub Actions cache service URL (pass as a secret)
    ACTIONS_RUNTIME_TOKEN   GitHub Actions cache token (pass as a secret)
    ARCH                    Architecture for S3 key prefix (fallback if not passed as parameter)
    SCCACHE_EXECUTABLE      Existing sccache binary to invoke explicitly.
                            Keeping it outside PATH prevents Meson from
                            automatically wrapping unsupported compilers.
    SCCACHE_BIN_DIR         Override the install dir for sccache and wrappers.
                            Default: /usr/local/bin when running as root
                            (image builds), \$HOME/.local/bin otherwise
                            (e.g. CI containers running as a non-root uid).

Examples:
    $0 install                     # architecture auto-detected via uname -m
    eval \$($0 setup-env)          # autotools / Meson
    eval \$($0 setup-env cmake)    # CMake builds
    $0 show-stats "UCX"
EOF
}

# Resolve the directory where sccache + wrapper scripts live for this
# invocation. Default keeps the historical image-build behavior (root writes to
# /usr/local/bin). Non-root callers (CI job containers running as the ARC
# runner uid) get $HOME/.local/bin so the script doesn't need root to install
# the wrappers.
sccache_bin_dir() {
    if [ -n "${SCCACHE_BIN_DIR:-}" ]; then
        printf '%s' "${SCCACHE_BIN_DIR}"
    elif [ "$(id -u)" -eq 0 ]; then
        printf '%s' "/usr/local/bin"
    else
        printf '%s' "${HOME:?HOME must be set to install sccache wrappers without root}/.local/bin"
    fi
}

# Resolve the sccache binary without requiring it to be on PATH. Image builds
# deliberately keep it under /opt/sccache because Meson otherwise detects the
# executable and automatically prepends it to nvcc.
sccache_executable() {
    if [ -n "${SCCACHE_EXECUTABLE:-}" ]; then
        if [ ! -x "${SCCACHE_EXECUTABLE}" ]; then
            echo "SCCACHE_EXECUTABLE is not executable: ${SCCACHE_EXECUTABLE}" >&2
            return 1
        fi
        printf '%s' "${SCCACHE_EXECUTABLE}"
    elif command -v sccache >/dev/null 2>&1; then
        command -v sccache
    else
        local candidate
        candidate="$(sccache_bin_dir)/sccache"
        if [ ! -x "${candidate}" ]; then
            echo "sccache executable not found" >&2
            return 1
        fi
        printf '%s' "${candidate}"
    fi
}

install_sccache() {
    # Derive arch from TARGETARCH (set by BuildKit) with uname -m fallback
    local arch_alt
    if [ -n "${TARGETARCH:-}" ]; then
        arch_alt=$([ "$TARGETARCH" = "amd64" ] && echo "x86_64" || echo "aarch64")
    else
        arch_alt=$(uname -m)
    fi

    local bin_dir
    bin_dir=$(sccache_bin_dir)
    mkdir -p "${bin_dir}"

    local sccache_exe
    if [ -n "${SCCACHE_EXECUTABLE:-}" ]; then
        sccache_exe=$(sccache_executable)
        echo "sccache already available at ${sccache_exe}, skipping download"
    elif command -v sccache >/dev/null 2>&1; then
        sccache_exe=$(command -v sccache)
        echo "sccache already installed at ${sccache_exe}, skipping download"
    else
        echo "Installing sccache ${SCCACHE_VERSION} for architecture ${arch_alt} into ${bin_dir}..."
        wget --tries=3 --waitretry=5 \
            "https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/sccache-${SCCACHE_VERSION}-${arch_alt}-unknown-linux-musl.tar.gz"
        tar -xzf "sccache-${SCCACHE_VERSION}-${arch_alt}-unknown-linux-musl.tar.gz"
        mv "sccache-${SCCACHE_VERSION}-${arch_alt}-unknown-linux-musl/sccache" "${bin_dir}/"
        rm -rf sccache*
        sccache_exe="${bin_dir}/sccache"
    fi

    # Create compiler wrapper scripts for autotools/Meson compatibility.
    # Autoconf breaks with CC="sccache gcc" (multi-word value), so we provide
    # single-binary wrappers that autoconf sees as a regular compiler.
    # The real compiler path is passed at runtime via SCCACHE_CC_REAL / SCCACHE_CXX_REAL.
    cat > "${bin_dir}/sccache-cc" <<'WRAPPER'
#!/bin/sh
# Only use sccache for pure compilations (-c flag).
# Autoconf tests the compiler with combined compile+link (no -c), and sccache
# can interfere with those tests. sccache only caches compilations anyway,
# so routing linking/other invocations directly to gcc loses nothing.
case " $* " in
    *" -c "*) exec "${SCCACHE_EXECUTABLE:-sccache}" "${SCCACHE_CC_REAL:-gcc}" "$@" ;;
    *)        exec "${SCCACHE_CC_REAL:-gcc}" "$@" ;;
esac
WRAPPER
    chmod +x "${bin_dir}/sccache-cc"

    cat > "${bin_dir}/sccache-cxx" <<'WRAPPER'
#!/bin/sh
case " $* " in
    *" -c "*) exec "${SCCACHE_EXECUTABLE:-sccache}" "${SCCACHE_CXX_REAL:-g++}" "$@" ;;
    *)        exec "${SCCACHE_CXX_REAL:-g++}" "$@" ;;
esac
WRAPPER
    chmod +x "${bin_dir}/sccache-cxx"

    echo "sccache installed successfully (executable=${sccache_exe}, bin_dir=${bin_dir})"
}

setup_env() {
    local mode="${1:-default}"

    local bin_dir
    bin_dir=$(sccache_bin_dir)
    local sccache_exe
    sccache_exe=$(sccache_executable)

    # Prepend bin_dir to PATH for the compiler wrappers. sccache itself may
    # intentionally remain outside PATH to avoid Meson's automatic detection.
    echo "export PATH=\"${bin_dir}:\${PATH}\";"
    echo "export SCCACHE_EXECUTABLE=\"${sccache_exe}\";"

    # Output a conditional block: only configure sccache if the server starts
    # successfully. The server needs working remote-cache credentials (for
    # example S3/IRSA credentials or a GitHub Actions runtime token, mounted
    # via --mount=type=secret); if they're missing or invalid, we skip sccache
    # entirely so the build continues with normal compilers.
    #
    # Use a per-step Unix domain socket so concurrent builds on the same
    # buildkit worker don't collide on the default TCP port 4226.
    echo 'export SCCACHE_SERVER_UDS="/tmp/sccache-$(mktemp -u XXXXXX).sock";'
    echo 'if "${SCCACHE_EXECUTABLE}" --start-server; then'
    echo '  export SCCACHE_IDLE_TIMEOUT=0;'
    echo '  export RUSTC_WRAPPER="${SCCACHE_EXECUTABLE}";'

    if [ "$mode" = "cmake" ]; then
        echo '  export CMAKE_C_COMPILER_LAUNCHER="${SCCACHE_EXECUTABLE}";'
        echo '  export CMAKE_CXX_COMPILER_LAUNCHER="${SCCACHE_EXECUTABLE}";'
        echo '  export CMAKE_CUDA_COMPILER_LAUNCHER="${SCCACHE_EXECUTABLE}";'
    else
        # Wrapper scripts (installed during sccache install) route only pure
        # compilations (-c flag) through sccache; linking goes directly to
        # the real compiler so autoconf's link tests pass.
        echo '  export SCCACHE_CC_REAL="${CC:-gcc}";'
        echo '  export SCCACHE_CXX_REAL="${CXX:-g++}";'
        echo "  export CC=\"${bin_dir}/sccache-cc\";"
        echo "  export CXX=\"${bin_dir}/sccache-cxx\";"
    fi

    echo 'else'
    echo '  echo "WARNING: sccache server failed to start, building without cache";'
    echo 'fi'
}

show_stats() {
    local sccache_exe
    if sccache_exe=$(sccache_executable 2>/dev/null); then
        # Generate timestamp in ISO 8601 format
        local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

        # Output human-readable text format first
        echo "=== sccache statistics AFTER $1 ==="
        "${sccache_exe}" --show-stats
        echo ""

        # Output JSON markers for deterministic parsing
        echo "=== SCCACHE_JSON_BEGIN ==="

        # Create JSON wrapper with section metadata
        cat <<EOF
{
  "section": "$1",
  "timestamp": "$timestamp",
  "sccache_stats": $("${sccache_exe}" --show-stats --stats-format json)
}
EOF

        echo "=== SCCACHE_JSON_END ==="
    else
        echo "sccache is not available"
    fi
}

main() {
    case "${1:-help}" in
        install)
            install_sccache
            ;;
        setup-env)
            shift
            setup_env "$@"
            ;;
        show-stats)
            shift  # Remove the command from arguments
            show_stats "$@"  # Pass all remaining arguments
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            echo "Unknown command: $1"
            usage
            exit 1
            ;;
    esac
}

main "$@"
