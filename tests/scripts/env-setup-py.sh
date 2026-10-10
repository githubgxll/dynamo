#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Prepare the selected interpreter; failures stop setup instead of skipping tests.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/python-env.sh"
if [[ ! -x "$PY" && "$PY" == "${TEST_VENV_DIR}/bin/python" ]]; then
    "${TEST_PYTHON_BOOTSTRAP:-python3.11}" -m venv "$TEST_VENV_DIR"
fi
"$PY" -c 'import sys; print("Test interpreter:", sys.executable)'
bash "$SCRIPT_DIR/env-setup.sh"
export PATH="$(dirname "$PY"):${HOME}/.cargo/bin:${HOME}/.local/bin:$PATH"
export RUSTUP_TOOLCHAIN="${RUST_TOOLCHAIN:-$(sed -n 's/^channel = "\(.*\)"/\1/p' "$SRC_ROOT/rust-toolchain.toml")}"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
export CARGO_INCREMENTAL=0
# Replay smoke tests have real-time deadlines; use optimized bindings while
# disabling LTO/debug symbols to keep development builds affordable.
export CARGO_PROFILE_RELEASE_DEBUG=0
export CARGO_PROFILE_RELEASE_LTO=false
export CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16
if [ -x /usr/local/gcc-12/bin/gcc ]; then
    export CC=/usr/local/gcc-12/bin/gcc CXX=/usr/local/gcc-12/bin/g++
    export LD_LIBRARY_PATH="/usr/local/gcc-12/lib64:${LD_LIBRARY_PATH:-}"
fi
export LIBCLANG_PATH="${LIBCLANG_PATH:-/usr/lib/x86_64-linux-gnu}"
unset RUSTFLAGS
"$PY" -m pip install pytest pytest-timeout pytest-benchmark pytest-asyncio pytest-xdist pytest-rerunfailures pytest-httpserver pytest-forked maturin patchelf tomli
# Always build this checkout, including uncommitted repairs. Never reuse an
# unrelated installed wheel based only on a successful import.
"$PY" -m pip install --force-reinstall --no-deps --config-settings 'build-args=--profile release --features slot-tracker,kv-indexer,select-service,ais-forward-pass' "$SRC_ROOT/lib/bindings/python"
"$PY" -m pip install -e "$SRC_ROOT[video-gateway]" scipy nats-py etcd3 psutil requests filelock huggingface_hub 'blinker>=1.9' uvloop
# Match repository test/runtime constraints instead of retaining stale system deps.
"$PY" -m pip install -r "$SRC_ROOT/container/deps/requirements.test.txt" -r "$SRC_ROOT/container/deps/requirements.common.txt" -r "$SRC_ROOT/container/deps/requirements.planner.txt"
# KServe tests need the gRPC client. 2.73 accepts the protobuf 6 runtime and
# grpcio used by requirements.common; older clients pin incompatible grpcio.
"$PY" -m pip install "${TEST_TRITONCLIENT_SPEC:-tritonclient[grpc]==2.73.0}"
if [[ "${INSTALL_CPU_TORCH:-auto}" == 1 ]] || { [[ "${INSTALL_CPU_TORCH:-auto}" == auto ]] && ! "$PY" -c 'import torch' >/dev/null 2>&1; }; then
    "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
# CPU video tests exercise the real encoder; do not download imageio's bundled binary.
if [[ -z "${IMAGEIO_FFMPEG_EXE:-}" ]] && ! command -v ffmpeg >/dev/null 2>&1; then
    if [[ "$(id -u)" == 0 ]] && command -v apt-get >/dev/null 2>&1; then
        apt-get install -y ffmpeg
    else
        echo "Install ffmpeg or set IMAGEIO_FFMPEG_EXE before running CPU video tests" >&2
        exit 1
    fi
fi
"$PY" -c 'import dynamo._core, dynamo.llm, dingo; from importlib.metadata import version; print("runtime", version("ai-dingo-runtime"), dynamo._core.__file__); print("dingo", dingo.__file__)'

"$PY" -m pip check
