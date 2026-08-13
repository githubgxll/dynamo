#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DingoRouter Python 集成测试环境准备脚本
# 在源码目录执行 Python 集成测试（命令 4-8）前先跑本脚本。
# 依赖 Python 单元测试环境（env-setup-py.sh），并额外安装 etcd/nats-server/HF 模型。
# 幂等：已满足的步骤会跳过。
set -euo pipefail

PY="/usr/bin/python3.11"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== [0/6] 调用 Python 单元测试环境准备（含通用 Rust 环境） ==="
bash "${SCRIPT_DIR}/env-setup-py.sh"

# SKIP_RUNTIME_BUILD=1（如 CI 无 libclang）时跳过一切 Python 集成测试环境准备
if [ "${SKIP_RUNTIME_BUILD:-0}" = "1" ]; then
    echo "SKIP: SKIP_RUNTIME_BUILD=1，跳过 Python 集成测试环境准备（etcd/nats-server/模型 均不安装）"
    echo
    echo "=== Python 集成测试环境准备跳过（SKIP_RUNTIME_BUILD=1）==="
    exit 0
fi

echo "=== [1/6] 安装 Python 集成测试依赖 ==="
# nats-py/etcd3/psutil/requests/aiohttp/filelock/huggingface_hub（下载模型用）
$PY -m pip install --quiet nats-py etcd3 psutil requests aiohttp filelock huggingface_hub 2>&1 | tail -2 || true
# etcd3 与新版 protobuf 不兼容，降级 protobuf（<3.21）使 import 通过。
$PY -m pip install --quiet "protobuf<3.21" 2>&1 | tail -1 || true
$PY -c "import nats, etcd3, psutil, requests, aiohttp, filelock, huggingface_hub; print('py integ deps OK')" 2>&1 | tail -1

echo "=== [2/6] 安装 etcd ==="
if command -v etcd >/dev/null 2>&1; then
    echo "OK: etcd 已存在 ($(etcd --version 2>&1 | head -1))"
else
    cd /tmp
    curl -sS -L -m 120 -o etcd.tar.gz https://github.com/etcd-io/etcd/releases/download/v3.5.21/etcd-v3.5.21-linux-amd64.tar.gz
    tar xzf etcd.tar.gz
    cp etcd-v3.5.21-linux-amd64/etcd etcd-v3.5.21-linux-amd64/etcdctl /usr/local/bin/
    echo "OK: etcd $(/usr/local/bin/etcd --version 2>&1 | head -1)"
fi

echo "=== [3/6] 安装 nats-server ==="
if command -v nats-server >/dev/null 2>&1; then
    echo "OK: nats-server 已存在 ($(nats-server --version 2>&1 | head -1))"
else
    cd /tmp
    curl -sS -L -m 120 -o nats-server.tar.gz https://github.com/nats-io/nats-server/releases/download/v2.10.27/nats-server-v2.10.27-linux-amd64.tar.gz
    tar xzf nats-server.tar.gz
    cp nats-server-v2.10.27-linux-amd64/nats-server /usr/local/bin/
    echo "OK: nats-server $(/usr/local/bin/nats-server --version 2>&1 | head -1)"
fi

echo "=== [4/6] 预下载 HuggingFace 模型（Qwen/Qwen3-0.6B） ==="
# 测试 conftest 在 HF_HUB_OFFLINE=1 下要求本地缓存存在；models-dir 指向该目录。
# 默认不下载模型（纯 CPU 节点不依赖）。
# 设 DOWNLOAD_MODEL=1 可启用下载（同时 run-python-integ-tests.sh 会跑命令 06/07/08）。
MODELS_DIR="/root/.cache/huggingface"
mkdir -p "${MODELS_DIR}"
if [ "${DOWNLOAD_MODEL:-0}" = "1" ]; then
    HF_DIR="${MODELS_DIR}/hub/models--Qwen--Qwen3-0.6B"
    if [ -d "${HF_DIR}" ] && [ "$(ls -A "${HF_DIR}/snapshots" 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "OK: Qwen/Qwen3-0.6B 已缓存"
    else
        echo "下载 Qwen/Qwen3-0.6B ..."
        HF_HUB_OFFLINE=0 $PY -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', ignore_patterns=['*.pth','*.onnx','*.gguf','original/*'])" 2>&1 | tail -2
        echo "OK: 已下载到 ${HF_DIR}"
    fi
else
    echo "默认不下载模型（DOWNLOAD_MODEL=1 可启用下载并跑命令 06/07/08）"
fi

echo "=== [5/6] （可选）重新构建含 slot-tracker 的 ai-dingo-runtime ==="
# test_standalone_slot_tracker 需要 dynamo.slot_tracker，要求 binding 以 --features slot-tracker 构建。
# 默认跳过（该功能可能拖入额外依赖）；如需启用，设环境变量 REBUILD_WITH_SLOT_TRACKER=1。
if [ "${REBUILD_WITH_SLOT_TRACKER:-0}" = "1" ]; then
    export PATH="${HOME}/.cargo/bin:${HOME}/.local/bin:${PATH}"
    if [ -x /usr/local/gcc-12/bin/gcc ]; then
        export LD_LIBRARY_PATH="/usr/local/gcc-12/lib64:${LD_LIBRARY_PATH:-}"
        export LIBRARY_PATH="/usr/local/gcc-12/lib64:${LIBRARY_PATH:-}"
    fi
    unset RUSTFLAGS
    SRC_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
    echo "重新构建 ai-dingo-runtime（包含 slot-tracker feature）..."
    $PY -m maturin build --manifest-path "${SRC_ROOT}/lib/bindings/python/Cargo.toml" --features slot-tracker --interpreter "$PY" 2>&1 | tail -3
    $PY -m pip install --quiet --force-reinstall "${SRC_ROOT}/lib/bindings/python/" 2>&1 | tail -2 || true
else
    echo "跳过（设 REBUILD_WITH_SLOT_TRACKER=1 可启用；test_standalone_slot_tracker 将因此失败）"
fi

echo "=== [6/6] 环境变量说明 ==="
echo "run-python-integ-tests.sh 已固化：PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python、HF_HUB_OFFLINE=1、--models-dir=${MODELS_DIR}"

echo
echo "=== Python 集成测试环境准备完成 ==="
echo "后续运行测试请执行同目录的 run-python-integ-tests.sh"
