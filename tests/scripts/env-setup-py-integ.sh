#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DingoRouter Python 集成测试环境准备脚本
# 在源码目录执行 Python 集成测试（命令 4-8）前先跑本脚本。
# 依赖 Python 单元测试环境（env-setup-py.sh），并额外安装 etcd/nats-server/HF 模型。
# 幂等：已满足的步骤会跳过。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/python-env.sh"

echo "=== [0/6] 调用 Python 单元测试环境准备（含通用 Rust 环境） ==="
bash "${SCRIPT_DIR}/env-setup-py.sh"

echo "=== [1/6] 安装 Python 集成测试依赖 ==="
# nats-py/etcd3/psutil/requests/aiohttp/filelock/huggingface_hub（下载模型用）
$PY -m pip install --quiet nats-py etcd3 psutil requests aiohttp filelock huggingface_hub
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
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

echo "slot-tracker feature was installed by env-setup-py.sh"

echo "=== [6/6] 环境变量说明 ==="
echo "run-python-integ-tests.sh 已固化：PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python、HF_HUB_OFFLINE=1、--models-dir=${MODELS_DIR}"

echo
echo "=== Python 集成测试环境准备完成 ==="
echo "后续运行测试请执行同目录的 run-python-integ-tests.sh"
