#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DingoRouter Python 单元测试环境准备脚本
# 在源码目录（/root/gjn/dynamo）执行 dingo 三组 Python 单元测试前，先跑本脚本。
# 幂等：已满足的步骤会跳过。
set -euo pipefail

PY="/usr/bin/python3.11"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== [0/5] 调用通用环境准备（protoc 21.12 + gcc-12 libstdc++ + tokio_unstable） ==="
# 构建 ai-dingo-runtime（Rust pyo3 binding）需要 protoc 和 gcc-12 libstdc++，复用单元测试环境。
bash "${SCRIPT_DIR}/env-setup.sh"

# SKIP_RUNTIME_BUILD=1（如 CI 无 libclang）：Rust 环境已就绪，Python 侧全部跳过，直接返回
if [ "${SKIP_RUNTIME_BUILD:-0}" = "1" ]; then
    echo "SKIP: SKIP_RUNTIME_BUILD=1，跳过 Python 环境准备（pytest 插件/绑定构建/editable 安装）"
    echo
    echo "=== Python 测试环境准备跳过（SKIP_RUNTIME_BUILD=1）==="
    exit 0
fi

echo "=== [1/5] 确认 python3.11 + ensurepip ==="
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "ERROR: ${PY} 不存在" >&2; exit 1
fi
if ! "$PY" -m pip --version >/dev/null 2>&1; then
    echo "bootstrap pip ..."
    "$PY" -m ensurepip >/dev/null 2>&1 || { echo "ERROR: ensurepip 失败" >&2; exit 1; }
fi
echo "OK: $(${PY} --version), pip $( ${PY} -m pip --version | awk '{print $2}')"

echo "=== [2/5] 安装 pytest 测试插件 ==="
# pyproject.toml [tool.pytest.ini_options] 引用了 pytest_benchmark/plugins，必须安装。
# asyncio_mode=auto 需要 pytest-asyncio。
"$PY" -m pip install --quiet --upgrade \
  pytest pytest-timeout pytest-benchmark pytest-asyncio \
  pytest-xdist pytest-rerunfailures 2>&1 | tail -2 || true
echo "OK: pytest 测试插件已安装"

echo "=== [3/5] 构建 ai-dingo-runtime（Rust Python binding，本地安装） ==="
# ai-dingo 依赖 ai-dingo-runtime，该包不在 PyPI，需从 lib/bindings/python maturin 构建并 pip 安装。
# 关键：必须通过 pip install 落地 .dist-info 元数据，否则 test_wheel_contents 查不到包会 fail。
# 幂等判断：pip show 能查到元数据 且 dynamo._core 能 import 才算已装。
# 设 SKIP_RUNTIME_BUILD=1 可跳过构建（如 CI 无 libclang 且无 root 权限装系统包），Python 测试将整体跳过。
runtime_ok=0
if [ "${SKIP_RUNTIME_BUILD:-0}" = "1" ]; then
    echo "SKIP: SKIP_RUNTIME_BUILD=1，跳过 ai-dingo-runtime 构建（Python 测试将跳过）"
elif "$PY" -m pip show ai-dingo-runtime >/dev/null 2>&1 && "$PY" -c "import dynamo._core" >/dev/null 2>&1; then
    echo "OK: ai-dingo-runtime 已安装（含元数据），跳过构建"
    runtime_ok=1
else
    export PATH="${HOME}/.cargo/bin:${HOME}/.local/bin:${PATH}"
    # gcc-12 路径（env-setup.sh 已检测；非 root 可能用系统 gcc）
    if [ -x /usr/local/gcc-12/bin/gcc ]; then
        export CC="/usr/local/gcc-12/bin/gcc"
        export CXX="/usr/local/gcc-12/bin/g++"
        export LD_LIBRARY_PATH="/usr/local/gcc-12/lib64:${LD_LIBRARY_PATH:-}"
        export LIBRARY_PATH="/usr/local/gcc-12/lib64:${LIBRARY_PATH:-}"
    elif [ -n "${CC:-}" ]; then
        :  # CC 已由 env-setup.sh 设置
    else
        export CC="$(which gcc)"
        export CXX="$(which g++)"
    fi
    # bindgen 需要 libclang
    [ -f /etc/profile.d/libclang_path.sh ] && source /etc/profile.d/libclang_path.sh 2>/dev/null || true
    export LIBCLANG_PATH="${LIBCLANG_PATH:-}"
    unset RUSTFLAGS
    if ! command -v maturin >/dev/null 2>&1; then
        echo "安装 maturin ..."
        "$PY" -m pip install --quiet maturin
    fi
    SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    echo "构建并安装 ai-dingo-runtime（首次约 5-8 分钟）..."
    # 不用管道（避免 pipefail 下 pip 失败导致 set -e 退出脚本），日志落盘后判断
    if "$PY" -m pip install --force-reinstall "${SRC_ROOT}/lib/bindings/python/" > /tmp/runtime-build.log 2>&1; then
        echo "OK: ai-dingo-runtime pip 安装返回 0"
    else
        echo "WARN: ai-dingo-runtime 构建失败（可能缺 libclang）；Python 测试将跳过" >&2
        echo "      详细日志: /tmp/runtime-build.log" >&2
    fi
    if "$PY" -m pip show ai-dingo-runtime >/dev/null 2>&1 && "$PY" -c "import dynamo._core" >/dev/null 2>&1; then
        echo "OK: ai-dingo-runtime 已安装（含元数据）"
        runtime_ok=1
    else
        echo "WARN: ai-dingo-runtime 不可用，Python 测试将跳过" >&2
    fi
fi

echo "=== [4/5] editable 安装 ai-dingo ==="
SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# 幂等判断：dingo 能 import 且 numpy 能 import（editable 应拉入的依赖）才算已装
if "$PY" -c "import dingo" >/dev/null 2>&1 && "$PY" -c "import numpy" >/dev/null 2>&1; then
    echo "OK: ai-dingo 已安装（含依赖），跳过"
elif [ "${runtime_ok:-0}" -eq 0 ]; then
    echo "SKIP: ai-dingo-runtime 不可用，跳过 ai-dingo editable 安装（Python 测试将跳过）"
else
    # 不用管道（避免 pipefail 下 pip 失败导致 set -e 退出脚本），日志落盘后判断
    if "$PY" -m pip install -e "${SRC_ROOT}" > /tmp/ai-dingo-install.log 2>&1; then
        echo "OK: ai-dingo editable pip 安装返回 0"
    else
        echo "WARN: ai-dingo editable 安装失败，详见 /tmp/ai-dingo-install.log" >&2
    fi
    # 以关键依赖能否 import 为准（pip 可能因依赖冲突告警返回非零）
    "$PY" -c "import dingo; import numpy; print('OK')" >/dev/null 2>&1 && echo "OK: ai-dingo editable 安装完成" || { echo "WARN: ai-dingo 不可用，Python 测试将跳过" >&2; }
fi

echo
echo "=== Python 测试环境准备完成 ==="
echo "后续运行测试请执行同目录的 run-python-unit-tests.sh（用 /usr/bin/python3.11）"
