#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DingoRouter Rust 集成测试环境准备脚本
# 环境依赖与单元测试相同（protoc 21.12 + gcc-12 libstdc++ + tokio_unstable cfg），
# 本脚本额外重新生成 kvbm-consolidator 的 e2e fixture（避免反序列化失败）。
# 幂等：已满足的步骤会跳过。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== [1/2] 调用通用环境准备（protoc / gcc-12 / tokio_unstable） ==="
bash "${SCRIPT_DIR}/env-setup.sh"

echo "=== [2/2] 重新生成 kvbm-consolidator e2e fixture ==="
export PATH="${HOME}/.cargo/bin:${HOME}/.local/bin:${PATH}"
# gcc-12 存在则优先（root 服务器），否则回退系统 gcc
if [ -x /usr/local/gcc-12/bin/gcc ]; then
    export LD_LIBRARY_PATH="/usr/local/gcc-12/lib64:${LD_LIBRARY_PATH:-}"
    export LIBRARY_PATH="/usr/local/gcc-12/lib64:${LIBRARY_PATH:-}"
    export CC="/usr/local/gcc-12/bin/gcc"
    export CXX="/usr/local/gcc-12/bin/g++"
else
    export CC="$(command -v gcc)"
    export CXX="$(command -v g++)"
fi
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
unset RUSTFLAGS

SRC_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SRC_ROOT}"
FIXTURE="lib/kvbm-consolidator/tests/fixtures/vllm_capture.msgpack"

# 总是重新生成 fixture：仓库内的 fixture 可能是旧格式（integer 而非 sequence），
# e2e.rs:198 用 rmp_serde 反序列化为 Vec<Vec<u8>> 会报 Syntax 错误。
# e2e_regenerate 是 #[ignore] 的测试，调用 regenerate_fixtures() 写出正确格式。
# 用 e2e test binary 的 --ignored e2e_regenerate 模式运行，代价低（只写 130 字节文件）。
# 先编译 e2e test target（如果已编译则秒过），再跑 regenerate。
echo "编译 e2e test target ..."
cargo +1.93.1 test -p kvbm-consolidator --test e2e --no-run 2>&1 | tail -3 || true
echo "重新生成 fixture ..."
cargo +1.93.1 test -p kvbm-consolidator --test e2e -- --ignored e2e_regenerate 2>&1 | tail -3
echo "fixture 大小: $(stat -c%s "$FIXTURE" 2>/dev/null || echo 0) 字节（正确应为 130）"

echo
echo "=== 集成测试环境准备完成 ==="
echo "后续运行测试请执行同目录的 run-rust-integ-tests.sh"
