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

# Fixture regeneration changes checked-in test data; never do it implicitly.
echo "Integration fixtures use the current checkout; no automatic regeneration."
