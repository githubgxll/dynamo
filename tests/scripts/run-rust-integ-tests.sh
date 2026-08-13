#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DingoRouter Rust 集成测试串行执行脚本
# 5 个 crate 的集成测试（--test '*'），串行执行，记录退出码/耗时/通过数。
# 用法：在源码根目录执行 ./run-rust-integ-tests.sh
# 前置：先执行 env-setup-integ.sh 完成环境准备（含 e2e fixture 重新生成）。
set -uo pipefail

# ---- 环境变量（与单元测试一致） ----
# 用 $HOME 相对路径：root 服务器为 /root/.cargo，CI 非 root runner 为 ~/.cargo
export PATH="${HOME}/.cargo/bin:${HOME}/.local/bin:${PATH}"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-16}"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
# gcc-12 存在则优先（root 服务器），否则回退系统 gcc
if [ -x /usr/local/gcc-12/bin/gcc ]; then
    export CC="/usr/local/gcc-12/bin/gcc"
    export CXX="/usr/local/gcc-12/bin/g++"
    export LD_LIBRARY_PATH="/usr/local/gcc-12/lib64:${LD_LIBRARY_PATH:-}"
    export LIBRARY_PATH="/usr/local/gcc-12/lib64:${LIBRARY_PATH:-}"
else
    export CC="$(command -v gcc)"
    export CXX="$(command -v g++)"
fi
unset RUSTFLAGS

TOOLCHAIN="1.93.1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SRC_ROOT}"

STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="/tmp/dingoRouter-rust-integ-tests-${STAMP}"
mkdir -p "${LOGDIR}"

# ---- 集成测试定义：crate | --test <文件名> ----
# 用 --test <具体文件名> 精确跑每个集成测试目标（避免 shell glob 展开问题，
# 且 passed 数与"仅集成测试"语义一致，不混入 lib/bin 单元测试）。
# 注意：dynamo-kv-router 需要 --features standalone-indexer
TESTS=(
  "dynamo-kv-hashing|-p dynamo-kv-hashing --test request_hashing"
  "dynamo-kv-hashing|-p dynamo-kv-hashing --test serde_roundtrip"
  "dynamo-kv-router|-p dynamo-kv-router --test standalone_indexer_http --features standalone-indexer"
  "dynamo-runtime|-p dynamo-runtime --test bidirectional_e2e"
  "dynamo-runtime|-p dynamo-runtime --test lifecycle"
  "dynamo-runtime|-p dynamo-runtime --test pipeline"
  "dynamo-runtime|-p dynamo-runtime --test pool"
  "dynamo-runtime|-p dynamo-runtime --test soak"
  "kvbm-consolidator|-p kvbm-consolidator --test chaos_properties"
  "kvbm-consolidator|-p kvbm-consolidator --test dedup"
  "kvbm-consolidator|-p kvbm-consolidator --test e2e"
  "kvbm-consolidator|-p kvbm-consolidator --test kvbm_bridge"
  "kvbm-consolidator|-p kvbm-consolidator --test lifecycle"
  "kvbm-consolidator|-p kvbm-consolidator --test output_contract"
  "kvbm-consolidator|-p kvbm-consolidator --test zmq_ingress"
  "kvbm-kernels|-p kvbm-kernels --test kernel_roundtrip"
  "kvbm-kernels|-p kvbm-kernels --test memcpy_batch"
  "kvbm-kernels|-p kvbm-kernels --test stub_build"
)

SUMMARY="${LOGDIR}/SUMMARY.txt"
{
  echo "# DingoRouter Rust 集成测试汇总"
  echo "# 时间: ${STAMP}"
  echo "# 源码: ${SRC_ROOT}"
  echo "# 工具链: cargo +${TOOLCHAIN}"
  echo "# 日志目录: ${LOGDIR}"
  echo "# 字段: crate | 退出码 | 耗时s | 通过/失败/跳过明细"
  echo
} > "${SUMMARY}"

# 按 crate 聚合
declare -A crate_pass crate_fail crate_ignored crate_code crate_elapsed
crate_order=()
for entry in "${TESTS[@]}"; do
  crate="${entry%%|*}"
  args="${entry#*|}"
  # 去重记录 crate 顺序
  if [ -z "${crate_code[$crate]+x}" ]; then
    crate_order+=("$crate")
    crate_pass[$crate]=0; crate_fail[$crate]=0; crate_ignored[$crate]=0
    crate_code[$crate]=0; crate_elapsed[$crate]=0
  fi
  log="${LOGDIR}/${crate}.log"
  start=$(date +%s)
  echo ">>> [${crate}] cargo +${TOOLCHAIN} test ${args}"
  cargo "+${TOOLCHAIN}" test ${args} >> "${log}" 2>&1
  code=$?
  end=$(date +%s)
  elapsed=$((end - start))
  crate_elapsed[$crate]=$(( ${crate_elapsed[$crate]} + elapsed ))
  if [ "$code" -ne 0 ]; then
    crate_code[$crate]=$code
  fi
  # 解析该次运行的 test result 行
  while IFS= read -r line; do
    p=$(echo "$line" | sed -n 's/.* \([0-9]*\) passed;.*/\1/p')
    f=$(echo "$line" | sed -n 's/.*; \([0-9]*\) failed;.*/\1/p')
    ig=$(echo "$line" | sed -n 's/.*; \([0-9]*\) ignored;.*/\1/p')
    [ -n "$p" ] && crate_pass[$crate]=$(( ${crate_pass[$crate]} + p ))
    [ -n "$f" ] && crate_fail[$crate]=$(( ${crate_fail[$crate]} + f ))
    [ -n "$ig" ] && crate_ignored[$crate]=$(( ${crate_ignored[$crate]} + ig ))
  done < <(grep -E '^test result:' "${log}" | tail -1 || true)
  p=${crate_pass[$crate]}; f=${crate_fail[$crate]}; ig=${crate_ignored[$crate]}
  echo "    (累计) pass=${p} fail=${f} ignored=${ig} (本次 exit=${code} +${elapsed}s)"
done

pass_count=0
fail_count=0
for crate in "${crate_order[@]}"; do
  code=${crate_code[$crate]}
  p=${crate_pass[$crate]}; f=${crate_fail[$crate]}; ig=${crate_ignored[$crate]}; el=${crate_elapsed[$crate]}
  status="PASS"
  if [ "$code" -ne 0 ]; then
    status="FAIL"
    fail_count=$((fail_count + 1))
  else
    pass_count=$((pass_count + 1))
  fi
  echo "${crate} | ${code} | ${el} | pass=${p} fail=${f} ignored=${ig}" >> "${SUMMARY}"
  echo "=== ${crate}: ${status} exit=${code} elapsed=${el}s pass=${p} fail=${f} ignored=${ig}"
done

echo
echo "================ 汇总 ================"
echo "通过: ${pass_count}/${#crate_order[@]}   失败: ${fail_count}/${#crate_order[@]}"
echo "汇总文件: ${SUMMARY}"
echo "日志目录: ${LOGDIR}"

# 统一标签行（供 oneclick 脚本解析）
_gt_p=0; _gt_f=0; _gt_i=0
while IFS='|' read -r _c _code _el rest; do
  _p=$(echo "$rest" | sed -n 's/.*pass=\([0-9]*\).*/\1/p'); [ -n "$_p" ] || _p=0
  _f=$(echo "$rest" | sed -n 's/.*fail=\([0-9]*\).*/\1/p'); [ -n "$_f" ] || _f=0
  _i=$(echo "$rest" | sed -n 's/.*ignored=\([0-9]*\).*/\1/p'); [ -n "$_i" ] || _i=0
  _gt_p=$((_gt_p + _p)); _gt_f=$((_gt_f + _f)); _gt_i=$((_gt_i + _i))
done < <(tail -n +3 "${SUMMARY}" 2>/dev/null)
echo "GROUP_TOTAL passed=${_gt_p} failed=${_gt_f} ignored=${_gt_i} skipped=0 deselected=0"
