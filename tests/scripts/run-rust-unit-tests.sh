#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DingoRouter Rust 单元测试串行执行脚本
# 23 行 cargo test 严格串行，记录退出码/耗时/通过数到日志。
# 用法：在本源码根目录（/root/gjn/dynamo）执行 ./run-rust-unit-tests.sh
# 前置：先执行 env-setup.sh 完成环境准备。
set -uo pipefail

# ---- 环境变量（关键：覆盖顺序见 env-setup.sh 说明） ----
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
# 关键：不要用 RUSTFLAGS 提供链接路径，否则会覆盖 .cargo/config.toml 的 tokio_unstable cfg
unset RUSTFLAGS

TOOLCHAIN="1.93.1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 源码根目录：脚本位于 <root>/tests/router，回退两级
SRC_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SRC_ROOT}"

STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="/tmp/dingoRouter-rust-tests-${STAMP}"
mkdir -p "${LOGDIR}"

# ---- 23 行测试定义：idx | log名 | cargo test 参数 ----
ALL_TESTS=(
  "01-dynamo-backend-common|-p dynamo-backend-common --lib"
  "02-dynamo-bench|-p dynamo-bench --lib"
  "03-dynamo-codegen|-p dynamo-codegen --lib --bins"
  "04-dynamo-data-gen|-p dynamo-data-gen --lib"
  "05-dynamo-ext-proc|-p dynamo-ext-proc --lib --bins"
  "06-dynamo-kv-hashing|-p dynamo-kv-hashing --lib"
  "07-dynamo-kv-router|-p dynamo-kv-router --lib"
  "08-dynamo-llm|-p dynamo-llm --lib"
  "09-dynamo-memory|-p dynamo-memory --lib"
  "10-dynamo-mocker|-p dynamo-mocker --lib"
  "11-dynamo-mocker-backend|-p dynamo-mocker-backend --bins"
  "12-dynamo-rl|-p dynamo-rl --lib"
  "13-dynamo-runtime|-p dynamo-runtime --lib"
  "14-dynamo-tokens|-p dynamo-tokens --lib"
  "15-dynamo-vllm-rs-backend|-p dynamo-vllm-rs-backend --bins"
  "16-kvbm-common|-p kvbm-common --lib"
  "17-kvbm-config|-p kvbm-config --lib"
  "18-kvbm-consolidator|-p kvbm-consolidator --lib"
  "19-kvbm-engine|-p kvbm-engine --lib"
  "20-kvbm-kernels|-p kvbm-kernels --lib"
  "21-kvbm-logical|-p kvbm-logical --lib"
  "22-kvbm-physical|-p kvbm-physical --lib"
  "23-libdynamo_llm|-p libdynamo_llm --lib"
)

# 支持 QUICK_PACKS 环境变量筛选测试包（逗号分隔，如 "dynamo-llm,dynamo-runtime"）
# 未设置时跑全部 23 个
if [ -n "${QUICK_PACKS:-}" ]; then
  TESTS=()
  IFS=',' read -ra _pkgs <<< "${QUICK_PACKS}"
  for entry in "${ALL_TESTS[@]}"; do
    _pkg_name=$(echo "$entry" | sed -n 's/.*-p \([^ ]*\).*/\1/p')
    for _q in "${_pkgs[@]}"; do
      if [ "$_pkg_name" = "$_q" ]; then
        TESTS+=("$entry")
        break
      fi
    done
  done
  echo "QUICK_PACKS 模式：只跑 ${#TESTS[@]} 个包：${QUICK_PACKS}"
else
  TESTS=("${ALL_TESTS[@]}")
fi

SUMMARY="${LOGDIR}/SUMMARY.txt"
{
  echo "# DingoRouter Rust 单元测试汇总"
  echo "# 时间: ${STAMP}"
  echo "# 源码: ${SRC_ROOT}"
  echo "# 工具链: cargo +${TOOLCHAIN}"
  echo "# 日志目录: ${LOGDIR}"
  echo "# 字段: 序号 | 退出码 | 耗时s | passed | failed | ignored"
  echo
} > "${SUMMARY}"

pass_count=0
fail_count=0
for entry in "${TESTS[@]}"; do
  idx="${entry%%|*}"
  args="${entry#*|}"
  log="${LOGDIR}/${idx}.log"
  start=$(date +%s)
  echo ">>> [${idx}] cargo +${TOOLCHAIN} test ${args}"
  cargo "+${TOOLCHAIN}" test ${args} > "${log}" 2>&1
  code=$?
  end=$(date +%s)
  elapsed=$((end - start))
  # 解析 test result 行（取最后一条）
  result_line=$(grep -E '^test result:' "${log}" | tail -1 || true)
  passed=$(echo "${result_line}" | sed -n 's/.* \([0-9]*\) passed;.*/\1/p')
  failed=$(echo "${result_line}" | sed -n 's/.*; \([0-9]*\) failed;.*/\1/p')
  ignored=$(echo "${result_line}" | sed -n 's/.*; \([0-9]*\) ignored;.*/\1/p')
  [ -z "${passed}" ] && passed=0
  [ -z "${failed}" ] && failed=0
  [ -z "${ignored}" ] && ignored=0
  status="PASS"
  if [ "${code}" -ne 0 ]; then
    status="FAIL"
    fail_count=$((fail_count + 1))
  else
    pass_count=$((pass_count + 1))
  fi
  echo "    ${status} exit=${code} elapsed=${elapsed}s passed=${passed} failed=${failed} ignored=${ignored}"
  echo "${idx} | ${code} | ${elapsed} | ${passed} | ${failed} | ${ignored}" >> "${SUMMARY}"
done

echo
echo "================ 汇总 ================"
echo "通过: ${pass_count}/23   失败: ${fail_count}/23"
echo "汇总文件: ${SUMMARY}"
echo "日志目录: ${LOGDIR}"

# 统一标签行（供 oneclick 脚本解析）：GROUP_TOTAL passed=P failed=F ignored=I skipped=S deselected=D
_gt_p=0; _gt_f=0; _gt_i=0
while IFS='|' read -r _idx _code _el p f i; do
  [ -n "${p:-}" ] && [ "$p" -ge 0 ] 2>/dev/null && _gt_p=$((_gt_p + p))
  [ -n "${f:-}" ] && [ "$f" -ge 0 ] 2>/dev/null && _gt_f=$((_gt_f + f))
  [ -n "${i:-}" ] && [ "$i" -ge 0 ] 2>/dev/null && _gt_i=$((_gt_i + i))
done < <(tail -n +3 "${SUMMARY}" 2>/dev/null)
echo "GROUP_TOTAL passed=${_gt_p} failed=${_gt_f} ignored=${_gt_i} skipped=0 deselected=0"
