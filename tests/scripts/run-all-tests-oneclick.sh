#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DingoRouter 一键全量测试脚本
# 顺序执行：环境准备 → Rust 单元测试 → Rust 集成测试 → Python 单元测试 → Python 集成测试
# 用法：
#   cd /root/gjn/dynamo && bash tests/router/run-all-tests-oneclick.sh
#   QUICK_MODE=smoke    bash tests/router/run-all-tests-oneclick.sh     # 冒烟：Rust 5核心包单测 + Python 3组单测
#   QUICK_MODE=standard bash tests/router/run-all-tests-oneclick.sh     # 中速：Rust 23包全量单测 + Python 3组单测（无集成测试）
#   QUICK_MODE=full bash tests/router/run-all-tests-oneclick.sh         # 全量（默认，不含模型）
#   DOWNLOAD_MODEL=1 QUICK_MODE=full bash tests/router/run-all-tests-oneclick.sh  # 全量+模型+依赖模型测试
# 前提：仅需源码已 clone；rustup/gcc-12/python3.11/protoc 均由 env-setup.sh 自动检测并安装。需 root + 外网。
# 默认不下载模型；DOWNLOAD_MODEL=1 时下载 Qwen3-0.6B 并跑依赖模型的集成测试(06/07/08)。
# 幂等：环境已就绪的步骤自动跳过；测试日志在 /tmp/dingoRouter-*-tests-<时间戳>/
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STAMP=$(date +%Y%m%d_%H%M%S)
QUICK_MODE="${QUICK_MODE:-full}"

# 颜色
G='\033[0;32m'; R='\033[0;31m'; C='\033[0;36m'; Y='\033[0;33m'; N='\033[0m'
info()  { echo -e "${C}[INFO]${N}  $1"; }
ok()    { echo -e "${G}[ OK ]${N}  $1"; }
fail()  { echo -e "${R}[FAIL]${N}  $1"; }
sec()   { echo -e "\n${C}========== $1 ==========${N}"; }

cd "${SRC_ROOT}"

# Python 解释器（优先 python3.11，其次 python3）
for _py in python3.11 python3; do command -v "$_py" >/dev/null 2>&1 && PY="$_py" && break; done

# ============== 模式说明 ==============
sec "DingoRouter 一键测试（${STAMP}）  QUICK_MODE=${QUICK_MODE}"
info "源码: ${SRC_ROOT}"
case "${QUICK_MODE}" in
    smoke)    info "冒烟模式：环境准备 → Rust 5 核心包单测（Python 单测需 binding，如装不了则跳过）" ;;
    standard) info "中速模式：环境准备 → Rust 23 包全量单测 → Python 3 组单测（跳过所有集成测试）" ;;
    full)     info "全量模式：环境准备 → Rust 单测 → Rust 集测 → Python 单测 → Python 集测"
              [ "${DOWNLOAD_MODEL:-0}" = "1" ] && info "  DOWNLOAD_MODEL=1：将下载模型并跑 06/07/08" || info "  默认不下载模型（06/07/08 跳过）" ;;
    *)        info "未知 QUICK_MODE=${QUICK_MODE}，回退到 full 模式"; QUICK_MODE="full" ;;
esac

TOTAL_PASS=0
TOTAL_FAIL=0
SUM_PASSED=0; SUM_FAILED=0; SUM_IGNORED=0; SUM_SKIPPED=0; SUM_DESELECTED=0
declare -a RESULTS=()

# 从日志解析子脚本末尾输出的 GROUP_TOTAL 行
parse_counts() {
  local logfile="$1"
  local line=$(grep -E '^GROUP_TOTAL ' "${logfile}" 2>/dev/null | tail -1)
  local p=0 f=0 i=0 s=0 d=0
  p=$(echo "$line" | sed -n 's/.*passed=\([0-9]*\).*/\1/p'); [ -n "$p" ] || p=0
  f=$(echo "$line" | sed -n 's/.*failed=\([0-9]*\).*/\1/p'); [ -n "$f" ] || f=0
  i=$(echo "$line" | sed -n 's/.*ignored=\([0-9]*\).*/\1/p'); [ -n "$i" ] || i=0
  s=$(echo "$line" | sed -n 's/.*skipped=\([0-9]*\).*/\1/p'); [ -n "$s" ] || s=0
  d=$(echo "$line" | sed -n 's/.*deselected=\([0-9]*\).*/\1/p'); [ -n "$d" ] || d=0
  echo "${p}|${f}|${i}|${s}|${d}"
}

# 运行某个测试脚本并解析汇总
run_test_script() {
  local label="$1"
  local script="$2"
  local start=$(date +%s)
  info "执行 ${label}: ${script}"
  bash "${script}" > /tmp/oneclick-${label}.log 2>&1
  local code=$?
  local end=$(date +%s)
  local elapsed=$((end - start))

  local counts=$(parse_counts "/tmp/oneclick-${label}.log")
  local p f i s d
  IFS='|' read -r p f i s d <<< "${counts}"
  [ -z "${p:-}" ] && p=0; [ -z "${f:-}" ] && f=0; [ -z "${i:-}" ] && i=0
  [ -z "${s:-}" ] && s=0; [ -z "${d:-}" ] && d=0

  local summary="passed=${p} failed=${f} ignored=${i} skipped=${s} deselected=${d}"

  if [ "${code}" -eq 0 ] && [ "${f}" -eq 0 ]; then
    ok "${label} 完成 (${elapsed}s)  ${summary}"
    TOTAL_PASS=$((TOTAL_PASS + 1))
    RESULTS+=("${label}|PASS|${elapsed}|${p}|${f}|${i}|${s}|${d}")
  else
    fail "${label} 退出码=${code} (${elapsed}s)  ${summary}"
    TOTAL_FAIL=$((TOTAL_FAIL + 1))
    RESULTS+=("${label}|FAIL(${code})|${elapsed}|${p}|${f}|${i}|${s}|${d}")
  fi
  SUM_PASSED=$((SUM_PASSED + p)); SUM_FAILED=$((SUM_FAILED + f)); SUM_IGNORED=$((SUM_IGNORED + i))
  SUM_SKIPPED=$((SUM_SKIPPED + s)); SUM_DESELECTED=$((SUM_DESELECTED + d))
  info "明细: ${summary}"
  info "完整日志: /tmp/oneclick-${label}.log"
}

# ============== 1. 环境准备 ==============
sec "第 0 步：环境准备（链式自包含，幂等）"
info "调用 env-setup-py-integ.sh（内部级联 env-setup-py.sh → env-setup.sh）"
bash "${SCRIPT_DIR}/env-setup-py-integ.sh" > /tmp/oneclick-envsetup.log 2>&1
env_code=$?
if [ "${env_code}" -ne 0 ]; then
  fail "环境准备失败（退出码 ${env_code}），终止"
  tail -20 /tmp/oneclick-envsetup.log
  exit 1
fi
ok "环境准备完成"
if [ "${QUICK_MODE}" = "full" ]; then
  info "调用 env-setup-integ.sh（Rust 集成测试 e2e fixture）"
  bash "${SCRIPT_DIR}/env-setup-integ.sh" > /tmp/oneclick-envsetup-integ.log 2>&1
  ok "Rust 集成测试 fixture 准备完成"
fi

# ============== 2. Rust 单元测试 ==============
if [ "${QUICK_MODE}" = "smoke" ]; then
  sec "第 1 步：Rust 单元测试（冒烟：5 核心包）"
  export QUICK_PACKS="dynamo-backend-common,dynamo-kv-router,dynamo-runtime,dynamo-llm,dynamo-mocker"
else
  sec "第 1 步：Rust 单元测试（23 包全量）"
  unset QUICK_PACKS 2>/dev/null || true
fi
run_test_script "rust-unit" "${SCRIPT_DIR}/run-rust-unit-tests.sh"

# ============== 3. Rust 集成测试 ==============
if [ "${QUICK_MODE}" = "full" ]; then
  sec "第 2 步：Rust 集成测试（5 crate）"
  run_test_script "rust-integ" "${SCRIPT_DIR}/run-rust-integ-tests.sh"
else
  sec "第 2 步：Rust 集成测试（${QUICK_MODE} 模式跳过）"
fi

# ============== 4. Python 单元测试 ==============
# Python 测试需要 ai-dingo-runtime（dynamo._core）和 ai-dingo（editable）。
# 如果 binding 构建失败（如缺 libclang），跳过 Python 测试而非整体失败。
if "${PY:-python3.11}" -c "import dynamo._core" >/dev/null 2>&1; then
  sec "第 3 步：Python 单元测试（3 组）"
  run_test_script "py-unit" "${SCRIPT_DIR}/run-python-unit-tests.sh"
else
  sec "第 3 步：Python 单元测试（跳过：dynamo._core 不可用，需 libclang 构建 binding）"
  info "如需跑 Python 测试，请在 runner 上预装 libclang（clang-devel / libclang-dev）"
fi

# ============== 5. Python 集成测试 ==============
if [ "${QUICK_MODE}" = "full" ]; then
  if "${PY:-python3.11}" -c "import dynamo._core" >/dev/null 2>&1; then
    sec "第 4 步：Python 集成测试（5 组）"
    run_test_script "py-integ" "${SCRIPT_DIR}/run-python-integ-tests.sh"
  else
    sec "第 4 步：Python 集成测试（跳过：dynamo._core 不可用）"
  fi
else
  sec "第 4 步：Python 集成测试（${QUICK_MODE} 模式跳过）"
fi

# ============== 汇总 ==============
TOTAL_ELAPSED=0
for r in "${RESULTS[@]}"; do
  _el=$(echo "$r" | awk -F'|' '{print $3}')
  TOTAL_ELAPSED=$((TOTAL_ELAPSED + _el))
done

sec "总览"
printf "%-12s | %-9s | %8s | %7s | %7s | %7s | %7s | %11s\n" "组别" "状态" "耗时s" "passed" "failed" "ignored" "skipped" "deselected"
echo "-------------|-----------|----------|---------|---------|---------|---------|-------------"
for r in "${RESULTS[@]}"; do
  echo "${r}" | awk -F'|' '{printf "%-12s | %-9s | %8s | %7s | %7s | %7s | %7s | %11s\n", $1, $2, $3, $4, $5, $6, $7, $8}'
done
echo "-------------|-----------|----------|---------|---------|---------|---------|-------------"
printf "%-12s | %-9s | %8s | %7s | %7s | %7s | %7s | %11s\n" "合计" "-" "${TOTAL_ELAPSED}" "${SUM_PASSED}" "${SUM_FAILED}" "${SUM_IGNORED}" "${SUM_SKIPPED}" "${SUM_DESELECTED}"
echo ""

# ============== 失败/错误汇总 ==============
if [ "${SUM_FAILED}" -gt 0 ] || [ "${TOTAL_FAIL}" -gt 0 ]; then
  sec "失败/错误详情"
  for r in "${RESULTS[@]}"; do
    _label=$(echo "$r" | awk -F'|' '{print $1}')
    _status=$(echo "$r" | awk -F'|' '{print $2}')
    _elapsed=$(echo "$r" | awk -F'|' '{print $3}')
    _passed=$(echo "$r" | awk -F'|' '{print $4}')
    _failed=$(echo "$r" | awk -F'|' '{print $5}')
    _ignored=$(echo "$r" | awk -F'|' '{print $6}')
    _skipped=$(echo "$r" | awk -F'|' '{print $7}')
    _deselected=$(echo "$r" | awk -F'|' '{print $8}')
    if [[ "${_status}" == FAIL* ]] || [ "${_failed}" -gt 0 ] 2>/dev/null; then
      echo "  ${_label}: ${_status}  passed=${_passed} failed=${_failed} elapsed=${_elapsed}s"
      echo "    日志: /tmp/oneclick-${_label}.log"
      # 提取具体 FAILED/ERROR 行（从 oneclick 日志或子脚本日志目录）
      _sublog="/tmp/oneclick-${_label}.log"
      if [ -f "${_sublog}" ]; then
        _fails=$(grep -E "^FAILED |ERROR at setup|ERROR collecting|tests/.* ERROR " "${_sublog}" 2>/dev/null | head -20)
        if [ -n "${_fails}" ]; then
          echo "    失败/错误用例:"
          echo "${_fails}" | sed 's/^/      /'
        fi
        # 子脚本日志目录
        _logdir=$(grep -E "日志目录:|LOG_DIR" "${_sublog}" 2>/dev/null | tail -1 | sed 's/.*: //')
        if [ -n "${_logdir}" ]; then
          echo "    完整日志目录: ${_logdir}"
        fi
      fi
      echo ""
    fi
  done
fi

if [ "${TOTAL_FAIL}" -eq 0 ] && [ "${SUM_FAILED}" -eq 0 ]; then
  ok "全部测试组通过（${TOTAL_PASS} 组，${SUM_PASSED} passed，0 failed，总耗时 ${TOTAL_ELAPSED}s）"
else
  fail "通过 ${TOTAL_PASS} 组，失败 ${TOTAL_FAIL} 组；用例 ${SUM_PASSED} passed / ${SUM_FAILED} failed（总耗时 ${TOTAL_ELAPSED}s）"
  echo "  各组日志: /tmp/oneclick-*.log"
fi
echo ""
echo "完成时间: $(date)"
