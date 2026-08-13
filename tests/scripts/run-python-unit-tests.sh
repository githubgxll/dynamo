#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DingoRouter Python 单元测试串行执行脚本（命令 1-3）
set -uo pipefail

export PATH="${HOME}/.cargo/bin:${HOME}/.local/bin:${PATH}"
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

PY="/usr/bin/python3.11"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SRC_ROOT}"

STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="/tmp/dingoRouter-py-unit-tests-${STAMP}"
mkdir -p "${LOGDIR}"

run() {
  local idx="$1"; shift
  local LOG="${LOGDIR}/${idx}.log"
  local start=$(date +%s)
  echo ">>> [${idx}]"
  $PY -m pytest "$@" > "${LOG}" 2>&1
  local code=$?
  local end=$(date +%s)
  local elapsed=$((end - start))
  local result_line=$(grep -E "={2,}.*passed.*in [0-9]" "${LOG}" | tail -1)
  [ -z "${result_line}" ] && result_line=$(grep -E "passed|failed" "${LOG}" | tail -1)
  echo "    ${result_line:-(no result line)}  | exit=${code} elapsed=${elapsed}s"
  echo "${idx} | ${code} | ${elapsed} | ${result_line}" >> "${LOGDIR}/SUMMARY.txt"
}

# 1. vllm_unit
run "01_vllm_unit" dingo/vllm/tests/ \
  -v --tb=long --timeout=60 --continue-on-collection-errors \
  --deselect dingo/vllm/tests/test_vllm_unit.py::test_custom_jinja_template_invalid_path \
  --deselect dingo/vllm/tests/test_vllm_unit.py::test_custom_jinja_template_valid_path \
  --deselect dingo/vllm/tests/test_vllm_unit.py::test_custom_jinja_template_env_var_expansion \
  --deselect dingo/vllm/tests/test_vllm_unit.py::test_endpoint_overrides_defaults \
  --deselect dingo/vllm/tests/test_vllm_unit.py::test_endpoint_not_provided_preserves_defaults \
  --deselect dingo/vllm/tests/test_vllm_unit.py::test_endpoint_overrides_with_prefill_worker \
  --deselect dingo/vllm/tests/test_vllm_unit.py::test_endpoint_invalid_format_raises \
  --deselect dingo/vllm/tests/test_vllm_unit.py::test_connector_nixl_raises_error_with_migration_hint \
  --deselect dingo/vllm/tests/test_vllm_unit.py::test_connector_none_raises_error \
  --deselect dingo/vllm/tests/test_vllm_unit.py::test_env_var_dyn_connector_raises_error \
  --deselect dingo/vllm/tests/test_vllm_unit.py::test_model_express_url_is_accepted_for_compatibility \
  --deselect dingo/vllm/tests/test_vllm_unit.py::test_model_express_url_env_is_accepted_for_compatibility \
  --deselect dingo/vllm/tests/test_vllm_unit.py::test_prefill_worker_without_kv_transfer_config_raises \
  --deselect dingo/vllm/tests/test_vllm_unit.py::test_disaggregation_mode_default \
  --deselect dingo/vllm/tests/test_vllm_unit.py::test_kv_events_disabled_by_default_without_explicit_config \
  --deselect dingo/vllm/tests/test_vllm_unit.py::test_disaggregation_mode_prefill \
  --deselect dingo/vllm/tests/test_vllm_unit.py::test_disaggregation_mode_decode \
  --deselect dingo/vllm/tests/test_vllm_unit.py::test_legacy_is_prefill_worker_emits_deprecation \
  --deselect dingo/vllm/tests/test_vllm_unit.py::test_legacy_is_decode_worker_emits_deprecation \
  --deselect dingo/vllm/tests/test_vllm_unit.py::test_conflicting_legacy_and_new_flags_raises \
  --deselect dingo/vllm/tests/test_vllm_unit.py::test_explicit_default_mode_with_legacy_flag_raises \
  --deselect dingo/vllm/tests/test_vllm_unit.py::test_parse_args_does_not_track_logprobs_mode_presence \
  --deselect dingo/vllm/tests/test_vllm_unit.py::TestEmbeddingWorkerFlag::test_default_false \
  --deselect dingo/vllm/tests/test_vllm_unit.py::TestEmbeddingWorkerFlag::test_flag_sets_true \
  --deselect dingo/vllm/tests/test_vllm_unit.py::TestEmbeddingWorkerFlag::test_rejects_prefill_disagg \
  --deselect dingo/vllm/tests/test_vllm_unit.py::TestEmbeddingWorkerFlag::test_rejects_decode_disagg \
  --deselect dingo/vllm/tests/test_vllm_unit.py::TestEmbeddingWorkerFlag::test_rejects_multimodal_combo \
  --deselect dingo/vllm/tests/test_vllm_unit.py::test_headless_namespace_has_required_fields \
  --deselect dingo/vllm/tests/test_vllm_engine.py::test_vllm_engine_all

# 2. frontend_unit
# 注：以下 6 个文件 import sglang/vllm 引擎模块，纯 CPU 节点未装这两个重型 GPU 包，
# 用 --ignore 跳过（等价于 vllm_unit/sglang_unit 对引擎依赖的处理）。
run "02_frontend_unit" dingo/frontend/tests/ \
  --ignore=dingo/frontend/tests/test_sglang_multimodal_prepost.py \
  --ignore=dingo/frontend/tests/test_sglang_processor_api.py \
  --ignore=dingo/frontend/tests/test_sglang_processor_metrics_unit.py \
  --ignore=dingo/frontend/tests/test_sglang_processor_unit.py \
  --ignore=dingo/frontend/tests/test_sglang_tool_calls.py \
  --ignore=dingo/frontend/tests/test_vllm_processor_unit.py \
  -v --tb=long --timeout=60 --continue-on-collection-errors

# 3. sglang_unit
# 注：test_fpm_contract.py 的 2 个用例直接 import sglang.srt.* 引擎模块，
# 本节点无 GPU/未装 sglang 引擎（sglang==0.5.17 需完整 GPU 栈），
# 在此处 deselect，等价于 vllm_unit deselect test_vllm_engine_all。
run "03_sglang_unit" dingo/sglang/tests/ \
  -v --tb=short --timeout=120 --continue-on-collection-errors \
  --deselect dingo/sglang/tests/test_sglang_unit.py::test_custom_jinja_template_valid_path \
  --deselect dingo/sglang/tests/test_sglang_unit.py::test_custom_jinja_template_env_var_expansion \
  --deselect dingo/sglang/tests/test_sglang_unit.py::test_tool_call_parser_valid_with_dynamo_tokenizer \
  --deselect dingo/sglang/tests/test_sglang_unit.py::test_namespace_flag_drives_default_endpoint_namespace \
  --deselect dingo/sglang/tests/test_sglang_unit.py::test_forward_pass_metrics_enabled_from_env \
  --deselect dingo/sglang/tests/test_sglang_unit.py::test_obsolete_dyn_endpoint_types_flag_is_supported \
  --deselect dingo/sglang/tests/test_sglang_unit.py::test_disagg_config_preserves_bootstrap_port \
  --deselect dingo/sglang/tests/test_sglang_engine.py::test_sglang_engine_all \
  --deselect dingo/sglang/tests/test_fpm_contract.py::test_sglang_fpm_decodes_with_dynamo_schema \
  --deselect dingo/sglang/tests/test_fpm_contract.py::test_sglang_fpm_field_order_matches_dynamo

echo
echo "================ 汇总 ================"
cat "${LOGDIR}/SUMMARY.txt" 2>/dev/null
echo "日志目录: ${LOGDIR}"

# 统一标签行（供 oneclick 脚本解析）：从 SUMMARY 第4列 result_line 提取
SUMMARY="${LOGDIR}/SUMMARY.txt"
_gt_p=0; _gt_f=0; _gt_s=0; _gt_d=0
while IFS='|' read -r _idx _code _el rest; do
  _p=$(echo "$rest" | sed -n 's/.* \([0-9][0-9]*\) passed.*/\1/p'); [ -n "$_p" ] || _p=0
  _f=$(echo "$rest" | sed -n 's/.* \([0-9][0-9]*\) failed.*/\1/p'); [ -n "$_f" ] || _f=0
  _s=$(echo "$rest" | sed -n 's/.* \([0-9][0-9]*\) skipped.*/\1/p'); [ -n "$_s" ] || _s=0
  _d=$(echo "$rest" | sed -n 's/.* \([0-9][0-9]*\) deselected.*/\1/p'); [ -n "$_d" ] || _d=0
  _gt_p=$((_gt_p + _p)); _gt_f=$((_gt_f + _f)); _gt_s=$((_gt_s + _s)); _gt_d=$((_gt_d + _d))
done < <(tail -n +1 "${SUMMARY}" 2>/dev/null)
echo "GROUP_TOTAL passed=${_gt_p} failed=${_gt_f} ignored=0 skipped=${_gt_s} deselected=${_gt_d}"
