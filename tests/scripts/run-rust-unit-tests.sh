#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -uo pipefail
TEST_GROUP=rust-unit
TEST_KIND=rust
source "$(dirname "${BASH_SOURCE[0]}")/test-common.sh"
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
# 未设置时跑全部 当前清单中的包
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


if [ -z "${QUICK_PACKS:-}" ]; then

# Newly added production crates; example policy crates are not included here.
TESTS+=(
 "24-truthy|-p dynamo-truthy --lib"
 "25-router-policies|-p dynamo-custom-policy-builtin --lib"
 "26-sidecar-common|-p dynamo-sidecar-common --lib"
 "27-sglang-sidecar|-p dynamo-sglang-sidecar --lib"
 "28-vllm-sidecar|-p dynamo-vllm-sidecar --lib"
 "29-trtllm-sidecar|-p dynamo-trtllm-sidecar --lib"
)
fi

for entry in "${TESTS[@]}"; do
    name="${entry%%|*}"
    read -r -a args <<< "${entry#*|}"
    # Each integration target gets its own log, so stale results cannot be counted again.
    for ((i=0; i<${#args[@]}; i++)); do
        if [[ "${args[i]}" == --test ]]; then name="${name}-${args[i+1]}"; fi
    done
    run "$name" cargo "+$TOOLCHAIN" test --locked "${args[@]}"
done
finish
exit $?
