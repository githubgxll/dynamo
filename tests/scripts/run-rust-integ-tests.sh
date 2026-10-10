#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -uo pipefail
TEST_GROUP=rust-integ
TEST_KIND=rust
source "$(dirname "${BASH_SOURCE[0]}")/test-common.sh"
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

# CPU-capable protocol regressions are required after an upstream merge.
for target in anthropic_http_replay responses_http_replay frontend_protocol_validation_http test_stop_behavior test_reasoning_parser test_streaming_tool_parsers parallel_tool_call_integration test_streaming_usage; do
 TESTS+=("dynamo-llm-${target}|-p dynamo-llm --test ${target}")
done

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
