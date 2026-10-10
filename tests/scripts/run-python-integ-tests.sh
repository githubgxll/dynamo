#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -uo pipefail
TEST_GROUP=py-integ
TEST_KIND=python
source "$(dirname "${BASH_SOURCE[0]}")/test-common.sh"
export HF_HUB_OFFLINE=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
extra_args=()
if [[ ! -d "${DYNAMO_WHEEL_SMOKE_WHEELHOUSE:-/opt/dynamo/wheelhouse}" ]]; then
    extra_args+=(--ignore=tests/wheels)
    omit wheel-artifact-tests "Runtime/KVBM distribution wheelhouse is unavailable; requires image build artifacts"
fi
if ! "$PY" -c 'import vllm' >/dev/null 2>&1; then
    extra_args+=(--ignore=tests/video_gateway)
    vllm_python="${VLLM_TEST_PY:-$PY}"
    if "$vllm_python" -c 'import vllm' >/dev/null 2>&1; then
        run_with_python "$vllm_python" video-gateway tests/video_gateway -m gpu_0 -v --tb=short --timeout=120 --continue-on-collection-errors
    else
        omit video-gateway "Requires vLLM protocol classes; set VLLM_TEST_PY"
    fi
fi
# 4. 集成测试主组
# GPU、集群和发布 wheel 验收依赖额外环境；CPU 测试失败直接计入结果。
run "04_integration_main" tests/ "${extra_args[@]}" -m gpu_0 \
  --ignore=tests/frontend/grpc/test_triton_identity.py \
  --ignore=tests/frontend/test_prompt_embeds.py \
  --ignore=tests/frontend/test_tool_calling_sglang.py \
  --ignore=tests/serve \
  --ignore=tests/utils/test_mock_gpu_alloc.py \
  --ignore=tests/fault_tolerance \
  --ignore=tests/kvbm_integration \
  --ignore=tests/router \
  --ignore=tests/frontend/test_vllm.py \
  --ignore=tests/test_predownload_models.py \
  --ignore=tests/mm_router/test_router_rust_mm_router_e2e.py \
  --ignore=tests/mm_router/test_vllm_mm_router_e2e.py \
  --ignore=tests/mm_router/test_router_rust_mm_frontend_decode_e2e.py \
  --ignore=tests/vllm_self_benchmark/test_self_benchmark_gpu.py \
  --ignore=tests/deploy \
  --ignore=tests/test_models_dir_flag.py \
  --deselect tests/frontend/test_frontend_api_surface_compliance.py::test_frontend_api_surface_compliance \
  --deselect tests/rl/test_worker_discovery.py::test_rl_worker_discovery_and_engine_admin_routes \
  --deselect tests/dependencies/test_kvbm_imports.py::test_kvbm_wheel_exists \
  --deselect tests/dependencies/test_kvbm_imports.py::test_kvbm_imports \
  -v --tb=short --timeout=300 \
  --models-dir=/root/.cache/huggingface \
  --continue-on-collection-errors

# 5. test_standalone_slot_tracker.py
# Requires a runtime built with --features slot-tracker.
# The current runtime is built with slot-tracker; all three cases run.
run "05_slot_tracker_standalone" tests/router/test_standalone_slot_tracker.py -v --tb=short --timeout=300

# 6. test_slot_tracker_e2e.py
# 依赖 Qwen3-0.6B 模型 tokenizer；默认不下载模型，DOWNLOAD_MODEL=1 时才跑。
if [ "${RUN_MODEL_TESTS:-${DOWNLOAD_MODEL:-0}}" != "1" ]; then
  omit "06_slot_tracker_e2e" "Set RUN_MODEL_TESTS=1 with a prepared model cache, or DOWNLOAD_MODEL=1 during setup"
else
  run "06_slot_tracker_e2e" tests/router/test_slot_tracker_e2e.py \
    -v --tb=short --timeout=300 --models-dir=/root/.cache/huggingface
fi

# 7. test_mocker_output_replay_e2e.py
# 依赖 Qwen3-0.6B 模型 tokenizer；默认不下载模型，DOWNLOAD_MODEL=1 时才跑。
if [ "${RUN_MODEL_TESTS:-${DOWNLOAD_MODEL:-0}}" != "1" ]; then
  omit "07_mocker_output_replay" "Set RUN_MODEL_TESTS=1 with a prepared model cache, or DOWNLOAD_MODEL=1 during setup"
else
  run "07_mocker_output_replay" tests/router/test_mocker_output_replay_e2e.py \
    -v --tb=short --timeout=300 --models-dir=/root/.cache/huggingface
fi

# 8. test_router_e2e_with_mockers.py
# 依赖 Qwen3-0.6B 模型 tokenizer（counter_worker.py）；默认不下载模型，DOWNLOAD_MODEL=1 时才跑。
# Run every selected router case; report failures instead of deselecting known scenarios.
if [ "${RUN_MODEL_TESTS:-${DOWNLOAD_MODEL:-0}}" != "1" ]; then
  omit "08_router_e2e_mockers" "Set RUN_MODEL_TESTS=1 with a prepared model cache, or DOWNLOAD_MODEL=1 during setup"
else
  run "08_router_e2e_mockers" tests/router/test_router_e2e_with_mockers.py \
    -v --tb=short --timeout=300 \
    --models-dir=/root/.cache/huggingface
fi

if [[ "${RUN_MODEL_TESTS:-${DOWNLOAD_MODEL:-0}}" == "1" ]]; then
  run kv-dc-relay-diagnostics tests/router/test_kv_dc_relay_e2e.py -m gpu_0 -v --tb=short --timeout=300 --models-dir=/root/.cache/huggingface
else
  omit kv-dc-relay-diagnostics "Requires a prepared tokenizer cache and a ckf-diagnostics runtime; set RUN_MODEL_TESTS=1"
fi

run router-cpu-utils tests/router/test_policy_class.py tests/router/test_mocker_config.py tests/router/test_kv_router_gil_release.py -m gpu_0 -v --tb=short --timeout=300 --models-dir=/root/.cache/huggingface
run fault-tolerance-cpu-utils tests/fault_tolerance/test_legacy_parse_results.py tests/fault_tolerance/test_worker_names.py -m gpu_0 -v --tb=short --timeout=120

omit additional-integration-coverage "GPU/cluster/fault-recovery acceptance, Triton server and optional diagnostic-feature scenarios still require their runtime infrastructure"
finish
exit $?
