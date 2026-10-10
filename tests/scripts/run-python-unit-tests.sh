#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -uo pipefail
TEST_GROUP=py-unit
TEST_KIND=python
source "$(dirname "${BASH_SOURCE[0]}")/test-common.sh"
# Do not deselect previously failing assertions. Backend suites require their own
# compatible engine environment (the vLLM and SGLang extras conflict).
declare -A backend_pys=()
for backend in vllm sglang; do
    interpreter_var="${backend^^}_TEST_PY"
    candidate="${!interpreter_var:-$PY}"
    if "$candidate" -c "import $backend" > "$LOGDIR/$backend-preflight.log" 2>&1; then
        backend_pys[$backend]="$candidate"
        run_with_python "$candidate" "${backend}-unit" "dingo/$backend/tests" -m gpu_0 -v --tb=short --timeout=120 --continue-on-collection-errors
    else
        omit "${backend}-unit" "$backend is unavailable in $candidate; set $interpreter_var to its test environment"
    fi
done
frontend_args=()
if ! "$PY" -c 'import sglang' >/dev/null 2>&1; then
    sglang_frontend=()
    for f in test_sglang_multimodal_prepost.py test_sglang_processor_api.py test_sglang_processor_metrics_unit.py test_sglang_processor_unit.py test_sglang_tool_calls.py; do
        frontend_args+=("--ignore=dingo/frontend/tests/$f")
        sglang_frontend+=("dingo/frontend/tests/$f")
        if [[ -z "${backend_pys[sglang]:-}" ]]; then
            omit "frontend-${f%.py}" "Requires SGLang backend environment"
        fi
    done
    if [[ -n "${backend_pys[sglang]:-}" ]]; then
        run_with_python "${backend_pys[sglang]}" frontend-sglang "${sglang_frontend[@]}" -m gpu_0 -v --tb=short --timeout=120 --continue-on-collection-errors
    fi
fi
if ! "$PY" -c 'import vllm' >/dev/null 2>&1; then
    frontend_args+=(--ignore=dingo/frontend/tests/test_vllm_processor_unit.py)
    if [[ -n "${backend_pys[vllm]:-}" ]]; then
        run_with_python "${backend_pys[vllm]}" frontend-vllm-processor dingo/frontend/tests/test_vllm_processor_unit.py -m gpu_0 -v --tb=short --timeout=120
    else
        omit frontend-vllm-processor "Requires vLLM backend environment"
    fi
fi
run frontend-unit dingo/frontend/tests "${frontend_args[@]}" -v --tb=short --timeout=120 --continue-on-collection-errors
for suite in common planner profiler global_router global_planner router replay mocker; do
    suite_args=()
    if [[ "$suite" == common ]] && ! "$PY" -c 'import vllm' >/dev/null 2>&1; then
        suite_args+=(--ignore=dingo/common/tests/multimodal/test_mm_kwargs_transfer.py)
        if [[ -n "${backend_pys[vllm]:-}" ]]; then
            run_with_python "${backend_pys[vllm]}" common-vllm-multimodal dingo/common/tests/multimodal/test_mm_kwargs_transfer.py -m gpu_0 -v --tb=short --timeout=120
        else
            omit common-vllm-multimodal "Requires vLLM MultiModalKwargsItem; set VLLM_TEST_PY"
        fi
    fi
    run "$suite-unit" "dingo/$suite" "${suite_args[@]}" -m gpu_0 -v --tb=short --timeout=120 --continue-on-collection-errors
done
# NIXL import mocks get their own process; the remaining bindings still run
# together to retain cross-test state coverage. Record both results on failure.
run python-bindings lib/bindings/python/tests \
    --ignore=lib/bindings/python/tests/test_nixl_connect_lazy_import.py \
    --ignore=lib/bindings/python/tests/test_nixl_connect_unit.py \
    -m gpu_0 -v --tb=short --timeout=120 --continue-on-collection-errors
run python-bindings-nixl \
    lib/bindings/python/tests/test_nixl_connect_lazy_import.py \
    lib/bindings/python/tests/test_nixl_connect_unit.py \
    -v --tb=short --timeout=120
finish
exit $?
