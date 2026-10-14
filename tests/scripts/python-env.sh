#!/usr/bin/env bash
# Source after SRC_ROOT is set. Explicit PY supports prebuilt backend images.
export TEST_VENV_DIR="${TEST_VENV_DIR:-${SRC_ROOT}/../dingo-test-venv}"
export PY="${PY:-${TEST_VENV_DIR}/bin/python}"

# Subprocess-based tests must inherit the selected virtual environment.
export PATH="$(dirname "$PY"):$PATH"
