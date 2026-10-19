#!/usr/bin/env bash
# H3 experimental overlay over the pinned gateway runtime; construction runs on Linux.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INFO="${H3_BUILD_INFO:-/opt/dynamo/build-info}"
mkdir -p "${INFO}"
if [ -f /run/secrets/pip-index-url ]; then
    export PIP_INDEX_URL="$(cat /run/secrets/pip-index-url)"
fi
if [ -f /run/secrets/uv-default-index ]; then
    export UV_DEFAULT_INDEX="$(cat /run/secrets/uv-default-index)"
fi

python3 "${SCRIPT_DIR}/audit_environment.py" --prepare-only
python3 "${SCRIPT_DIR}/runtime_dependencies.py" before \
    --directory "${INFO}" --protected-packages "${SCRIPT_DIR}/protected_packages.txt" \
    --omni-version 0.31.0rc1

# The inherited file is validated before generating this exact single NCCL override.
# A constraints file alone cannot express an upstream intentional metadata override.
export UV_OVERRIDE="${INFO}/h3-reviewed-overrides.txt"
export UV_LINK_MODE=copy
uv --version > "${INFO}/uv-version.txt"
OMNI_REQUIREMENT="$(python3 "${SCRIPT_DIR}/prepare_omni_wheel.py" \
    --policy "${SCRIPT_DIR}/gateway-dependency-policy.json" --directory "${INFO}")"
uv pip install --system --prerelease=allow \
    --constraints "${INFO}/protected-before-omni.txt" \
    --constraints "${INFO}/h3-shared-constraints.txt" \
    --requirements "${INFO}/dynamo-vllm-default-requirements.txt" \
    --requirements "${INFO}/inherited-telemetry-requirements.txt" \
    --requirements "${INFO}/h3-backend-helpers.txt" \
    --reinstall-package vllm-omni \
    "${OMNI_REQUIREMENT}" \
    av==16.1.0 soxr==1.1.0 torchcodec==0.14.0 \
    opencv-python==4.13.0.92 opencv-python-headless==4.13.0.92

# Two upstream default requirement graphs request different distribution names
# for cv2. Keep both metadata, choose one actual provider, and verify its RECORD.
uv pip install --system --no-deps --reinstall-package opencv-python-headless \
    --only-binary :all: opencv-python-headless==4.13.0.92

python3 "${SCRIPT_DIR}/runtime_dependencies.py" after --directory "${INFO}"
python3 "${SCRIPT_DIR}/audit_environment.py"
python3 "${SCRIPT_DIR}/media_smoke.py"
python3 "${SCRIPT_DIR}/service_smoke.py"
# Final non-root audit after assets/patches uses fresh evidence in a writable path.
chmod -R a+rX "${INFO}"
