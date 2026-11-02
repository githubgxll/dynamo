#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INFO="${1:?Provide the writable environment evidence directory}"
mkdir -p "$INFO"
if [[ -f /run/secrets/pip-index-url ]]; then
    export PIP_INDEX_URL="$(cat /run/secrets/pip-index-url)"
fi
if [[ -f /run/secrets/uv-default-index ]]; then
    export UV_DEFAULT_INDEX="$(cat /run/secrets/uv-default-index)"
fi
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=500
export UV_HTTP_RETRIES=4
export UV_NO_PROGRESS=1
export UV_OVERRIDE="$INFO/reviewed-overrides.txt"
if [[ ! -f "$INFO/parent-inventory.json" ]]; then
    python3 -B "$HERE/environment.py" prepare --output-dir "$INFO"
fi
python3 -B "$HERE/environment.py" download --output-dir "$INFO"
# No --upgrade: retain compatible installed packages. Every inherited effective
# pip capability is a solve root, with all current default dependency edges and
# recursive extras; only reviewed native/framework changes are unlocked.
uv pip install --system --python "$(python3 -c 'import sys; print(sys.executable)')" \
    --constraint "$INFO/constraints.txt" --requirement "$INFO/roots.in" \
    --requirement "$HERE/requirements.lock" \
    --reinstall-package vllm --reinstall-package vllm-omni \
    "$INFO/wheels/vllm-0.32.0-cp38-abi3-manylinux_2_28_x86_64.whl" \
    "$INFO/wheels/vllm_omni-0.32.0-py3-none-any.whl"
python3 -B "$HERE/environment.py" reconcile --output-dir "$INFO" --inputs "$INFO"
# Both distribution names are required upstream. Headless owns the real cv2
# namespace; validate its complete hashed RECORD in the post-install audit.
uv pip install --system --python "$(python3 -c 'import sys; print(sys.executable)')" \
    --no-deps --only-binary :all: --reinstall-package opencv-python-headless \
    opencv-python-headless==4.13.0.92
python3 -B "$HERE/environment.py" audit --output-dir "$INFO" --inputs "$INFO"
chmod -R a+rX "$INFO"
