#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

: "${VLLM_OMNI_REF:?VLLM_OMNI_REF must be set}"

VLLM_OMNI_PROTECTED_PACKAGES_FILE="${VLLM_OMNI_PROTECTED_PACKAGES_FILE:-/tmp/vllm_omni_protected_packages.txt}"

PROTECTED_CONSTRAINTS="$(mktemp /tmp/vllm-openai-protected.XXXXXX.txt)"
VLLM_OMNI_VERSION="${VLLM_OMNI_REF#v}"

cleanup() {
  rm -rf "${PROTECTED_CONSTRAINTS}"
}

trap cleanup EXIT

python3 - "${VLLM_OMNI_PROTECTED_PACKAGES_FILE}" "${VLLM_OMNI_VERSION}" <<'PY' > "${PROTECTED_CONSTRAINTS}"
import importlib.metadata as md
from pathlib import Path
import sys

for raw_line in Path(sys.argv[1]).read_text().splitlines():
    name = raw_line.strip()
    if not name or name.startswith("#"):
        continue
    # Omni 0.29.0rc1 and 0.30.0 require transformers>=5.13,<5.15.
    # Permit their API stack and paired tokenizer to resolve while retaining
    # the compiled core pins. Do not generalize this to unreviewed releases.
    if sys.argv[2] in {"0.29.0rc1", "0.30.0"} and name in {"transformers", "tokenizers"}:
        continue
    try:
        dist = md.distribution(name)
    except Exception:
        continue
    project_name = dist.metadata.get("Name") or name
    print(f"{project_name}=={dist.version}")
PY

export VLLM_OMNI_TARGET_DEVICE

# Keep the released package bytes reproducible as well as the version label.
# Upstream v0.30.0 source tag: a8576ccb725c4e21cd13c3eb5f9a546b21149d2b.
OMNI_REQUIREMENT="vllm-omni==${VLLM_OMNI_VERSION}"
if [ "${VLLM_OMNI_VERSION}" = "0.30.0" ]; then
  OMNI_REQUIREMENT='https://files.pythonhosted.org/packages/b0/42/6068bdd37584af1d96156c42f44f8da340f5184abb9b0e8cdd0799b62766/vllm_omni-0.30.0-py3-none-any.whl#sha256=141cb0c7b9c07970e5c92a99aab9696e81682dbfa3d2bfcde359c3c1533385f4'
fi

mkdir -p /opt/dynamo/build-info
cp "${PROTECTED_CONSTRAINTS}" /opt/dynamo/build-info/protected-before-omni.txt

# Use --system flag only for CUDA (system Python), omit for CPU/XPU (venv)
if [ "${VLLM_OMNI_TARGET_DEVICE}" = "cuda" ]; then
  uv pip install --system \
    --prerelease=allow \
    --constraints "${PROTECTED_CONSTRAINTS}" \
    "${OMNI_REQUIREMENT}"
else
  uv pip install \
    --prerelease=allow \
    --constraints "${PROTECTED_CONSTRAINTS}" \
    "${OMNI_REQUIREMENT}"
fi

# Verify the protected runtime solve survived dependency installation.
python3 - "${PROTECTED_CONSTRAINTS}" <<'PY'
import importlib.metadata as md
from pathlib import Path
import sys

for line in Path(sys.argv[1]).read_text().splitlines():
    name, expected = line.split("==", 1)
    actual = md.version(name)
    if actual != expected:
        raise RuntimeError(f"Protected package changed: {name}: {expected} -> {actual}")
PY
