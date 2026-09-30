#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

: "${VLLM_OMNI_REF:?VLLM_OMNI_REF must be set}"

VLLM_OMNI_PROTECTED_PACKAGES_FILE="${VLLM_OMNI_PROTECTED_PACKAGES_FILE:-/tmp/vllm_omni_protected_packages.txt}"
RUNTIME_DEPENDENCY_HELPER="${RUNTIME_DEPENDENCY_HELPER:-/tmp/runtime_dependency_requirements.py}"
BUILD_INFO=/opt/dynamo/build-info
PROTECTED_CONSTRAINTS="${BUILD_INFO}/protected-before-omni.txt"
RUNTIME_REQUIREMENTS="${BUILD_INFO}/dynamo-vllm-default-requirements.txt"
VLLM_OMNI_VERSION="${VLLM_OMNI_REF#v}"

python3 "${RUNTIME_DEPENDENCY_HELPER}" before \
  --directory "${BUILD_INFO}" \
  --protected-packages "${VLLM_OMNI_PROTECTED_PACKAGES_FILE}" \
  --omni-version "${VLLM_OMNI_VERSION}"

export VLLM_OMNI_TARGET_DEVICE

# Keep the released package bytes reproducible as well as the version label.
# Upstream v0.30.0 source tag: a8576ccb725c4e21cd13c3eb5f9a546b21149d2b.
OMNI_REQUIREMENT="vllm-omni==${VLLM_OMNI_VERSION}"
if [ "${VLLM_OMNI_VERSION}" = "0.30.0" ]; then
  OMNI_REQUIREMENT='https://files.pythonhosted.org/packages/b0/42/6068bdd37584af1d96156c42f44f8da340f5184abb9b0e8cdd0799b62766/vllm_omni-0.30.0-py3-none-any.whl#sha256=141cb0c7b9c07970e5c92a99aab9696e81682dbfa3d2bfcde359c3c1533385f4'
fi

# Jointly resolve the complete Omni default dependency graph and actual installed
# Dynamo/vLLM default requirements. This fills dependencies skipped by the local
# Dynamo wheel's --no-deps install without replacing the compiled GPU stack.
# Keep inherited UV_OVERRIDE intact: the official base deliberately overrides
# Torch's NCCL metadata pin for its DeepEP support. Constraints freeze that NCCL.
# Use --system flag only for CUDA (system Python), omit for CPU/XPU (venv).
if [ "${VLLM_OMNI_TARGET_DEVICE}" = "cuda" ]; then
  uv pip install --system \
    --prerelease=allow \
    --constraints "${PROTECTED_CONSTRAINTS}" \
    --requirements "${RUNTIME_REQUIREMENTS}" \
    "${OMNI_REQUIREMENT}"
else
  uv pip install \
    --prerelease=allow \
    --constraints "${PROTECTED_CONSTRAINTS}" \
    --requirements "${RUNTIME_REQUIREMENTS}" \
    "${OMNI_REQUIREMENT}"
fi

# Record path-level evidence and verify that no protected version drifted.
python3 "${RUNTIME_DEPENDENCY_HELPER}" after --directory "${BUILD_INFO}"
