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
    # Omni 0.29.0rc1 requires transformers>=5.13,<5.15, while the pinned
    # upstream vLLM 0.29 image ships 5.16.1. Let Omni resolve this pure-Python
    # API stack and its paired tokenizer, retaining the compiled core pins.
    if sys.argv[2] == "0.29.0rc1" and name in {"transformers", "tokenizers"}:
        continue
    try:
        dist = md.distribution(name)
    except Exception:
        continue
    project_name = dist.metadata.get("Name") or name
    print(f"{project_name}=={dist.version}")
PY

# Match the verified H3 0.32 service's pure-Python frontend pair while retaining
# the compiled core versions solved by the pinned vLLM base image.
if [ "${VLLM_OMNI_VERSION}" = "0.32.0" ]; then
  python3 - "${PROTECTED_CONSTRAINTS}" <<'PY_CONSTRAINTS'
from pathlib import Path
import sys
p = Path(sys.argv[1])
lines = [line for line in p.read_text().splitlines() if line.split("==", 1)[0].lower() not in {"transformers", "tokenizers"}]
lines += ["transformers==5.16.1", "tokenizers==0.23.2"]
p.write_text("\n".join(lines) + "\n")
PY_CONSTRAINTS
fi

export VLLM_OMNI_TARGET_DEVICE

# Bytecode compilation can compete with the concurrent Rust wheel builds.
# Keep it enabled, but allow slower builders more than uv's default 60 seconds.
export UV_COMPILE_BYTECODE_TIMEOUT="${UV_COMPILE_BYTECODE_TIMEOUT:-300}"

# Use --system flag only for CUDA (system Python), omit for CPU/XPU (venv)
if [ "${VLLM_OMNI_TARGET_DEVICE}" = "cuda" ]; then
  uv pip install --system \
    --prerelease=allow \
    --constraints "${PROTECTED_CONSTRAINTS}" \
    "vllm-omni==${VLLM_OMNI_VERSION}"
else
  uv pip install \
    --prerelease=allow \
    --constraints "${PROTECTED_CONSTRAINTS}" \
    "vllm-omni==${VLLM_OMNI_VERSION}"
fi
