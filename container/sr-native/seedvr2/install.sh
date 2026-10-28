#!/usr/bin/env bash
set -euo pipefail
root=/opt/sr-native/seedvr2
# This is an independent venv: never import the Omni 2.13 native stack.
python -m pip install --require-hashes --only-binary=:all: --no-deps -r "$root/bootstrap.lock"
python -m pip install --require-hashes --no-build-isolation \
  --extra-index-url https://download.pytorch.org/whl/cu121 -r "$root/requirements.lock"
python -c 'import sys,torch; assert sys.version_info[:2] == (3,10); assert torch.__version__ == "2.3.0+cu121"; assert torch.version.cuda == "12.1"'
# Build against the installed Torch ABI, not the upstream Torch2.4-specific wheel.
cd /opt/sources/apex
python -m pip install --no-build-isolation --no-deps --no-cache-dir \
  --config-settings="--build-option=--cpp_ext" \
  --config-settings="--build-option=--cuda_ext" .
cd /opt/sources/flash-attn
# The pinned PyPI sdist includes csrc/cutlass. Force compilation to suppress
# setup.py's attempt to fetch a guessed release wheel from GitHub.
FLASH_ATTENTION_FORCE_BUILD=TRUE python -m pip install --no-build-isolation --no-deps --no-cache-dir .
python -m pip check
