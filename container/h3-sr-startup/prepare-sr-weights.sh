#!/bin/sh
set -eu
export PYTHONSAFEPATH=1
export CUDA_VISIBLE_DEVICES=""
source_dir=${SR_SOURCE_DIR:-/models/SeedVR2-3B}
output_dir=${SR_OUTPUT_DIR:-/opt/sr/seedvr2-3b}
exec python3 -B /opt/h3-weight-init/convert_seedvr2.py \
  --source-dir "$source_dir" --output-dir "$output_dir" "$@"
