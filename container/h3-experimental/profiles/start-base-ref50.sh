#!/usr/bin/env bash
set -euo pipefail
exec python3 -B /opt/h3-experimental/profiles/launch.py --profile base-ref50 --start "$@"
