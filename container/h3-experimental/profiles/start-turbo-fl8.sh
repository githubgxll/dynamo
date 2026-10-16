#!/usr/bin/env bash
set -euo pipefail
exec python3 -B /opt/h3-experimental/profiles/launch.py --profile turbo-fl8 --start "$@"
