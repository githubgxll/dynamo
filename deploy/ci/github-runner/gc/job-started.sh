#!/usr/bin/env bash
set -euxo pipefail
exec python3 /etc/dingo-gc/manager.py started
