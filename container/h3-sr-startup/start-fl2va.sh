#!/bin/sh
set -eu
export PYTHONSAFEPATH=1
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
config_dir=${H3SR_CONFIG_DIR:-/etc/h3-sr}
python3 -B "$script_dir/configure.py" check --variant fl2va \
  --profile "$config_dir/profile.json" --gateway-command "$config_dir/gateway-command.json" \
  --mounted-dir "$config_dir" --runtime-identity
exec python3 -B /opt/h3-sr/service/launcher.py \
  --profile "$config_dir/profile.json" --gateway-command-file "$config_dir/gateway-command.json"
