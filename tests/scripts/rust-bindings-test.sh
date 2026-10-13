#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# The Python bindings are a separate Cargo workspace. Extension modules do not
# link libpython, but a Rust test executable must do so on Linux.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/python-env.sh"
TOOLCHAIN="${RUST_TOOLCHAIN:-$(sed -n 's/^channel = "\(.*\)"/\1/p' "$SRC_ROOT/rust-toolchain.toml")}"
mapfile -t python_link < <("$PY" - <<'PY_CONFIG'
import sysconfig
from pathlib import Path
libdir = sysconfig.get_config_var("LIBDIR")
library = sysconfig.get_config_var("LDLIBRARY")
if not libdir or not library or not (Path(libdir) / library).exists():
    raise SystemExit("Python development library is required for bindings tests")
name = library.removeprefix("lib").split(".so")[0].split(".dylib")[0]
print(libdir)
print(name)
PY_CONFIG
)
if [[ ${#python_link[@]} != 2 ]]; then
    echo "Unable to determine Python development library" >&2
    exit 1
fi
export PYO3_PYTHON="$PY"
# CLI config arrays extend the repository target flags (including tokio_unstable).
# RUSTFLAGS would replace those flags, so keep the shared runner's clean policy.
unset RUSTFLAGS CARGO_ENCODED_RUSTFLAGS
host=$(rustc "+$TOOLCHAIN" -vV | sed -n 's/^host: //p')
link_config=$("$PY" -c 'import json,sys; print("target." + sys.argv[1] + ".rustflags=" + json.dumps(["-Lnative="+sys.argv[2], "-Clink-arg=-l"+sys.argv[3]]))' "$host" "${python_link[0]}" "${python_link[1]}")
export LD_LIBRARY_PATH="${python_link[0]}:${LD_LIBRARY_PATH:-}"
cd "$SRC_ROOT"
exec cargo "+$TOOLCHAIN" --config "$link_config" test --locked --manifest-path "$SRC_ROOT/lib/bindings/python/Cargo.toml" --lib "$@"
