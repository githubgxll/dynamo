#!/usr/bin/env bash
# Shared test environment. Source from a tests/scripts runner.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PATH="${HOME}/.cargo/bin:${HOME}/.local/bin:${PATH}"
source "$SCRIPT_DIR/python-env.sh"
TOOLCHAIN="${RUST_TOOLCHAIN:-$(sed -n 's/^channel = "\(.*\)"/\1/p' "${SRC_ROOT}/rust-toolchain.toml")}"
export RUSTUP_TOOLCHAIN="$TOOLCHAIN"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
# Avoid hundreds of GB of debug symbols/incremental caches during broad test runs.
export CARGO_PROFILE_DEV_DEBUG="${CARGO_PROFILE_DEV_DEBUG:-0}"
export CARGO_PROFILE_TEST_DEBUG="${CARGO_PROFILE_TEST_DEBUG:-0}"
export CARGO_INCREMENTAL="${CARGO_INCREMENTAL:-0}"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="$NO_PROXY"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
if [ -x /usr/local/gcc-12/bin/gcc ]; then
    export CC=/usr/local/gcc-12/bin/gcc CXX=/usr/local/gcc-12/bin/g++
    export LD_LIBRARY_PATH="/usr/local/gcc-12/lib64:${LD_LIBRARY_PATH:-}"
    export LIBRARY_PATH="/usr/local/gcc-12/lib64:${LIBRARY_PATH:-}"
fi
unset RUSTFLAGS
cd "$SRC_ROOT" || exit 1
STAMP="$(date +%Y%m%d_%H%M%S)-$$"
LOGDIR="${TEST_RESULTS_DIR:-/tmp}/dingoRouter-${TEST_GROUP}-tests-${STAMP}"
mkdir -p "$LOGDIR" || exit 1
: > "$LOGDIR/commands.jsonl"
echo "Source: $SRC_ROOT; commit: $(git rev-parse HEAD); toolchain: $TOOLCHAIN; python: $PY"
echo "日志目录: $LOGDIR"
record() {
    "$PY" - "$LOGDIR/commands.jsonl" "$@" <<'END_RECORD'
import json, sys
path,name,code,elapsed,log,xml,reason=sys.argv[1:]
row=dict(name=name,exit_code=int(code),elapsed=int(elapsed),log=log,xml=xml)
if reason: row.update(omitted=True,reason=reason)
with open(path,'a') as f: f.write(json.dumps(row)+'\n')
END_RECORD
}
run() {
    local name="$1"; shift
    local log="$LOGDIR/$name.log" xml="" start=$SECONDS code
    local -a command=("$@")
    if [[ "$TEST_KIND" == python ]]; then
        xml="$LOGDIR/$name.xml"
        command=("$PY" -m pytest "$@" "--junitxml=$xml")
    fi
    printf '>>> [%s]' "$name"; printf ' %q' "${command[@]}"; echo
    timeout --signal=TERM --kill-after=30 "${TEST_COMMAND_TIMEOUT:-1800}" "${command[@]}" > "$log" 2>&1
    code=$?
    record "$name" "$code" "$((SECONDS-start))" "$log" "$xml" "" || exit 1
    echo "<<< [$name] exit=$code elapsed=$((SECONDS-start))s; $log"
}
# Backend stacks require separate interpreters; propagate the same Python to
# subprocess tests and result recording without changing the caller's environment.
run_with_python() {
    local PY="$1"; shift
    local PATH="$(dirname "$PY"):$PATH"
    export PY PATH
    run "$@"
}
omit() {
    local name="$1" reason="$2"
    printf '%s\n' "$reason" > "$LOGDIR/$name.log"
    record "$name" 0 0 "$LOGDIR/$name.log" "" "$reason" || exit 1
    echo "NOT_RUN [$name]: $reason"
}
finish() { "$PY" "$SCRIPT_DIR/test_results.py" "$LOGDIR" "$TEST_KIND"; }
