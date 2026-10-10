#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Runs the configured regression suites; unavailable coverage is INCOMPLETE.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SRC_ROOT" || exit 1
source "$SCRIPT_DIR/python-env.sh"
export PATH="$(dirname "$PY"):${HOME}/.cargo/bin:$PATH"
export TEST_RESULTS_DIR="${TEST_RESULTS_DIR:-/tmp/dingoRouter-all-$(date +%Y%m%d_%H%M%S)-$$}"
mkdir -p "$TEST_RESULTS_DIR" || exit 1
exec > >(tee -a "$TEST_RESULTS_DIR/result.log") 2>&1
printf 'Started: %s\nSource: %s\nCommit: %s\nResults: %s\n' "$(date -Is)" "$SRC_ROOT" "$(git rev-parse HEAD)" "$TEST_RESULTS_DIR"
git diff --stat
QUICK_MODE="${QUICK_MODE:-full}"
case "$QUICK_MODE" in smoke|standard|full) ;; *) echo "Invalid QUICK_MODE=$QUICK_MODE"; exit 1;; esac
# A caller may prepare this exact interpreter separately, but its version must match.
if [[ "${SKIP_ENV_SETUP:-0}" != 1 ]]; then
    bash "$SCRIPT_DIR/env-setup-py-integ.sh" > "$TEST_RESULTS_DIR/envsetup.log" 2>&1
    code=$?
    if (( code != 0 )); then
        echo "SETUP_FAILED exit=$code; see $TEST_RESULTS_DIR/envsetup.log"
        echo "$code" > "$TEST_RESULTS_DIR/exit-code.txt"
        exit "$code"
    fi
fi
"$PY" - "$SRC_ROOT" <<'CHECK'
import importlib.metadata as m, pathlib, sys, tomli
expected=tomli.loads(pathlib.Path(sys.argv[1],"pyproject.toml").read_text())["project"]["version"]
assert m.version("ai-dingo-runtime")==expected, "Installed runtime does not match source version"
import dynamo._core
print("Runtime:", expected, dynamo._core.__file__)
CHECK
if (( $? != 0 )); then echo 1 > "$TEST_RESULTS_DIR/exit-code.txt"; exit 1; fi
if [[ "$QUICK_MODE" == smoke ]]; then
    export QUICK_PACKS=dynamo-backend-common,dynamo-kv-router,dynamo-runtime,dynamo-llm,dynamo-mocker
else
    unset QUICK_PACKS
fi
status=0
: > "$TEST_RESULTS_DIR/groups.tsv"
for group in rust-unit rust-integ python-unit python-integ; do
    if [[ "$QUICK_MODE" != full && "$group" == *integ ]]; then
        printf '%s\tNOT_RUN\tmode=%s\n' "$group" "$QUICK_MODE" >> "$TEST_RESULTS_DIR/groups.tsv"
        (( status == 0 )) && status=2
        continue
    fi
    echo "=== $group $(date -Is) ==="
    bash "$SCRIPT_DIR/run-$group-tests.sh" > "$TEST_RESULTS_DIR/$group.log" 2>&1
    code=$?
    tail -n 40 "$TEST_RESULTS_DIR/$group.log"
    if (( code == 0 )); then state=PASS
    elif (( code == 2 )); then state=INCOMPLETE; (( status == 0 )) && status=2
    else state=FAIL; status=1
    fi
    printf '%s\t%s\texit=%s\n' "$group" "$state" "$code" >> "$TEST_RESULTS_DIR/groups.tsv"
done
# Keep errors, infrastructure failures and omissions separate in the final report.
"$PY" - "$TEST_RESULTS_DIR" <<'REPORT'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); totals={}; groups=[]
for p in sorted(root.glob("*/summary.json")):
    data=json.loads(p.read_text()); groups.append(dict(path=str(p),**data))
    for k,v in data["totals"].items():totals[k]=totals.get(k,0)+v
(root/"summary.json").write_text(json.dumps(dict(totals=totals,groups=groups),indent=2)+"\n")
text=(root/"groups.tsv").read_text()+"\nTOTAL "+" ".join(f"{k}={v}" for k,v in totals.items())+"\n"
(root/"SUMMARY.txt").write_text(text)
print(text)
REPORT
if (( $? != 0 )); then status=1; fi
echo "$status" > "$TEST_RESULTS_DIR/exit-code.txt"
echo "Finished: $(date -Is); exit=$status (0=PASS, 1=FAIL, 2=INCOMPLETE)"
echo "Summary: $TEST_RESULTS_DIR/SUMMARY.txt"
echo "Full output: $TEST_RESULTS_DIR/result.log"
exit "$status"
