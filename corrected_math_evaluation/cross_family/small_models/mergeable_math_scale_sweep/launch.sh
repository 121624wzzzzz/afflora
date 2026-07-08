#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

EXP=corrected_math_evaluation/cross_family/small_models/mergeable_math_scale_sweep
PY="${PYTHON_BIN:-python}"

mkdir -p "$EXP/logs"

"$PY" "$EXP/run_experiment.py" > "$EXP/logs/orchestrator.log" 2>&1 &
ORCH_PID=$!
echo "$ORCH_PID" > "$EXP/orchestrator.pid"

"$PY" "$EXP/monitor.py" --orchestrator-pid "$ORCH_PID" --interval 3600 > "$EXP/logs/monitor.log" 2>&1 &
MON_PID=$!
echo "$MON_PID" > "$EXP/monitor.pid"

echo "launched small mergeable math sweep: orchestrator=$ORCH_PID monitor=$MON_PID"
wait "$ORCH_PID"
STATUS=$?
wait "$MON_PID" || true
exit "$STATUS"
