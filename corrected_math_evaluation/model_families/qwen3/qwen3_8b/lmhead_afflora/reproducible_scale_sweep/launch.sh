#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
PY=/home/wz/anaconda3/envs/torch24/bin/python

if [[ -f "$HERE/orchestrator.pid" ]] && kill -0 "$(cat "$HERE/orchestrator.pid")" 2>/dev/null; then
  echo "orchestrator already running: $(cat "$HERE/orchestrator.pid")"
  exit 0
fi

"$PY" "$HERE/run_experiment.py" > "$HERE/orchestrator.log" 2>&1 &
ORCHESTRATOR_PID=$!
printf '%s\n' "$ORCHESTRATOR_PID" > "$HERE/orchestrator.pid"

"$PY" "$HERE/monitor.py" --orchestrator-pid "$ORCHESTRATOR_PID" --interval 3600 \
  > "$HERE/monitor_stdout.log" 2>&1 &
MONITOR_PID=$!
printf '%s\n' "$MONITOR_PID" > "$HERE/monitor.pid"

cleanup() {
  kill "$ORCHESTRATOR_PID" "$MONITOR_PID" 2>/dev/null || true
  wait "$ORCHESTRATOR_PID" "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

echo "orchestrator_pid=$ORCHESTRATOR_PID monitor_pid=$MONITOR_PID"
set +e
wait "$ORCHESTRATOR_PID"
STATUS=$?
wait "$MONITOR_PID"
set -e
trap - EXIT
exit "$STATUS"
