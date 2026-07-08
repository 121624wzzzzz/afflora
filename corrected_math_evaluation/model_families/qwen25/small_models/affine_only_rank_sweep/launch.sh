#!/usr/bin/env bash
set -euo pipefail

cd /commondocument/wz/cross_encoder_workspace/im_exp/lora

EXP=corrected_math_evaluation/model_families/qwen25/small_models/affine_only_rank_sweep
PY=/home/wz/anaconda3/envs/torch24/bin/python

mkdir -p "$EXP/logs"

"$PY" "$EXP/run_experiment.py" > "$EXP/logs/orchestrator.log" 2>&1 &
ORCH_PID=$!
echo "$ORCH_PID" > "$EXP/orchestrator.pid"

"$PY" "$EXP/monitor.py" --orchestrator-pid "$ORCH_PID" --interval 3600 > "$EXP/logs/monitor.log" 2>&1 &
MON_PID=$!
echo "$MON_PID" > "$EXP/monitor.pid"

echo "launched small affine-only rank sweep: orchestrator=$ORCH_PID monitor=$MON_PID"
wait "$ORCH_PID"
STATUS=$?
wait "$MON_PID" || true
exit "$STATUS"
