#!/usr/bin/env bash
set -euo pipefail

cd /commondocument/wz/cross_encoder_workspace/im_exp/lora

EXP=corrected_math_evaluation/model_families/qwen3/qwen3_8b/lmhead_afflora/rank1_small_scale_sweep
PY=/home/wz/anaconda3/envs/torch24/bin/python

mkdir -p "$EXP/logs"

"$PY" "$EXP/run_experiment.py" > "$EXP/logs/orchestrator.log" 2>&1 &
ORCH_PID=$!
echo "$ORCH_PID" > "$EXP/orchestrator.pid"

"$PY" "$EXP/monitor.py" "$ORCH_PID" > "$EXP/logs/monitor.log" 2>&1 &
MON_PID=$!
echo "$MON_PID" > "$EXP/monitor.pid"

echo "launched afflora small-scale sweep: orchestrator=$ORCH_PID monitor=$MON_PID"
wait "$ORCH_PID"
STATUS=$?
wait "$MON_PID" || true
exit "$STATUS"
