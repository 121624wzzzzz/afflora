#!/usr/bin/env bash
set -euo pipefail

cd /commondocument/wz/cross_encoder_workspace/im_exp/lora

PY=/home/wz/anaconda3/envs/torch24/bin/python
PREV=corrected_math_evaluation/model_families/qwen3/qwen3_8b/lmhead_afflora/reproducible_scale_sweep/state.json
EXP=corrected_math_evaluation/model_families/qwen3/qwen3_8b/lmhead_afflora/rank1_small_scale_sweep

mkdir -p "$EXP/logs"

while true; do
  PHASE=$("$PY" - "$PREV" <<'PY'
import json, sys
path = sys.argv[1]
try:
    print(json.load(open(path)).get("phase", "missing"))
except FileNotFoundError:
    print("missing")
PY
)
  echo "$(date '+%F %T %Z') previous_phase=$PHASE"
  if [[ "$PHASE" == "complete" ]]; then
    break
  fi
  if [[ "$PHASE" == *"failed"* ]]; then
    echo "previous sweep failed; not launching small-scale sweep"
    exit 1
  fi
  sleep 60
done

exec bash "$EXP/launch.sh"
