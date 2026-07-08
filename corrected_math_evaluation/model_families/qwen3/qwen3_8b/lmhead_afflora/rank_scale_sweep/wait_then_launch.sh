#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

PY="${PYTHON_BIN:-python}"
PREV=corrected_math_evaluation/model_families/qwen3/qwen3_8b/lmhead_afflora/rank1_small_scale_sweep/state.json
EXP=corrected_math_evaluation/model_families/qwen3/qwen3_8b/lmhead_afflora/rank_scale_sweep

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
    echo "previous sweep failed; not launching rank-scale sweep"
    exit 1
  fi
  sleep 60
done

exec bash "$EXP/launch.sh"
