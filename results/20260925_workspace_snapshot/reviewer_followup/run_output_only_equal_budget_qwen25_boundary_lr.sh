#!/usr/bin/env bash
set -euo pipefail

# Seed-42 boundary-LR extension of the strict output-only equal-budget sweep.
# This launcher intentionally evaluates corrected dev only.  It is restartable:
# completed training/evaluation artifacts are skipped, while incomplete jobs are
# rerun under the same explicit run name.

ROOT=/commondocument/wz/cross_encoder_workspace/im_exp/lora
PY=/home/wz/anaconda3/envs/torch24/bin/python
export CUDA_HOME=/home/wz/anaconda3/envs/torch24
export LD_LIBRARY_PATH=/home/wz/anaconda3/envs/torch24/lib:${LD_LIBRARY_PATH:-}
export DS_IGNORE_CUDA_DETECTION=1

MODEL=../models/Qwen2.5-1.5B-Base
TRAIN=corrected_sft_experiment/data/train.jsonl
DEV=corrected_sft_experiment/data/dev.jsonl
EXP=reviewer_followup/output_only_equal_budget_qwen25
OUT=$EXP/checkpoints
LOG=$EXP/logs

cd "$ROOT"
mkdir -p "$OUT" "$LOG"

training_complete() {
  local method=$1
  local run_dir=$2
  local -a common=(
    "$run_dir/adapter_model.safetensors"
    "$run_dir/adapter_config.json"
    "$run_dir/run_args.json"
    "$run_dir/trainable_summary.json"
  )
  local artifact
  for artifact in "${common[@]}"; do
    [[ -s "$artifact" ]] || return 1
  done
  if [[ "$method" == "aff_r50" ]]; then
    [[ -s "$run_dir/affine_vocab_adapter.safetensors" ]] || return 1
    [[ -s "$run_dir/affine_vocab_config.json" ]] || return 1
  else
    [[ -s "$run_dir/optimizer_groups.json" ]] || return 1
  fi
}

dev_complete() {
  local report=$1
  [[ -s "$report" ]] || return 1
  "$PY" -c '
import json
import math
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
rows = report.get("per_example")
valid = (
    report.get("num_examples") == 1000
    and report.get("source_start_index") == 0
    and report.get("source_end_index") == 1000
    and isinstance(rows, list)
    and len(rows) == 1000
    and isinstance(report.get("avg_ce"), (int, float))
    and math.isfinite(float(report["avg_ce"]))
)
raise SystemExit(0 if valid else 1)
' "$report" >/dev/null 2>&1
}

train_and_eval() {
  local method=$1
  local scale=$2
  local boundary_lr_scale=$3
  local gpu=$4
  local seed=42
  local boundary_lr_tag=${boundary_lr_scale//./p}
  local run=qwen25_15b_out_${method}_s${scale}_blr${boundary_lr_tag}_sd${seed}
  local run_dir=$OUT/$run
  local train_log=$LOG/${run}.train.log
  local dev_log=$LOG/${run}.dev.log
  local train_script=corrected_sft_experiment/train_corrected_sft.py

  if [[ "$method" == "vocab_r1" ]]; then
    train_script=reviewer_followup/train_corrected_sft_output_vocab_boundary_lr.py
  fi

  local -a cmd=(
    "$PY" "$train_script"
    --model-path "$MODEL"
    --train-data "$TRAIN"
    --eval-data "$DEV"
    --output-dir "$run_dir"
    --hidden-lora-rank 8
    --hidden-lora-alpha 16
    --hidden-lora-dropout 0.05
    --hidden-lora-target-modules q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj
    --max-seq-len 1024
    --per-device-train-batch-size 8
    --gradient-accumulation-steps 2
    --learning-rate 2e-4
    --num-train-epochs 1
    --eval-samples 1000
    --eval-steps 250
    --logging-steps 10
    --save-strategy no
    --bf16
    --master-dtype fp32
    --seed "$seed"
  )

  if [[ "$method" == "aff_r50" ]]; then
    cmd+=(
      --variant affine_lm_head_plus_hidden_lora
      --affine-rank 50
      --affine-alpha "$((50 * scale))"
      --affine-dropout 0
      --no-affine-input-bias
      --affine-learning-rate-scale "$boundary_lr_scale"
    )
  else
    cmd+=(
      --variant hidden_lora
      --include-emb-lmh-lora-rank 1
      --emb-lmh-lora-alpha "$scale"
      --output-vocab-lr-scale "$boundary_lr_scale"
    )
  fi

  if training_complete "$method" "$run_dir"; then
    echo "SKIP complete train run=$run"
  else
    echo "START train run=$run gpu=$gpu time=$(date --iso-8601=seconds)"
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" >"$train_log" 2>&1
    if ! training_complete "$method" "$run_dir"; then
      echo "INCOMPLETE train artifacts run=$run" >&2
      return 1
    fi
    echo "DONE train run=$run time=$(date --iso-8601=seconds)"
  fi

  if dev_complete "$run_dir/dev_report.json"; then
    echo "SKIP complete dev run=$run"
  else
    echo "START dev run=$run gpu=$gpu time=$(date --iso-8601=seconds)"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" corrected_sft_experiment/evaluate_corrected_sft.py \
      --run-dir "$run_dir" \
      --data "$DEV" \
      --output "$run_dir/dev_report.json" \
      --batch-size 8 \
      >"$dev_log" 2>&1
    dev_complete "$run_dir/dev_report.json" || {
      echo "INCOMPLETE dev artifact run=$run" >&2
      return 1
    }
    echo "DONE dev run=$run time=$(date --iso-8601=seconds)"
  fi
}

# method, functional scale alpha/rank, boundary-only LR multiplier.
TASKS=(
  "aff_r50 2 0.5"
  "aff_r50 2 2"
  "aff_r50 8 0.5"
  "aff_r50 8 2"
  "vocab_r1 2 0.5"
  "vocab_r1 2 2"
  "vocab_r1 8 0.5"
  "vocab_r1 8 2"
)

pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  read -r method scale boundary_lr_scale <<<"${TASKS[$gpu]}"
  train_and_eval "$method" "$scale" "$boundary_lr_scale" "$gpu" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if (( failed != 0 )); then
  echo "BOUNDARY_LR_SWEEP_FAILED time=$(date --iso-8601=seconds)" >&2
  exit 1
fi

"$PY" reviewer_followup/summarize_output_only_equal_budget_qwen25_joint_lr.py
echo "BOUNDARY_LR_SWEEP_COMPLETE time=$(date --iso-8601=seconds)"
