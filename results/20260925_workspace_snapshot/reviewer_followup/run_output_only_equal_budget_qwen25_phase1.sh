#!/usr/bin/env bash
set -euo pipefail

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

train_and_eval() {
  local method=$1
  local scale=$2
  local gpu=$3
  local seed=42
  local run=qwen25_15b_out_${method}_s${scale}_sd${seed}
  local run_dir=$OUT/$run
  local train_log=$LOG/${run}.train.log
  local dev_log=$LOG/${run}.dev.log
  local train_script=corrected_sft_experiment/train_corrected_sft.py

  if [[ "$method" == "vocab_r1" ]]; then
    train_script=reviewer_followup/train_corrected_sft_output_vocab.py
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
    )
  else
    cmd+=(
      --variant hidden_lora
      --include-emb-lmh-lora-rank 1
      --emb-lmh-lora-alpha "$scale"
    )
  fi

  if [[ ! -s "$run_dir/adapter_model.safetensors" || ! -s "$run_dir/run_args.json" ]]; then
    echo "START train run=$run gpu=$gpu time=$(date --iso-8601=seconds)"
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" >"$train_log" 2>&1
    echo "DONE train run=$run time=$(date --iso-8601=seconds)"
  else
    echo "SKIP complete train run=$run"
  fi

  if [[ ! -s "$run_dir/dev_report.json" ]]; then
    echo "START dev run=$run gpu=$gpu time=$(date --iso-8601=seconds)"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" corrected_sft_experiment/evaluate_corrected_sft.py \
      --run-dir "$run_dir" \
      --data "$DEV" \
      --output "$run_dir/dev_report.json" \
      --batch-size 8 \
      >"$dev_log" 2>&1
    echo "DONE dev run=$run time=$(date --iso-8601=seconds)"
  else
    echo "SKIP complete dev run=$run"
  fi
}

TASKS=(
  "aff_r50 1"
  "aff_r50 2"
  "aff_r50 4"
  "aff_r50 8"
  "vocab_r1 1"
  "vocab_r1 2"
  "vocab_r1 4"
  "vocab_r1 8"
)

pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  read -r method scale <<<"${TASKS[$gpu]}"
  train_and_eval "$method" "$scale" "$gpu" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if (( failed != 0 )); then
  echo "PHASE1_FAILED time=$(date --iso-8601=seconds)" >&2
  exit 1
fi

echo "PHASE1_COMPLETE time=$(date --iso-8601=seconds)"
