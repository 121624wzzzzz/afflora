#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "Usage: $0 <selected-aff-r50-scale> <selected-vocab-r1-scale>" >&2
  echo "Both scales must be one of: 1 2 4 8" >&2
  exit 2
fi

AFF_SCALE=$1
VOCAB_SCALE=$2
case "$AFF_SCALE" in
  1|2|4|8) ;;
  *)
    echo "Invalid A-LoRA scale: $AFF_SCALE (expected 1, 2, 4, or 8)" >&2
    exit 2
    ;;
esac
case "$VOCAB_SCALE" in
  1|2|4|8) ;;
  *)
    echo "Invalid Vocab-LoRA scale: $VOCAB_SCALE (expected 1, 2, 4, or 8)" >&2
    exit 2
    ;;
esac

ROOT=/commondocument/wz/cross_encoder_workspace/im_exp/lora
PY=/home/wz/anaconda3/envs/torch24/bin/python
export CUDA_HOME=/home/wz/anaconda3/envs/torch24
export LD_LIBRARY_PATH=/home/wz/anaconda3/envs/torch24/lib:${LD_LIBRARY_PATH:-}
export DS_IGNORE_CUDA_DETECTION=1

MODEL=../models/Qwen2.5-1.5B-Base
TRAIN=corrected_sft_experiment/data/train.jsonl
DEV=corrected_sft_experiment/data/dev.jsonl
TEST=corrected_sft_experiment/data/test.jsonl
EXP=reviewer_followup/output_only_equal_budget_qwen25
OUT=$EXP/checkpoints
LOG=$EXP/logs
COMPARE=$EXP/phase2_comparisons

# Six independent single-GPU jobs are run concurrently. Override with a
# whitespace-separated list, e.g. GPU_IDS="2 3 4 5 6 7".
read -r -a AVAILABLE_GPUS <<<"${GPU_IDS:-0 1 2 3 4 5}"
if (( ${#AVAILABLE_GPUS[@]} < 6 )); then
  echo "Need at least six entries in GPU_IDS; got ${#AVAILABLE_GPUS[@]}" >&2
  exit 2
fi

cd "$ROOT"
mkdir -p "$OUT" "$LOG" "$COMPARE"

is_complete_run() {
  local method=$1
  local run_dir=$2
  [[ -s "$run_dir/run_args.json" && -s "$run_dir/adapter_model.safetensors" ]] || return 1
  if [[ "$method" == "aff_r50" ]]; then
    [[ -s "$run_dir/affine_vocab_adapter.safetensors" ]] || return 1
  fi
}

train_and_eval() {
  local method=$1
  local scale=$2
  local seed=$3
  local gpu=$4
  local run=qwen25_15b_out_${method}_s${scale}_sd${seed}
  local run_dir=$OUT/$run
  local train_log=$LOG/${run}.train.log
  local dev_log=$LOG/${run}.dev.log
  local test_log=$LOG/${run}.test.log
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

  if ! is_complete_run "$method" "$run_dir"; then
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

  if [[ ! -s "$run_dir/test_report.json" ]]; then
    echo "START test run=$run gpu=$gpu time=$(date --iso-8601=seconds)"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" corrected_sft_experiment/evaluate_corrected_sft.py \
      --run-dir "$run_dir" \
      --data "$TEST" \
      --output "$run_dir/test_report.json" \
      --batch-size 8 \
      >"$test_log" 2>&1
    echo "DONE test run=$run time=$(date --iso-8601=seconds)"
  else
    echo "SKIP complete test run=$run"
  fi
}

TASKS=(
  "aff_r50 $AFF_SCALE 43"
  "vocab_r1 $VOCAB_SCALE 43"
  "aff_r50 $AFF_SCALE 44"
  "vocab_r1 $VOCAB_SCALE 44"
  "aff_r50 $AFF_SCALE 45"
  "vocab_r1 $VOCAB_SCALE 45"
)

pids=()
for task_index in "${!TASKS[@]}"; do
  read -r method scale seed <<<"${TASKS[$task_index]}"
  train_and_eval "$method" "$scale" "$seed" "${AVAILABLE_GPUS[$task_index]}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if (( failed != 0 )); then
  echo "PHASE2_TRAIN_OR_EVAL_FAILED time=$(date --iso-8601=seconds)" >&2
  exit 1
fi

for seed in 43 44 45; do
  aff_dir=$OUT/qwen25_15b_out_aff_r50_s${AFF_SCALE}_sd${seed}
  vocab_dir=$OUT/qwen25_15b_out_vocab_r1_s${VOCAB_SCALE}_sd${seed}
  for split in dev test; do
    comparison=$COMPARE/vocab_vs_aff_s${VOCAB_SCALE}_vs_s${AFF_SCALE}_sd${seed}_${split}.json
    "$PY" corrected_sft_experiment/compare_results.py \
      --baseline "$vocab_dir/${split}_report.json" \
      --treatment "$aff_dir/${split}_report.json" \
      --output "$comparison" \
      --samples 10000 \
      --seed "$seed"
  done
done

echo "PHASE2_COMPLETE aff_scale=$AFF_SCALE vocab_scale=$VOCAB_SCALE time=$(date --iso-8601=seconds)"
