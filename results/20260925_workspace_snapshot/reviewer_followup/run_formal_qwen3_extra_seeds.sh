#!/usr/bin/env bash
set -euo pipefail

ROOT=/commondocument/wz/cross_encoder_workspace/im_exp/lora
PY=/home/wz/anaconda3/envs/torch24/bin/python
export CUDA_HOME=/home/wz/anaconda3/envs/torch24
export LD_LIBRARY_PATH=/home/wz/anaconda3/envs/torch24/lib:${LD_LIBRARY_PATH:-}
export DS_IGNORE_CUDA_DETECTION=1
MODEL=../models/Qwen3-0.6B-Base
TRAIN=corrected_sft_experiment/data/train.jsonl
DEV=corrected_sft_experiment/data/dev.jsonl
TEST=corrected_sft_experiment/data/test.jsonl
OUT=corrected_sft_experiment/outputs/formal
LOG=reviewer_followup/logs/formal_qwen3_extra_seeds

cd "$ROOT"
mkdir -p "$LOG"

train_one() {
  local seed=$1
  local variant=$2
  local gpu=$3
  local tag
  if [[ "$variant" == "hidden_lora" ]]; then
    tag=hidden
  else
    tag=afflora
  fi
  local run=qwen3_06b_${tag}_sd${seed}
  local run_dir=$OUT/$run
  local log_file=$LOG/${run}.train.log

  if [[ -s "$run_dir/adapter_model.safetensors" && -s "$run_dir/run_args.json" ]]; then
    echo "SKIP complete training $run"
    return
  fi

  local -a cmd=(
    "$PY" corrected_sft_experiment/train_corrected_sft.py
    --model-path "$MODEL"
    --train-data "$TRAIN"
    --eval-data "$DEV"
    --output-dir "$run_dir"
    --variant "$variant"
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
  if [[ "$variant" == "affine_input_lm_head_plus_hidden_lora" ]]; then
    cmd+=(--affine-rank 16 --affine-alpha 128)
  fi
  echo "START training $run gpu=$gpu $(date --iso-8601=seconds)"
  CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" >"$log_file" 2>&1
  echo "DONE training $run $(date --iso-8601=seconds)"
}

eval_one() {
  local seed=$1
  local tag=$2
  local gpu=$3
  local run=qwen3_06b_${tag}_sd${seed}
  local run_dir=$OUT/$run
  for split in dev test; do
    local data=$DEV
    [[ "$split" == "test" ]] && data=$TEST
    local report=$run_dir/${split}_report.json
    if [[ -s "$report" ]]; then
      echo "SKIP existing $report"
      continue
    fi
    echo "START eval $run $split gpu=$gpu $(date --iso-8601=seconds)"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" corrected_sft_experiment/evaluate_corrected_sft.py \
      --run-dir "$run_dir" --data "$data" --output "$report" --batch-size 8 \
      >"$LOG/${run}.${split}.log" 2>&1
  done
}

train_one 43 hidden_lora 0 & p0=$!
train_one 43 affine_input_lm_head_plus_hidden_lora 1 & p1=$!
train_one 44 hidden_lora 2 & p2=$!
train_one 44 affine_input_lm_head_plus_hidden_lora 3 & p3=$!
wait "$p0" "$p1" "$p2" "$p3"

eval_one 43 hidden 0 & p0=$!
eval_one 43 afflora 1 & p1=$!
eval_one 44 hidden 2 & p2=$!
eval_one 44 afflora 3 & p3=$!
wait "$p0" "$p1" "$p2" "$p3"

for seed in 43 44; do
  for split in dev test; do
    "$PY" corrected_sft_experiment/compare_results.py \
      --baseline "$OUT/qwen3_06b_hidden_sd${seed}/${split}_report.json" \
      --treatment "$OUT/qwen3_06b_afflora_sd${seed}/${split}_report.json" \
      --output "$OUT/qwen3_06b_sd${seed}_${split}_comparison.json" \
      --samples 10000 --seed "$seed" \
      >"$LOG/qwen3_06b_sd${seed}.${split}.compare.log" 2>&1
  done
done

echo "FORMAL_QWEN3_EXTRA_SEEDS_COMPLETE $(date --iso-8601=seconds)"
