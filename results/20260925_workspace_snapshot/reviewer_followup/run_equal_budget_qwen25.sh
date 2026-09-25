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
TEST=corrected_sft_experiment/data/test.jsonl
OUT=reviewer_followup/equal_budget_qwen25/checkpoints
LOG=reviewer_followup/equal_budget_qwen25/logs

cd "$ROOT"
mkdir -p "$OUT" "$LOG"

train_one() {
  local seed=$1
  local method=$2
  local gpu=$3
  local run=qwen25_15b_${method}_sd${seed}
  local run_dir=$OUT/$run
  local train_script=corrected_sft_experiment/train_corrected_sft.py
  if [[ "$method" == "vocab_lora_r1" ]]; then
    train_script=reviewer_followup/train_corrected_sft_isolated_vocab.py
  fi
  if [[ -s "$run_dir/adapter_model.safetensors" && -s "$run_dir/run_args.json" ]]; then
    echo "SKIP complete training $run"
    return
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
  if [[ "$method" == "afflora_r48" ]]; then
    cmd+=(
      --variant affine_input_lm_head_plus_hidden_lora
      --affine-rank 48
      --affine-alpha 384
    )
  else
    cmd+=(
      --variant hidden_lora
      --include-emb-lmh-lora-rank 1
      --emb-lmh-lora-alpha 2
    )
  fi
  echo "START training $run gpu=$gpu $(date --iso-8601=seconds)"
  CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" >"$LOG/${run}.train.log" 2>&1
  echo "DONE training $run $(date --iso-8601=seconds)"
}

eval_one() {
  local seed=$1
  local method=$2
  local gpu=$3
  local run=qwen25_15b_${method}_sd${seed}
  local run_dir=$OUT/$run
  for split in dev test; do
    local data=$DEV
    [[ "$split" == "test" ]] && data=$TEST
    local report=$run_dir/${split}_report.json
    [[ -s "$report" ]] && continue
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" corrected_sft_experiment/evaluate_corrected_sft.py \
      --run-dir "$run_dir" --data "$data" --output "$report" --batch-size 8 \
      >"$LOG/${run}.${split}.log" 2>&1
  done
}

TASKS=(
  "42 afflora_r48" "42 vocab_lora_r1"
  "43 afflora_r48" "43 vocab_lora_r1"
  "44 afflora_r48" "44 vocab_lora_r1"
)

run_waves() {
  local stage=$1
  local counter=$LOG/.${stage}.next
  local lock=$LOG/.${stage}.lock
  printf '0\n' >"$counter"

  worker() {
    local gpu=$1
    while true; do
      local index
      {
        flock 9
        index=$(<"$counter")
        if (( index >= ${#TASKS[@]} )); then
          return
        fi
        printf '%s\n' "$((index + 1))" >"$counter"
      } 9>"$lock"
      read -r seed method <<<"${TASKS[$index]}"
      if [[ "$stage" == "train" ]]; then
        train_one "$seed" "$method" "$gpu" &
      else
        eval_one "$seed" "$method" "$gpu" &
      fi
      wait "$!"
    done
  }

  local pids=()
  for gpu in 0 1 2 3; do
    worker "$gpu" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
}

run_waves train
run_waves eval

for seed in 42 43 44; do
  for split in dev test; do
    base=corrected_sft_experiment/outputs/formal/qwen25_15b_hidden_sd${seed}/${split}_report.json
    a48=$OUT/qwen25_15b_afflora_r48_sd${seed}/${split}_report.json
    vocab=$OUT/qwen25_15b_vocab_lora_r1_sd${seed}/${split}_report.json
    "$PY" corrected_sft_experiment/compare_results.py --baseline "$base" --treatment "$a48" \
      --output reviewer_followup/equal_budget_qwen25/hidden_vs_a48_sd${seed}_${split}.json \
      --samples 10000 --seed "$seed" >/dev/null
    "$PY" corrected_sft_experiment/compare_results.py --baseline "$base" --treatment "$vocab" \
      --output reviewer_followup/equal_budget_qwen25/hidden_vs_vocab_sd${seed}_${split}.json \
      --samples 10000 --seed "$seed" >/dev/null
    "$PY" corrected_sft_experiment/compare_results.py --baseline "$vocab" --treatment "$a48" \
      --output reviewer_followup/equal_budget_qwen25/vocab_vs_a48_sd${seed}_${split}.json \
      --samples 10000 --seed "$seed" >/dev/null
  done
done

echo "EQUAL_BUDGET_QWEN25_COMPLETE $(date --iso-8601=seconds)"
