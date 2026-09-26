#!/usr/bin/env bash
set -euo pipefail

ROOT=/commondocument/wz/cross_encoder_workspace/im_exp/lora
cd "$ROOT"

PY=/home/wz/anaconda3/envs/torch24/bin/python
TORCHRUN=/home/wz/anaconda3/envs/torch24/bin/torchrun
export CUDA_HOME=/home/wz/anaconda3/envs/torch24
export LD_LIBRARY_PATH=/home/wz/anaconda3/envs/torch24/lib:${LD_LIBRARY_PATH:-}

MODEL=../models/Qwen2.5-1.5B-Base
DATA_DIR=corrected_sft_experiment/data_ultrachat_100k_len4096
TRAIN_DATA=$DATA_DIR/train.jsonl
EVAL_DATA=$DATA_DIR/eval.jsonl
OUT_BASE=corrected_sft_experiment/outputs/ultrachat100k
LOG_DIR=corrected_sft_experiment/logs/ultrachat_tied_bias_rank_seed_sweep
mkdir -p "$LOG_DIR" "$OUT_BASE"

RUN_LOG=$LOG_DIR/scheduler.log
TASKS_DONE=$LOG_DIR/tasks_done.tsv
POST_LOG=$LOG_DIR/posteval.log

# Fields: name|variant|hidden_rank|seed
# Existing completed seed42/hr8 tied+bias runs are reused for comparisons.
TASKS=(
  "qwen25_15b_hidden_hr8_sd43_len4096_ddp4_bs2_acc2|hidden_lora|8|43"
  "qwen25_15b_afflora_tied_bias_hr8_sd43_len4096_ddp4_bs2_acc2|affine_input_lm_head_plus_hidden_lora|8|43"
  "qwen25_15b_hidden_hr8_sd44_len4096_ddp4_bs2_acc2|hidden_lora|8|44"
  "qwen25_15b_afflora_tied_bias_hr8_sd44_len4096_ddp4_bs2_acc2|affine_input_lm_head_plus_hidden_lora|8|44"
  "qwen25_15b_hidden_hr1_sd42_len4096_ddp4_bs2_acc2|hidden_lora|1|42"
  "qwen25_15b_afflora_tied_bias_hr1_sd42_len4096_ddp4_bs2_acc2|affine_input_lm_head_plus_hidden_lora|1|42"
  "qwen25_15b_hidden_hr2_sd42_len4096_ddp4_bs2_acc2|hidden_lora|2|42"
  "qwen25_15b_afflora_tied_bias_hr2_sd42_len4096_ddp4_bs2_acc2|affine_input_lm_head_plus_hidden_lora|2|42"
  "qwen25_15b_hidden_hr4_sd42_len4096_ddp4_bs2_acc2|hidden_lora|4|42"
  "qwen25_15b_afflora_tied_bias_hr4_sd42_len4096_ddp4_bs2_acc2|affine_input_lm_head_plus_hidden_lora|4|42"
)

gpu_group() {
  if [[ "$1" == "0" ]]; then
    printf '0,1,2,3'
  else
    printf '4,5,6,7'
  fi
}

port_for_group() {
  if [[ "$1" == "0" ]]; then
    printf '29620'
  else
    printf '29621'
  fi
}

run_train() {
  local name=$1
  local variant=$2
  local rank=$3
  local seed=$4
  local group=$5
  local out_dir=$OUT_BASE/$name
  local log_file=$LOG_DIR/${name}.log
  local gpus
  local port
  gpus=$(gpu_group "$group")
  port=$(port_for_group "$group")

  if [[ -f "$out_dir/final_eval_results.json" && -f "$out_dir/run_args.json" ]]; then
    echo "$(date '+%F %T') SKIP existing $name" | tee -a "$RUN_LOG"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$name" "$variant" "$rank" "$seed" "skipped" >> "$TASKS_DONE"
    return 0
  fi

  rm -rf "$out_dir"
  echo "$(date '+%F %T') START group=$group gpus=$gpus name=$name variant=$variant rank=$rank seed=$seed" | tee -a "$RUN_LOG"

  local -a cmd=(
    "$TORCHRUN" --nproc_per_node=4 --master_port="$port"
    corrected_sft_experiment/train_corrected_sft.py
    --model-path "$MODEL"
    --train-data "$TRAIN_DATA"
    --eval-data "$EVAL_DATA"
    --output-dir "$out_dir"
    --variant "$variant"
    --hidden-lora-rank "$rank"
    --hidden-lora-alpha "$((rank * 2))"
    --hidden-lora-dropout 0.05
    --hidden-lora-target-modules q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj
    --max-seq-len 4096
    --per-device-train-batch-size 2
    --gradient-accumulation-steps 2
    --learning-rate 2e-4
    --num-train-epochs 1
    --eval-samples 2000
    --eval-steps 1250
    --logging-steps 50
    --save-strategy no
    --bf16
    --master-dtype fp32
    --seed "$seed"
  )

  if [[ "$variant" == "affine_input_lm_head_plus_hidden_lora" ]]; then
    cmd+=(
      --affine-rank 16
      --affine-alpha 128
      --affine-lm-head-bias
      --tie-affine-input-lm-head-adapters
    )
  fi

  if CUDA_VISIBLE_DEVICES="$gpus" "${cmd[@]}" > "$log_file" 2>&1; then
    echo "$(date '+%F %T') DONE name=$name" | tee -a "$RUN_LOG"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$name" "$variant" "$rank" "$seed" "done" >> "$TASKS_DONE"
  else
    echo "$(date '+%F %T') FAIL name=$name see=$log_file" | tee -a "$RUN_LOG"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$name" "$variant" "$rank" "$seed" "failed" >> "$TASKS_DONE"
    return 1
  fi
}

pid_alive() {
  local pid=${1:-}
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

main_train() {
  : > "$RUN_LOG"
  : > "$TASKS_DONE"
  local next=0
  local pid0=""
  local pid1=""
  local fail=0

  while (( next < ${#TASKS[@]} )) || pid_alive "$pid0" || pid_alive "$pid1"; do
    if ! pid_alive "$pid0" && (( next < ${#TASKS[@]} )); then
      IFS='|' read -r name variant rank seed <<< "${TASKS[$next]}"
      next=$((next + 1))
      run_train "$name" "$variant" "$rank" "$seed" 0 &
      pid0=$!
    fi
    if ! pid_alive "$pid1" && (( next < ${#TASKS[@]} )); then
      IFS='|' read -r name variant rank seed <<< "${TASKS[$next]}"
      next=$((next + 1))
      run_train "$name" "$variant" "$rank" "$seed" 1 &
      pid1=$!
    fi
    sleep 60
    if [[ -n "$pid0" ]] && ! pid_alive "$pid0"; then
      wait "$pid0" || fail=1
      pid0=""
    fi
    if [[ -n "$pid1" ]] && ! pid_alive "$pid1"; then
      wait "$pid1" || fail=1
      pid1=""
    fi
  done

  return "$fail"
}

eval_report() {
  local run_dir=$1
  local gpu=$2
  if [[ -f "$run_dir/eval_report.json" ]]; then
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" corrected_sft_experiment/evaluate_corrected_sft.py \
    --run-dir "$run_dir" \
    --data "$EVAL_DATA" \
    --output "$run_dir/eval_report.json" \
    --max-seq-len 4096 \
    --batch-size 4 \
    --device cuda
}

compare_pair() {
  local baseline=$1
  local treatment=$2
  local output=$3
  "$PY" corrected_sft_experiment/compare_results.py \
    --baseline "$baseline" \
    --treatment "$treatment" \
    --output "$output" \
    --samples 10000 \
    --seed 42
}

post_eval() {
  {
    echo "$(date '+%F %T') post_eval start"

    for item in "${TASKS[@]}"; do
      IFS='|' read -r name _variant _rank _seed <<< "$item"
      if [[ -f "$OUT_BASE/$name/final_eval_results.json" ]]; then
        eval_report "$OUT_BASE/$name" 0
      else
        echo "missing trained output for $name"
      fi
    done

    # Multi-seed hr8 comparisons.
    compare_pair \
      "$OUT_BASE/qwen25_15b_hidden_sd42_len4096_ddp4_bs2_acc2/eval_report.json" \
      "$OUT_BASE/qwen25_15b_afflora_tied_bias_sd42_len4096_ddp4_bs2_acc2/eval_report.json" \
      "$OUT_BASE/qwen25_15b_hidden_vs_afflora_tied_bias_hr8_sd42_len4096_comparison.json"
    for seed in 43 44; do
      compare_pair \
        "$OUT_BASE/qwen25_15b_hidden_hr8_sd${seed}_len4096_ddp4_bs2_acc2/eval_report.json" \
        "$OUT_BASE/qwen25_15b_afflora_tied_bias_hr8_sd${seed}_len4096_ddp4_bs2_acc2/eval_report.json" \
        "$OUT_BASE/qwen25_15b_hidden_vs_afflora_tied_bias_hr8_sd${seed}_len4096_comparison.json"
    done

    # Low hidden-rank comparisons.
    for rank in 1 2 4; do
      compare_pair \
        "$OUT_BASE/qwen25_15b_hidden_hr${rank}_sd42_len4096_ddp4_bs2_acc2/eval_report.json" \
        "$OUT_BASE/qwen25_15b_afflora_tied_bias_hr${rank}_sd42_len4096_ddp4_bs2_acc2/eval_report.json" \
        "$OUT_BASE/qwen25_15b_hidden_vs_afflora_tied_bias_hr${rank}_sd42_len4096_comparison.json"
    done

    "$PY" - <<'PY'
import json
from pathlib import Path

base = Path("corrected_sft_experiment/outputs/ultrachat100k")
items = []
for path in sorted(base.glob("qwen25_15b_hidden_vs_afflora_tied_bias_hr*_sd*_len4096_comparison.json")):
    report = json.loads(path.read_text())
    parts = path.stem.split("_")
    hr = next(part for part in parts if part.startswith("hr"))
    sd = next(part for part in parts if part.startswith("sd"))
    items.append({
        "comparison": path.name,
        "hidden_rank": hr,
        "seed": sd,
        "delta_ce": report["delta_ce_treatment_minus_baseline"],
        "ci95": report["ci95"],
        "probability_treatment_better": report["probability_treatment_better"],
    })
out = base / "qwen25_15b_ultrachat_tied_bias_rank_seed_summary.json"
out.write_text(json.dumps(items, indent=2), encoding="utf-8")
print(json.dumps(items, indent=2))
PY

    echo "$(date '+%F %T') post_eval done"
  } >> "$POST_LOG" 2>&1
}

main_train
post_eval
