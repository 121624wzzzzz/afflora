#!/usr/bin/env bash
# Matched short continuation from the same completed no-energy corrected-SFT run.
set -euo pipefail
ROOT=/commondocument/wz/cross_encoder_workspace/im_exp/lora
PYTHON=/home/wz/anaconda3/envs/torch24/bin/python
OUT="$ROOT/reviewer_followup/tied_energy_lambda_sweep_qwen3"
cd "$ROOT"; export DS_IGNORE_CUDA_DETECTION=1
seeds=(43 44 45 46)
for i in {0..3}; do
  seed=${seeds[$i]}
  if (( seed <= 44 )); then
    init="$ROOT/reviewer_followup/tied_no_energy_qwen3/checkpoints/qwen3_06b_tied_noenergy_sd${seed}"
  else
    init="$OUT/checkpoints/qwen3_06b_tied_forward_tau0p00625_l0_sd${seed}"
  fi
  for arm in l0 l100; do
    gpu=$i
    name="continuation_${arm}_sd${seed}"
    extra=()
    if [[ "$arm" == l100 ]]; then
      gpu=$((i + 4))
      extra=(--affine-energy-lambda 100 --affine-energy-tau 0.00625)
    fi
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" corrected_sft_experiment/train_corrected_sft.py \
      --model-path ../models/Qwen3-0.6B-Base --train-data corrected_sft_experiment/data/train.jsonl \
      --output-dir "$OUT/checkpoints/$name" --variant affine_input_lm_head_plus_hidden_lora \
      --tie-affine-input-lm-head-adapters --affine-lm-head-bias --affine-rank 16 --affine-alpha 128 \
      --initial-affine-adapter "$init" --initial-hidden-lora-adapter "$init" \
      --hidden-lora-rank 8 --hidden-lora-alpha 16 --hidden-lora-dropout 0.05 \
      --max-seq-len 1024 --per-device-train-batch-size 8 --gradient-accumulation-steps 2 \
      --learning-rate 5e-5 --max-steps 300 --logging-steps 10 --bf16 --master-dtype fp32 --base-dtype auto \
      --save-strategy no --lr-scheduler-type cosine --warmup-ratio 0.03 --max-grad-norm 1.0 --seed "$seed" \
      "${extra[@]}" > "$OUT/logs/${name}.train.log" 2>&1 &
  done
done
wait
echo "$(date -Is) continuation l0/l100 screen training complete"
