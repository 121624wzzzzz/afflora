#!/usr/bin/env bash
# Pre-specified independent replication of the explicit-constraint 10% anchor result.
# This keeps the total number of SFT rows fixed and varies only seed / energy lambda.
set -euo pipefail
ROOT=/commondocument/wz/cross_encoder_workspace/im_exp/lora
PYTHON=/home/wz/anaconda3/envs/torch24/bin/python
OUT="$ROOT/reviewer_followup/tied_energy_lambda_sweep_qwen3"
cd "$ROOT"; export DS_IGNORE_CUDA_DETECTION=1
mkdir -p "$OUT/logs" "$OUT/checkpoints"
seeds=(47 48 49 50)
for i in {0..3}; do
  seed=${seeds[$i]}
  for energy in 0 100; do
    gpu=$i; tag=l0
    if (( energy > 0 )); then gpu=$((i + 4)); tag=l100; fi
    name="qwen3_06b_tied_forward_tau0p00625_${tag}_constraintanchor10_sd${seed}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" corrected_sft_experiment/train_corrected_sft.py \
      --model-path ../models/Qwen3-0.6B-Base --train-data corrected_sft_experiment/data/train.jsonl \
      --anchor-data "$OUT/ultrachat_explicit_constraints_len1024.jsonl" --anchor-fraction 0.1 --anchor-sampling-seed 1729 \
      --eval-data corrected_sft_experiment/data/dev.jsonl --eval-samples 1000 --eval-steps 250 \
      --output-dir "$OUT/checkpoints/$name" --variant affine_input_lm_head_plus_hidden_lora \
      --tie-affine-input-lm-head-adapters --affine-lm-head-bias --affine-rank 16 --affine-alpha 128 --affine-dropout 0 \
      --affine-energy-lambda "$energy" --affine-energy-tau 0.00625 \
      --hidden-lora-rank 8 --hidden-lora-alpha 16 --hidden-lora-dropout 0.05 --max-seq-len 1024 \
      --per-device-train-batch-size 8 --gradient-accumulation-steps 2 --learning-rate 2e-4 --num-train-epochs 1 \
      --logging-steps 10 --bf16 --master-dtype fp32 --base-dtype auto --save-strategy no \
      --lr-scheduler-type cosine --warmup-ratio 0.03 --max-grad-norm 1.0 --seed "$seed" \
      > "$OUT/logs/${tag}_constraintanchor10_sd${seed}.train.log" 2>&1 &
  done
done
wait
echo "$(date -Is) constraint-anchor10 replication 47--50 training complete"
