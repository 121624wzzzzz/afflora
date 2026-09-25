#!/usr/bin/env bash
# Pre-specified intermediate-energy replication: λ=30 lies between the
# behavior-neutral λ=10 and CE-effective λ=100 regimes, with the selected cap.
set -euo pipefail

ROOT=/commondocument/wz/cross_encoder_workspace/im_exp/lora
PYTHON=/home/wz/anaconda3/envs/torch24/bin/python
OUT="$ROOT/reviewer_followup/tied_energy_lambda_sweep_qwen3"
cd "$ROOT"
export DS_IGNORE_CUDA_DETECTION=1

# Seed 42 was used for cap selection; use independently matched 43--50 only.
seeds=(43 44 45 46 47 48 49 50)
for gpu in {0..7}; do
  seed=${seeds[$gpu]}
  name="qwen3_06b_tied_forward_tau0p00625_l30_biascap0p0016_corrected_sd${seed}"
  run="$OUT/checkpoints/$name"
  echo "$(date -Is) starting $name on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" corrected_sft_experiment/train_corrected_sft.py \
    --model-path ../models/Qwen3-0.6B-Base --train-data corrected_sft_experiment/data/train.jsonl \
    --eval-data corrected_sft_experiment/data/dev.jsonl --eval-samples 1000 --eval-steps 250 --output-dir "$run" \
    --variant affine_input_lm_head_plus_hidden_lora --tie-affine-input-lm-head-adapters --affine-lm-head-bias \
    --affine-rank 16 --affine-alpha 128 --affine-dropout 0 --affine-energy-lambda 30 --affine-energy-tau 0.00625 \
    --affine-bias-energy-lambda 100 --affine-bias-energy-tau 0.0016 \
    --hidden-lora-rank 8 --hidden-lora-alpha 16 --hidden-lora-dropout 0.05 --max-seq-len 1024 \
    --per-device-train-batch-size 8 --gradient-accumulation-steps 2 --learning-rate 2e-4 --num-train-epochs 1 \
    --logging-steps 10 --bf16 --master-dtype fp32 --base-dtype auto --save-strategy steps --save-steps 250 --save-total-limit 6 \
    --lr-scheduler-type cosine --warmup-ratio 0.03 --max-grad-norm 1.0 --seed "$seed" \
    > "$OUT/logs/l30_biascap0p0016_corrected_sd${seed}.train.log" 2>&1 &
done
wait
echo "$(date -Is) l30 bias-cap trainings complete"
