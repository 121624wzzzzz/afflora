#!/usr/bin/env bash
# Matched affine-only screen: isolate energy from the hidden-LoRA co-adaptation.
set -euo pipefail
ROOT=/commondocument/wz/cross_encoder_workspace/im_exp/lora
PYTHON=/home/wz/anaconda3/envs/torch24/bin/python
OUT="$ROOT/reviewer_followup/tied_energy_lambda_sweep_qwen3"
cd "$ROOT"; export DS_IGNORE_CUDA_DETECTION=1
seeds=(43 44 45 46)
for gpu in {0..7}; do
  pair=$((gpu / 2)); kind=$((gpu % 2)); seed=${seeds[$pair]}
  if [ "$kind" -eq 0 ]; then label=l0; lambda=0; else label=l100; lambda=100; fi
  # Affine-only strict tying contains shared adapter tensors.  The generic
  # Trainer checkpoint serializer cannot represent that sharing; final adapter
  # saving below is handled by train_corrected_sft.py, so disable native saves.
  name="qwen3_06b_tied_affineonly_tau0p00625_${label}_corrected_retry1_sd${seed}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" corrected_sft_experiment/train_corrected_sft.py \
    --model-path ../models/Qwen3-0.6B-Base --train-data corrected_sft_experiment/data/train.jsonl \
    --eval-data corrected_sft_experiment/data/dev.jsonl --eval-samples 1000 --eval-steps 250 \
    --output-dir "$OUT/checkpoints/$name" --variant affine_input_lm_head \
    --tie-affine-input-lm-head-adapters --affine-lm-head-bias --affine-rank 16 --affine-alpha 128 \
    --affine-dropout 0 --affine-energy-lambda "$lambda" --affine-energy-tau 0.00625 \
    --max-seq-len 1024 --per-device-train-batch-size 8 --gradient-accumulation-steps 2 \
    --learning-rate 2e-4 --num-train-epochs 1 --logging-steps 10 --bf16 --master-dtype fp32 \
    --base-dtype auto --save-strategy no \
    --lr-scheduler-type cosine --warmup-ratio 0.03 --max-grad-norm 1.0 --seed "$seed" \
    > "$OUT/logs/affineonly_${label}_sd${seed}.train.log" 2>&1 &
done
wait
echo "$(date -Is) affine-only matched screen training complete"
