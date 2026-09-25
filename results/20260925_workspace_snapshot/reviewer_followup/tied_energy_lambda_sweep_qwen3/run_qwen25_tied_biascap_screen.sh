#!/usr/bin/env bash
# Cross-model paired screen: Qwen2.5-1.5B, strict tied lambda=0 vs lambda=100+bias cap.
set -euo pipefail
ROOT=/commondocument/wz/cross_encoder_workspace/im_exp/lora
PYTHON=/home/wz/anaconda3/envs/torch24/bin/python
OUT="$ROOT/reviewer_followup/tied_energy_lambda_sweep_qwen3/qwen25_crosscheck"
cd "$ROOT"
export DS_IGNORE_CUDA_DETECTION=1
mkdir -p "$OUT/checkpoints" "$OUT/logs"
for seed in 42 43 44; do
  for kind in l0 cap; do
    if [[ "$kind" == l0 ]]; then gpu=$((seed - 42)); lambda=0; cap_lambda=0; cap_tau=0.0016; suffix='l0'; else gpu=$((seed - 39)); lambda=100; cap_lambda=100; cap_tau=0.0016; suffix='l100_biascap0p0016'; fi
    name="qwen25_15b_tied_tau0p00625_${suffix}_sd${seed}"
    run="$OUT/checkpoints/$name"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" corrected_sft_experiment/train_corrected_sft.py \
      --model-path ../models/Qwen2.5-1.5B-Base --train-data corrected_sft_experiment/data/train.jsonl \
      --eval-data corrected_sft_experiment/data/dev.jsonl --eval-samples 1000 --eval-steps 250 --output-dir "$run" \
      --variant affine_input_lm_head_plus_hidden_lora --tie-affine-input-lm-head-adapters --affine-lm-head-bias \
      --affine-rank 16 --affine-alpha 128 --affine-dropout 0 --affine-energy-lambda "$lambda" --affine-energy-tau 0.00625 \
      --affine-bias-energy-lambda "$cap_lambda" --affine-bias-energy-tau "$cap_tau" \
      --hidden-lora-rank 8 --hidden-lora-alpha 16 --hidden-lora-dropout 0.05 --max-seq-len 1024 \
      --per-device-train-batch-size 8 --gradient-accumulation-steps 2 --learning-rate 2e-4 --num-train-epochs 1 \
      --logging-steps 10 --bf16 --master-dtype fp32 --base-dtype auto --save-strategy steps --save-steps 250 --save-total-limit 6 \
      --lr-scheduler-type cosine --warmup-ratio 0.03 --max-grad-norm 1.0 --seed "$seed" \
      > "$OUT/logs/${name}.train.log" 2>&1 &
  done
done
wait
echo "$(date -Is) Qwen2.5 tied crosscheck training complete"
