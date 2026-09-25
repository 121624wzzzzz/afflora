#!/usr/bin/env bash
# Eight matched replications of the selected joint energy/bias trust region.
set -euo pipefail

ROOT=/commondocument/wz/cross_encoder_workspace/im_exp/lora
PYTHON=/home/wz/anaconda3/envs/torch24/bin/python
OUT="$ROOT/reviewer_followup/tied_energy_lambda_sweep_qwen3"
cd "$ROOT"
export DS_IGNORE_CUDA_DETECTION=1

seeds=(43 44 45 46 47 48 49 50)
for gpu in {0..7}; do
  seed=${seeds[$gpu]}
  name="qwen3_06b_tied_forward_tau0p00625_l100_biascap0p0008_sd${seed}"
  run="$OUT/checkpoints/$name"
  if [[ -f "$run/run_args.json" && -f "$run/adapter_model.safetensors" && -f "$run/affine_vocab_adapter.safetensors" ]]; then
    echo "$(date -Is) $name already complete"
    continue
  fi
  echo "$(date -Is) starting $name on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/train_affine_vocab_lora.py \
    --model-path ../models/Qwen3-0.6B-Base \
    --train-data corrected_sft_experiment/data/train.jsonl \
    --eval-data corrected_sft_experiment/data/dev.jsonl --eval-samples 1000 --eval-steps 250 \
    --output-dir "$run" \
    --variant affine_input_lm_head_plus_hidden_lora \
    --tie-affine-input-lm-head-adapters --affine-lm-head-bias \
    --affine-rank 16 --affine-alpha 128 --affine-dropout 0 \
    --affine-energy-lambda 100 --affine-energy-tau 0.00625 \
    --affine-bias-energy-lambda 100 --affine-bias-energy-tau 0.0008 \
    --hidden-lora-rank 8 --hidden-lora-alpha 16 --hidden-lora-dropout 0.05 \
    --max-seq-len 1024 --per-device-train-batch-size 8 --gradient-accumulation-steps 2 \
    --learning-rate 2e-4 --num-train-epochs 1 --logging-steps 10 \
    --bf16 --master-dtype fp32 --base-dtype auto \
    --save-strategy steps --save-steps 250 --save-total-limit 6 \
    --lr-scheduler-type cosine --warmup-ratio 0.03 --max-grad-norm 1.0 --seed "$seed" \
    > "$OUT/logs/l100_biascap0p0008_sd${seed}.train.log" 2>&1 &
done
wait
echo "$(date -Is) all selected bias-cap replications completed"
