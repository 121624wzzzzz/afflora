#!/usr/bin/env bash
# Exact early-energy + dual-domain-KL treatment, paired with the broad controls.
set -euo pipefail

ROOT=/commondocument/wz/cross_encoder_workspace/im_exp/lora
PYTHON=/home/wz/anaconda3/envs/torch24/bin/python
OUT="$ROOT/reviewer_followup/language_injection_telugu_broad"
DATA="$OUT/data_qwen25_len2048"
ANCHOR="$ROOT/reviewer_followup/tied_energy_lambda_sweep_qwen3/qwen25_crosscheck/constraint_anchor_qwen25_len1024/train.jsonl"

cd "$ROOT"
export DS_IGNORE_CUDA_DETECTION=1
mkdir -p "$OUT/checkpoints" "$OUT/logs"
for gpu in {0..7}; do
  seed=$((59 + gpu))
  control="qwen25_15b_broadtelugu_tied_l0_anchor0p1_sd${seed}"
  name="qwen25_15b_broadtelugu_tied_l100early400_anchor0p1_dualkl0p05_sd${seed}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" corrected_sft_experiment/train_corrected_sft.py \
    --model-path ../models/Qwen2.5-1.5B-Base --train-data "$DATA/train.jsonl" \
    --auxiliary-anchor-data "$ANCHOR" --auxiliary-anchor-lambda .1 --auxiliary-anchor-samples 2278 --auxiliary-anchor-batch-size 1 --anchor-sampling-seed 1729 \
    --reference-run-dir "$OUT/checkpoints/$control" --reference-kl-lambda .05 --reference-kl-on-main-and-auxiliary-anchor \
    --output-dir "$OUT/checkpoints/$name" --variant affine_input_lm_head_plus_hidden_lora \
    --tie-affine-input-lm-head-adapters --affine-lm-head-bias --affine-rank 16 --affine-alpha 128 --affine-dropout 0 \
    --affine-energy-lambda 100 --affine-energy-tau .00625 --affine-energy-end-step 400 \
    --hidden-lora-rank 8 --hidden-lora-alpha 16 --hidden-lora-dropout .05 --max-seq-len 2048 \
    --per-device-train-batch-size 2 --gradient-accumulation-steps 8 --learning-rate 2e-4 --num-train-epochs 1 \
    --logging-steps 10 --bf16 --master-dtype fp32 --base-dtype auto --save-strategy no --lr-scheduler-type cosine --warmup-ratio .03 --max-grad-norm 1.0 --seed "$seed" \
    > "$OUT/logs/${name}.train.log" 2>&1 &
done
wait
echo "$(date -Is) broad Telugu candidates 59--66 complete"
