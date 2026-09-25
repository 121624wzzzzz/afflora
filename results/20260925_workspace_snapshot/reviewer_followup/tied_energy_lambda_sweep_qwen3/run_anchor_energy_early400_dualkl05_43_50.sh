#!/usr/bin/env bash
# Independent replication of the selected early-only, dual-domain KL objective
# on pre-existing matched controls 43--50.  No result-dependent changes are
# made to the objective, data, schedule, or evaluation protocol.
set -euo pipefail
ROOT=/commondocument/wz/cross_encoder_workspace/im_exp/lora
PYTHON=/home/wz/anaconda3/envs/torch24/bin/python
OUT="$ROOT/reviewer_followup/tied_energy_lambda_sweep_qwen3"
cd "$ROOT"; export DS_IGNORE_CUDA_DETECTION=1
mkdir -p "$OUT/logs" "$OUT/checkpoints"
for gpu in {0..7}; do
  seed=$((43 + gpu))
  teacher="$OUT/checkpoints/qwen3_06b_tied_forward_tau0p00625_l0_auxconstraint10a0p1_sd${seed}"
  name="qwen3_06b_tied_forward_tau0p00625_l100early400_auxconstraint10a0p1_dualkl0p05_sd${seed}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" corrected_sft_experiment/train_corrected_sft.py \
    --model-path ../models/Qwen3-0.6B-Base --train-data corrected_sft_experiment/data/train.jsonl \
    --auxiliary-anchor-data "$OUT/ultrachat_explicit_constraints_len1024.jsonl" \
    --auxiliary-anchor-lambda 0.1 --auxiliary-anchor-samples 2278 --auxiliary-anchor-batch-size 2 \
    --anchor-sampling-seed 1729 --eval-data corrected_sft_experiment/data/dev.jsonl --eval-samples 1000 --eval-steps 250 \
    --reference-run-dir "$teacher" --reference-kl-lambda 0.05 --reference-kl-on-main-and-auxiliary-anchor \
    --output-dir "$OUT/checkpoints/$name" --variant affine_input_lm_head_plus_hidden_lora \
    --tie-affine-input-lm-head-adapters --affine-lm-head-bias --affine-rank 16 --affine-alpha 128 --affine-dropout 0 \
    --affine-energy-lambda 100 --affine-energy-tau 0.00625 --affine-energy-end-step 400 \
    --hidden-lora-rank 8 --hidden-lora-alpha 16 --hidden-lora-dropout 0.05 --max-seq-len 1024 \
    --per-device-train-batch-size 8 --gradient-accumulation-steps 2 --learning-rate 2e-4 --num-train-epochs 1 \
    --logging-steps 10 --bf16 --master-dtype fp32 --base-dtype auto --save-strategy no \
    --lr-scheduler-type cosine --warmup-ratio 0.03 --max-grad-norm 1.0 --seed "$seed" \
    > "$OUT/logs/l100early400_auxconstraint10a0p1_dualkl0p05_sd${seed}.train.log" 2>&1 &
done
wait
echo "$(date -Is) early-only dual-KL 43--50 complete"
