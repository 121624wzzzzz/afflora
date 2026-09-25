#!/usr/bin/env bash
set -euo pipefail
ROOT=/commondocument/wz/cross_encoder_workspace/im_exp/lora
PYTHON=/home/wz/anaconda3/envs/torch24/bin/python
OUT="$ROOT/reviewer_followup/language_injection_telugu_broad"
TEST="$ROOT/reviewer_followup/language_injection_telugu/data_qwen25_len2048/test.jsonl"
EVAL="$ROOT/reviewer_followup/language_injection_telugu/evaluate_belebele_mc.py"
mkdir -p "$OUT/belebele_scores" "$OUT/logs"
for offset in {0..3}; do
  seed=$((59 + offset))
  energy="qwen25_15b_broadtelugu_energyonly_l100early400_anchor0p1_sd${seed}"
  kl="qwen25_15b_broadtelugu_klonly_anchor0p1_dualkl0p05_sd${seed}"
  CUDA_VISIBLE_DEVICES="$offset" "$PYTHON" "$EVAL" --run-dir "$OUT/checkpoints/$energy" --data "$TEST" --output "$OUT/belebele_scores/$energy.json" --batch-size 32 > "$OUT/logs/$energy.belebele.log" 2>&1 &
  CUDA_VISIBLE_DEVICES="$((offset + 4))" "$PYTHON" "$EVAL" --run-dir "$OUT/checkpoints/$kl" --data "$TEST" --output "$OUT/belebele_scores/$kl.json" --batch-size 32 > "$OUT/logs/$kl.belebele.log" 2>&1 &
done
wait
echo "$(date -Is) broad Telugu ablation evaluation complete"
