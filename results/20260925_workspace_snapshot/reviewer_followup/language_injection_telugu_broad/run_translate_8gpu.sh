#!/usr/bin/env bash
# Translate the frozen 30k UltraChat selection to Telugu on all eight GPUs.
set -euo pipefail

ROOT=/commondocument/wz/cross_encoder_workspace/im_exp/lora
PYTHON=/home/wz/anaconda3/envs/torch24/bin/python
OUT="$ROOT/reviewer_followup/language_injection_telugu_broad"
SOURCE="$ROOT/corrected_sft_experiment/data_ultrachat_100k_len1024/train.jsonl"
MODEL=/home/wz/.cache/modelscope/hub/models/facebook/nllb-200-distilled-600M

cd "$ROOT"
mkdir -p "$OUT/translated_shards" "$OUT/logs"
for gpu in {0..7}; do
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$OUT/prepare_translate_ultrachat.py" \
    --source "$SOURCE" --model "$MODEL" \
    --selection-file "$OUT/selection_manifest.json" --selection-limit 30000 \
    --shard-index "$gpu" --num-shards 8 --batch-size 128 \
    --output "$OUT/translated_shards/part_${gpu}.jsonl" \
    > "$OUT/logs/translate_gpu${gpu}.log" 2>&1 &
done
wait
echo "$(date -Is) all Telugu translation shards complete"
