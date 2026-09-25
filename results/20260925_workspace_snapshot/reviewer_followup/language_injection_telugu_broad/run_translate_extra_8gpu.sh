#!/usr/bin/env bash
# Extend the frozen selection from 30k to 40k without retranslating its first 30k rows.
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
    --selection-file "$OUT/selection_manifest_40k.json" --selection-limit 40000 --selection-start 30000 \
    --shard-index "$gpu" --num-shards 8 --batch-size 128 \
    --output "$OUT/translated_shards/extra_${gpu}.jsonl" \
    > "$OUT/logs/translate_extra_gpu${gpu}.log" 2>&1 &
done
wait
echo "$(date -Is) extra Telugu translation shards complete"
