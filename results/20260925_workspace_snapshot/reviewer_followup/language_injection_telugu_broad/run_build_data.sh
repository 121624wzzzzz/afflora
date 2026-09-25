#!/usr/bin/env bash
# Run token-budget filtering as a detached child so it survives launcher teardown.
set -euo pipefail
ROOT=/commondocument/wz/cross_encoder_workspace/im_exp/lora
PYTHON=/home/wz/anaconda3/envs/torch24/bin/python
OUT="$ROOT/reviewer_followup/language_injection_telugu_broad"
cd "$ROOT"
"$PYTHON" "$OUT/build_sft_data.py" --shards-dir "$OUT/translated_shards" \
  --tokenizer ../models/Qwen2.5-1.5B-Base --output-dir "$OUT/data_qwen25_len2048" \
  --target-rows 22780 --max-seq-len 2048 > "$OUT/logs/build_sft_data.log" 2>&1 &
pid=$!
wait "$pid"
status=$?
printf 'build_sft_data exit=%s\n' "$status" >> "$OUT/logs/build_sft_data.log"
exit "$status"
