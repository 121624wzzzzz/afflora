#!/usr/bin/env bash
# Continue only after all eight translation shards are complete and well formed.
set -euo pipefail

ROOT=/commondocument/wz/cross_encoder_workspace/im_exp/lora
PYTHON=/home/wz/anaconda3/envs/torch24/bin/python
OUT="$ROOT/reviewer_followup/language_injection_telugu_broad"
SHARDS="$OUT/translated_shards"
TEST="$ROOT/reviewer_followup/language_injection_telugu/data_qwen25_len2048/test.jsonl"
EVAL="$ROOT/reviewer_followup/language_injection_telugu/evaluate_belebele_mc.py"

cd "$ROOT"
mkdir -p "$OUT/logs" "$OUT/belebele_scores"
while true; do
  ready=1
  for shard in {0..7}; do
    file="$SHARDS/part_${shard}.jsonl"
    if [[ ! -f "$file" ]] || [[ $(wc -l < "$file") -ne 3750 ]]; then
      ready=0
    fi
    extra="$SHARDS/extra_${shard}.jsonl"
    if [[ ! -f "$extra" ]] || [[ $(wc -l < "$extra") -ne 1250 ]]; then
      ready=0
    fi
  done
  if [[ "$ready" -eq 1 ]]; then
    break
  fi
  if rg -q 'Traceback|RuntimeError|CUDA out of memory' "$OUT"/logs/translate_gpu*.log "$OUT"/logs/translate_extra_gpu*.log 2>/dev/null; then
    echo "Translation error detected; refusing to advance." >&2
    exit 1
  fi
  sleep 60
done

"$PYTHON" "$OUT/build_sft_data.py" --shards-dir "$SHARDS" --tokenizer ../models/Qwen2.5-1.5B-Base --output-dir "$OUT/data_qwen25_len2048" --target-rows 22780 --max-seq-len 2048 > "$OUT/logs/build_sft_data.log" 2>&1

bash "$OUT/run_controls59_66.sh" > "$OUT/logs/controls_launcher.log" 2>&1
bash "$OUT/run_candidates59_66.sh" > "$OUT/logs/candidates_launcher.log" 2>&1

for kind in control candidate; do
  for gpu in {0..7}; do
    seed=$((59 + gpu))
    if [[ "$kind" == control ]]; then
      name="qwen25_15b_broadtelugu_tied_l0_anchor0p1_sd${seed}"
    else
      name="qwen25_15b_broadtelugu_tied_l100early400_anchor0p1_dualkl0p05_sd${seed}"
    fi
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$EVAL" --run-dir "$OUT/checkpoints/$name" --data "$TEST" --output "$OUT/belebele_scores/$name.json" --batch-size 32 > "$OUT/logs/$name.belebele.log" 2>&1 &
  done
  wait
done

"$PYTHON" "$OUT/summarize_belebele.py" --scores-dir "$OUT/belebele_scores" --output "$OUT/belebele_summary.json" > "$OUT/logs/summarize_belebele.log" 2>&1
echo "$(date -Is) broad Telugu pipeline complete"
