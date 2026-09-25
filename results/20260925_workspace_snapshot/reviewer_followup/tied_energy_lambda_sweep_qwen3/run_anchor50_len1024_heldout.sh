#!/usr/bin/env bash
# Full 1,000-row corrected-SFT held-out CE for both arms of the anchor screen.
set -euo pipefail

ROOT=/commondocument/wz/cross_encoder_workspace/im_exp/lora
PYTHON=/home/wz/anaconda3/envs/torch24/bin/python
OUT="$ROOT/reviewer_followup/tied_energy_lambda_sweep_qwen3"
cd "$ROOT"
mkdir -p "$OUT/heldout_shards" "$OUT/logs"
tb=(0 125 250 375 500 625 750 875 1000)

for seed in 43 44 45 46; do
  for tag in l0 l100; do
    name="${tag}_anchor50_len1024_sd${seed}"
    run="$OUT/checkpoints/qwen3_06b_tied_forward_tau0p00625_${name}"
    echo "$(date -Is) $name heldout"
    pids=()
    for gpu in {0..7}; do
      start=${tb[$gpu]}; end=${tb[$((gpu + 1))]}
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" corrected_sft_experiment/evaluate_corrected_sft.py \
        --run-dir "$run" --data corrected_sft_experiment/data/test.jsonl \
        --output "$OUT/heldout_shards/${name}_${start}_${end}.json" \
        --batch-size 8 --max-seq-len 1024 --start-index "$start" --end-index "$end" \
        > "$OUT/logs/${name}_${start}_${end}.heldout.log" 2>&1 &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
  done
done
echo "$(date -Is) anchor50-len1024 heldout complete"
