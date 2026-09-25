#!/usr/bin/env bash
set -euo pipefail
ROOT=/commondocument/wz/cross_encoder_workspace/im_exp/lora
PYTHON=/home/wz/anaconda3/envs/torch24/bin/python
OUT="$ROOT/reviewer_followup/tied_energy_lambda_sweep_qwen3"
cd "$ROOT"; mkdir -p "$OUT"/{heldout_shards,logs}
tb=(0 125 250 375 500 625 750 875 1000)
for tag in kl0p03 kl0p1; do for seed in 43 44 45 46; do
  name="l100_ref${tag}_sd${seed}"; run="$OUT/checkpoints/qwen3_06b_tied_forward_tau0p00625_${name}"; pids=(); echo "$(date -Is) $name heldout"
  for gpu in {0..7}; do start=${tb[$gpu]}; end=${tb[$((gpu+1))]}; CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" corrected_sft_experiment/evaluate_corrected_sft.py --run-dir "$run" --data corrected_sft_experiment/data/test.jsonl --output "$OUT/heldout_shards/${name}_${start}_${end}.json" --batch-size 8 --max-seq-len 1024 --start-index "$start" --end-index "$end" > "$OUT/logs/${name}_${start}_${end}.heldout.log" 2>&1 & pids+=("$!"); done
  for pid in "${pids[@]}"; do wait "$pid"; done
done; done
