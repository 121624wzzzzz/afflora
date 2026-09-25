#!/usr/bin/env bash
# Evaluate the four independent completion seeds with the same eight-way
# held-out and IFEval shard layout used by the matched baselines.
set -euo pipefail
ROOT=/commondocument/wz/cross_encoder_workspace/im_exp/lora
PYTHON=/home/wz/anaconda3/envs/torch24/bin/python
OUT="$ROOT/reviewer_followup/tied_energy_lambda_sweep_qwen3"
cd "$ROOT"
mkdir -p "$OUT"/{heldout_shards,ifeval_shards,ifeval_responses,ifeval_scores,logs}
tb=(0 125 250 375 500 625 750 875 1000)
ib=(0 68 136 204 271 339 407 475 541)
for seed in 47 48 49 50; do
  name="l100_biaslr0p3_corrected_sd${seed}"
  run="$OUT/checkpoints/qwen3_06b_tied_forward_tau0p00625_l100_biaslr0p3_corrected_sd${seed}"
  echo "$(date -Is) $name heldout"
  pids=()
  for gpu in {0..7}; do
    start=${tb[$gpu]}; end=${tb[$((gpu+1))]}
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" corrected_sft_experiment/evaluate_corrected_sft.py \
      --run-dir "$run" --data corrected_sft_experiment/data/test.jsonl \
      --output "$OUT/heldout_shards/${name}_${start}_${end}.json" \
      --batch-size 8 --max-seq-len 1024 --start-index "$start" --end-index "$end" \
      > "$OUT/logs/${name}_${start}_${end}.heldout.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  echo "$(date -Is) $name ifeval"
  pids=()
  for gpu in {0..7}; do
    start=${ib[$gpu]}; end=${ib[$((gpu+1))]}
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" reviewer_followup/evaluate_ifeval.py \
      --run-dir "$run" --output "$OUT/ifeval_shards/${name}_${start}_${end}.jsonl" \
      --batch-size 8 --max-new-tokens 512 --start-index "$start" --end-index "$end" \
      > "$OUT/logs/${name}_${start}_${end}.ifeval.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  "$PYTHON" reviewer_followup/merge_ifeval_shards.py --shard-dir "$OUT/ifeval_shards" \
    --name-prefix 'l100_biaslr0p3_corrected_' --seed "$seed" \
    --output "$OUT/ifeval_responses/${name}.jsonl" > "$OUT/logs/${name}.merge.log" 2>&1
  mkdir -p "$OUT/ifeval_scores/$name"
  NLTK_DATA=/tmp/ifeval_nltk PYTHONPATH=/tmp/ifeval_deps:/tmp/ifevalsrc5.dJyI4k "$PYTHON" \
    -m instruction_following_eval.evaluation_main \
    --input_data "$OUT/ifeval_responses/${name}.jsonl" \
    --input_response_data "$OUT/ifeval_responses/${name}.jsonl" \
    --output_dir "$OUT/ifeval_scores/$name" > "$OUT/logs/${name}.score.log" 2>&1
done
