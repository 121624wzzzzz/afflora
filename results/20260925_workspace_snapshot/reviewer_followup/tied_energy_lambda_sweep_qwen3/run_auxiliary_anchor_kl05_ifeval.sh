#!/usr/bin/env bash
set -euo pipefail
ROOT=/commondocument/wz/cross_encoder_workspace/im_exp/lora
PYTHON=/home/wz/anaconda3/envs/torch24/bin/python
OUT="$ROOT/reviewer_followup/tied_energy_lambda_sweep_qwen3"
cd "$ROOT"; mkdir -p "$OUT/ifeval_shards" "$OUT/ifeval_responses" "$OUT/ifeval_scores" "$OUT/logs"
boundaries=(0 68 136 204 271 339 407 475 541)
for seed in 43 44 45 46; do
  name="l100_auxconstraint10a0p1_anchorkl0p05_sd${seed}"
  run="$OUT/checkpoints/qwen3_06b_tied_forward_tau0p00625_${name}"
  echo "$(date -Is) $name IFEval"; pids=()
  for gpu in {0..7}; do
    start=${boundaries[$gpu]}; end=${boundaries[$((gpu + 1))]}
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" reviewer_followup/evaluate_ifeval.py \
      --run-dir "$run" --output "$OUT/ifeval_shards/${name}_${start}_${end}.jsonl" \
      --batch-size 8 --max-new-tokens 512 --start-index "$start" --end-index "$end" \
      > "$OUT/logs/${name}_${start}_${end}.ifeval.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  "$PYTHON" reviewer_followup/merge_ifeval_shards.py --shard-dir "$OUT/ifeval_shards" \
    --name-prefix 'l100_auxconstraint10a0p1_anchorkl0p05_' --seed "$seed" \
    --output "$OUT/ifeval_responses/${name}.jsonl" > "$OUT/logs/${name}.merge.log" 2>&1
  mkdir -p "$OUT/ifeval_scores/$name"
  NLTK_DATA=/tmp/ifeval_nltk PYTHONPATH=/tmp/ifeval_deps:/tmp/ifevalsrc5.dJyI4k "$PYTHON" \
    -m instruction_following_eval.evaluation_main --input_data "$OUT/ifeval_responses/${name}.jsonl" \
    --input_response_data "$OUT/ifeval_responses/${name}.jsonl" --output_dir "$OUT/ifeval_scores/$name" \
    > "$OUT/logs/${name}.score.log" 2>&1
done
echo "$(date -Is) auxiliary-anchor KL IFEval complete"
