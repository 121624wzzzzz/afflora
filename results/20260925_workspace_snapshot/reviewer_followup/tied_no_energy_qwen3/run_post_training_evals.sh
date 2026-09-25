#!/usr/bin/env bash
# Run only after all three paired no-energy tied checkpoints are complete.
set -euo pipefail

ROOT=/commondocument/wz/cross_encoder_workspace/im_exp/lora
PYTHON=/home/wz/anaconda3/envs/torch24/bin/python
OUT="$ROOT/reviewer_followup/tied_no_energy_qwen3"
cd "$ROOT"

if (( $# )); then
  seeds=("$@")
else
  seeds=(42 43 44)
fi
test_bounds=(0 125 250 375 500 625 750 875 1000)
ifeval_bounds=(0 68 136 204 271 339 407 475 541)

for seed in "${seeds[@]}"; do
  run="$OUT/checkpoints/qwen3_06b_tied_noenergy_sd${seed}"
  until [[ -f "$run/adapter_model.safetensors" && -f "$run/affine_vocab_adapter.safetensors" ]]; do
    echo "$(date -Is) waiting for seed $seed checkpoint"
    sleep 60
  done
done

for seed in "${seeds[@]}"; do
  run="$OUT/checkpoints/qwen3_06b_tied_noenergy_sd${seed}"
  echo "$(date -Is) starting 8-GPU held-out CE for seed $seed"
  pids=()
  for gpu in {0..7}; do
    start=${test_bounds[$gpu]}
    end=${test_bounds[$((gpu + 1))]}
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" corrected_sft_experiment/evaluate_corrected_sft.py \
      --run-dir "$run" --data corrected_sft_experiment/data/test.jsonl \
      --output "$OUT/test_shards/qwen3_06b_tied_noenergy_sd${seed}_${start}_${end}.json" \
      --batch-size 8 --max-seq-len 1024 --start-index "$start" --end-index "$end" \
      > "$OUT/test_logs/qwen3_06b_tied_noenergy_sd${seed}_${start}_${end}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done

  echo "$(date -Is) starting 8-GPU matched-512 IFEval for seed $seed"
  pids=()
  for gpu in {0..7}; do
    start=${ifeval_bounds[$gpu]}
    end=${ifeval_bounds[$((gpu + 1))]}
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" reviewer_followup/evaluate_ifeval.py \
      --run-dir "$run" \
      --output "$OUT/ifeval_shards/qwen3_06b_tied_noenergy_sd${seed}_${start}_${end}.jsonl" \
      --batch-size 8 --max-new-tokens 512 --start-index "$start" --end-index "$end" \
      > "$OUT/ifeval_logs/qwen3_06b_tied_noenergy_sd${seed}_${start}_${end}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  "$PYTHON" reviewer_followup/merge_ifeval_shards.py \
    --shard-dir "$OUT/ifeval_shards" --name-prefix qwen3_06b_tied_noenergy_ \
    --seed "$seed" --output "$OUT/ifeval_responses/qwen3_06b_tied_noenergy_sd${seed}.jsonl" \
    > "$OUT/ifeval_logs/qwen3_06b_tied_noenergy_sd${seed}.merge.log" 2>&1
  mkdir -p "$OUT/ifeval_scores/qwen3_06b_tied_noenergy_sd${seed}"
  NLTK_DATA=/tmp/ifeval_nltk PYTHONPATH=/tmp/ifeval_deps:/tmp/ifevalsrc5.dJyI4k "$PYTHON" \
    -m instruction_following_eval.evaluation_main \
    --input_data "$OUT/ifeval_responses/qwen3_06b_tied_noenergy_sd${seed}.jsonl" \
    --input_response_data "$OUT/ifeval_responses/qwen3_06b_tied_noenergy_sd${seed}.jsonl" \
    --output_dir "$OUT/ifeval_scores/qwen3_06b_tied_noenergy_sd${seed}" \
    > "$OUT/ifeval_logs/qwen3_06b_tied_noenergy_sd${seed}.score.log" 2>&1
done

echo "$(date -Is) all no-energy tied evaluations completed"
