#!/usr/bin/env bash
# Restartable eight-GPU IFEval for the six frozen fixed-hidden endpoints.
#
# IFEval is evaluation-only here: the endpoint set is hard-coded to paired
# hidden seeds 42/43/44 with A-LoRA r50/s16 and Vocab-LoRA r1/s32.
set -euo pipefail

ROOT=/commondocument/wz/cross_encoder_workspace/im_exp/lora
PYTHON_BIN=${PYTHON_BIN:-/home/wz/anaconda3/envs/torch24/bin/python}
EXP="$ROOT/reviewer_followup/fixed_hidden_boundary_fp32_qwen25"
CHECKPOINT_ROOT="$EXP/checkpoints"
OUTPUT_ROOT=${OUTPUT_ROOT:-"$EXP/ifeval_main"}
IFEVAL_PYTHONPATH=${IFEVAL_PYTHONPATH:-/tmp/ifeval_deps:/tmp/ifevalsrc5.dJyI4k}
IFEVAL_NLTK_DATA=${IFEVAL_NLTK_DATA:-/tmp/ifeval_nltk}
GENERATOR="$ROOT/reviewer_followup/evaluate_fixed_hidden_boundary_ifeval.py"
MERGER="$ROOT/reviewer_followup/merge_fixed_hidden_boundary_ifeval_shards.py"
VALIDATOR="$ROOT/reviewer_followup/validate_fixed_hidden_boundary_ifeval.py"
SUMMARIZER="$ROOT/reviewer_followup/summarize_fixed_hidden_boundary_fp32_qwen25_ifeval.py"

BOUNDS=(0 68 136 204 271 339 407 475 541)
BATCH_SIZE=8
MAX_NEW_TOKENS=512
gpus_csv=0,1,2,3,4,5,6,7
validate_only=0

usage() {
  cat <<'EOF'
Usage:
  reviewer_followup/run_fixed_hidden_boundary_fp32_qwen25_ifeval.sh [OPTIONS]

Options:
  --gpus CSV       Exactly eight distinct physical GPU IDs
                   (default: 0,1,2,3,4,5,6,7).
  --validate-only  Launch nothing; require and validate all final artifacts,
                   then rebuild the statistical summary.
  -h, --help       Show this message.

Environment:
  PYTHON_BIN, OUTPUT_ROOT, IFEVAL_PYTHONPATH, IFEVAL_NLTK_DATA

Frozen run set:
  hsd42/43/44 x {A-LoRA r50/s16, Vocab-LoRA r1/s32}.

Protocol:
  541 canonical google/IFEval prompts; native chat template; greedy decoding;
  max_new_tokens=512; batch size 8 per shard; exact eight-way boundaries
  [0,68,136,204,271,339,407,475,541]. IFEval never selects an endpoint.

Restart contract:
  A shard is reused only if the dedicated validator confirms its checkpoint
  binding, exact canonical interval, generation metadata, and immutable
  protocol manifest. Merge is rebuilt. Official scoring is reused only when
  both 541-row strict/loose files pass content validation.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

while (($# > 0)); do
  case "$1" in
    --gpus)
      (($# >= 2)) || die "--gpus requires a value"
      gpus_csv=$2
      shift 2
      ;;
    --validate-only)
      validate_only=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ -x "$PYTHON_BIN" ]] || die "Python executable not found: $PYTHON_BIN"
for required in "$GENERATOR" "$MERGER" "$VALIDATOR" "$SUMMARIZER"; do
  [[ -f "$required" ]] || die "missing implementation: $required"
done
command -v flock >/dev/null || die "flock is required"

IFS=',' read -r -a gpus <<<"$gpus_csv"
((${#gpus[@]} == 8)) || die "--gpus must contain exactly eight IDs"
declare -A seen_gpus=()
for gpu in "${gpus[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || die "invalid GPU ID: $gpu"
  [[ -z "${seen_gpus[$gpu]+x}" ]] || die "duplicate GPU ID: $gpu"
  seen_gpus[$gpu]=1
done
if ((!validate_only)); then
  command -v nvidia-smi >/dev/null || die "nvidia-smi is required"
  mapfile -t installed_gpus < <(
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits
  )
  for gpu in "${gpus[@]}"; do
    installed=0
    for candidate in "${installed_gpus[@]}"; do
      if [[ "$gpu" == "$candidate" ]]; then
        installed=1
        break
      fi
    done
    ((installed == 1)) || die "GPU index is not installed: $gpu"
  done
fi

OUTPUT_ROOT=$("$PYTHON_BIN" -c \
  'import sys; from pathlib import Path; print(Path(sys.argv[1]).resolve())' \
  "$OUTPUT_ROOT")
PROTOCOL_MANIFEST="$OUTPUT_ROOT/protocol_manifest.json"

export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false

cd "$ROOT"
"$PYTHON_BIN" -m py_compile "$GENERATOR" "$MERGER" "$VALIDATOR" "$SUMMARIZER"
SCORER_PATH=$(
  PYTHONPATH="$IFEVAL_PYTHONPATH" NLTK_DATA="$IFEVAL_NLTK_DATA" \
    "$PYTHON_BIN" -c \
    'import inspect; import instruction_following_eval.evaluation_main as m; print(inspect.getsourcefile(m))'
)
[[ -f "$SCORER_PATH" ]] || die "cannot resolve official scorer source"

run_names=()
for seed in 42 43 44; do
  run_names+=(
    "qwen25_15b_fhfp32_hsd${seed}_alora_r50_s16_bsd${seed}"
    "qwen25_15b_fhfp32_hsd${seed}_vocab_lora_r1_s32_bsd${seed}"
  )
done

# Validate all six endpoints before creating output state or spending GPU time.
for run_name in "${run_names[@]}"; do
  run_dir="$CHECKPOINT_ROOT/$run_name"
  echo "$(date --iso-8601=seconds) CHECK checkpoint=$run_name"
  "$PYTHON_BIN" "$VALIDATOR" checkpoint --run-dir "$run_dir" >/dev/null
done

exec {launcher_lock_fd}>"$EXP/ifeval_main.launch.lock"
flock -n "$launcher_lock_fd" || die "another fixed-hidden IFEval launcher is active"

if ((validate_only)); then
  "$PYTHON_BIN" "$VALIDATOR" protocol \
    --path "$PROTOCOL_MANIFEST" \
    --scorer-path "$SCORER_PATH"
else
  "$PYTHON_BIN" "$VALIDATOR" protocol \
    --path "$PROTOCOL_MANIFEST" \
    --scorer-path "$SCORER_PATH" \
    --create
fi

mkdir -p \
  "$OUTPUT_ROOT/shards" \
  "$OUTPUT_ROOT/responses" \
  "$OUTPUT_ROOT/scores" \
  "$OUTPUT_ROOT/logs"

active_pids=()
cleanup_children() {
  if ((${#active_pids[@]} > 0)); then
    kill "${active_pids[@]}" >/dev/null 2>&1 || true
    wait "${active_pids[@]}" >/dev/null 2>&1 || true
  fi
}
trap cleanup_children EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for run_name in "${run_names[@]}"; do
  run_dir="$CHECKPOINT_ROOT/$run_name"
  pids=()
  labels=()

  for shard_index in 0 1 2 3 4 5 6 7; do
    start=${BOUNDS[$shard_index]}
    end=${BOUNDS[$((shard_index + 1))]}
    gpu=${gpus[$shard_index]}
    shard="$OUTPUT_ROOT/shards/${run_name}_${start}_${end}.jsonl"
    log="$OUTPUT_ROOT/logs/${run_name}_${start}_${end}.ifeval.log"

    if "$PYTHON_BIN" "$VALIDATOR" shard \
      --path "$shard" \
      --run-dir "$run_dir" \
      --protocol-manifest "$PROTOCOL_MANIFEST" \
      --start-index "$start" \
      --end-index "$end" \
      >/dev/null 2>&1; then
      echo "$(date --iso-8601=seconds) SKIP valid shard=$run_name[$start,$end)"
      continue
    fi
    if ((validate_only)); then
      die "missing or invalid shard in --validate-only mode: $shard"
    fi

    echo "$(date --iso-8601=seconds) START shard=$run_name[$start,$end) gpu=$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$GENERATOR" \
      --run-dir "$run_dir" \
      --output "$shard" \
      --batch-size "$BATCH_SIZE" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --start-index "$start" \
      --end-index "$end" \
      --affine-ablation none \
      >"$log" 2>&1 &
    pids+=("$!")
    labels+=("$run_name[$start,$end)")
  done

  active_pids=("${pids[@]}")
  failed=0
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
      echo "$(date --iso-8601=seconds) DONE shard=${labels[$index]}"
    else
      status=$?
      echo \
        "$(date --iso-8601=seconds) FAILED status=$status shard=${labels[$index]}" \
        >&2
      failed=1
    fi
  done
  active_pids=()
  ((failed == 0)) || exit 1

  for shard_index in 0 1 2 3 4 5 6 7; do
    start=${BOUNDS[$shard_index]}
    end=${BOUNDS[$((shard_index + 1))]}
    shard="$OUTPUT_ROOT/shards/${run_name}_${start}_${end}.jsonl"
    "$PYTHON_BIN" "$VALIDATOR" shard \
      --path "$shard" \
      --run-dir "$run_dir" \
      --protocol-manifest "$PROTOCOL_MANIFEST" \
      --start-index "$start" \
      --end-index "$end"
  done

  response="$OUTPUT_ROOT/responses/${run_name}.jsonl"
  merge_log="$OUTPUT_ROOT/logs/${run_name}.merge.log"
  "$PYTHON_BIN" "$MERGER" \
    --shard-dir "$OUTPUT_ROOT/shards" \
    --run-dir "$run_dir" \
    --protocol-manifest "$PROTOCOL_MANIFEST" \
    --output "$response" \
    >"$merge_log" 2>&1
  "$PYTHON_BIN" "$VALIDATOR" merged \
    --path "$response" \
    --run-dir "$run_dir" \
    --protocol-manifest "$PROTOCOL_MANIFEST"

  score_dir="$OUTPUT_ROOT/scores/$run_name"
  score_log="$OUTPUT_ROOT/logs/${run_name}.score.log"
  if "$PYTHON_BIN" "$VALIDATOR" scores \
    --response-path "$response" \
    --score-dir "$score_dir" \
    --run-dir "$run_dir" \
    --protocol-manifest "$PROTOCOL_MANIFEST" \
    >/dev/null 2>&1; then
    echo "$(date --iso-8601=seconds) SKIP valid official scores=$run_name"
  elif ((validate_only)); then
    die "missing or invalid official scores in --validate-only mode: $score_dir"
  else
    echo "$(date --iso-8601=seconds) START official scoring=$run_name"
    mkdir -p "$score_dir"
    NLTK_DATA="$IFEVAL_NLTK_DATA" PYTHONPATH="$IFEVAL_PYTHONPATH" \
      "$PYTHON_BIN" -m instruction_following_eval.evaluation_main \
      --input_data "$response" \
      --input_response_data "$response" \
      --output_dir "$score_dir" \
      >"$score_log" 2>&1
    "$PYTHON_BIN" "$VALIDATOR" scores \
      --response-path "$response" \
      --score-dir "$score_dir" \
      --run-dir "$run_dir" \
      --protocol-manifest "$PROTOCOL_MANIFEST"
    echo "$(date --iso-8601=seconds) DONE official scoring=$run_name"
  fi
done

"$PYTHON_BIN" "$SUMMARIZER" \
  --experiment-dir "$EXP" \
  --ifeval-dir "$OUTPUT_ROOT"
echo "$(date --iso-8601=seconds) IFEVAL_COMPLETE runs=${#run_names[@]}"
