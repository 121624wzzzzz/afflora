#!/usr/bin/env bash
# Restartable eight-GPU matched IFEval for fixed output-only equal-budget runs.
#
# This launcher never selects a configuration from IFEval.  Give it either
# explicit completed run directories, or the dev-selected A-LoRA/Vocab-LoRA
# functional scales, boundary-only LR multipliers, and a seed list.  It runs
# all 541 prompts using the native model chat template, greedy decoding, and
# max_new_tokens=512.
set -euo pipefail

ROOT=/commondocument/wz/cross_encoder_workspace/im_exp/lora
PYTHON_BIN=${PYTHON_BIN:-/home/wz/anaconda3/envs/torch24/bin/python}
EXP="$ROOT/reviewer_followup/output_only_equal_budget_qwen25"
OUTPUT_ROOT=${OUTPUT_ROOT:-"$EXP/ifeval_final"}
IFEVAL_PYTHONPATH=${IFEVAL_PYTHONPATH:-/tmp/ifeval_deps:/tmp/ifevalsrc5.dJyI4k}
IFEVAL_NLTK_DATA=${IFEVAL_NLTK_DATA:-/tmp/ifeval_nltk}
GENERATOR="$ROOT/reviewer_followup/evaluate_ifeval.py"
MERGER="$ROOT/reviewer_followup/merge_ifeval_shards.py"
VALIDATOR="$ROOT/reviewer_followup/validate_output_only_ifeval.py"

BOUNDS=(0 68 136 204 271 339 407 475 541)
EXPECTED_COUNT=541
BATCH_SIZE=8
MAX_NEW_TOKENS=512

run_dirs=()
run_dirs_file=
aff_scale=
vocab_scale=
aff_boundary_lr=
vocab_boundary_lr=
seeds_csv=
gpus_csv=0,1,2,3,4,5,6,7
validate_only=0

usage() {
  cat <<'EOF'
Usage:
  reviewer_followup/run_output_only_equal_budget_qwen25_ifeval.sh \
    --run-dir RUN_DIR [--run-dir RUN_DIR ...]

  reviewer_followup/run_output_only_equal_budget_qwen25_ifeval.sh \
    --run-dirs-file PATH

  reviewer_followup/run_output_only_equal_budget_qwen25_ifeval.sh \
    --aff-scale {1,2,4,8,16,32} --aff-boundary-lr {0.5,1,2} \
    --vocab-scale {1,2,4,8,16,32} --vocab-boundary-lr {0.5,1,2} \
    --seeds CSV

Options:
  --run-dir PATH       Completed fixed-config run directory; repeatable.
  --run-dirs-file PATH One run directory per line; blank/comment lines ignored.
  --aff-scale N        Dev-selected output A-LoRA rank-50 functional scale.
  --aff-boundary-lr N  A-LoRA boundary-only LR multiplier (default: 1).
  --vocab-scale N      Dev-selected output Vocab-LoRA rank-1 functional scale.
  --vocab-boundary-lr N
                       Vocab-LoRA boundary-only LR multiplier (default: 1).
  --seeds CSV          Seeds used to construct both selected-scale run names.
  --gpus CSV           Exactly eight physical GPU IDs (default: 0,1,2,3,4,5,6,7).
  --validate-only      Validate all expected artifacts; launch nothing.
  -h, --help           Show this message.

Environment overrides:
  PYTHON_BIN, OUTPUT_ROOT, IFEVAL_PYTHONPATH, IFEVAL_NLTK_DATA

Restart behavior:
  A shard is reused only after content-level validation against its exact
  canonical IFEval interval and checkpoint metadata. Missing/invalid shards
  are regenerated on their fixed GPU. Merge is always rebuilt and checked.
  Official scoring is skipped only when both 541-row score files validate.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

while (($# > 0)); do
  case "$1" in
    --run-dir)
      (($# >= 2)) || die "--run-dir requires a value"
      run_dirs+=("$2")
      shift 2
      ;;
    --run-dirs-file)
      (($# >= 2)) || die "--run-dirs-file requires a value"
      [[ -z "$run_dirs_file" ]] || die "--run-dirs-file may be specified only once"
      run_dirs_file=$2
      shift 2
      ;;
    --aff-scale)
      (($# >= 2)) || die "--aff-scale requires a value"
      aff_scale=$2
      shift 2
      ;;
    --vocab-scale)
      (($# >= 2)) || die "--vocab-scale requires a value"
      vocab_scale=$2
      shift 2
      ;;
    --aff-boundary-lr)
      (($# >= 2)) || die "--aff-boundary-lr requires a value"
      aff_boundary_lr=$2
      shift 2
      ;;
    --vocab-boundary-lr)
      (($# >= 2)) || die "--vocab-boundary-lr requires a value"
      vocab_boundary_lr=$2
      shift 2
      ;;
    --seeds)
      (($# >= 2)) || die "--seeds requires a value"
      seeds_csv=$2
      shift 2
      ;;
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
[[ -f "$GENERATOR" ]] || die "missing generator: $GENERATOR"
[[ -f "$MERGER" ]] || die "missing merger: $MERGER"
[[ -f "$VALIDATOR" ]] || die "missing validator: $VALIDATOR"
OUTPUT_ROOT=$("$PYTHON_BIN" -c \
  'import sys; from pathlib import Path; print(Path(sys.argv[1]).resolve())' \
  "$OUTPUT_ROOT")

explicit_mode=0
if ((${#run_dirs[@]} > 0)) || [[ -n "$run_dirs_file" ]]; then
  explicit_mode=1
fi
scale_mode=0
if [[
  -n "$aff_scale" || -n "$vocab_scale" ||
  -n "$aff_boundary_lr" || -n "$vocab_boundary_lr" ||
  -n "$seeds_csv"
]]; then
  scale_mode=1
fi
((explicit_mode + scale_mode == 1)) || die \
  "choose exactly one input mode: explicit run dirs, or selected scales plus seeds"

if ((explicit_mode)); then
  [[
    -z "$aff_scale" && -z "$vocab_scale" &&
    -z "$aff_boundary_lr" && -z "$vocab_boundary_lr" &&
    -z "$seeds_csv"
  ]] || die "explicit run dirs cannot be combined with scale/LR/seed options"
  if [[ -n "$run_dirs_file" ]]; then
    [[ -f "$run_dirs_file" ]] || die "run-dir list does not exist: $run_dirs_file"
    while IFS= read -r line || [[ -n "$line" ]]; do
      line=${line%$'\r'}
      [[ -n "$line" ]] || continue
      [[ "$line" =~ ^[[:space:]]*# ]] && continue
      run_dirs+=("$line")
    done <"$run_dirs_file"
  fi
else
  [[ "$aff_scale" =~ ^(1|2|4|8|16|32)$ ]] || die \
    "--aff-scale must be one of 1,2,4,8,16,32"
  [[ "$vocab_scale" =~ ^(1|2|4|8|16|32)$ ]] || die \
    "--vocab-scale must be one of 1,2,4,8,16,32"
  aff_boundary_lr=${aff_boundary_lr:-1}
  vocab_boundary_lr=${vocab_boundary_lr:-1}
  [[ "$aff_boundary_lr" =~ ^(0[.]5|1|2)$ ]] || die \
    "--aff-boundary-lr must be one of 0.5,1,2"
  [[ "$vocab_boundary_lr" =~ ^(0[.]5|1|2)$ ]] || die \
    "--vocab-boundary-lr must be one of 0.5,1,2"
  [[ -n "$seeds_csv" ]] || die "--seeds is required in selected-scale mode"

  case "$aff_boundary_lr" in
    1) aff_lr_tag= ;;
    0.5) aff_lr_tag=_blr0p5 ;;
    2) aff_lr_tag=_blr2 ;;
    *) die "unreachable A-LoRA boundary LR: $aff_boundary_lr" ;;
  esac
  case "$vocab_boundary_lr" in
    1) vocab_lr_tag= ;;
    0.5) vocab_lr_tag=_blr0p5 ;;
    2) vocab_lr_tag=_blr2 ;;
    *) die "unreachable Vocab-LoRA boundary LR: $vocab_boundary_lr" ;;
  esac

  IFS=',' read -r -a selected_seeds <<<"$seeds_csv"
  ((${#selected_seeds[@]} > 0)) || die "empty --seeds list"
  for seed in "${selected_seeds[@]}"; do
    [[ "$seed" =~ ^[0-9]+$ ]] || die "invalid seed in --seeds: $seed"
    run_dirs+=(
      "$EXP/checkpoints/qwen25_15b_out_aff_r50_s${aff_scale}${aff_lr_tag}_sd${seed}"
      "$EXP/checkpoints/qwen25_15b_out_vocab_r1_s${vocab_scale}${vocab_lr_tag}_sd${seed}"
    )
  done
fi

((${#run_dirs[@]} > 0)) || die "no run directories selected"

IFS=',' read -r -a gpus <<<"$gpus_csv"
((${#gpus[@]} == 8)) || die "--gpus must contain exactly eight IDs"
declare -A seen_gpus=()
for gpu in "${gpus[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || die "invalid GPU ID: $gpu"
  [[ -z "${seen_gpus[$gpu]+x}" ]] || die "duplicate GPU ID: $gpu"
  seen_gpus[$gpu]=1
done

# Normalize and de-duplicate now, before any generation starts.
normalized_runs=()
declare -A seen_runs=()
for run_dir in "${run_dirs[@]}"; do
  [[ -d "$run_dir" ]] || die "run directory does not exist: $run_dir"
  normalized=$("$PYTHON_BIN" -c \
    'import sys; from pathlib import Path; print(Path(sys.argv[1]).resolve())' \
    "$run_dir")
  [[ -z "${seen_runs[$normalized]+x}" ]] || continue
  seen_runs[$normalized]=1
  normalized_runs+=("$normalized")
done
run_dirs=("${normalized_runs[@]}")

mkdir -p \
  "$OUTPUT_ROOT/shards" \
  "$OUTPUT_ROOT/responses" \
  "$OUTPUT_ROOT/scores" \
  "$OUTPUT_ROOT/logs"

cd "$ROOT"

# Import the official scorer before spending GPU time. The validator itself is
# also compiled/imported here so a broken local environment fails immediately.
PYTHONPATH="$IFEVAL_PYTHONPATH" NLTK_DATA="$IFEVAL_NLTK_DATA" \
  "$PYTHON_BIN" -c \
  'import instruction_following_eval.evaluation_main' \
  >/dev/null
"$PYTHON_BIN" -m py_compile "$GENERATOR" "$MERGER" "$VALIDATOR"

for run_dir in "${run_dirs[@]}"; do
  run_name=$(basename "$run_dir")
  if [[ "$run_name" =~ _sd([0-9]+)$ ]]; then
    seed=${BASH_REMATCH[1]}
  else
    die "run name lacks required _sdSEED suffix: $run_name"
  fi

  echo "$(date --iso-8601=seconds) CHECK checkpoint=$run_name"
  "$PYTHON_BIN" "$VALIDATOR" checkpoint --run-dir "$run_dir"

  pids=()
  pid_labels=()
  for shard_index in 0 1 2 3 4 5 6 7; do
    start=${BOUNDS[$shard_index]}
    end=${BOUNDS[$((shard_index + 1))]}
    gpu=${gpus[$shard_index]}
    shard="$OUTPUT_ROOT/shards/${run_name}_${start}_${end}.jsonl"
    log="$OUTPUT_ROOT/logs/${run_name}_${start}_${end}.ifeval.log"

    if "$PYTHON_BIN" "$VALIDATOR" shard \
      --path "$shard" \
      --run-dir "$run_dir" \
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
      >"$log" 2>&1 &
    pids+=("$!")
    pid_labels+=("$run_name[$start,$end)")
  done

  failed=0
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
      echo "$(date --iso-8601=seconds) DONE shard=${pid_labels[$index]}"
    else
      status=$?
      echo "$(date --iso-8601=seconds) FAILED status=$status shard=${pid_labels[$index]}" >&2
      failed=1
    fi
  done
  ((failed == 0)) || exit 1

  # Recheck all eight exact intervals even if they predated this invocation.
  for shard_index in 0 1 2 3 4 5 6 7; do
    start=${BOUNDS[$shard_index]}
    end=${BOUNDS[$((shard_index + 1))]}
    shard="$OUTPUT_ROOT/shards/${run_name}_${start}_${end}.jsonl"
    "$PYTHON_BIN" "$VALIDATOR" shard \
      --path "$shard" \
      --run-dir "$run_dir" \
      --start-index "$start" \
      --end-index "$end"
  done

  response="$OUTPUT_ROOT/responses/${run_name}.jsonl"
  merge_log="$OUTPUT_ROOT/logs/${run_name}.merge.log"
  "$PYTHON_BIN" "$MERGER" \
    --shard-dir "$OUTPUT_ROOT/shards" \
    --name-prefix "${run_name}_" \
    --seed "$seed" \
    --expected-count "$EXPECTED_COUNT" \
    --output "$response" \
    >"$merge_log" 2>&1
  "$PYTHON_BIN" "$VALIDATOR" merged \
    --path "$response" \
    --run-dir "$run_dir"

  score_dir="$OUTPUT_ROOT/scores/$run_name"
  score_log="$OUTPUT_ROOT/logs/${run_name}.score.log"
  if "$PYTHON_BIN" "$VALIDATOR" scores \
    --response-path "$response" \
    --score-dir "$score_dir" \
    --run-dir "$run_dir" \
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
      --run-dir "$run_dir"
    echo "$(date --iso-8601=seconds) DONE official scoring=$run_name"
  fi
done

echo "$(date --iso-8601=seconds) IFEVAL_COMPLETE runs=${#run_dirs[@]}"
