#!/usr/bin/env bash
set -euo pipefail

if (( $# != 4 )); then
  echo "Usage: $0 <aff-scale> <aff-boundary-lr-multiplier> <vocab-scale> <vocab-boundary-lr-multiplier>" >&2
  echo "Allowed scales: 1 2 4 8 16 32; allowed boundary-LR multipliers: 0.5 1 2" >&2
  exit 2
fi

AFF_SCALE=$1
AFF_BOUNDARY_LR=$2
VOCAB_SCALE=$3
VOCAB_BOUNDARY_LR=$4

validate_scale() {
  local method=$1
  local value=$2
  case "$value" in
    1|2|4|8|16|32) ;;
    *)
      echo "Invalid $method scale: $value (expected 1, 2, 4, 8, 16, or 32)" >&2
      exit 2
      ;;
  esac
}

validate_boundary_lr() {
  local method=$1
  local value=$2
  case "$value" in
    0.5|1|2) ;;
    *)
      echo "Invalid $method boundary-LR multiplier: $value (expected 0.5, 1, or 2)" >&2
      exit 2
      ;;
  esac
}

validate_scale "A-LoRA" "$AFF_SCALE"
validate_boundary_lr "A-LoRA" "$AFF_BOUNDARY_LR"
validate_scale "Vocab-LoRA" "$VOCAB_SCALE"
validate_boundary_lr "Vocab-LoRA" "$VOCAB_BOUNDARY_LR"

ROOT=/commondocument/wz/cross_encoder_workspace/im_exp/lora
PY=/home/wz/anaconda3/envs/torch24/bin/python
export CUDA_HOME=/home/wz/anaconda3/envs/torch24
export LD_LIBRARY_PATH=/home/wz/anaconda3/envs/torch24/lib:${LD_LIBRARY_PATH:-}
export DS_IGNORE_CUDA_DETECTION=1

MODEL=../models/Qwen2.5-1.5B-Base
TRAIN=corrected_sft_experiment/data/train.jsonl
DEV=corrected_sft_experiment/data/dev.jsonl
TEST=corrected_sft_experiment/data/test.jsonl
EXP=reviewer_followup/output_only_equal_budget_qwen25
OUT=$EXP/checkpoints
LOG=$EXP/logs
COMPARE=$EXP/phase2_v2_comparisons

# Six phase-2 train/dev/test jobs occupy the first six cards.  The final two
# cards concurrently evaluate the frozen selected seed-42 checkpoints on test.
# GPU_IDS is a whitespace-separated list of eight distinct integer GPU IDs.
read -r -a AVAILABLE_GPUS <<<"${GPU_IDS:-0 1 2 3 4 5 6 7}"
if (( ${#AVAILABLE_GPUS[@]} != 8 )); then
  echo "GPU_IDS must contain exactly eight GPU IDs; got ${#AVAILABLE_GPUS[@]}" >&2
  exit 2
fi
declare -A SEEN_GPUS=()
for gpu in "${AVAILABLE_GPUS[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "GPU_IDS entries must be non-negative integers; got $gpu" >&2
    exit 2
  fi
  if [[ -n "${SEEN_GPUS[$gpu]:-}" ]]; then
    echo "GPU_IDS entries must be distinct; duplicate $gpu" >&2
    exit 2
  fi
  SEEN_GPUS[$gpu]=1
done

cd "$ROOT"
mkdir -p "$OUT" "$LOG" "$COMPARE"

boundary_lr_suffix() {
  local value=$1
  if [[ "$value" == "1" ]]; then
    return 0
  fi
  printf '_blr%s' "${value//./p}"
}

run_name_for() {
  local method=$1
  local scale=$2
  local boundary_lr=$3
  local seed=$4
  printf 'qwen25_15b_out_%s_s%s%s_sd%s\n' \
    "$method" "$scale" "$(boundary_lr_suffix "$boundary_lr")" "$seed"
}

training_complete() {
  local method=$1
  local boundary_lr=$2
  local run_dir=$3
  local -a common=(
    "$run_dir/adapter_model.safetensors"
    "$run_dir/adapter_config.json"
    "$run_dir/run_args.json"
    "$run_dir/trainable_summary.json"
  )
  local artifact
  for artifact in "${common[@]}"; do
    [[ -s "$artifact" ]] || return 1
  done
  if [[ "$method" == "aff_r50" ]]; then
    [[ -s "$run_dir/affine_vocab_adapter.safetensors" ]] || return 1
    [[ -s "$run_dir/affine_vocab_config.json" ]] || return 1
  elif [[ "$boundary_lr" != "1" ]]; then
    [[ -s "$run_dir/optimizer_groups.json" ]] || return 1
  fi
}

validate_run_config() {
  local method=$1
  local scale=$2
  local boundary_lr=$3
  local seed=$4
  local run_dir=$5
  "$PY" - "$method" "$scale" "$boundary_lr" "$seed" "$run_dir" <<'PY'
import json
import math
import sys
from pathlib import Path

method, scale_text, boundary_lr_text, seed_text, run_dir_text = sys.argv[1:]
scale = int(scale_text)
boundary_lr = float(boundary_lr_text)
seed = int(seed_text)
run_dir = Path(run_dir_text)
args = json.loads((run_dir / "run_args.json").read_text(encoding="utf-8"))
params = json.loads(
    (run_dir / "trainable_summary.json").read_text(encoding="utf-8")
)
adapter = json.loads(
    (run_dir / "adapter_config.json").read_text(encoding="utf-8")
)

def exact(source, actual, expected):
    if actual != expected:
        raise ValueError(f"{run_dir.name}: {source}={actual!r}, expected {expected!r}")

def close(source, actual, expected):
    if not isinstance(actual, (int, float)) or not math.isclose(
        float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"{run_dir.name}: {source}={actual!r}, expected {expected!r}")

common = {
    "seed": seed,
    "learning_rate": 2e-4,
    "hidden_lora_rank": 8,
    "hidden_lora_alpha": 16,
    "hidden_lora_dropout": 0.05,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "num_train_epochs": 1,
    "save_strategy": "no",
    "master_dtype": "fp32",
}
for key, expected in common.items():
    exact(f"run_args.{key}", args.get(key), expected)

hidden_targets = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "up_proj", "down_proj", "gate_proj",
}
exact("adapter_config.r", adapter.get("r"), 8)
exact("adapter_config.lora_alpha", adapter.get("lora_alpha"), 16)
close("adapter_config.lora_dropout", adapter.get("lora_dropout"), 0.05)

if method == "aff_r50":
    expected_targets = hidden_targets
    exact("run_args.variant", args.get("variant"), "affine_lm_head_plus_hidden_lora")
    exact("run_args.affine_rank", args.get("affine_rank"), 50)
    close("run_args.affine_alpha", args.get("affine_alpha"), 50 * scale)
    close("run_args.affine_dropout", args.get("affine_dropout"), 0.0)
    exact("run_args.no_affine_input_bias", args.get("no_affine_input_bias"), True)
    exact("run_args.affine_lm_head_bias", args.get("affine_lm_head_bias"), False)
    close(
        "run_args.affine_learning_rate_scale",
        args.get("affine_learning_rate_scale"),
        boundary_lr,
    )
    exact("trainable_summary.trainable", params.get("trainable"), 9_385_984)
    affine = json.loads(
        (run_dir / "affine_vocab_config.json").read_text(encoding="utf-8")
    )
    for key, expected in {
        "hidden_size": 1536,
        "rank": 50,
        "alpha": 50 * scale,
        "dropout": 0,
        "use_input": False,
        "use_lm_head": True,
        "use_lm_head_bias": False,
    }.items():
        exact(f"affine_vocab_config.{key}", affine.get(key), expected)
else:
    expected_targets = hidden_targets | {"lm_head"}
    exact("run_args.variant", args.get("variant"), "hidden_lora")
    exact("run_args.include_emb_lmh_lora_rank", args.get("include_emb_lmh_lora_rank"), 1)
    exact("run_args.emb_lmh_lora_alpha", args.get("emb_lmh_lora_alpha"), scale)
    exact("adapter_config.rank_pattern.lm_head", adapter.get("rank_pattern", {}).get("lm_head"), 1)
    exact("adapter_config.alpha_pattern.lm_head", adapter.get("alpha_pattern", {}).get("lm_head"), scale)
    exact("trainable_summary.trainable", params.get("trainable"), 9_385_856)
    if boundary_lr != 1.0:
        audit = json.loads(
            (run_dir / "optimizer_groups.json").read_text(encoding="utf-8")
        )
        close("optimizer.hidden_learning_rate", audit.get("hidden_learning_rate"), 2e-4)
        close(
            "optimizer.output_vocab_learning_rate",
            audit.get("output_vocab_learning_rate"),
            2e-4 * boundary_lr,
        )
        close(
            "optimizer.output_vocab_lr_scale",
            audit.get("output_vocab_lr_scale"),
            boundary_lr,
        )
        exact("optimizer.coverage_assertion", audit.get("coverage_assertion"), "passed")
        exact("optimizer.hidden_lora_params", audit.get("hidden_lora_params"), 9_232_384)
        exact("optimizer.output_vocab_lora_params", audit.get("output_vocab_lora_params"), 153_472)

exact("adapter_config.target_modules", set(adapter.get("target_modules", [])), expected_targets)
if "embed_tokens" in set(adapter.get("target_modules", [])):
    raise ValueError(f"{run_dir.name}: input embeddings were unexpectedly targeted")
PY
}

report_complete() {
  local report_path=$1
  local run_dir=$2
  local data_path=$3
  local seed=$4
  [[ -s "$report_path" ]] || return 1
  "$PY" - "$report_path" "$run_dir" "$data_path" "$seed" <<'PY' >/dev/null 2>&1
import json
import math
import sys
from pathlib import Path

report_path, run_dir, data_path, seed_text = sys.argv[1:]
report = json.loads(Path(report_path).read_text(encoding="utf-8"))
rows = report.get("per_example")
valid = (
    report.get("num_examples") == 1000
    and report.get("source_start_index") == 0
    and report.get("source_end_index") == 1000
    and report.get("seed") == int(seed_text)
    and Path(report.get("run_dir", "")).resolve() == Path(run_dir).resolve()
    and Path(report.get("data", "")).resolve() == Path(data_path).resolve()
    and isinstance(rows, list)
    and len(rows) == 1000
    and isinstance(report.get("avg_ce"), (int, float))
    and math.isfinite(float(report["avg_ce"]))
    and all(
        isinstance(row.get("record_id"), str)
        and isinstance(row.get("token_count"), int)
        and row["token_count"] > 0
        for row in rows
    )
)
raise SystemExit(0 if valid else 1)
PY
}

evaluate_report() {
  local run=$1
  local seed=$2
  local data=$3
  local split=$4
  local gpu=$5
  local run_dir=$OUT/$run
  local report=$run_dir/${split}_report.json
  local eval_log=$LOG/${run}.${split}.log

  if report_complete "$report" "$run_dir" "$data" "$seed"; then
    echo "SKIP complete $split run=$run"
    return
  fi
  echo "START $split run=$run gpu=$gpu time=$(date --iso-8601=seconds)"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" corrected_sft_experiment/evaluate_corrected_sft.py \
    --run-dir "$run_dir" \
    --data "$data" \
    --output "$report" \
    --batch-size 8 \
    >"$eval_log" 2>&1
  report_complete "$report" "$run_dir" "$data" "$seed" || {
    echo "INCOMPLETE $split artifact run=$run" >&2
    return 1
  }
  echo "DONE $split run=$run time=$(date --iso-8601=seconds)"
}

train_and_eval() {
  local method=$1
  local scale=$2
  local boundary_lr=$3
  local seed=$4
  local gpu=$5
  local run
  run=$(run_name_for "$method" "$scale" "$boundary_lr" "$seed")
  local run_dir=$OUT/$run
  local train_log=$LOG/${run}.train.log
  local train_script=corrected_sft_experiment/train_corrected_sft.py

  if [[ "$method" == "vocab_r1" ]]; then
    train_script=reviewer_followup/train_corrected_sft_output_vocab.py
    if [[ "$boundary_lr" != "1" ]]; then
      train_script=reviewer_followup/train_corrected_sft_output_vocab_boundary_lr.py
    fi
  fi

  local -a cmd=(
    "$PY" "$train_script"
    --model-path "$MODEL"
    --train-data "$TRAIN"
    --eval-data "$DEV"
    --output-dir "$run_dir"
    --hidden-lora-rank 8
    --hidden-lora-alpha 16
    --hidden-lora-dropout 0.05
    --hidden-lora-target-modules q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj
    --max-seq-len 1024
    --per-device-train-batch-size 8
    --gradient-accumulation-steps 2
    --learning-rate 2e-4
    --num-train-epochs 1
    --eval-samples 1000
    --eval-steps 250
    --logging-steps 10
    --save-strategy no
    --bf16
    --master-dtype fp32
    --seed "$seed"
  )

  if [[ "$method" == "aff_r50" ]]; then
    cmd+=(
      --variant affine_lm_head_plus_hidden_lora
      --affine-rank 50
      --affine-alpha "$((50 * scale))"
      --affine-dropout 0
      --no-affine-input-bias
      --affine-learning-rate-scale "$boundary_lr"
    )
  else
    cmd+=(
      --variant hidden_lora
      --include-emb-lmh-lora-rank 1
      --emb-lmh-lora-alpha "$scale"
    )
    if [[ "$boundary_lr" != "1" ]]; then
      cmd+=(--output-vocab-lr-scale "$boundary_lr")
    fi
  fi

  if training_complete "$method" "$boundary_lr" "$run_dir"; then
    validate_run_config "$method" "$scale" "$boundary_lr" "$seed" "$run_dir"
    echo "SKIP complete train run=$run"
  else
    echo "START train run=$run gpu=$gpu time=$(date --iso-8601=seconds)"
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" >"$train_log" 2>&1
    training_complete "$method" "$boundary_lr" "$run_dir" || {
      echo "INCOMPLETE train artifacts run=$run" >&2
      return 1
    }
    validate_run_config "$method" "$scale" "$boundary_lr" "$seed" "$run_dir"
    echo "DONE train run=$run time=$(date --iso-8601=seconds)"
  fi

  evaluate_report "$run" "$seed" "$DEV" dev "$gpu"
  evaluate_report "$run" "$seed" "$TEST" test "$gpu"
}

# Resolve and validate the selected seed-42 runs before starting any phase-2
# mutation.  Seed-42 dev was used for selection; test is still untouched here.
AFF_RUN_42=$(run_name_for aff_r50 "$AFF_SCALE" "$AFF_BOUNDARY_LR" 42)
VOCAB_RUN_42=$(run_name_for vocab_r1 "$VOCAB_SCALE" "$VOCAB_BOUNDARY_LR" 42)
AFF_DIR_42=$OUT/$AFF_RUN_42
VOCAB_DIR_42=$OUT/$VOCAB_RUN_42

for selected in \
  "aff_r50 $AFF_SCALE $AFF_BOUNDARY_LR $AFF_RUN_42 $AFF_DIR_42" \
  "vocab_r1 $VOCAB_SCALE $VOCAB_BOUNDARY_LR $VOCAB_RUN_42 $VOCAB_DIR_42"; do
  read -r method scale boundary_lr run run_dir <<<"$selected"
  if ! training_complete "$method" "$boundary_lr" "$run_dir"; then
    echo "Selected seed-42 training artifacts are incomplete: $run_dir" >&2
    exit 1
  fi
  validate_run_config "$method" "$scale" "$boundary_lr" 42 "$run_dir"
  if ! report_complete "$run_dir/dev_report.json" "$run_dir" "$DEV" 42; then
    echo "Selected seed-42 dev report is incomplete: $run_dir/dev_report.json" >&2
    exit 1
  fi
done

TASKS=(
  "aff_r50 $AFF_SCALE $AFF_BOUNDARY_LR 43"
  "vocab_r1 $VOCAB_SCALE $VOCAB_BOUNDARY_LR 43"
  "aff_r50 $AFF_SCALE $AFF_BOUNDARY_LR 44"
  "vocab_r1 $VOCAB_SCALE $VOCAB_BOUNDARY_LR 44"
  "aff_r50 $AFF_SCALE $AFF_BOUNDARY_LR 45"
  "vocab_r1 $VOCAB_SCALE $VOCAB_BOUNDARY_LR 45"
)

# Launch all eight GPU workers together.
pids=()
for task_index in "${!TASKS[@]}"; do
  read -r method scale boundary_lr seed <<<"${TASKS[$task_index]}"
  train_and_eval \
    "$method" "$scale" "$boundary_lr" "$seed" \
    "${AVAILABLE_GPUS[$task_index]}" &
  pids+=("$!")
done
evaluate_report "$AFF_RUN_42" 42 "$TEST" test "${AVAILABLE_GPUS[6]}" &
pids+=("$!")
evaluate_report "$VOCAB_RUN_42" 42 "$TEST" test "${AVAILABLE_GPUS[7]}" &
pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if (( failed != 0 )); then
  echo "PHASE2_V2_TRAIN_OR_EVAL_FAILED time=$(date --iso-8601=seconds)" >&2
  exit 1
fi

# No selection occurs below: the frozen configurations are compared on both
# splits for seeds 42-45, with test serving only as held-out confirmation.
AFF_SUFFIX=$(boundary_lr_suffix "$AFF_BOUNDARY_LR")
VOCAB_SUFFIX=$(boundary_lr_suffix "$VOCAB_BOUNDARY_LR")
compare_pids=()
for seed in 42 43 44 45; do
  aff_run=$(run_name_for aff_r50 "$AFF_SCALE" "$AFF_BOUNDARY_LR" "$seed")
  vocab_run=$(run_name_for vocab_r1 "$VOCAB_SCALE" "$VOCAB_BOUNDARY_LR" "$seed")
  aff_dir=$OUT/$aff_run
  vocab_dir=$OUT/$vocab_run
  for split in dev test; do
    comparison=$COMPARE/vocab_s${VOCAB_SCALE}${VOCAB_SUFFIX}_vs_aff_s${AFF_SCALE}${AFF_SUFFIX}_sd${seed}_${split}.json
    "$PY" corrected_sft_experiment/compare_results.py \
      --baseline "$vocab_dir/${split}_report.json" \
      --treatment "$aff_dir/${split}_report.json" \
      --output "$comparison" \
      --samples 10000 \
      --seed "$seed" &
    compare_pids+=("$!")
  done
done

failed=0
for pid in "${compare_pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if (( failed != 0 )); then
  echo "PHASE2_V2_COMPARISON_FAILED time=$(date --iso-8601=seconds)" >&2
  exit 1
fi

echo "PHASE2_V2_COMPLETE aff_scale=$AFF_SCALE aff_boundary_lr=$AFF_BOUNDARY_LR vocab_scale=$VOCAB_SCALE vocab_boundary_lr=$VOCAB_BOUNDARY_LR time=$(date --iso-8601=seconds)"
