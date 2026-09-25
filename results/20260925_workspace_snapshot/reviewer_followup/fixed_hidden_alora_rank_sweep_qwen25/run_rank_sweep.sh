#!/usr/bin/env bash
set -euo pipefail

ROOT=/commondocument/wz/cross_encoder_workspace/im_exp/lora
PY=/home/wz/anaconda3/envs/torch24/bin/python
MODEL=$ROOT/../models/Qwen2.5-1.5B-Base
TRAIN=$ROOT/corrected_sft_experiment/data/train.jsonl
DEV=$ROOT/corrected_sft_experiment/data/dev.jsonl
TEST=$ROOT/corrected_sft_experiment/data/test.jsonl
SOURCE=$ROOT/corrected_sft_experiment/outputs/formal/qwen25_15b_hidden_sd42
ENTRY=$ROOT/reviewer_followup/train_fixed_hidden_alora_rank_sweep.py
BASE_ENTRY=$ROOT/reviewer_followup/train_corrected_sft_fixed_hidden_boundary_fp32.py
EVALUATOR=$ROOT/reviewer_followup/evaluate_fixed_hidden_alora_rank_sweep.py
BASE_EVALUATOR=$ROOT/corrected_sft_experiment/evaluate_corrected_sft.py
SUMMARIZER=$ROOT/reviewer_followup/summarize_fixed_hidden_alora_rank_sweep.py
EXP=$ROOT/reviewer_followup/fixed_hidden_alora_rank_sweep_qwen25
OUT=$EXP/checkpoints
LOG=$EXP/logs
RANK50=$ROOT/reviewer_followup/fixed_hidden_boundary_fp32_qwen25/checkpoints/qwen25_15b_fhfp32_hsd42_alora_r50_s16_bsd42

EXPECTED_TRAIN_SHA=e56c2eb28a09aa409015b9b809d79403c0e43f33741d56d7df28bf76e11a00ed
EXPECTED_DEV_SHA=6435023ea79c637f1307dbe7de32891406d3f5722ed3c1a02c1c290394da82ea
EXPECTED_TEST_SHA=25b6bae84df967eb15a5fdd205df15c55b41e2a35ea0956f598bf4e0b645fdf1
EXPECTED_MODEL_SHA=0e8c8aa86468aba09c9d32157ff4bc2301c7e6c50e4398960425b2ea71e66f77
EXPECTED_SOURCE_SHA=354ed4961fb4d6dc7cf77d5e1e0ef528726b90cd71776e533088fa6d9b77e167
EXPECTED_HIDDEN_CANONICAL_SHA=92fe20f386d39b89e99591108b5fbcf464b0705423fa379e48e6361950381b44
SCALE=16
RANKS=(2 4 8 16 32)

export CUDA_HOME=/home/wz/anaconda3/envs/torch24
export LD_LIBRARY_PATH=/home/wz/anaconda3/envs/torch24/lib:${LD_LIBRARY_PATH:-}
export DS_IGNORE_CUDA_DETECTION=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

die() {
  echo "ERROR: $*" >&2
  exit 1
}

sha() {
  sha256sum "$1" | awk '{print $1}'
}

require_sha() {
  local path=$1 expected=$2 actual
  [[ -s "$path" ]] || die "missing file: $path"
  actual=$(sha "$path")
  [[ "$actual" == "$expected" ]] || die "SHA mismatch: $path ($actual != $expected)"
}

run_name() {
  printf 'qwen25_15b_fhfp32_hsd42_alora_r%s_s16_bsd42\n' "$1"
}

validate_run() {
  local rank=$1 run_dir=$2 entry_sha=$3
  "$PY" - "$rank" "$run_dir" "$ENTRY" "$entry_sha" "$SOURCE" "$EXPECTED_HIDDEN_CANONICAL_SHA" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path
from safetensors.torch import load_file

rank = int(sys.argv[1])
run = Path(sys.argv[2]).resolve()
entry = Path(sys.argv[3]).resolve()
entry_sha = sys.argv[4]
source = Path(sys.argv[5]).resolve()
hidden_sha = sys.argv[6]
count = 2 * 1536 * rank
alpha = 16.0 * rank
required = [
    "adapter_model.safetensors", "adapter_config.json", "run_args.json",
    "trainable_summary.json", "fixed_boundary_adapter.safetensors",
    "fixed_boundary_config.json", "fixed_hidden_boundary_audit.json",
]
missing = [name for name in required if not (run / name).is_file()]
if missing:
    raise RuntimeError(f"{run.name}: missing {missing}")
args = json.loads((run / "run_args.json").read_text())
cfg = json.loads((run / "fixed_boundary_config.json").read_text())
audit = json.loads((run / "fixed_hidden_boundary_audit.json").read_text())
summary = json.loads((run / "trainable_summary.json").read_text())
expected_args = {
    "variant": "affine_lm_head_plus_hidden_lora",
    "seed": 42,
    "affine_rank": rank,
    "freeze_initial_hidden_lora": True,
    "max_seq_len": 1024,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "num_train_epochs": 1.0,
    "max_steps": -1,
    "max_train_samples": None,
    "learning_rate": 2e-4,
    "max_grad_norm": 0.0,
    "master_dtype": "fp32",
}
for key, expected in expected_args.items():
    if args.get(key) != expected:
        raise RuntimeError(f"{run.name}: args.{key}={args.get(key)!r} != {expected!r}")
if not math.isclose(float(args["affine_alpha"]), alpha, rel_tol=0, abs_tol=1e-12):
    raise RuntimeError("alpha mismatch")
if Path(args["initial_hidden_lora_adapter"]).resolve() != source:
    raise RuntimeError("source hidden path mismatch")
if cfg.get("marker") != "fixed_hidden_output_boundary_fp32_rank_sweep_v1":
    raise RuntimeError("rank-sweep marker mismatch")
for key, expected in {
    "kind": "alora", "rank": rank, "trainable_parameters": count,
    "boundary_dtype": "torch.float32", "output_dtype": "torch.float32",
    "input_adapter": False, "boundary_bias": False,
    "zero_initialized_residual": True,
}.items():
    if cfg.get(key) != expected:
        raise RuntimeError(f"config.{key} mismatch")
if not math.isclose(float(cfg["scale"]), 16.0, rel_tol=0, abs_tol=1e-12):
    raise RuntimeError("config scale mismatch")
if summary.get("trainable") != count:
    raise RuntimeError("trainable summary mismatch")
if audit.get("implementation") != str(entry) or audit.get("implementation_sha256") != entry_sha:
    raise RuntimeError("implementation binding mismatch")
if audit.get("saved_hidden_canonical_tensor_sha256") != hidden_sha:
    raise RuntimeError("frozen hidden digest mismatch")
if audit.get("saved_hidden_unchanged_assertion") != "passed":
    raise RuntimeError("hidden unchanged assertion missing")
optimizer = audit.get("boundary_optimizer", {})
if optimizer.get("trainable_parameters") != count:
    raise RuntimeError("optimizer parameter count mismatch")
for key in ("only_boundary_trainable_assertion", "hidden_frozen_assertion",
            "optimizer_parameter_coverage_assertion"):
    if optimizer.get(key) != "passed":
        raise RuntimeError(f"optimizer audit failed: {key}")
state_path = run / "fixed_boundary_adapter.safetensors"
state = load_file(str(state_path), device="cpu")
expected_shapes = {
    "affine.down.weight": (rank, 1536),
    "affine.up.weight": (1536, rank),
}
if set(state) != set(expected_shapes):
    raise RuntimeError("boundary tensor names mismatch")
for name, shape in expected_shapes.items():
    if tuple(state[name].shape) != shape or str(state[name].dtype) != "torch.float32":
        raise RuntimeError(f"{name}: shape/dtype mismatch")
if sum(t.numel() for t in state.values()) != count:
    raise RuntimeError("saved boundary count mismatch")
if hashlib.sha256(state_path.read_bytes()).hexdigest() != audit["saved_boundary_file_sha256"]:
    raise RuntimeError("saved boundary hash mismatch")
PY
}

validate_report() {
  local report=$1 run_dir=$2 data=$3 start=$4 end=$5
  "$PY" - "$report" "$run_dir" "$data" "$start" "$end" <<'PY'
import json
import math
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
run = Path(sys.argv[2]).resolve()
data = Path(sys.argv[3]).resolve()
start, end = map(int, sys.argv[4:6])
payload = json.loads(report_path.read_text())
source = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
rows = payload.get("per_example")
if not isinstance(rows, list) or len(rows) != end - start:
    raise RuntimeError(f"{report_path}: row count mismatch")
if payload.get("source_start_index") != start or payload.get("source_end_index") != end:
    raise RuntimeError(f"{report_path}: source interval mismatch")
if Path(payload["run_dir"]).resolve() != run or Path(payload["data"]).resolve() != data:
    raise RuntimeError(f"{report_path}: run/data binding mismatch")
selected = source[start:end]
if [row.get("record_id") for row in rows] != [row.get("record_id") for row in selected]:
    raise RuntimeError(f"{report_path}: record order mismatch")
nll = sum(float(row["nll_sum"]) for row in rows)
tokens = sum(int(row["token_count"]) for row in rows)
if payload.get("supervised_tokens") != tokens:
    raise RuntimeError(f"{report_path}: token count mismatch")
if not math.isclose(float(payload["total_nll"]), nll, rel_tol=0, abs_tol=1e-5):
    raise RuntimeError(f"{report_path}: total NLL mismatch")
if not math.isclose(float(payload["avg_ce"]), nll / tokens, rel_tol=0, abs_tol=1e-10):
    raise RuntimeError(f"{report_path}: CE mismatch")
PY
}

evaluate_one() {
  local gpu=$1 run_dir=$2 split=$3 data=$4 start=$5 end=$6 report=$7
  local log=$LOG/$(basename "$run_dir").${split}.log
  if [[ -s "$report" ]] && validate_report "$report" "$run_dir" "$data" "$start" "$end" >/dev/null 2>&1; then
    echo "SKIP valid eval run=$(basename "$run_dir") split=$split"
    return
  fi
  echo "START eval run=$(basename "$run_dir") split=$split gpu=$gpu time=$(date --iso-8601=seconds)"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$EVALUATOR" \
    --model-path "$MODEL" --run-dir "$run_dir" --data "$data" \
    --output "$report" --batch-size 8 --max-seq-len 1024 \
    --start-index "$start" --end-index "$end" --device cuda >"$log" 2>&1
  validate_report "$report" "$run_dir" "$data" "$start" "$end"
  echo "DONE eval run=$(basename "$run_dir") split=$split time=$(date --iso-8601=seconds)"
}

run_rank() {
  local rank=$1 gpu=$2 entry_sha=$3
  local name run_dir alpha train_log
  name=$(run_name "$rank")
  run_dir=$OUT/$name
  alpha=$((rank * SCALE))
  train_log=$LOG/$name.train.log
  exec {lock_fd}>"$OUT/$name.lock"
  flock -n "$lock_fd" || die "another launcher owns $name"
  if [[ -e "$run_dir/fixed_hidden_boundary_audit.json" ]]; then
    validate_run "$rank" "$run_dir" "$entry_sha"
    echo "SKIP valid train run=$name"
  else
    echo "START train run=$name gpu=$gpu time=$(date --iso-8601=seconds)"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ENTRY" \
      --sweep-alora-rank "$rank" --sweep-alora-scale "$SCALE" \
      --fixed-boundary-kind alora --fixed-boundary-tf32 deny \
      --model-path "$MODEL" --train-data "$TRAIN" --output-dir "$run_dir" \
      --variant affine_lm_head_plus_hidden_lora \
      --initial-hidden-lora-adapter "$SOURCE" --freeze-initial-hidden-lora \
      --affine-rank "$rank" --affine-alpha "$alpha" --affine-dropout 0 \
      --no-affine-input-bias --affine-learning-rate-scale 1 \
      --max-seq-len 1024 --per-device-train-batch-size 8 \
      --gradient-accumulation-steps 2 --learning-rate 2e-4 \
      --num-train-epochs 1 --lr-scheduler-type cosine --warmup-ratio 0.03 \
      --max-grad-norm 0 --logging-steps 10 --save-strategy no --bf16 \
      --base-dtype auto --master-dtype fp32 --seed 42 >"$train_log" 2>&1
    validate_run "$rank" "$run_dir" "$entry_sha"
    echo "DONE train run=$name time=$(date --iso-8601=seconds)"
  fi
  evaluate_one "$gpu" "$run_dir" train1000 "$TRAIN" 0 1000 "$run_dir/train1000_report.json"
  evaluate_one "$gpu" "$run_dir" dev "$DEV" 0 1000 "$run_dir/dev_report.json"
  evaluate_one "$gpu" "$run_dir" test "$TEST" 0 1000 "$run_dir/test_report.json"
}

[[ -x "$PY" ]] || die "missing Python environment"
for file in "$ENTRY" "$BASE_ENTRY" "$EVALUATOR" "$BASE_EVALUATOR" "$SUMMARIZER"; do
  [[ -s "$file" ]] || die "missing implementation: $file"
done
require_sha "$TRAIN" "$EXPECTED_TRAIN_SHA"
require_sha "$DEV" "$EXPECTED_DEV_SHA"
require_sha "$TEST" "$EXPECTED_TEST_SHA"
require_sha "$MODEL/config.json" "$EXPECTED_MODEL_SHA"
require_sha "$SOURCE/adapter_model.safetensors" "$EXPECTED_SOURCE_SHA"

read -r -a GPUS <<<"${GPU_IDS:-0 1 2 3 4 5}"
(( ${#GPUS[@]} == 6 )) || die "GPU_IDS must contain exactly six indices"
declare -A SEEN=()
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || die "invalid GPU: $gpu"
  [[ -z "${SEEN[$gpu]:-}" ]] || die "duplicate GPU: $gpu"
  SEEN[$gpu]=1
done

mkdir -p "$OUT" "$LOG"
ENTRY_SHA=$(sha "$ENTRY")

"$PY" - "$EXP/launch_manifest.json" "$ENTRY" "$BASE_ENTRY" "$EVALUATOR" \
  "$BASE_EVALUATOR" "$SUMMARIZER" "$ENTRY_SHA" "${GPUS[*]}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

target = Path(sys.argv[1])
entry, base_entry, evaluator, base_evaluator, summarizer = map(Path, sys.argv[2:7])
entry_sha, gpu_text = sys.argv[7:9]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
payload = {
    "marker": "fixed_hidden_alora_rank_sweep_launch_v1",
    "ranks": [2, 4, 8, 16, 32, 50],
    "new_training_ranks": [2, 4, 8, 16, 32],
    "functional_scale": 16,
    "seed": 42,
    "training_entry": str(entry.resolve()),
    "training_entry_sha256": entry_sha,
    "base_training_implementation_sha256": sha(base_entry),
    "evaluator_sha256": sha(evaluator),
    "base_evaluator_sha256": sha(base_evaluator),
    "summarizer_sha256": sha(summarizer),
    "gpu_ids": [int(value) for value in gpu_text.split()],
    "rank50_policy": "reuse prior validated fixed-hidden endpoint",
}
if target.exists():
    if json.loads(target.read_text()) != payload:
        raise RuntimeError("existing launch manifest differs")
else:
    target.write_text(json.dumps(payload, indent=2) + "\n")
PY

# Rank 50 already exists; only its train-subset diagnostic is missing.
validate_report "$RANK50/dev_report.json" "$RANK50" "$DEV" 0 1000
validate_report "$RANK50/test_report.json" "$RANK50" "$TEST" 0 1000
evaluate_one "${GPUS[5]}" "$RANK50" train1000 "$TRAIN" 0 1000 "$EXP/rank50_train1000_report.json" &
rank50_pid=$!

pids=()
labels=()
for index in "${!RANKS[@]}"; do
  rank=${RANKS[$index]}
  run_rank "$rank" "${GPUS[$index]}" "$ENTRY_SHA" &
  pids+=("$!")
  labels+=("r$rank/gpu${GPUS[$index]}")
done

failed=0
if ! wait "$rank50_pid"; then
  echo "FAILED r50 reference eval" >&2
  failed=1
fi
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "FAILED ${labels[$index]}" >&2
    failed=1
  fi
done
(( failed == 0 )) || die "one or more rank-sweep tasks failed"

for rank in "${RANKS[@]}"; do
  directory=$OUT/$(run_name "$rank")
  validate_run "$rank" "$directory" "$ENTRY_SHA"
  validate_report "$directory/train1000_report.json" "$directory" "$TRAIN" 0 1000
  validate_report "$directory/dev_report.json" "$directory" "$DEV" 0 1000
  validate_report "$directory/test_report.json" "$directory" "$TEST" 0 1000
done
validate_report "$EXP/rank50_train1000_report.json" "$RANK50" "$TRAIN" 0 1000

"$PY" "$SUMMARIZER" --experiment-dir "$EXP" --rank50-dir "$RANK50" \
  >"$EXP/summarizer.stdout.json"
echo "RANK_SWEEP_COMPLETE time=$(date --iso-8601=seconds)"
