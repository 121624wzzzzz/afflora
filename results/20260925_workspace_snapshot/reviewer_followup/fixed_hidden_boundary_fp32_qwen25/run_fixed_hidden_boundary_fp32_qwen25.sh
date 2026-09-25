#!/usr/bin/env bash
set -euo pipefail

ROOT=/commondocument/wz/cross_encoder_workspace/im_exp/lora
PY=/home/wz/anaconda3/envs/torch24/bin/python
MODEL=$ROOT/../models/Qwen2.5-1.5B-Base
TRAIN=$ROOT/corrected_sft_experiment/data/train.jsonl
DEV=$ROOT/corrected_sft_experiment/data/dev.jsonl
TEST=$ROOT/corrected_sft_experiment/data/test.jsonl
ENTRY=$ROOT/reviewer_followup/train_corrected_sft_fixed_hidden_boundary_fp32.py
EVALUATOR=$ROOT/reviewer_followup/evaluate_fixed_hidden_boundary_corrected_sft.py
EXP=$ROOT/reviewer_followup/fixed_hidden_boundary_fp32_qwen25
OUT=$EXP/checkpoints
LOG=$EXP/logs
LAUNCH_MANIFEST=$EXP/launch_manifest.json
COMPLETION_MANIFEST=$EXP/completion_manifest.json

EXPECTED_TRAIN_SHA=e56c2eb28a09aa409015b9b809d79403c0e43f33741d56d7df28bf76e11a00ed
EXPECTED_DEV_SHA=6435023ea79c637f1307dbe7de32891406d3f5722ed3c1a02c1c290394da82ea
EXPECTED_TEST_SHA=25b6bae84df967eb15a5fdd205df15c55b41e2a35ea0956f598bf4e0b645fdf1
EXPECTED_MODEL_CONFIG_SHA=0e8c8aa86468aba09c9d32157ff4bc2301c7e6c50e4398960425b2ea71e66f77
EXPECTED_PIPELINE_SHA=47ef96c81e32234d220765279ba634e710162518f08678dc3a32309176c6e460

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

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

require_sha() {
  local path=$1
  local expected=$2
  [[ -s "$path" ]] || die "required file is absent or empty: $path"
  local actual
  actual=$(sha256_file "$path")
  [[ "$actual" == "$expected" ]] || die "SHA256 mismatch for $path: $actual != $expected"
}

source_dir_for() {
  local seed=$1
  printf '%s/corrected_sft_experiment/outputs/formal/qwen25_15b_hidden_sd%s\n' "$ROOT" "$seed"
}

source_adapter_sha_for() {
  case "$1" in
    42) printf '%s\n' 354ed4961fb4d6dc7cf77d5e1e0ef528726b90cd71776e533088fa6d9b77e167 ;;
    43) printf '%s\n' 16958535a58d7cb3dd6c536fd1ff89beb7f88e053a9a67fb20eeb8a90135a093 ;;
    44) printf '%s\n' 2890f3f5c0eef7104616179d36916aa3ce1999047247e27e06d2c31b4d379476 ;;
    *) die "unsupported source seed: $1" ;;
  esac
}

source_config_sha_for() {
  case "$1" in
    42) printf '%s\n' d5d6c8d1d2b30f5ca8cc5ac9287dad0d0b2044becd6ec686823ea88c41ca4a96 ;;
    43) printf '%s\n' 6c9ca4bd6d30ef54cefe4e82de61cd020708aa4a444b0a8282513ec5a1a90d2f ;;
    44) printf '%s\n' 1434787a9c4769dfb16c822b20009d4983ae13fecea2e0ba0aeac028675b12ef ;;
    *) die "unsupported source seed: $1" ;;
  esac
}

source_args_sha_for() {
  case "$1" in
    42) printf '%s\n' cfc5f457fbf9dffb05379f7cbb0e703c90e2894cc39a57707d8ac3dd26c101ee ;;
    43) printf '%s\n' c160c377e73a28f73dc7548bb6f59f88ed645a18095baa994cc8726d91b7c9ae ;;
    44) printf '%s\n' ded771fb1de07ec0adc408035d2d843746dabbb358dd64db82b8bac553d05f37 ;;
    *) die "unsupported source seed: $1" ;;
  esac
}

source_tensor_sha_for() {
  case "$1" in
    42) printf '%s\n' 92fe20f386d39b89e99591108b5fbcf464b0705423fa379e48e6361950381b44 ;;
    43) printf '%s\n' 5485e347c6315cbef11099fd17e6a9b03431b33b3beaa0b15e00a9791f0f3d1f ;;
    44) printf '%s\n' baceb5f2f02f3327aeae089f103e7faa792121b732d3b55b1c5078f189404d62 ;;
    *) die "unsupported source seed: $1" ;;
  esac
}

run_name_for() {
  local kind=$1
  local seed=$2
  if [[ "$kind" == "alora" ]]; then
    printf 'qwen25_15b_fhfp32_hsd%s_alora_r50_s16_bsd%s\n' "$seed" "$seed"
  else
    printf 'qwen25_15b_fhfp32_hsd%s_vocab_lora_r1_s32_bsd%s\n' "$seed" "$seed"
  fi
}

validate_source_checkpoint() {
  local seed=$1
  local source_dir
  source_dir=$(source_dir_for "$seed")
  require_sha "$source_dir/adapter_model.safetensors" "$(source_adapter_sha_for "$seed")"
  require_sha "$source_dir/adapter_config.json" "$(source_config_sha_for "$seed")"
  require_sha "$source_dir/run_args.json" "$(source_args_sha_for "$seed")"
  "$PY" - "$ROOT" "$source_dir" "$seed" "$EXPECTED_PIPELINE_SHA" <<'PY'
import json
import math
import sys
from pathlib import Path
from safetensors.torch import load_file

root, source_text, seed_text, pipeline_sha = sys.argv[1:]
root = Path(root).resolve()
source = Path(source_text).resolve()
seed = int(seed_text)
args = json.loads((source / "run_args.json").read_text(encoding="utf-8"))
cfg = json.loads((source / "adapter_config.json").read_text(encoding="utf-8"))
state = load_file(str(source / "adapter_model.safetensors"), device="cpu")

def exact(label, actual, expected):
    if actual != expected:
        raise ValueError(f"{source.name}: {label}={actual!r}, expected {expected!r}")

def close(label, actual, expected):
    if not isinstance(actual, (int, float)) or not math.isclose(
        float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"{source.name}: {label}={actual!r}, expected {expected!r}")

for key, expected in {
    "variant": "hidden_lora",
    "seed": seed,
    "hidden_lora_rank": 8,
    "hidden_lora_alpha": 16,
    "hidden_lora_dropout": 0.05,
    "max_seq_len": 1024,
    "master_dtype": "fp32",
    "include_emb_lmh_lora_rank": 0,
}.items():
    exact(f"run_args.{key}", args.get(key), expected)
exact(
    "run_args.corrected_data_pipeline.implementation_sha256",
    args.get("corrected_data_pipeline", {}).get("implementation_sha256"),
    pipeline_sha,
)
exact("adapter_config.r", cfg.get("r"), 8)
exact("adapter_config.lora_alpha", cfg.get("lora_alpha"), 16)
close("adapter_config.lora_dropout", cfg.get("lora_dropout"), 0.05)
exact("adapter_config.bias", cfg.get("bias"), "none")
exact(
    "adapter_config.target_modules",
    set(cfg.get("target_modules", [])),
    {"q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"},
)
if len(state) != 392 or sum(t.numel() for t in state.values()) != 9_232_384:
    raise ValueError(f"{source.name}: unexpected hidden tensor budget")
bad = [
    name for name in state
    if "lora_" not in name or "lm_head" in name or "embed_tokens" in name
]
if bad:
    raise ValueError(f"{source.name}: source is not pure hidden LoRA: {bad[:8]}")
PY
}

validate_run() {
  local kind=$1
  local seed=$2
  local run_dir=$3
  local entry_sha=$4
  local source_dir
  source_dir=$(source_dir_for "$seed")
  "$PY" - \
    "$kind" "$seed" "$run_dir" "$source_dir" \
    "$(source_adapter_sha_for "$seed")" "$(source_tensor_sha_for "$seed")" \
    "$entry_sha" "$ENTRY" "$MODEL" "$TRAIN" "$EXPECTED_PIPELINE_SHA" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path
from safetensors.torch import load_file

(
    kind, seed_text, run_text, source_text, source_file_sha,
    source_tensor_sha, entry_sha, entry_text, model_text, train_text,
    pipeline_sha,
) = sys.argv[1:]
seed = int(seed_text)
run = Path(run_text).resolve()
source = Path(source_text).resolve()
entry = Path(entry_text).resolve()
model = Path(model_text).resolve()
train = Path(train_text).resolve()

required = [
    "adapter_model.safetensors", "adapter_config.json",
    "fixed_boundary_adapter.safetensors", "fixed_boundary_config.json",
    "fixed_hidden_boundary_audit.json", "run_args.json",
    "trainable_summary.json",
]
missing = [name for name in required if not (run / name).is_file()]
if missing:
    raise ValueError(f"{run.name}: missing completion artifacts: {missing}")

args = json.loads((run / "run_args.json").read_text(encoding="utf-8"))
adapter = json.loads((run / "adapter_config.json").read_text(encoding="utf-8"))
config = json.loads((run / "fixed_boundary_config.json").read_text(encoding="utf-8"))
audit = json.loads((run / "fixed_hidden_boundary_audit.json").read_text(encoding="utf-8"))
summary = json.loads((run / "trainable_summary.json").read_text(encoding="utf-8"))

rank = 50 if kind == "alora" else 1
alpha = 800.0 if kind == "alora" else 32.0
scale = 16.0 if kind == "alora" else 32.0
count = 153_600 if kind == "alora" else 153_472
shapes = (
    {"affine.down.weight": (50, 1536), "affine.up.weight": (1536, 50)}
    if kind == "alora"
    else {"affine.down.weight": (1, 1536), "affine.up.weight": (151936, 1)}
)

def exact(label, actual, expected):
    if actual != expected:
        raise ValueError(f"{run.name}: {label}={actual!r}, expected {expected!r}")

def close(label, actual, expected):
    if not isinstance(actual, (int, float)) or not math.isclose(
        float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"{run.name}: {label}={actual!r}, expected {expected!r}")

def resolved(label, actual, expected):
    if not actual or Path(actual).resolve() != expected:
        raise ValueError(f"{run.name}: {label}={actual!r}, expected {str(expected)!r}")

for key, expected in {
    "variant": "affine_lm_head_plus_hidden_lora",
    "seed": seed,
    # These parser fields are inert when the audited initial adapter is loaded.
    # Keep the legacy defaults used by the already-started formal jobs; the
    # saved adapter and tensor audit enforce the actual r=8/alpha=16 source.
    "hidden_lora_rank": 16,
    "hidden_lora_alpha": 32,
    "hidden_lora_dropout": 0.05,
    "hidden_lora_target_modules": "q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
    "freeze_initial_hidden_lora": True,
    "affine_rank": rank,
    "affine_dropout": 0.0,
    "no_affine_input_bias": True,
    "affine_lm_head_bias": False,
    "tie_affine_input_lm_head_adapters": False,
    "include_emb_lmh_lora_rank": 0,
    "initial_affine_adapter": None,
    "max_seq_len": 1024,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "num_train_epochs": 1.0,
    "max_steps": -1,
    "max_train_samples": None,
    "dataset_split": "train",
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "max_grad_norm": 0.0,
    "bf16": True,
    "fp16": False,
    "gradient_checkpointing": False,
    "trust_remote_code": False,
    "base_dtype": "auto",
    "master_dtype": "fp32",
    "save_strategy": "no",
    "skip_final_model_save": False,
    "resume_from_checkpoint": None,
    "eval_data": None,
    "eval_samples": 0,
    "reference_run_dir": None,
    "anchor_data": None,
    "auxiliary_anchor_data": None,
}.items():
    exact(f"run_args.{key}", args.get(key), expected)
for key, expected in {
    "learning_rate": 2e-4,
    "affine_alpha": alpha,
    "affine_bias_scale": 1.0,
    "affine_learning_rate_scale": 1.0,
    "affine_bias_learning_rate_scale": 1.0,
    "affine_energy_lambda": 0.0,
    "affine_energy_tau": 0.0,
    "affine_bias_energy_lambda": 0.0,
    "affine_bias_energy_tau": 0.0,
    "reference_kl_lambda": 0.0,
    "anchor_fraction": 0.0,
    "auxiliary_anchor_lambda": 0.0,
}.items():
    close(f"run_args.{key}", args.get(key), expected)
resolved("run_args.model_path", args.get("model_path"), model)
resolved("run_args.train_data", args.get("train_data"), train)
resolved("run_args.initial_hidden_lora_adapter", args.get("initial_hidden_lora_adapter"), source)
exact(
    "run_args.corrected_data_pipeline.implementation_sha256",
    args.get("corrected_data_pipeline", {}).get("implementation_sha256"),
    pipeline_sha,
)

exact("adapter.r", adapter.get("r"), 8)
exact("adapter.lora_alpha", adapter.get("lora_alpha"), 16)
close("adapter.lora_dropout", adapter.get("lora_dropout"), 0.05)
exact(
    "adapter.target_modules",
    set(adapter.get("target_modules", [])),
    {"q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"},
)
exact("trainable_summary.trainable", summary.get("trainable"), count)

for payload_name, payload in (("config", config), ("audit", audit)):
    exact(f"{payload_name}.marker", payload.get("marker"), "fixed_hidden_output_boundary_fp32_v1")
    exact(f"{payload_name}.kind", payload.get("kind"), kind)
exact("config.rank", config.get("rank"), rank)
close("config.alpha", config.get("alpha"), alpha)
close("config.scale", config.get("scale"), scale)
exact("config.hidden_size", config.get("hidden_size"), 1536)
exact("config.vocab_size", config.get("vocab_size"), 151936)
exact("config.trainable_parameters", config.get("trainable_parameters"), count)
for key, expected in {
    "boundary_dtype": "torch.float32",
    "output_dtype": "torch.float32",
    "autocast_disabled": True,
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "input_adapter": False,
    "boundary_bias": False,
    "boundary_dropout": 0.0,
    "frozen_lm_head_weight_dtype": "torch.bfloat16",
    "zero_initialized_residual": True,
}.items():
    exact(f"config.{key}", config.get(key), expected)
exact("audit.tf32_mode", audit.get("tf32_mode"), "deny")
exact("audit.cuda_matmul_allow_tf32", audit.get("cuda_matmul_allow_tf32"), False)
exact("audit.cudnn_allow_tf32", audit.get("cudnn_allow_tf32"), False)
exact("audit.completion_audit_assertion", audit.get("completion_audit_assertion"), "passed")
resolved("audit.implementation", audit.get("implementation"), entry)
exact("audit.implementation_sha256", audit.get("implementation_sha256"), entry_sha)
exact("audit.saved_boundary_parameter_count", audit.get("saved_boundary_parameter_count"), count)
exact("audit.saved_hidden_unchanged_assertion", audit.get("saved_hidden_unchanged_assertion"), "passed")

hidden = audit.get("hidden_checkpoint", {})
resolved("audit.hidden.source_adapter_dir", hidden.get("source_adapter_dir"), source)
exact("audit.hidden.source_adapter_file_sha256", hidden.get("source_adapter_file_sha256"), source_file_sha)
exact("audit.hidden.canonical_tensor_sha256", hidden.get("canonical_tensor_sha256"), source_tensor_sha)
exact("audit.hidden.tensor_count", hidden.get("tensor_count"), 392)
exact("audit.hidden.parameter_count", hidden.get("parameter_count"), 9_232_384)
exact("audit.hidden.tensor_exact_assertion", hidden.get("tensor_exact_assertion"), "passed")
exact("audit.hidden.dropout_identity_assertion", hidden.get("dropout_identity_assertion"), "passed")
exact("audit.hidden.dropout_modules_replaced_with_identity", hidden.get("dropout_modules_replaced_with_identity"), 196)
exact("config.hidden_checkpoint_audit", config.get("hidden_checkpoint_audit"), hidden)
exact("audit.saved_hidden_canonical_tensor_sha256", audit.get("saved_hidden_canonical_tensor_sha256"), source_tensor_sha)

optimizer = audit.get("boundary_optimizer", {})
exact("optimizer.only_boundary_trainable_assertion", optimizer.get("only_boundary_trainable_assertion"), "passed")
exact("optimizer.hidden_frozen_assertion", optimizer.get("hidden_frozen_assertion"), "passed")
exact("optimizer.optimizer_parameter_coverage_assertion", optimizer.get("optimizer_parameter_coverage_assertion"), "passed")
exact("optimizer.trainable_parameters", optimizer.get("trainable_parameters"), count)
exact("optimizer.trainable_dtypes", set(optimizer.get("trainable_dtypes", {}).values()), {"torch.float32"})
exact("optimizer.optimizer_parameter_dtypes", optimizer.get("optimizer_parameter_dtypes"), ["torch.float32"])
exact("optimizer.learning_rates", optimizer.get("learning_rates"), [2e-4])
close("optimizer.max_grad_norm", optimizer.get("max_grad_norm"), 0.0)

state_path = run / "fixed_boundary_adapter.safetensors"
state = load_file(str(state_path), device="cpu")
exact("boundary tensor names", set(state), set(shapes))
for name, expected_shape in shapes.items():
    exact(f"{name}.shape", tuple(state[name].shape), expected_shape)
    exact(f"{name}.dtype", str(state[name].dtype), "torch.float32")
exact("boundary tensor count", sum(t.numel() for t in state.values()), count)
digest = hashlib.sha256(state_path.read_bytes()).hexdigest()
exact("audit.saved_boundary_file_sha256", audit.get("saved_boundary_file_sha256"), digest)
PY
}

validate_report() {
  local report=$1
  local run_dir=$2
  local data=$3
  local seed=$4
  "$PY" - "$report" "$run_dir" "$data" "$MODEL" "$seed" <<'PY'
import json
import math
import sys
from pathlib import Path

report_text, run_text, data_text, model_text, seed_text = sys.argv[1:]
report_path = Path(report_text)
run = Path(run_text).resolve()
data = Path(data_text).resolve()
model = Path(model_text).resolve()
seed = int(seed_text)
report = json.loads(report_path.read_text(encoding="utf-8"))
rows = report.get("per_example")
source_rows = [
    json.loads(line)
    for line in data.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if len(source_rows) != 1000:
    raise ValueError(f"{data}: expected exactly 1000 source rows")
if not isinstance(rows, list) or len(rows) != 1000:
    raise ValueError(f"{report_path}: expected 1000 per-example rows")
expected = {
    "num_examples": 1000,
    "source_start_index": 0,
    "source_end_index": 1000,
    "seed": seed,
}
for key, value in expected.items():
    if report.get(key) != value:
        raise ValueError(f"{report_path}: {key}={report.get(key)!r}, expected {value!r}")
for key, actual, target in (
    ("run_dir", report.get("run_dir"), run),
    ("data", report.get("data"), data),
    ("model_path", report.get("model_path"), model),
):
    if not actual or Path(actual).resolve() != target:
        raise ValueError(f"{report_path}: {key}={actual!r}, expected {str(target)!r}")

total_nll = 0.0
total_tokens = 0
for index, (row, source) in enumerate(zip(rows, source_rows)):
    if row.get("record_id") != source.get("record_id"):
        raise ValueError(f"{report_path}: record_id mismatch at row {index}")
    count = row.get("token_count")
    nll = row.get("nll_sum")
    mean = row.get("mean_ce")
    if not isinstance(count, int) or count <= 0:
        raise ValueError(f"{report_path}: invalid token_count at row {index}")
    if not all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in (nll, mean)):
        raise ValueError(f"{report_path}: non-finite loss at row {index}")
    if not math.isclose(float(mean), float(nll) / count, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"{report_path}: inconsistent mean_ce at row {index}")
    total_nll += float(nll)
    total_tokens += count
if report.get("supervised_tokens") != total_tokens:
    raise ValueError(f"{report_path}: supervised token total mismatch")
if not math.isclose(float(report.get("total_nll")), total_nll, rel_tol=1e-12, abs_tol=1e-5):
    raise ValueError(f"{report_path}: total NLL mismatch")
if not math.isclose(float(report.get("avg_ce")), total_nll / total_tokens, rel_tol=1e-12, abs_tol=1e-10):
    raise ValueError(f"{report_path}: average CE mismatch")
PY
}

report_is_valid() {
  validate_report "$@" >/dev/null 2>&1
}

evaluate_split() {
  local kind=$1
  local seed=$2
  local gpu=$3
  local run_dir=$4
  local split=$5
  local data=$6
  local report=$run_dir/${split}_report.json
  local log=$LOG/$(basename "$run_dir").${split}.log
  if [[ -s "$report" ]] && report_is_valid "$report" "$run_dir" "$data" "$seed"; then
    echo "SKIP valid $split run=$(basename "$run_dir")"
    return 0
  fi
  echo "START $split run=$(basename "$run_dir") gpu=$gpu time=$(date --iso-8601=seconds)"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$EVALUATOR" \
    --model-path "$MODEL" \
    --run-dir "$run_dir" \
    --data "$data" \
    --output "$report" \
    --batch-size 8 \
    --max-seq-len 1024 \
    --device cuda >"$log" 2>&1
  validate_report "$report" "$run_dir" "$data" "$seed"
  echo "DONE $split run=$(basename "$run_dir") time=$(date --iso-8601=seconds)"
}

run_one() {
  local kind=$1
  local seed=$2
  local gpu=$3
  local entry_sha=$4
  local run_name
  run_name=$(run_name_for "$kind" "$seed")
  local run_dir=$OUT/$run_name
  local source_dir
  source_dir=$(source_dir_for "$seed")
  local train_log=$LOG/${run_name}.train.log
  local rank alpha
  if [[ "$kind" == "alora" ]]; then
    rank=50
    alpha=800
  else
    rank=1
    alpha=32
  fi

  exec {lock_fd}>"$OUT/${run_name}.lock"
  flock -n "$lock_fd" || {
    echo "ERROR: another launcher owns run $run_name" >&2
    return 1
  }

  if [[ -e "$run_dir/fixed_hidden_boundary_audit.json" ]]; then
    validate_run "$kind" "$seed" "$run_dir" "$entry_sha"
    echo "SKIP valid train run=$run_name"
  else
    if pgrep -af "$ENTRY" | grep -F -- "/$run_name" >/dev/null; then
      echo "ERROR: training is already active outside this launcher for $run_name" >&2
      return 1
    fi
    echo "START train run=$run_name gpu=$gpu time=$(date --iso-8601=seconds)"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ENTRY" \
      --fixed-boundary-kind "$kind" \
      --fixed-boundary-tf32 deny \
      --model-path "$MODEL" \
      --train-data "$TRAIN" \
      --output-dir "$run_dir" \
      --variant affine_lm_head_plus_hidden_lora \
      --initial-hidden-lora-adapter "$source_dir" \
      --freeze-initial-hidden-lora \
      --affine-rank "$rank" \
      --affine-alpha "$alpha" \
      --affine-dropout 0 \
      --no-affine-input-bias \
      --affine-learning-rate-scale 1 \
      --max-seq-len 1024 \
      --per-device-train-batch-size 8 \
      --gradient-accumulation-steps 2 \
      --learning-rate 2e-4 \
      --num-train-epochs 1 \
      --lr-scheduler-type cosine \
      --warmup-ratio 0.03 \
      --max-grad-norm 0 \
      --logging-steps 10 \
      --save-strategy no \
      --bf16 \
      --base-dtype auto \
      --master-dtype fp32 \
      --seed "$seed" >"$train_log" 2>&1
    validate_run "$kind" "$seed" "$run_dir" "$entry_sha"
    echo "DONE train run=$run_name time=$(date --iso-8601=seconds)"
  fi

  evaluate_split "$kind" "$seed" "$gpu" "$run_dir" dev "$DEV"
  evaluate_split "$kind" "$seed" "$gpu" "$run_dir" test "$TEST"
}

[[ -x "$PY" ]] || die "Python environment is not executable: $PY"
command -v sha256sum >/dev/null || die "sha256sum is required"
command -v flock >/dev/null || die "flock is required"
command -v nvidia-smi >/dev/null || die "nvidia-smi is required"
[[ -s "$ENTRY" ]] || die "training entry is missing: $ENTRY"
[[ -s "$EVALUATOR" ]] || die "dedicated evaluator is missing: $EVALUATOR"

require_sha "$TRAIN" "$EXPECTED_TRAIN_SHA"
require_sha "$DEV" "$EXPECTED_DEV_SHA"
require_sha "$TEST" "$EXPECTED_TEST_SHA"
require_sha "$MODEL/config.json" "$EXPECTED_MODEL_CONFIG_SHA"
require_sha "$ROOT/corrected_sft_experiment/data_pipeline.py" "$EXPECTED_PIPELINE_SHA"

read -r -a GPUS <<<"${GPU_IDS:-0 1 2 3 4 5}"
(( ${#GPUS[@]} == 6 )) || die "GPU_IDS must contain exactly six GPU indices"
declare -A SEEN_GPUS=()
mapfile -t INSTALLED_GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
(( ${#INSTALLED_GPUS[@]} > 0 )) || die "nvidia-smi returned no installed GPUs"
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || die "GPU index is not a non-negative integer: $gpu"
  [[ -z "${SEEN_GPUS[$gpu]:-}" ]] || die "GPU_IDS contains duplicate index: $gpu"
  SEEN_GPUS[$gpu]=1
  found=false
  for installed in "${INSTALLED_GPUS[@]}"; do
    [[ "$gpu" == "$installed" ]] && found=true
  done
  [[ "$found" == true ]] || die "GPU index $gpu is not installed"
done

for seed in 42 43 44; do
  validate_source_checkpoint "$seed"
done

ENTRY_SHA=$(sha256_file "$ENTRY")
EVALUATOR_SHA=$(sha256_file "$EVALUATOR")
mkdir -p "$OUT" "$LOG"

"$PY" - \
  "$LAUNCH_MANIFEST" "$ROOT" "$ENTRY" "$ENTRY_SHA" "$EVALUATOR" "$EVALUATOR_SHA" \
  "$EXPECTED_TRAIN_SHA" "$EXPECTED_DEV_SHA" "$EXPECTED_TEST_SHA" \
  "$EXPECTED_MODEL_CONFIG_SHA" "${GPUS[*]}" <<'PY'
import json
import sys
from pathlib import Path

(
    manifest_text, root_text, entry_text, entry_sha, evaluator_text, evaluator_sha,
    train_sha, dev_sha, test_sha, model_config_sha, gpu_text,
) = sys.argv[1:]
root = Path(root_text).resolve()
payload = {
    "marker": "fixed_hidden_boundary_fp32_qwen25_launch_v1",
    "protocol": str((root / "reviewer_followup/fixed_hidden_boundary_fp32_qwen25/DESIGN.md").resolve()),
    "training_entry": str(Path(entry_text).resolve()),
    "training_entry_sha256": entry_sha,
    "evaluator": str(Path(evaluator_text).resolve()),
    "evaluator_sha256": evaluator_sha,
    "data_sha256": {"train": train_sha, "dev": dev_sha, "test": test_sha},
    "model_config_sha256": model_config_sha,
    "gpu_ids": [int(value) for value in gpu_text.split()],
    "source_adapter_sha256": {
        "42": "354ed4961fb4d6dc7cf77d5e1e0ef528726b90cd71776e533088fa6d9b77e167",
        "43": "16958535a58d7cb3dd6c536fd1ff89beb7f88e053a9a67fb20eeb8a90135a093",
        "44": "2890f3f5c0eef7104616179d36916aa3ce1999047247e27e06d2c31b4d379476",
    },
    "fixed_endpoints": {
        "alora": {"rank": 50, "alpha": 800, "scale": 16},
        "vocab_lora": {"rank": 1, "alpha": 32, "scale": 32},
    },
    "endpoint_selection": "none_in_this_experiment",
}
manifest = Path(manifest_text)
if manifest.exists():
    existing = json.loads(manifest.read_text(encoding="utf-8"))
    if existing != payload:
        raise RuntimeError(
            "Existing launch_manifest.json does not match the current protocol; "
            "use a new experiment directory rather than mixing artifacts"
        )
else:
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest)
PY

TASKS=(
  "alora 42 ${GPUS[0]}"
  "vocab_lora 42 ${GPUS[1]}"
  "alora 43 ${GPUS[2]}"
  "vocab_lora 43 ${GPUS[3]}"
  "alora 44 ${GPUS[4]}"
  "vocab_lora 44 ${GPUS[5]}"
)

pids=()
labels=()
for task in "${TASKS[@]}"; do
  read -r kind seed gpu <<<"$task"
  run_one "$kind" "$seed" "$gpu" "$ENTRY_SHA" &
  pids+=("$!")
  labels+=("$kind/seed$seed/gpu$gpu")
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "FAILED ${labels[$index]}" >&2
    failed=1
  fi
done
(( failed == 0 )) || die "one or more fixed-hidden jobs failed"

for seed in 42 43 44; do
  for kind in alora vocab_lora; do
    run_dir=$OUT/$(run_name_for "$kind" "$seed")
    validate_run "$kind" "$seed" "$run_dir" "$ENTRY_SHA"
    validate_report "$run_dir/dev_report.json" "$run_dir" "$DEV" "$seed"
    validate_report "$run_dir/test_report.json" "$run_dir" "$TEST" "$seed"
  done
done

"$PY" - "$OUT" "$COMPLETION_MANIFEST" "$ENTRY_SHA" "$EVALUATOR_SHA" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

out = Path(sys.argv[1]).resolve()
target = Path(sys.argv[2])
entry_sha, evaluator_sha = sys.argv[3:]
runs = {}
for seed in (42, 43, 44):
    pair_frozen_hashes = set()
    pair_hidden_hashes = set()
    for kind in ("alora", "vocab_lora"):
        suffix = "alora_r50_s16" if kind == "alora" else "vocab_lora_r1_s32"
        name = f"qwen25_15b_fhfp32_hsd{seed}_{suffix}_bsd{seed}"
        directory = out / name
        config = json.loads((directory / "fixed_boundary_config.json").read_text(encoding="utf-8"))
        audit = json.loads((directory / "fixed_hidden_boundary_audit.json").read_text(encoding="utf-8"))
        dev = json.loads((directory / "dev_report.json").read_text(encoding="utf-8"))
        test = json.loads((directory / "test_report.json").read_text(encoding="utf-8"))
        pair_frozen_hashes.add(config["frozen_native_weight_sha256"])
        pair_hidden_hashes.add(audit["saved_hidden_canonical_tensor_sha256"])
        runs[name] = {
            "seed": seed,
            "kind": kind,
            "source_hidden_canonical_sha256": audit["saved_hidden_canonical_tensor_sha256"],
            "frozen_native_weight_sha256": config["frozen_native_weight_sha256"],
            "boundary_file_sha256": audit["saved_boundary_file_sha256"],
            "dev_avg_ce": dev["avg_ce"],
            "test_avg_ce": test["avg_ce"],
            "dev_report_sha256": hashlib.sha256((directory / "dev_report.json").read_bytes()).hexdigest(),
            "test_report_sha256": hashlib.sha256((directory / "test_report.json").read_bytes()).hexdigest(),
        }
    if len(pair_frozen_hashes) != 1:
        raise RuntimeError(f"seed {seed}: paired treatments used different frozen lm_head weights")
    if len(pair_hidden_hashes) != 1:
        raise RuntimeError(f"seed {seed}: paired treatments used different frozen hidden LoRA")

payload = {
    "marker": "fixed_hidden_boundary_fp32_qwen25_complete_v1",
    "training_entry_sha256": entry_sha,
    "evaluator_sha256": evaluator_sha,
    "pair_identity_assertion": "passed",
    "all_artifact_validation_assertion": "passed",
    "runs": runs,
}
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
temporary.replace(target)
PY

echo "ALL DONE: $COMPLETION_MANIFEST"
