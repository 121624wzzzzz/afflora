#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="${IM_EXP_ENV:-$REPO_ROOT/../set}"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi
set -u
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
MODEL_ROOT="${MODEL_ROOT:-$REPO_ROOT/../models}"
TRAIN_DATA="${TRAIN_DATA:-$REPO_ROOT/../minimind/dataset/lora_exam.jsonl}"
PY="${PYTHON_BIN:-python}"

cd "$REPO_ROOT"
"$PY" scripts/train_affine_vocab_lora.py \
  --model-path "$MODEL_ROOT/Qwen3-0.6B-Base" \
  --train-data "$TRAIN_DATA" \
  --output-dir outputs/affine_vocab/smoke/qwen3_0_6b/affine_input \
  --variant affine_input \
  --max-seq-len 256 \
  --max-train-samples 8 \
  --max-steps 2 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --learning-rate 2e-4 \
  --affine-rank 4 \
  --affine-alpha 8 \
  --bf16 \
  --gradient-checkpointing
