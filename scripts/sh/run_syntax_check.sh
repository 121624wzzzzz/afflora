#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="${IM_EXP_ENV:-$REPO_ROOT/../set}"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE" >/dev/null
fi
PY="${PYTHON_BIN:-python}"
cd "$REPO_ROOT"
"$PY" -m py_compile \
  scripts/train_affine_vocab_lora.py \
  scripts/eval_base_loss.py \
  scripts/eval_merge_equivalence.py
