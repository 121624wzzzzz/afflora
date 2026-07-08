#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
"${PYTHON_BIN:-python}" corrected_math_evaluation/model_families/qwen25/qwen25_0p5b/hidden_rank_then_mergeable/run_hr8_ar4_multiseed.py
