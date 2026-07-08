#!/usr/bin/env bash
set -euo pipefail
cd /commondocument/wz/cross_encoder_workspace/im_exp/lora
/home/wz/anaconda3/envs/torch24/bin/python corrected_math_evaluation/model_families/qwen3/close_size/hidden_then_mergeable/run_targeted_multiseed.py
