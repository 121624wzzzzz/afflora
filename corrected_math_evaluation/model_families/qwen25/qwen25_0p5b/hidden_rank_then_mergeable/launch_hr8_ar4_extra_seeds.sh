#!/usr/bin/env bash
set -euo pipefail
cd /commondocument/wz/cross_encoder_workspace/im_exp/lora
/home/wz/anaconda3/envs/torch24/bin/python corrected_math_evaluation/model_families/qwen25/qwen25_0p5b/hidden_rank_then_mergeable/run_hr8_ar4_extra_seeds.py
