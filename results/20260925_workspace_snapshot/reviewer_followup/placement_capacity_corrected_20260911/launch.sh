#!/usr/bin/env bash
set -euo pipefail
cd /commondocument/wz/cross_encoder_workspace/im_exp/lora/reviewer_followup/placement_capacity_corrected_20260911
export CUDA_HOME=/home/wz/anaconda3/envs/torch24
export LD_LIBRARY_PATH=/home/wz/anaconda3/envs/torch24/lib:${LD_LIBRARY_PATH:-}
export DS_IGNORE_CUDA_DETECTION=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec /home/wz/anaconda3/envs/torch24/bin/python -u run.py run --gpus 1,2,3,4 >> scheduler.log 2>&1
