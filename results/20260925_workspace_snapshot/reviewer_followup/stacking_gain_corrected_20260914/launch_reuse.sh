#!/usr/bin/env bash
set -euo pipefail
cd /commondocument/wz/cross_encoder_workspace/im_exp/lora/reviewer_followup/stacking_gain_corrected_20260914
export CUDA_HOME=/home/wz/anaconda3/envs/torch24
export LD_LIBRARY_PATH=/home/wz/anaconda3/envs/torch24/lib:${LD_LIBRARY_PATH:-}
export DS_IGNORE_CUDA_DETECTION=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec /home/wz/anaconda3/envs/torch24/bin/python -u audit_reuse.py run --gpus 0,1,2,3,4,5,6,7 >> reuse_audit/scheduler.log 2>&1
