#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

source scripts/stop_progress/gpu_guard.sh
export MAGNUM_LOG="${MAGNUM_LOG:-quiet}"
export HABITAT_SIM_LOG="${HABITAT_SIM_LOG:-quiet}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:64}"
export VGGT_CACHE_START_SIZE="${VGGT_CACHE_START_SIZE:-4}"
export VGGT_CACHE_RECENT_SIZE="${VGGT_CACHE_RECENT_SIZE:-8}"

PYTHON="${PYTHON:-/data3/psy_code/conda_envs/janusvln/bin/python}"
MODEL_PATH="${MODEL_PATH:-JanusVLN_Base}"
HEAD_PATH="${HEAD_PATH:-outputs/stop_head_stop_ready_geo_2k/stop_ready_head.pt}"
OUTPUT_PATH="${OUTPUT_PATH:-evaluation_stop_progress/dev20_stop_ready_geo_2k_score}"
EPISODE_SUBSET_JSON="${EPISODE_SUBSET_JSON:-data/eval_subsets/val_unseen_dev20_probe.json}"
MAX_STEPS="${MAX_STEPS:-30}"
MAX_PIXELS="${MAX_PIXELS:-100352}"
MODEL_DEVICE_MAP="${MODEL_DEVICE_MAP:-janusvln_2gpu}"
STOP_PROGRESS_THRESHOLD="${STOP_PROGRESS_THRESHOLD:-0.0}"

"${PYTHON}" src/evaluation.py \
  --model_path "${MODEL_PATH}" \
  --habitat_config_path config/vln_r2r.yaml \
  --eval_split val_unseen \
  --output_path "${OUTPUT_PATH}" \
  --episode_subset_json "${EPISODE_SUBSET_JSON}" \
  --max_steps "${MAX_STEPS}" \
  --model_device_map "${MODEL_DEVICE_MAP}" \
  --max_pixels "${MAX_PIXELS}" \
  --force_add_stop_progress_head \
  --stop_progress_head_path "${HEAD_PATH}" \
  --use_stop_progress_gate \
  --stop_progress_threshold "${STOP_PROGRESS_THRESHOLD}" \
  --invalid_action_fallback MOVE_FORWARD \
  --stop_gate_fallback MOVE_FORWARD \
  --save_step_logs
