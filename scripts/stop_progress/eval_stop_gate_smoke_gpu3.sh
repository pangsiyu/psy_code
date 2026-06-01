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
CHECKPOINT="${CHECKPOINT:-./JanusVLN_Base}"
STOP_PROGRESS_HEAD_PATH="${STOP_PROGRESS_HEAD_PATH:-./JanusVLN_StopHead_Smoke_No37/stop_progress_head.pt}"
OUTPUT_PATH="${OUTPUT_PATH:-evaluation_stop_progress/smoke_no37_stop_gate}"
CONFIG="${CONFIG:-config/vln_r2r.yaml}"
EVAL_SPLIT="${EVAL_SPLIT:-val_unseen}"
NUM_HISTORY="${NUM_HISTORY:-4}"
MAX_PIXELS="${MAX_PIXELS:-50176}"
MAX_STEPS="${MAX_STEPS:-10}"
MAX_EVAL_EPISODES="${MAX_EVAL_EPISODES:-3}"
EPISODE_SUBSET_JSON="${EPISODE_SUBSET_JSON:-data/eval_subsets/val_unseen_smoke20.json}"
MODEL_DEVICE_MAP="${MODEL_DEVICE_MAP:-janusvln_2gpu}"
STOP_PROGRESS_THRESHOLD="${STOP_PROGRESS_THRESHOLD:-0.85}"
STOP_SUCCESS_RADIUS="${STOP_SUCCESS_RADIUS:-3.0}"

"${PYTHON}" src/evaluation.py \
  --model_path "${CHECKPOINT}" \
  --habitat_config_path "${CONFIG}" \
  --eval_split "${EVAL_SPLIT}" \
  --num_history "${NUM_HISTORY}" \
  --max_pixels "${MAX_PIXELS}" \
  --max_steps "${MAX_STEPS}" \
  --max_eval_episodes "${MAX_EVAL_EPISODES}" \
  --model_device_map "${MODEL_DEVICE_MAP}" \
  --use_stop_progress_gate \
  --stop_progress_threshold "${STOP_PROGRESS_THRESHOLD}" \
  --stop_success_radius "${STOP_SUCCESS_RADIUS}" \
  --invalid_action_fallback MOVE_FORWARD \
  --stop_gate_fallback MOVE_FORWARD \
  --force_add_stop_progress_head \
  --stop_progress_head_path "${STOP_PROGRESS_HEAD_PATH}" \
  --episode_subset_json "${EPISODE_SUBSET_JSON}" \
  --save_step_logs \
  --output_path "${OUTPUT_PATH}"
