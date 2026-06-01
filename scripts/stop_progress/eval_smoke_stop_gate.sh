#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

source scripts/stop_progress/gpu_guard.sh
export MAGNUM_LOG="${MAGNUM_LOG:-quiet}"
export HABITAT_SIM_LOG="${HABITAT_SIM_LOG:-quiet}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:64}"
export VGGT_CACHE_START_SIZE="${VGGT_CACHE_START_SIZE:-8}"
export VGGT_CACHE_RECENT_SIZE="${VGGT_CACHE_RECENT_SIZE:-16}"

PYTHON="${PYTHON:-/data3/psy_code/conda_envs/janusvln/bin/python}"
CHECKPOINT="${CHECKPOINT:-JanusVLN_StopHead}"
OUTPUT_PATH="${OUTPUT_PATH:-evaluation_smoke_stop_gate}"
CONFIG="${CONFIG:-config/vln_r2r.yaml}"
EVAL_SPLIT="${EVAL_SPLIT:-val_unseen}"
NUM_HISTORY="${NUM_HISTORY:-8}"
MAX_PIXELS="${MAX_PIXELS:-100352}"
MAX_STEPS="${MAX_STEPS:-400}"
MAX_EVAL_EPISODES="${MAX_EVAL_EPISODES:-20}"
EPISODES_PER_SCENE="${EPISODES_PER_SCENE:-}"
EPISODE_SUBSET_JSON="${EPISODE_SUBSET_JSON:-data/eval_subsets/val_unseen_smoke20.json}"
MODEL_DEVICE_MAP="${MODEL_DEVICE_MAP:-janusvln_2gpu}"
MAX_MEMORY_PER_GPU="${MAX_MEMORY_PER_GPU:-22GiB}"
STOP_PROGRESS_THRESHOLD="${STOP_PROGRESS_THRESHOLD:-0.85}"
STOP_SUCCESS_RADIUS="${STOP_SUCCESS_RADIUS:-3.0}"
FORCE_ADD_STOP_PROGRESS_HEAD="${FORCE_ADD_STOP_PROGRESS_HEAD:-0}"
STOP_PROGRESS_HEAD_PATH="${STOP_PROGRESS_HEAD_PATH:-}"
SAVE_STEP_LOGS="${SAVE_STEP_LOGS:-1}"

args=(
  src/evaluation.py
  --model_path "${CHECKPOINT}"
  --habitat_config_path "${CONFIG}"
  --eval_split "${EVAL_SPLIT}"
  --num_history "${NUM_HISTORY}"
  --max_pixels "${MAX_PIXELS}"
  --max_steps "${MAX_STEPS}"
  --max_eval_episodes "${MAX_EVAL_EPISODES}"
  --model_device_map "${MODEL_DEVICE_MAP}"
  --max_memory_per_gpu "${MAX_MEMORY_PER_GPU}"
  --use_stop_progress_gate
  --stop_progress_threshold "${STOP_PROGRESS_THRESHOLD}"
  --stop_success_radius "${STOP_SUCCESS_RADIUS}"
  --invalid_action_fallback MOVE_FORWARD
  --stop_gate_fallback MOVE_FORWARD
  --output_path "${OUTPUT_PATH}"
)

if [ "${FORCE_ADD_STOP_PROGRESS_HEAD}" = "1" ]; then
  args+=(--force_add_stop_progress_head)
fi

if [ -n "${STOP_PROGRESS_HEAD_PATH}" ]; then
  args+=(--stop_progress_head_path "${STOP_PROGRESS_HEAD_PATH}")
fi

if [ -n "${EPISODES_PER_SCENE}" ]; then
  args+=(--episodes_per_scene "${EPISODES_PER_SCENE}")
fi

if [ -n "${EPISODE_SUBSET_JSON}" ]; then
  args+=(--episode_subset_json "${EPISODE_SUBSET_JSON}")
fi

if [ "${SAVE_STEP_LOGS}" = "1" ]; then
  args+=(--save_step_logs)
fi

"${PYTHON}" "${args[@]}"
