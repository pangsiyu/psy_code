#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

source scripts/stop_progress/gpu_guard.sh
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:64}"

PYTHON="${PYTHON:-/data3/psy_code/conda_envs/janusvln/bin/python}"
MODEL_PATH="${MODEL_PATH:-JanusVLN_Base}"
DATA_PATH="${DATA_PATH:-data/train_r2r_only_stop_ready_geo.jsonl}"
CACHE_DIR="${CACHE_DIR:-data/stop_head_cache/r2r_stop_ready_geo_2k}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/stop_head_stop_ready_geo_2k}"
MAX_SAMPLES="${MAX_SAMPLES:-2000}"
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-1}"
MAX_PIXELS="${MAX_PIXELS:-100352}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-4096}"
VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-4}"
VGGT_CACHE_START_SIZE="${VGGT_CACHE_START_SIZE:-4}"
VGGT_CACHE_RECENT_SIZE="${VGGT_CACHE_RECENT_SIZE:-8}"
DEVICE_MAP="${DEVICE_MAP:-janusvln_2gpu}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-1e-4}"

mkdir -p "${CACHE_DIR}" "${OUTPUT_DIR}"

"${PYTHON}" tools/extract_stop_head_hidden.py \
  --model_path "${MODEL_PATH}" \
  --data_path "${DATA_PATH}" \
  --output_dir "${CACHE_DIR}" \
  --max_samples "${MAX_SAMPLES}" \
  --batch_size "${EXTRACT_BATCH_SIZE}" \
  --max_pixels "${MAX_PIXELS}" \
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --video_max_frames "${VIDEO_MAX_FRAMES}" \
  --vggt_cache_start_size "${VGGT_CACHE_START_SIZE}" \
  --vggt_cache_recent_size "${VGGT_CACHE_RECENT_SIZE}" \
  --device_map "${DEVICE_MAP}" \
  --target_keys stop_ready,distance_to_goal,distance_progress \
  --sample_strategy balanced_stop_ready \
  --split_by traj_id

"${PYTHON}" tools/train_stop_head_from_hidden.py \
  --cache_dir "${CACHE_DIR}" \
  --output_path "${OUTPUT_DIR}/stop_ready_head.pt" \
  --target_key stop_ready \
  --loss_type bce \
  --stop_head_hidden_dim 1024 \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --split_by traj_id
