#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

source scripts/stop_progress/gpu_guard.sh
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:64}"

PYTHON="${PYTHON:-/data3/psy_code/conda_envs/janusvln/bin/python}"
CACHE_DIR="${CACHE_DIR:-data/stop_head_cache/r2r_256}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/stop_head_hidden_256}"
MODEL_PATH="${MODEL_PATH:-JanusVLN_Base}"
DATA_PATH="${DATA_PATH:-data/train_r2r_only_stop_progress.jsonl}"
MAX_SAMPLES="${MAX_SAMPLES:-256}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_PIXELS="${MAX_PIXELS:-501760}"
VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-4}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-4096}"
VGGT_CACHE_START_SIZE="${VGGT_CACHE_START_SIZE:-4}"
VGGT_CACHE_RECENT_SIZE="${VGGT_CACHE_RECENT_SIZE:-8}"
DEVICE_MAP="${DEVICE_MAP:-janusvln_2gpu}"
MAX_MEMORY_PER_GPU="${MAX_MEMORY_PER_GPU:-}"
CPU_MEMORY="${CPU_MEMORY:-80GiB}"
OFFLOAD_FOLDER="${OFFLOAD_FOLDER:-}"
STOP_HEAD_HIDDEN_DIM="${STOP_HEAD_HIDDEN_DIM:-1024}"
EPOCHS="${EPOCHS:-20}"
HEAD_BATCH_SIZE="${HEAD_BATCH_SIZE:-256}"
LR="${LR:-1e-4}"
VAL_RATIO="${VAL_RATIO:-0.1}"

mkdir -p "${CACHE_DIR}"
mkdir -p "${OUTPUT_DIR}"

extract_args=(
  tools/extract_stop_head_hidden.py
  --model_path "${MODEL_PATH}"
  --data_path "${DATA_PATH}"
  --output_dir "${CACHE_DIR}"
  --max_samples "${MAX_SAMPLES}"
  --batch_size "${BATCH_SIZE}"
  --max_pixels "${MAX_PIXELS}"
  --video_max_frames "${VIDEO_MAX_FRAMES}"
  --model_max_length "${MODEL_MAX_LENGTH}"
  --vggt_cache_start_size "${VGGT_CACHE_START_SIZE}"
  --vggt_cache_recent_size "${VGGT_CACHE_RECENT_SIZE}"
  --device_map "${DEVICE_MAP}"
)

if [ -n "${MAX_MEMORY_PER_GPU}" ]; then
  extract_args+=(--max_memory_per_gpu "${MAX_MEMORY_PER_GPU}" --cpu_memory "${CPU_MEMORY}")
fi

if [ -n "${OFFLOAD_FOLDER}" ]; then
  extract_args+=(--offload_folder "${OFFLOAD_FOLDER}")
fi

"${PYTHON}" tools/extract_stop_head_hidden.py \
  "${extract_args[@]:1}"

"${PYTHON}" tools/train_stop_head_from_hidden.py \
  --cache_dir "${CACHE_DIR}" \
  --output_path "${OUTPUT_DIR}/stop_progress_head.pt" \
  --stop_head_hidden_dim "${STOP_HEAD_HIDDEN_DIM}" \
  --epochs "${EPOCHS}" \
  --batch_size "${HEAD_BATCH_SIZE}" \
  --lr "${LR}" \
  --val_ratio "${VAL_RATIO}"
