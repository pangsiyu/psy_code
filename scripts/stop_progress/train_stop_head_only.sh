#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

source scripts/stop_progress/gpu_guard.sh
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-14400}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.6,max_split_size_mb:128}"
export VGGT_CACHE_START_SIZE="${VGGT_CACHE_START_SIZE:-4}"
export VGGT_CACHE_RECENT_SIZE="${VGGT_CACHE_RECENT_SIZE:-8}"
export JANUSVLN_SAVE_FULL_MODEL="${JANUSVLN_SAVE_FULL_MODEL:-0}"

TORCHRUN="${TORCHRUN:-torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MODEL_PATH="${MODEL_PATH:-./JanusVLN_Base}"
VGGT_MODEL_PATH="${VGGT_MODEL_PATH:-none}"
DATASET_USE="${DATASET_USE:-train_r2r_only_stop_progress}"
OUTPUT_DIR="${OUTPUT_DIR:-./JanusVLN_StopHead_Only}"
CACHE_DIR="${CACHE_DIR:-./cache}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/zero3_offload.json}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1000}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
STOP_LOSS_WEIGHT="${STOP_LOSS_WEIGHT:-10.0}"
STOP_HEAD_HIDDEN_DIM="${STOP_HEAD_HIDDEN_DIM:-1024}"

mkdir -p "${OUTPUT_DIR}"

vggt_args=()
if [ "${VGGT_MODEL_PATH}" = "none" ]; then
  vggt_args=(--vggt_model_path "")
elif [ -n "${VGGT_MODEL_PATH}" ]; then
  vggt_args=(--vggt_model_path "${VGGT_MODEL_PATH}")
fi

"${TORCHRUN}" --nproc_per_node="${NPROC_PER_NODE}" \
  src/qwen_vl/train/train_qwen.py \
  --model_name_or_path "${MODEL_PATH}" \
  "${vggt_args[@]}" \
  --dataset_use "${DATASET_USE}" \
  --output_dir "${OUTPUT_DIR}" \
  --cache_dir "${CACHE_DIR}" \
  --bf16 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate "${LEARNING_RATE}" \
  --optim adamw_torch \
  --model_max_length 2048 \
  --data_flatten False \
  --max_samples "${MAX_SAMPLES}" \
  --max_pixels $((60*28*28)) \
  --min_pixels $((16*28*28)) \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --max_steps "${MAX_TRAIN_STEPS}" \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --weight_decay 0.01 \
  --logging_steps 1 \
  --save_strategy epoch \
  --save_total_limit 1 \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --gradient_checkpointing \
  --dataloader_num_workers 0 \
  --group_by_modality_length true \
  --seed 42 \
  --report_to none \
  --reference_frame first \
  --add_stop_progress_head True \
  --stop_head_hidden_dim "${STOP_HEAD_HIDDEN_DIM}" \
  --stop_loss_weight "${STOP_LOSS_WEIGHT}" \
  --trainable_scope stop_head_only \
  > "${OUTPUT_DIR}/train.log" 2>&1 &

echo "Training started in background. Log: ${OUTPUT_DIR}/train.log"
echo "Tail with: tail -f ${OUTPUT_DIR}/train.log"
