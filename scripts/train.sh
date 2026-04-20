#!/bin/bash

# ================= 指定显卡配置 =================
export CUDA_VISIBLE_DEVICES=0,1,4,5
NPROC_PER_NODE=4

# === 环境变量：允许超长超时，并放开通信限制 ===
export NCCL_TIMEOUT=14400
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.6,max_split_size_mb:128"

# ================= 路径配置 =================
MODEL_PATH="./checkpoints/Qwen2.5-VL-7B" 
VGGT_MODEL_PATH="./checkpoints/VGGT/vggt_base.pth"

# 🟢 当前默认进行的是【带 Topo 头 (WithTopo)】的实验
OUTPUT_DIR="./JanusVLN_Ablation_NoTopo"                  
CACHE_DIR="./cache"                        
mkdir -p $OUTPUT_DIR

# ================= 动态生成 终极 ZeRO-3 配置文件 =================
# 策略：ZeRO-3 极致切割显存，坚决不 offload 到 CPU
cat <<EOT > scripts/zero3_no_offload.json
{
  "fp16": {
    "enabled": "auto",
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "bf16": {
    "enabled": "auto"
  },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": false
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "steps_per_print": 10,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false
}
EOT

# ================= 启动训练 =================
torchrun --nproc_per_node=$NPROC_PER_NODE \
            src/qwen_vl/train/train_qwen.py \
            --model_name_or_path $MODEL_PATH \
            --vggt_model_path $VGGT_MODEL_PATH \
            --tune_mm_llm False \
            --tune_mm_vision False \
            --tune_mm_mlp True \
            --dataset_use "train_r2r_rxr" \
            --unlabeled_data_path "" \
            --output_dir $OUTPUT_DIR \
            --cache_dir $CACHE_DIR \
            --bf16 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 4 \
            --learning_rate 2e-5 \
            --mm_projector_lr 1e-5 \
            --vision_tower_lr 1e-6 \
            --optim adamw_bnb_8bit \
            --model_max_length 2048 \
            --data_flatten False \
            --max_pixels $((60*28*28)) \
            --min_pixels $((16*28*28)) \
            --num_train_epochs 1 \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --weight_decay 0.01 \
            --logging_steps 1 \
            --save_strategy "epoch" \
            --save_total_limit 1 \
            --deepspeed "scripts/zero3_no_offload.json" \
            --gradient_checkpointing \
            --dataloader_num_workers 0 \
            --group_by_modality_length true \
            --seed 42 \
            --report_to "none" \
            --reference_frame first \
            > ${OUTPUT_DIR}/train.log 2>&1 &