# 🔴 指定你要使用的 3 张显卡的物理序号（根据你的实际空闲显卡修改）
export CUDA_VISIBLE_DEVICES=3,4,5,6

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
MASTER_PORT=$((RANDOM % 101 + 20000))

# 指向你的权重路径
CHECKPOINT="./JanusVLN_Ablation_WithTopo/checkpoint-187"
echo "CHECKPOINT: ${CHECKPOINT}"

# 输出路径
OUTPUT_PATH="evaluation_WithTopo_results"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"

CONFIG="config/vln_r2r.yaml"
echo "CONFIG: ${CONFIG}"

# 这里保持 nproc_per_node=3，它会自动映射到你上面指定的 3 张卡
torchrun --nproc_per_node=4 --master_port=$MASTER_PORT src/evaluation.py \
    --model_path $CHECKPOINT \
    --habitat_config_path $CONFIG \
    --num_history 8 \
    --model_max_length 8192 \
    --output_path $OUTPUT_PATH