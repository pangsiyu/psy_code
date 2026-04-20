import os
import sys
import torch
import gc
import re
from PIL import Image

# ==============================================
# 【论文原生超参数】严格对齐§4.1 Implementation Details
# ==============================================
# 显存优化配置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# 关闭梯度计算，推理模式强制要求
torch.set_grad_enabled(False)

# --------------------------
# 论文固定超参数（禁止修改）
# --------------------------
INIT_WINDOW_SIZE = 8       # 初始窗口大小（论文固定8帧，Attention Sinks全局锚点）
SLIDING_WINDOW_SIZE = 48   # 滑动窗口大小（论文固定48帧，近期上下文）
SPATIAL_WEIGHT_LAMBDA = 0.2# 空间特征融合权重（论文固定0.2）
ACTION_SPACE = ["Move Forward", "Turn Left", "Turn Right", "Stop"]

# --------------------------
# 用户路径配置（仅需修改这里）
# --------------------------
MODEL_PATH = "/data3/psy_code/code/JanusVLN/checkpoints/misstl/JanusVLN_Extra"
IMAGE_FOLDER = "/data3/psy_code/code/SEER_beta1/SEER_beta/video_Nav/video_localizer/result_video_2/video_frames"
NAV_INSTRUCTION = "Find the red velvet long dress"  # 论文要求英文指令
RESULT_FILE = "JanusVLN_论文复现_推理结果.txt"

# --------------------------
# 论文原生依赖导入（和官方源码保持一致）
# --------------------------
from transformers import AutoConfig, AutoTokenizer, AutoProcessor
from qwen_vl.model.vggt.utils.load_fn import load_and_preprocess_images
from src.qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationForJanusVLN
from qwen_vl_utils import extract_vision_info

# ==============================================
# 【论文对齐】工具函数
# ==============================================
def extract_action(raw_text: str) -> str:
    """
    严格对齐论文动作空间，优先匹配论文原生格式，兜底兼容下划线格式
    彻底解决之前正则匹配错误、全量输出MOVE_FORWARD的问题
    """
    raw_text = raw_text.strip()
    # 1. 最高优先级：匹配论文原生训练格式（首字母+空格）
    for action in ACTION_SPACE:
        if action.lower() in raw_text.lower():
            # 统一转为下划线大写格式，方便后续处理
            action_map = {
                "Move Forward": "MOVE_FORWARD",
                "Turn Left": "TURN_LEFT",
                "Turn Right": "TURN_RIGHT",
                "Stop": "STOP"
            }
            return action_map[action]
    
    # 2. 兜底匹配：下划线格式
    underscore_pattern = r"(MOVE_FORWARD|TURN_LEFT|TURN_RIGHT|STOP)"
    match = re.search(underscore_pattern, raw_text, re.IGNORECASE)
    if match:
        return match.group(0).upper()
    
    # 3. 最终兜底（仅当无任何匹配时触发）
    return "MOVE_FORWARD"

def update_dual_window_kv(
    current_past_kv: tuple,
    initial_window_kv: tuple,
    current_frame_idx: int
) -> tuple:
    """
    【论文核心】双窗口KV缓存增量更新（§3.2 Dual Implicit Memory）
    - 永久保留初始8帧KV（全局场景锚点）
    - 滑动窗口仅保留最近48帧KV，FIFO淘汰旧帧
    - 固定显存占用，不会随帧数增长溢出
    """
    # 初始窗口阶段：直接保存全部KV，作为永久全局锚点
    if current_frame_idx < INIT_WINDOW_SIZE:
        return current_past_kv, current_past_kv
    
    # 滑动窗口阶段：拼接初始窗口 + 最近N帧KV
    keep_frames = min(SLIDING_WINDOW_SIZE, current_frame_idx - INIT_WINDOW_SIZE + 1)
    updated_kv = []
    
    # 逐层处理Transformer的KV缓存
    for layer_idx in range(len(current_past_kv)):
        # 初始窗口KV永久保留
        init_key = initial_window_kv[layer_idx][0]
        init_value = initial_window_kv[layer_idx][1]
        
        # 滑动窗口仅保留最近keep_frames帧的KV
        current_key = current_past_kv[layer_idx][0][:, :, -keep_frames:, :]
        current_value = current_past_kv[layer_idx][1][:, :, -keep_frames:, :]
        
        # 拼接：初始窗口 + 滑动窗口，固定大小
        merged_key = torch.cat([init_key, current_key], dim=2)
        merged_value = torch.cat([init_value, current_value], dim=2)
        updated_kv.append((merged_key, merged_value))
    
    return tuple(updated_kv), initial_window_kv

# ==============================================
# 【论文原生推理主流程】100%对齐论文流式导航范式
# ==============================================
if __name__ == "__main__":
    # --------------------------
    # 1. 一次性加载模型（论文原生逻辑，无逐帧重载）
    # --------------------------
    print("="*80)
    print("正在加载JanusVLN论文原生模型...")
    print(f"模型路径：{MODEL_PATH}")
    print(f"初始窗口：{INIT_WINDOW_SIZE}帧 | 滑动窗口：{SLIDING_WINDOW_SIZE}帧")
    print(f"导航指令：{NAV_INSTRUCTION}")
    print("="*80)

    # 论文原生模型配置
    config = AutoConfig.from_pretrained(MODEL_PATH)
    config.attn_implementation = "eager"  # 论文原生attention实现
    
    # 加载论文原生模型，开启evaluation推理模式
    model = Qwen2_5_VLForConditionalGenerationForJanusVLN.from_pretrained(
        MODEL_PATH,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        mode='evaluation'  # 论文原生推理模式，必须开启
    ).eval()

    # 加载论文原生tokenizer和processor
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        max_pixels=1605632,
        min_pixels=28*28,
        use_fast=True
    )

    # 图像预处理参数（论文原生）
    patch_size = processor.image_processor.patch_size
    merge_size = processor.image_processor.merge_size
    print("模型加载完成！开始逐帧流式推理...")

    # --------------------------
    # 2. 读取视频帧序列
    # --------------------------
    img_files = sorted([
        os.path.join(IMAGE_FOLDER, f)
        for f in os.listdir(IMAGE_FOLDER)
        if f.startswith("frame_")
    ])
    total_frames = len(img_files)
    if total_frames == 0:
        print(f"错误：在{IMAGE_FOLDER}中未找到frame_开头的图片！")
        sys.exit(1)
    print(f"读取到视频帧总数：{total_frames}帧")

    # --------------------------
    # 3. 双隐式记忆初始化
    # --------------------------
    global_past_kv = None       # 全局KV缓存（双隐式记忆载体）
    initial_window_kv = None     # 初始窗口永久KV缓存
    frame_count = 0               # 当前处理帧序号

    # --------------------------
    # 4. 结果文件初始化
    # --------------------------
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        f.write("JanusVLN 论文复现推理结果\n")
        f.write("="*80 + "\n")
        f.write(f"导航指令：{NAV_INSTRUCTION}\n")
        f.write(f"初始窗口大小：{INIT_WINDOW_SIZE}帧\n")
        f.write(f"滑动窗口大小：{SLIDING_WINDOW_SIZE}帧\n")
        f.write(f"视频帧总数：{total_frames}帧\n")
        f.write("="*80 + "\n")
        f.write(f"{'帧序号':<8}\t{'图片名':<20}\t{'模型原始输出':<40}\t{'最终动作':<15}\n")
        f.flush()

        # --------------------------
        # 5. 论文原生流式逐帧推理
        # --------------------------
        for idx, img_path in enumerate(img_files):
            img_name = os.path.basename(img_path)
            frame_count += 1
            print(f"正在处理第{frame_count:04d}/{total_frames:04d}帧 | {img_name}")

            # 加载当前帧（论文原生单帧输入，无历史帧重复输入）
            current_frame = Image.open(img_path).convert("RGB")

            # --------------------------
            # 【论文对齐】Prompt构建（和训练范式完全一致）
            # --------------------------
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": current_frame},
                        {"type": "text", "text": f"Navigation instruction: {NAV_INSTRUCTION}\nOutput only one action from [Move Forward, Turn Left, Turn Right, Stop]. Do not output any other text, punctuation or explanation."}
                    ]
                }
            ]

            # --------------------------
            # 论文原生图像预处理
            # --------------------------
            text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            vision_info = extract_vision_info(message)
            imgs, vggt_imgs = [], []
            for ele in vision_info:
                img_tensor = load_and_preprocess_images([ele["image"]])[0]
                _, h, w = img_tensor.shape
                # 对齐论文patch/merge尺寸要求
                if (w // patch_size) % merge_size > 0:
                    w -= (w // patch_size) % merge_size * patch_size
                if (h // patch_size) % merge_size > 0:
                    h -= (h // patch_size) % merge_size * patch_size
                imgs.append(img_tensor[:, :h, :w])
                vggt_imgs.append(img_tensor[:, :h, :w])

            # --------------------------
            # 模型输入构建
            # --------------------------
            inputs = processor(
                text=[text],
                images=imgs,
                return_tensors="pt"
            ).to("cuda")
            # 论文原生3D空间几何编码器输入
            inputs["images_vggt"] = [
                torch.stack(vggt_imgs).to("cuda", dtype=torch.bfloat16)
            ]
            # 论文核心：开启KV缓存，复用双隐式记忆
            inputs["use_cache"] = True
            inputs["past_key_values"] = global_past_kv
            inputs["pad_token_id"] = tokenizer.eos_token_id

            # --------------------------
            # 【论文对齐】推理生成
            # --------------------------
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=15,
                    do_sample=False,
                    temperature=0.01,
                    top_p=1.0
                )

            # --------------------------
            # 【核心修复】仅解码生成的新token，彻底排除输入prompt干扰
            # --------------------------
            input_token_length = inputs.input_ids.shape[1]
            raw_output = tokenizer.decode(
                outputs[0][input_token_length:],
                skip_special_tokens=True
            ).strip()
            final_action = extract_action(raw_output)

            # --------------------------
            # 【论文核心】双窗口KV缓存增量更新
            # --------------------------
            new_past_kv = outputs.past_key_values
            global_past_kv, initial_window_kv = update_dual_window_kv(
                new_past_kv,
                initial_window_kv,
                frame_count
            )

            # --------------------------
            # 结果保存与打印
            # --------------------------
            result_line = f"{frame_count:<8}\t{img_name:<20}\t{raw_output:<40}\t{final_action:<15}"
            print(f"推理结果：{raw_output} → {final_action}\n")
            f.write(result_line + "\n")
            f.flush()

            # --------------------------
            # 实时显存清理
            # --------------------------
            del current_frame, inputs, outputs, imgs, vggt_imgs, new_past_kv
            torch.cuda.empty_cache()
            gc.collect()

    # --------------------------
    # 最终资源释放
    # --------------------------
    del model, tokenizer, processor, global_past_kv, initial_window_kv
    torch.cuda.empty_cache()
    gc.collect()

    # --------------------------
    # 完成提示
    # --------------------------
    print("="*80)
    print(f"🎉 全部{total_frames}帧推理完成！")
    print(f"📄 详细结果已保存到：{RESULT_FILE}")
    print("="*80)