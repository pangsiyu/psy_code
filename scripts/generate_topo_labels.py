import os
import json
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import cv2

# ================= 1. 配置参数 (根据实际路径修改) =================
# 假设你在 JanusVLN 根目录运行此脚本
VIDEO_DIR = "./data/unlabeled_videos" # 存放无标注 MP4 和生成的 JPG 图片的目录
OUTPUT_JSON = "./data/topo_pseudo_labels.json"

# 【关键修改 1】使用你已经下载好的本地模型路径，避免联网报错！
MODEL_PATH = "./checkpoints/Qwen2.5-VL-7B" 

FRAME_INTERVAL = 30 # 假设30fps，每秒抽1帧

TOPO_CLASSES = {
    "0": "Narrow Corridor (狭窄走廊/通道)",
    "1": "Open Area (开阔区域/房间内部)",
    "2": "Intersection (十字路口/岔路)",
    "3": "Dead End (死胡同/前方障碍)",
    "4": "Doorway (门道/区域过渡)"
}

# ================= 2. 加载模型 =================
print(f"正在从本地 {MODEL_PATH} 加载 VLM 模型...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# ================= 3. 核心推理逻辑 =================
def extract_frames_and_save(video_path, interval):
    """提取视频帧，转为RGB供推理使用，同时【保存为JPG图片】供后续训练使用"""
    cap = cv2.VideoCapture(video_path)
    video_basename = os.path.basename(video_path).split('.')[0] # 获取不带后缀的视频名
    
    frames_for_inference = []
    count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if count % interval == 0:
            # 【关键修改 2】将抽取的帧保存到本地硬盘
            img_filename = f"{video_basename}_frame_{count}.jpg"
            img_save_path = os.path.join(VIDEO_DIR, img_filename)
            # cv2.imwrite 需要 BGR 格式，直接存 frame 即可
            cv2.imwrite(img_save_path, frame)
            
            # 转换为 RGB 存入内存供 Qwen 模型推理
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_for_inference.append({
                "frame_idx": count, 
                "image_rgb": frame_rgb,
                "saved_path": img_save_path # 记录保存的路径
            })
            
        count += 1
    cap.release()
    return frames_for_inference

def predict_topology(image_rgb):
    prompt_text = (
        "You are an embodied agent. From this first-person perspective, "
        "classify the spatial topology of the scene into exactly one of the following IDs:\n"
        "0: Narrow Corridor\n1: Open Area\n2: Intersection\n3: Dead End\n4: Doorway\n"
        "Output ONLY the integer ID (0, 1, 2, 3, or 4) without any other text."
    )
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_rgb},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=5)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
    
    # 提取输出中的第一个符合条件的数字
    for char in output_text:
        if char in TOPO_CLASSES.keys():
            return int(char)
    return 1 # 默认兜底为 Open Area

# ================= 4. 执行主循环 =================
if __name__ == "__main__":
    # 确保目录存在
    os.makedirs(VIDEO_DIR, exist_ok=True)
    
    # 最终保存为 { "图片路径": 类别ID } 的扁平化字典
    final_labels = {}
    
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
    print(f"找到 {len(video_files)} 个视频文件待处理。")
    
    for video_name in video_files:
        video_path = os.path.join(VIDEO_DIR, video_name)
        print(f"\n正在处理视频: {video_name}")
        
        # 抽帧并保存到本地
        frames_data = extract_frames_and_save(video_path, FRAME_INTERVAL)
        print(f"  已成功提取并保存 {len(frames_data)} 张图片到 {VIDEO_DIR}")
        
        # 遍历推理每一帧图片
        for f_data in frames_data:
            topo_id = predict_topology(f_data["image_rgb"])
            
            # 【关键修改 3】直接把图片的物理路径作为 Key，方便 Dataset 读取
            final_labels[f_data["saved_path"]] = topo_id
            
            print(f"  - 帧 {f_data['frame_idx']} 预测完成: {TOPO_CLASSES[str(topo_id)]}")

    # 将所有带标签的路径保存为 JSON
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_labels, f, indent=4)
        
    print(f"\n大功告成！所有伪标签已保存至 {OUTPUT_JSON}！")
    print(f"你可以检查 {VIDEO_DIR} 目录，现在里面不仅有视频，还有被模型打过标签的一张张 JPG 图片了。")