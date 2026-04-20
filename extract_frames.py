import cv2
import os
import json
import random

print("🚀 开始执行视频抽帧与无标注数据集构建...")

# ================= 配置区 =================
# 输入视频路径
VIDEO_PATH = "/data3/psy_code/code/JanusVLN/datasets/unlabeled_videos/868769379-1-208.mp4"

# 图片保存目录 (与我们之前 train.sh 中约定的 data/ 目录保持一致)
OUTPUT_IMG_DIR = "/data3/psy_code/code/JanusVLN/data/unlabeled_images"

# 自动生成的伪标签 JSON 保存路径
JSON_OUTPUT_PATH = "/data3/psy_code/code/JanusVLN/data/topo_pseudo_labels.json"

# 抽帧频率：每秒抽几帧？(推荐 1，即每秒 1 帧，保证空间变化足够大)
EXTRACT_FPS = 1  
# ==========================================

def main():
    # 确保输出目录存在
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ 错误：无法打开视频文件 {VIDEO_PATH}")
        return

    # 获取视频的原始帧率 (通常是 30 或 60)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0:
        original_fps = 30 # 兜底默认值
        
    # 计算每隔多少帧抽一次
    frame_interval = max(1, int(original_fps / EXTRACT_FPS))
    
    # 获取视频去后缀的名字，方便给图片命名
    video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    
    topo_dict = {}
    frame_count = 0
    saved_count = 0

    print(f"🎬 视频原始帧率: {original_fps:.2f} FPS")
    print(f"✂️  设置抽帧频率: {EXTRACT_FPS} FPS (每隔 {frame_interval} 帧抽取一张)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break # 视频读取完毕
            
        # 达到间隔要求，保存图片
        if frame_count % frame_interval == 0:
            # 命名格式: 视频名_frame_00001.jpg
            img_name = f"{video_name}_frame_{saved_count:05d}.jpg"
            img_abs_path = os.path.join(OUTPUT_IMG_DIR, img_name)
            
            # 保存图片到硬盘
            cv2.imwrite(img_abs_path, frame)
            
            # 记录到 JSON 字典中 (使用相对路径，适配模型训练读取逻辑)
            rel_path = f"data/unlabeled_images/{img_name}"
            
            # ⚠️ 这里给图片随机打上 0~4 的伪标签作为占位符。
            # 后续你可以用视觉大模型 (如 CLIP/Qwen) 批量推理这些图片，替换这里的随机值！
            topo_dict[rel_path] = random.randint(0, 4) 
            
            saved_count += 1
            if saved_count % 50 == 0:
                print(f"   已抽取 {saved_count} 张图片...")
                
        frame_count += 1

    cap.release()
    print(f"✅ 抽帧完成！共提取了 {saved_count} 张单帧图片，存放在: {OUTPUT_IMG_DIR}")

    # 将字典写入 JSON 文件
    with open(JSON_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(topo_dict, f, indent=4)
        
    print(f"✅ 伪标签 JSON 文件已自动生成: {JSON_OUTPUT_PATH}")
    print("💡 提示: 现在你可以直接运行 bash scripts/train.sh 测试完整 Pipeline 啦！")

if __name__ == "__main__":
    main()