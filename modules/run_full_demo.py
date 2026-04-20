"""
完整可执行脚本：一键生成导航演示视频（适配你的箭头图片）
适配你的JSON路径：/data3/psy_code/code/SEER_beta1/SEER_beta/path/result/3/navigation_result.json
适配你的箭头路径：/data3/psy_code/code/SEER_beta1/SEER_beta/video_Nav/arrows/
"""
import sys
import os
import cv2
import json
import numpy as np

# ==================== 1. 路径配置（请根据实际情况修改） ====================
JANUSVLN_CODE_PATH = "/data3/psy_code/code/JanusVLN"
JANUSVLN_WEIGHTS_PATH = "/data3/psy_code/code/JanusVLN/checkpoints/misstl/JanusVLN_Extra/"

# 你的JSON文件路径
INPUT_JSON_PATH = "/data3/psy_code/code/SEER_beta1/SEER_beta/path/result/3/navigation_result.json"

# 你的录制视频路径（请修改为你实际的视频路径）
INPUT_VIDEO_PATH = "/data3/psy_code/code/JanusVLN/modules/demo.mp4"

# 你的箭头图片文件夹路径
ARROWS_FOLDER = "/data3/psy_code/code/SEER_beta1/SEER_beta/video_Nav/arrows/"

# 输出视频路径
OUTPUT_VIDEO_PATH = "/data3/psy_code/code/JanusVLN/final_navigation_demo_with_arrows.mp4"

# ==================== 2. 环境初始化 ====================
sys.path.insert(0, JANUSVLN_CODE_PATH)

# 检查文件是否存在
def check_file_exists(file_path, desc):
    if not os.path.exists(file_path):
        print(f"❌ 错误：{desc}不存在：{file_path}")
        return False
    print(f"✅ {desc}存在：{file_path}")
    return True

print("="*70)
print("🚀 JanusVLN 完整导航演示生成程序（带自定义箭头）")
print("="*70)

# 检查所有必要文件
all_exist = True
all_exist &= check_file_exists(INPUT_JSON_PATH, "JSON路线文件")
all_exist &= check_file_exists(INPUT_VIDEO_PATH, "录制视频")
all_exist &= check_file_exists(JANUSVLN_WEIGHTS_PATH, "JanusVLN权重目录")
all_exist &= check_file_exists(ARROWS_FOLDER, "箭头图片文件夹")

# 检查箭头图片是否存在
arrow_files = ["0.png", "45.png", "90.png", "135.png", "180.png"]
for arrow_file in arrow_files:
    arrow_path = os.path.join(ARROWS_FOLDER, arrow_file)
    all_exist &= check_file_exists(arrow_path, f"箭头图片 {arrow_file}")

if not all_exist:
    print("\n❌ 请检查路径配置后重新运行")
    sys.exit(1)

# 导入模块
try:
    from modules.janusvln_plugin import JanusVLNPlugin
    from modules.route_converter import generate_advanced_navigation_prompt
    print("✅ 模块导入成功")
except Exception as e:
    print(f"❌ 模块导入失败：{e}")
    sys.exit(1)

# ==================== 3. 核心功能函数 ====================
def load_json_data(json_path):
    """加载JSON数据"""
    print(f"\n[1/5] 正在加载JSON数据...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    path_len = len(data.get("path_details", []))
    user_query = data.get("user_query", "")[:50] + "..."
    print(f"✅ JSON加载成功！包含 {path_len} 个路径点")
    print(f"✅ 用户查询：{user_query}")
    return data

def load_arrow_images(arrows_folder):
    """预加载所有箭头图片"""
    arrows = {}
    # 映射关系：方向 -> 文件名
    arrow_mapping = {
        "向左": "0.png",
        "左前": "45.png",
        "向前": "90.png",
        "右前": "135.png",
        "向右": "180.png",
        "停止": "90.png"  # 停止时也用向前的箭头，或者你可以加一个停止的箭头
    }
    
    for direction, filename in arrow_mapping.items():
        arrow_path = os.path.join(arrows_folder, filename)
        # 读取箭头图片（支持透明通道）
        arrow_img = cv2.imread(arrow_path, cv2.IMREAD_UNCHANGED)
        if arrow_img is not None:
            arrows[direction] = arrow_img
            print(f"✅ 已加载箭头：{direction} -> {filename}")
        else:
            print(f"⚠️ 警告：无法加载箭头 {filename}")
    
    return arrows

def overlay_arrow_on_frame(frame, arrow_img, position=(0.5, 0.7), scale=0.3):
    """
    将箭头图片叠加到视频帧上（支持透明背景）
    
    参数:
        frame: 原始视频帧
        arrow_img: 箭头图片（带alpha通道）
        position: 箭头在帧上的位置 (x比例, y比例)，(0.5, 0.7) 表示居中偏下
        scale: 箭头大小缩放比例
    """
    h, w = frame.shape[:2]
    ah, aw = arrow_img.shape[:2]
    
    # 计算箭头的目标大小
    target_w = int(w * scale)
    target_h = int(ah * (target_w / aw))
    
    # 调整箭头大小
    arrow_resized = cv2.resize(arrow_img, (target_w, target_h))
    
    # 计算箭头位置
    x = int(w * position[0] - target_w / 2)
    y = int(h * position[1] - target_h / 2)
    
    # 确保不超出边界
    x = max(0, min(x, w - target_w))
    y = max(0, min(y, h - target_h))
    
    # 分离颜色通道和alpha通道
    if arrow_resized.shape[2] == 4:
        # 有透明通道
        b, g, r, a = cv2.split(arrow_resized)
        arrow_rgb = cv2.merge((b, g, r))
        alpha_mask = a / 255.0
        
        # 提取ROI
        roi = frame[y:y+target_h, x:x+target_w]
        
        # 混合：箭头 * alpha + 背景 * (1-alpha)
        for c in range(0, 3):
            roi[:, :, c] = (alpha_mask * arrow_rgb[:, :, c] + 
                            (1 - alpha_mask) * roi[:, :, c])
        
        # 放回原帧
        frame[y:y+target_h, x:x+target_w] = roi
    else:
        # 没有透明通道，直接覆盖
        frame[y:y+target_h, x:x+target_w] = arrow_resized
    
    return frame

def visualization_with_custom_arrows(frame, arrow_direction, janusvln_action, user_query, arrow_images):
    """
    使用你的自定义箭头的可视化函数
    """
    height, width = frame.shape[:2]
    
    # 1. 画目标文本
    if user_query:
        display_q = user_query[:60] + "..." if len(user_query) > 60 else user_query
        cv2.putText(frame, f"Target: {display_q}", (50, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    
    # 2. 画动作文本
    cv2.putText(frame, f"Direction: {arrow_direction}", (50, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(frame, f"Action: {janusvln_action}", (50, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # 3. 【核心】叠加你的箭头图片
    if arrow_direction in arrow_images:
        arrow_img = arrow_images[arrow_direction]
        # 叠加箭头：位置在画面下方居中，大小占画面宽度的30%
        frame = overlay_arrow_on_frame(frame, arrow_img, position=(0.5, 0.75), scale=0.3)
    
    return frame

# ==================== 4. 主执行流程 ====================
def main():
    # 1. 加载JSON
    json_data = load_json_data(INPUT_JSON_PATH)
    user_query = json_data.get("user_query", "")
    
    # 2. 生成导航Prompt
    print("\n[2/5] 正在生成导航Prompt...")
    nav_prompt = generate_advanced_navigation_prompt(json_data)
    print("✅ Prompt生成成功（前300字符）：")
    print(nav_prompt[:300] + "..." if len(nav_prompt) > 300 else nav_prompt)
    
    # 3. 预加载箭头图片
    print("\n[3/5] 正在预加载箭头图片...")
    arrow_images = load_arrow_images(ARROWS_FOLDER)
    if not arrow_images:
        print("❌ 错误：未加载到任何箭头图片")
        sys.exit(1)
    
    # 4. 初始化JanusVLN
    print("\n[4/5] 正在初始化JanusVLN...")
    try:
        agent = JanusVLNPlugin(JANUSVLN_WEIGHTS_PATH)
        agent.reset(nav_prompt)
        print("✅ JanusVLN初始化成功")
    except Exception as e:
        print(f"❌ JanusVLN初始化失败：{e}")
        print("💡 提示：请检查权重文件是否完整，CUDA是否可用")
        sys.exit(1)
    
    # 5. 打开视频
    print("\n[5/5] 正在打开视频...")
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ 无法打开视频：{INPUT_VIDEO_PATH}")
        sys.exit(1)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"✅ 视频信息：{width}x{height}, {fps}fps, 共 {total_frames} 帧")
    
    # 6. 创建输出视频
    print("\n[6/6] 正在创建输出视频...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
    print(f"✅ 输出视频路径：{OUTPUT_VIDEO_PATH}")
    
    # 7. 逐帧处理
    print("\n" + "="*70)
    print("开始逐帧处理...")
    print("="*70)
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # JanusVLN预测
        janusvln_action, arrow_direction = agent.predict(frame)
        
        # 【核心】使用你的自定义箭头可视化
        frame = visualization_with_custom_arrows(
            frame, arrow_direction, janusvln_action, user_query, arrow_images
        )
        
        # 写入视频
        out.write(frame)
        
        # 进度打印
        if frame_idx % 30 == 0:
            print(f"处理进度：{frame_idx}/{total_frames} | 当前方向：{arrow_direction}")
        
        # 提前停止
        if janusvln_action == "Stop":
            print("\n✅ JanusVLN决策停止，提前结束")
            break
    
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # 完成
    print("\n" + "="*70)
    print("🎉 处理完成！")
    print(f"📽️ 输出视频：{OUTPUT_VIDEO_PATH}")
    print("="*70)

if __name__ == "__main__":
    main()