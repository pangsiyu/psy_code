import os
import json
import numpy as np
import cv2

print("🚀 开始生成极简验证数据集 (Dummy Data)...")

# ==========================================
# 1. 生成有监督数据 (骗过原生 JanusVLN 逻辑)
# ==========================================
sup_base_dir = "data/trajectory_data/R2R-CE-640x480"
sup_img_dir = os.path.join(sup_base_dir, "images")
os.makedirs(sup_img_dir, exist_ok=True)

sup_images = []
# 生成 4 张纯灰色的假图片
for i in range(4):
    img_path = os.path.join(sup_img_dir, f"mock_sup_{i}.jpg")
    dummy_img = np.full((480, 640, 3), 128, dtype=np.uint8) # 128 灰色，防止全0引发梯度异常
    cv2.imwrite(img_path, dummy_img)
    # data_qwen.py 中通常会加上 data_path 拼接，所以这里存相对名称或视你 __init__.py 配置而定
    # 为了最保险，我们填相对工作目录的路径
    sup_images.append(img_path)

# 伪造符合 Qwen-VL SFT 格式的 JSON
dummy_supervised_data = [
    {
        "id": "mock_r2r_001",
        "image": [sup_images[0], sup_images[1]],
        "conversations": [
            {"from": "human", "value": "<image><image>\nWalk forward and turn left at the kitchen."},
            {"from": "gpt", "value": "Action 1, Action 2"}
        ]
    },
    {
        "id": "mock_r2r_002",
        "image": [sup_images[2], sup_images[3]],
        "conversations": [
            {"from": "human", "value": "<image><image>\nStop near the sofa."},
            {"from": "gpt", "value": "Action 0, Action 0"}
        ]
    }
]

sup_json_path = "data/train_r2r_rxr_mock.json"
with open(sup_json_path, "w", encoding="utf-8") as f:
    json.dump(dummy_supervised_data, f, indent=4)
print(f"✅ 有监督假数据生成完毕 -> {sup_json_path}")


# ==========================================
# 2. 生成半监督数据 (骗过你新增的 Topo 分支)
# ==========================================
unsup_img_dir = "data/unlabeled_videos"
os.makedirs(unsup_img_dir, exist_ok=True)

topo_labels_dict = {}
# 生成 2 张半监督假图片
for i in range(2):
    img_path = os.path.join(unsup_img_dir, f"mock_unsup_{i}.jpg")
    dummy_img = np.full((480, 640, 3), 64, dtype=np.uint8) # 64 暗灰色
    cv2.imwrite(img_path, dummy_img)
    # 随机分配拓扑类别 0~4
    topo_labels_dict[img_path] = i % 5

unsup_json_path = "data/topo_pseudo_labels.json"
with open(unsup_json_path, "w", encoding="utf-8") as f:
    json.dump(topo_labels_dict, f, indent=4)
print(f"✅ 半监督假数据生成完毕 -> {unsup_json_path}")
print("🎉 所有 Dummy Data 准备就绪，可以进行 Dry Run 训练啦！")