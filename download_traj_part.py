from modelscope.hub.snapshot_download import snapshot_download
import os

print("🚀 开始通过 SDK 连接 ModelScope 服务器，精准拉取 R2R 数据...")
print("💡 提示：ModelScope SDK 原生支持断点续传！")
print("💡 只要 './JanusVLN_Trajectory_Data' 目录还在，重新运行就会自动接着下。")

# 核心修改：定义白名单过滤规则
# '*train_r2r_rxr.json'  匹配你的 5GB 标注文件
# '*R2R-CE-640x480*'     匹配 trajectory_data 目录下的 R2R 图像分卷压缩包 (part00, part01)
target_files = [
    '*train_r2r_rxr.json',
    '*R2R-CE-640x480*'
]

# 执行下载
dataset_dir = snapshot_download(
    'misstl/JanusVLN_Trajectory_Data', 
    repo_type='dataset', 
    local_dir='./JanusVLN_Trajectory_Data',
    allow_file_pattern=target_files  # <--- 注入白名单
)

print(f"✅ R2R 专属数据集已成功完整下载（或断点续传完成）！文件保存在: {dataset_dir}")