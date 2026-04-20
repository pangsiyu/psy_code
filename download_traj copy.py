from modelscope.hub.snapshot_download import snapshot_download
import os

print("🚀 开始通过 SDK 连接 ModelScope 服务器...")
print("💡 提示：如果中途网络断开报错，只要重新运行此脚本，就会自动断点续传！")

# local_dir 参数的作用是让下载下来的文件直接放在这个目录，
# 完全模拟 git clone 的效果，避免默认下载到系统 C盘/根目录 撑爆硬盘。
dataset_dir = snapshot_download(
    'misstl/JanusVLN_Trajectory_Data', 
    repo_type='dataset', 
    local_dir='./JanusVLN_Trajectory_Data' 
)

print(f"✅ 数据集已成功完整下载！文件保存在: {dataset_dir}")