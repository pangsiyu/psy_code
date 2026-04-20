from modelscope import snapshot_download

# 下载模型到指定目录（这里设为项目的 checkpoints/JanusVLN_Extra 下）
model_dir = snapshot_download(
    'misstl/JanusVLN_Extra',  # 模型在 ModelScope 的 ID
    cache_dir='./checkpoints',  # 本地保存的根目录
    revision='master'  # 版本（默认 master 即可）
)

print(f"模型已下载至：{model_dir}")