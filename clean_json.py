import json

input_json = "data/train_r2r_rxr.json"
output_json = "data/train_r2r_only.json"

print("🚀 开始读取并清洗原作者的 JSON 标注文件...")

with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"📦 原始数据总量: {len(data)} 条")

filtered_data = []
author_prefix = "/mnt/nas-data-5/zengshuang.zs/data/"
our_prefix = "data/trajectory_data/"

for item in data:
    # 转成字符串，方便暴力全文替换和查找
    item_str = json.dumps(item)
    
    # 1. 过滤掉所有 RxR 的数据
    if "RxR-CE-640x480" in item_str:
        continue
        
    # 2. 擦除原作者的 NAS 绝对路径，换成我们的本地路径
    if author_prefix in item_str:
        item_str = item_str.replace(author_prefix, our_prefix)
        
    filtered_data.append(json.loads(item_str))

print(f"🎯 清洗完成！保留了纯 R2R 数据: {len(filtered_data)} 条")

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, indent=2)
    
print(f"✅ 干净的标注文件已保存至: {output_json}")