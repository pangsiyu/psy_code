import os
import re
import json

# ================= 配置搜索区域 =================
SEARCH_DIRS = ["src", "configs", "scripts"]
KEYWORDS = [
    "max_pixels", "min_pixels", "video_max_frame_pixels",
    "image_processor", "tokenizer",
    "ACTION", "STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT",
    "from_pretrained", "load_state_dict",
    "def forward", "class JanusVLN", "class Agent"
]

# 存储结果
findings = {
    "image_params": [],
    "tokenizer_info": [],
    "action_space": [],
    "model_loading": [],
    "model_definition": []
}

def search_in_file(filepath):
    """在单个文件中搜索关键词"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except:
        return

    for i, line in enumerate(lines):
        line_lower = line.lower()
        # 搜索图像参数
        if any(k in line for k in ["max_pixels", "min_pixels", "video_max_frame_pixels", "merge_size", "patch_size"]):
            # 提取前后3行作为上下文
            context = lines[max(0, i-3): min(len(lines), i+4)]
            findings["image_params"].append({
                "file": filepath,
                "line": i+1,
                "code": "".join(context)
            })
        
        # 搜索动作空间
        if any(k in line for k in ["ACTION", "STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]) and "=" in line:
            findings["action_space"].append({
                "file": filepath,
                "line": i+1,
                "code": line.strip()
            })

        # 搜索模型加载
        if "load_state_dict" in line or "from_pretrained" in line:
            context = lines[max(0, i-2): min(len(lines), i+5)]
            findings["model_loading"].append({
                "file": filepath,
                "line": i+1,
                "code": "".join(context)
            })
            
        # 搜索模型定义
        if "class " in line and ("Model" in line or "Agent" in line or "Janus" in line):
            findings["model_definition"].append({
                "file": filepath,
                "line": i+1,
                "code": line.strip()
            })

def main():
    print("🔍 Starting to probe the project structure...")
    print(f"📂 Searching in directories: {SEARCH_DIRS}")
    
    for search_dir in SEARCH_DIRS:
        if not os.path.exists(search_dir):
            continue
        for root, _, files in os.walk(search_dir):
            for file in files:
                if file.endswith('.py') or file.endswith('.yaml') or file.endswith('.yml') or file.endswith('.sh'):
                    search_in_file(os.path.join(root, file))

    # 生成报告
    report_file = "project_probe_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(findings, f, indent=4, ensure_ascii=False)
    
    print("✅ Probe finished!")
    print(f"📄 Report generated: {report_file}")
    print("\n" + "="*50)
    print("📋 Quick Summary:")
    
    print(f"  - Image/Processor params found in: {len(findings['image_params'])} places")
    if findings['image_params']:
        print(f"    Example: {findings['image_params'][0]['file']}")
        
    print(f"  - Action space clues found in: {len(findings['action_space'])} places")
    if findings['action_space']:
        print(f"    Example: {findings['action_space'][0]['code']}")
        
    print(f"  - Model loading logic found in: {len(findings['model_loading'])} places")
    print(f"  - Model classes found in: {len(findings['model_definition'])} places")
    
    print("\nPlease send me the content of 'project_probe_report.json' or just paste the 'Quick Summary' part here, and I will write the perfect inference code for you!")

if __name__ == "__main__":
    main()