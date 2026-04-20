"""
全局路线转换工具：完全适配你的详细路径JSON结构
JSON路径示例：/data3/psy_code/code/SEER_beta1/SEER_beta/path/result/3/navigation_result.json
【核心修改】generate_advanced_navigation_prompt 改为直接返回用户原始指令，符合原论文JanusVLN要求
"""
from typing import List, Dict, Any

def generate_advanced_navigation_prompt(json_data: Dict[str, Any]) -> str:
    """
    【核心修改】直接返回用户的原始指令，完全符合原论文JanusVLN的输入要求
    原论文的标准输入：目标描述型指令（如"Go to the kitchen"），不是路线约束型指令
    """
    # 1. 从JSON中提取用户的原始导航指令
    user_query = json_data.get("user_query", "Go to the destination")
    
    # 2. 直接返回，不做任何路线信息的拼接（完全符合原论文JanusVLN的训练和推理逻辑）
    final_prompt = user_query.strip()
    
    # 3. 调试打印：确认生成的Prompt是用户原始指令
    print("="*70)
    print("【调试】生成的导航Prompt（符合原论文JanusVLN要求）：")
    print(final_prompt)
    print("="*70)
    
    return final_prompt

# -------------------------- 保留原有兼容函数（不影响之前的代码） --------------------------
def parse_your_json_data(json_data: Dict[str, Any]) -> List[Dict]:
    """
    解析你的JSON数据为标准化格式（向后兼容）
    """
    path_details = json_data.get("path_details", [])
    standard_route = []
    for pose in path_details:
        standard_route.append({
            "step_number": pose.get("step_number"),
            "view_id": pose.get("view_id"),
            "desc": pose.get("description", f"路径点{pose.get('step_number', 0)}")
        })
    return standard_route

def generate_smart_description(json_data: Dict[str, Any], use_user_query: bool = True) -> str:
    """
    简化版描述生成（向后兼容）
    """
    path_details = json_data.get("path_details", [])
    user_query = json_data.get("user_query", "")
    
    if not path_details:
        return "从当前位置出发，沿路线前进，到达终点后停下"
    
    if len(path_details) <= 2:
        start_desc = path_details[0].get('description', '起点')
        end_desc = path_details[-1].get('description', '终点')
        if use_user_query and user_query:
            return f"从{start_desc}出发，找到以下目标后停下：{user_query}"
        else:
            return f"从{start_desc}出发，到达{end_desc}后停下"
    else:
        start_desc = path_details[0].get('description', '起点')
        end_desc = path_details[-1].get('description', '终点')
        middle_descs = [pose.get('description', '') for pose in path_details[1:-1]][:2]
        middle_str = "、".join(middle_descs) if middle_descs else ""
        
        if use_user_query and user_query:
            if middle_str:
                return f"从{start_desc}出发，途经{middle_str}，找到以下目标后停下：{user_query}"
            else:
                return f"从{start_desc}出发，找到以下目标后停下：{user_query}"
        else:
            if middle_str:
                return f"从{start_desc}出发，途经{middle_str}，到达{end_desc}后停下"
            else:
                return f"从{start_desc}出发，到达{end_desc}后停下"

def convert_route_to_text(route_sampling_points: List[Dict], start_key: str = "desc", point_key: str = "desc") -> str:
    """
    通用路线转文本函数（向后兼容）
    """
    if not route_sampling_points:
        return "从当前位置出发，沿路线前进，到达终点后停下"
    
    descriptions = []
    for i, point in enumerate(route_sampling_points):
        desc = point.get(point_key, f"路径点{i}")
        descriptions.append(desc)
    
    if len(descriptions) == 1:
        return f"到达{descriptions[0]}后停下"
    elif len(descriptions) == 2:
        return f"从{descriptions[0]}出发，到达{descriptions[1]}后停下"
    else:
        start_desc = descriptions[0]
        end_desc = descriptions[-1]
        middle_desc = "、".join(descriptions[1:-1])
        return f"从{start_desc}出发，途经{middle_desc}，到达{end_desc}后停下"