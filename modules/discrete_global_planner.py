import os
import torch
from PIL import Image
from typing import Optional
from src.qwen_vl.model import JanusVLN
from src.qwen_vl.data.processors import JanusVLNImageProcessor, JanusVLNTextProcessor
from src.qwen_vl.model.dual_memory import DualImplicitMemory

class JanusVLNAgent:
    def __init__(self, model_weights_path: str, device: str = "cuda"):
        self.device = device
        self.model_weights_path = model_weights_path
        
        # 论文原生超参数（保持不变）
        self.initial_window_size = 8
        self.sliding_window_size = 48
        self.image_resolution = (640, 480)
        self.action_space = ["Move Forward", "Turn Left", "Turn Right", "Stop"]
        
        # 加载模型（完全复用原生代码）
        print("[JanusVLN] 正在加载模型...")
        self.model = JanusVLN.from_pretrained(
            model_weights_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(device)
        self.model.eval()
        
        self.image_processor = JanusVLNImageProcessor(image_size=self.image_resolution)
        self.text_processor = JanusVLNTextProcessor.from_pretrained(model_weights_path)
        
        self.dual_memory = None
        self.history = None
        self.frame_idx = 0
        self.target_object = None  # 新增：保存目标物体名称

    def reset(self, target_object: str):
        """
        重置智能体，仅需目标物体名称，无需详细路线指令
        :param target_object: 目标物体，如"红丝绒长裙"
        """
        self.target_object = target_object
        self.dual_memory = DualImplicitMemory(
            initial_window_size=self.initial_window_size,
            sliding_window_size=self.sliding_window_size
        )
        self.history = None
        self.frame_idx = 0
        print(f"[JanusVLN] 智能体已重置，导航目标：{self.target_object}")

    def _build_constrained_prompt(self) -> str:
        """
        【核心】构造约束性 Prompt，强制 Qwen 按我们的要求输出
        无需详细路线指令，仅关注目标物体和当前视觉
        """
        prompt = f"""你是一个专业的视觉导航智能体。
【核心约束】
1. 你的唯一导航目标是：找到并靠近{self.target_object}。
2. 你不需要预先知道路线，只需根据当前看到的场景，自主判断下一步动作。
3. 如果你在当前画面中看到了{self.target_object}，请向它前进；如果没看到，请探索前进。
4. 当你认为已经非常靠近{self.target_object}时，输出"Stop"。
5. 你只能输出以下4种动作中的一种，禁止输出其他任何内容：Move Forward, Turn Left, Turn Right, Stop。

【当前任务】
找到{self.target_object}并在它旁边停下。

下一步动作："""
        return prompt

    def predict_action(self, image: Image.Image) -> str:
        """
        输入当前帧图像，输出动作
        """
        # 图像预处理（复用原生代码）
        pixel_values = self.image_processor(image, return_tensors="pt").to(
            self.device, torch.bfloat16
        )
        
        # 【核心】使用约束性 Prompt 替代原有的详细路线指令
        constrained_prompt = self._build_constrained_prompt()
        text_inputs = self.text_processor(
            constrained_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        # 模型推理（完全复用原生前向逻辑）
        with torch.no_grad():
            outputs = self.model.predict_step(
                pixel_values=pixel_values,
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
                dual_memory=self.dual_memory,
                frame_idx=self.frame_idx,
                history=self.history
            )
        
        # 解析动作
        action_logits = outputs.action_logits
        pred_action_idx = torch.argmax(action_logits, dim=-1).item()
        pred_action = self.action_space[pred_action_idx]
        
        # 更新状态
        self.history = outputs.history
        self.frame_idx += 1
        
        print(f"[JanusVLN] 第 {self.frame_idx} 帧，预测动作：{pred_action}")
        return pred_action