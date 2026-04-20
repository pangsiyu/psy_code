"""
JanusVLN 插件：【极致显存压缩版】
优化目标：在不影响别人的前提下，把自己的显存占用压到最低
"""
import sys
import os
import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List

# ==================== 【极致显存压缩】全局配置 ====================
# 仅GPU加载
ONLY_GPU_LOAD = True
# 深度禁用FlashAttention
DEPTH_DISABLE_FLASH_ATTENTION = True
# 禁用梯度检查点
USE_GRAD_CHECKPOINT = False
# 半精度
USE_HALF_PRECISION = True
# 【1】极致低分辨率
INFERENCE_RESOLUTION = (64, 48)
# 【2】彻底禁用KV Cache（省最多显存）
DISABLE_ALL_CACHE = True
# 【3】最小记忆窗口
MEMORY_WINDOW_SIZE = 4

# ==================== 依赖检查 ====================
def check_dependencies():
    required_libs = {
        "fastdtw": "fastdtw",
        "gymnasium": "gymnasium",
        "transformers": "transformers",
        "torch": "torch",
        "numpy": "numpy"
    }
    missing_libs = []
    install_commands = []
    for import_name, install_name in required_libs.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_libs.append(import_name)
            install_commands.append(install_name)
    
    if missing_libs:
        print(f"❌ 缺失依赖库：{', '.join(missing_libs)}")
        sys.exit(1)

check_dependencies()

# ==================== 全局深度禁用FlashAttention ====================
def compat_autocast(dtype=None):
    if torch.__version__ >= "2.0.0":
        return torch.amp.autocast('cuda', dtype=dtype)
    else:
        return torch.cuda.amp.autocast(dtype=dtype)

if DEPTH_DISABLE_FLASH_ATTENTION:
    from transformers import AutoConfig, AutoModelForCausalLM
    original_config_from_pretrained = AutoConfig.from_pretrained
    def patched_config_from_pretrained(*args, **kwargs):
        config = original_config_from_pretrained(*args, **kwargs)
        if hasattr(config, "attn_implementation"):
            config.attn_implementation = "eager"
        config.use_cache = not DISABLE_ALL_CACHE
        config.gradient_checkpointing = USE_GRAD_CHECKPOINT
        return config
    AutoConfig.from_pretrained = patched_config_from_pretrained

    original_model_from_pretrained = AutoModelForCausalLM.from_pretrained
    def patched_model_from_pretrained(*args, **kwargs):
        kwargs.pop("attn_implementation", None)
        kwargs["attn_implementation"] = "eager"
        kwargs["device_map"] = {"": "cuda"}
        kwargs["torch_dtype"] = torch.float16
        kwargs["use_cache"] = not DISABLE_ALL_CACHE
        return original_model_from_pretrained(*args, **kwargs)
    AutoModelForCausalLM.from_pretrained = patched_model_from_pretrained

# 固定数值变量
MAX_PIXELS = INFERENCE_RESOLUTION[0] * INFERENCE_RESOLUTION[1]
MIN_PIXELS = 64 * 48
DEFAULT_PATCH_SIZE = 14
DEFAULT_MERGE_SIZE = 2

JANUSVLN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, JANUSVLN_ROOT)

# ==================== 核心导入 + 模型补丁 ====================
try:
    from src.dagger import JanusVLN_Inference as JanusVLN
    from src.qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationForJanusVLN
    
    def patched_qwen_from_pretrained(cls, pretrained, **kwargs):
        kwargs["device_map"] = {"": "cuda"}
        kwargs["attn_implementation"] = "eager"
        kwargs["torch_dtype"] = torch.float16
        kwargs["use_cache"] = not DISABLE_ALL_CACHE
        return super(Qwen2_5_VLForConditionalGenerationForJanusVLN, cls).from_pretrained(pretrained,** kwargs)
    
    Qwen2_5_VLForConditionalGenerationForJanusVLN.from_pretrained = classmethod(patched_qwen_from_pretrained)
    
    original_call_model = JanusVLN.call_model
    def patched_call_model(self, observations, task, step_id, add_frame_index=False, gen_kwargs={}):
        step_id = step_id if isinstance(step_id, int) else 0
        if hasattr(self.processor, "image_processor"):
            self.processor.image_processor.patch_size = DEFAULT_PATCH_SIZE
            self.processor.image_processor.merge_size = DEFAULT_MERGE_SIZE
            self.processor.image_processor.max_pixels = MAX_PIXELS
            self.processor.image_processor.min_pixels = MIN_PIXELS
        gen_kwargs["use_cache"] = not DISABLE_ALL_CACHE
        return original_call_model(self, observations, task, step_id, add_frame_index, gen_kwargs)
    
    JanusVLN.call_model = patched_call_model

except ImportError as e:
    print(f"[JanusVLN] 导入失败：{e}")
    raise

# ==================== 【极致压缩】主插件类 ====================
class JanusVLNPlugin:
    def __init__(
        self,
        model_weights_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        if ONLY_GPU_LOAD:
            if not torch.cuda.is_available():
                raise RuntimeError("仅GPU加载模式已启用，但GPU不可用！")
            device = "cuda"
        self.device = device
        
        # 【极致清理】初始化前彻底清空
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
            free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            print(f"[JanusVLN] 初始可用显存：{free_mem/1024/1024:.2f} MiB")
        
        self.image_resolution = INFERENCE_RESOLUTION
        self.action_space = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"]
        self.frame_idx = 0
        self.action_history = []
        self.current_prompt = ""
        
        # 最小化记忆
        self.memory_window_size = MEMORY_WINDOW_SIZE
        self.rgb_history: List[Image.Image] = []
        self.past_key_values = None
        
        print(f"[JanusVLN] 加载模型（极致压缩模式）...")
        try:
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            self.model = JanusVLN(
                pretrained=model_weights_path,
                device=self.device
            )
            
            if self.device == "cuda":
                used_mem = torch.cuda.memory_allocated() / 1024 / 1024
                print(f"[JanusVLN] 加载完成！显存占用：{used_mem:.2f} MiB")
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"[JanusVLN] 显存仍不足，请尝试在空闲时段运行")
                raise e
            else:
                raise e

    def reset(self, user_instruction: str):
        self.current_prompt = user_instruction.strip() if user_instruction else "Go to the destination"
        self.frame_idx = 0
        self.action_history = []
        self.rgb_history = []
        self.past_key_values = None
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print(f"[JanusVLN] 已重置，目标：{self.current_prompt}")

    def predict(self, frame_bgr: np.ndarray) -> Tuple[str, str]:
        if frame_bgr is None or frame_bgr.size == 0:
            return "MOVE_FORWARD", "向前"
        
        # 【4】用最快最省显存的图像缩放
        try:
            frame_rgb = frame_bgr[:, :, ::-1]
            pil_image = Image.fromarray(frame_rgb).resize(
                self.image_resolution, 
                Image.Resampling.NEAREST  # 【关键】用NEAREST，最省显存
            )
        except:
            pil_image = Image.new("RGB", self.image_resolution, (255,255,255))
        
        # 维护最小化历史
        self.rgb_history.append(pil_image)
        if len(self.rgb_history) > self.memory_window_size:
            self.rgb_history = self.rgb_history[-self.memory_window_size:]
        
        # 【5】每帧强制同步+清空显存
        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        step_id = int(self.frame_idx)
        task = self.current_prompt
        gen_kwargs = {
            "max_new_tokens": 8,  # 【6】减少生成token数
            "do_sample": False,
            "num_beams": 1,
            "use_cache": not DISABLE_ALL_CACHE
        }
        
        janusvln_action = "MOVE_FORWARD"
        with torch.no_grad():
            try:
                dtype = torch.float16 if self.device == "cuda" else torch.float32
                with compat_autocast(dtype=dtype):
                    outputs = self.model.call_model(
                        observations=self.rgb_history,
                        task=task,
                        step_id=step_id,
                        add_frame_index=False,  # 【7】禁用帧索引，省一点
                        gen_kwargs=gen_kwargs
                    )
                
                janusvln_action = self._parse_model_output(outputs)
                
            except Exception as e:
                if self.frame_idx == 0:
                    janusvln_action = "MOVE_FORWARD"
                else:
                    janusvln_action = self.action_history[-1] if self.action_history else "MOVE_FORWARD"
        
        self.frame_idx += 1
        self.action_history.append(janusvln_action)
        arrow_direction = self._map_to_5_arrow(janusvln_action)
        
        return janusvln_action, arrow_direction

    def _parse_model_output(self, outputs) -> str:
        if outputs is None:
            return "MOVE_FORWARD"
        try:
            if isinstance(outputs, str):
                output_text = outputs.strip().upper()
            elif isinstance(outputs, (list, tuple)):
                output_text = outputs[0].strip().upper() if outputs else "MOVE_FORWARD"
            else:
                output_text = str(outputs).strip().upper()
            for action in self.action_space:
                if action in output_text:
                    return action
            return "MOVE_FORWARD"
        except:
            return "MOVE_FORWARD"

    def _map_to_5_arrow(self, janusvln_action: str) -> str:
        if janusvln_action is None:
            return "向前"
        janusvln_action = janusvln_action.upper()
        if janusvln_action == "MOVE_FORWARD":
            return "向前"
        elif janusvln_action == "TURN_LEFT":
            return "向左"
        elif janusvln_action == "TURN_RIGHT":
            return "向右"
        elif janusvln_action == "STOP":
            return "停止"
        else:
            return "向前"