import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import re
import tqdm
import torch
import copy
import cv2
import json
import random
import argparse
import itertools
import quaternion
import transformers
import numpy as np

from typing import Any
from omegaconf import OmegaConf
from PIL import Image, ImageFile
from collections import OrderedDict
from torch.nn.utils.rnn import pad_sequence
from transformers.image_utils import to_numpy_array

import habitat
from habitat import logger, Env
from habitat_extensions import measures
from habitat.config.default import get_agent_config
from habitat_baselines.config.default import get_config as get_habitat_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video, observations_to_image

from utils.dist import *
import base64
from datetime import datetime
from io import BytesIO
from qwen_vl_utils import extract_vision_info
from transformers import AutoConfig, AutoTokenizer, AutoProcessor
from qwen_vl.model.vggt.utils.load_fn import load_and_preprocess_images
# 注意：如果你之前把模型修改写在了 _yuan 文件中，这里请确保导入的是包含了 dist_head 和 topo_head 的类
from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationForJanusVLN
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

min_pixels: int = 28 * 28
max_pixels: int = 1605632

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class VLNEvaluator:
    def __init__(
        self,
        config_path: str,
        split: str = "val_seen",
        env_num: int = 8,
        output_path: str = None,
        model: Any = None,
        epoch: int = 0,
        args: argparse.Namespace = None,
    ):
        self.args = args
        self.device = torch.device('cuda')
        self.split = split
        self.env_num = env_num
        self.save_video = args.save_video
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.epoch = epoch
        self.config_path = config_path
        self.config = get_habitat_config(config_path)
        self.agent_config = get_agent_config(self.config.habitat.simulator)
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors
        self.save_video_ratio = args.save_video_ratio

        with habitat.config.read_write(self.config):
            self.config.habitat.dataset.split = self.split
            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(
                            draw=True,
                            visibility_dist=5.0,
                            fov=90,
                        ),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )

        self.image_processor = model.processor
        self.model = model
        self.tokenizer = model.tokenizer
        
        self.actions2idx = OrderedDict({
            'STOP': [0],
            "MOVE_FORWARD": [1],
            "TURN_LEFT": [2],
            "TURN_RIGHT": [3]
        })

        self.num_history = args.num_history

    def config_env(self) -> Env:
        env = Env(config=self.config)
        return env

    def eval_action(self, idx) -> None:
        env = self.config_env()
        scene_episode_dict = {}
        for episode in env.episodes:
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)

        sucs, spls, oss, ones = [], [], [], []
        done_res = []
        if os.path.exists(os.path.join(self.output_path, f'result.json')):
            with open(os.path.join(self.output_path, f'result.json'),'r') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    done_res.append([res["scene_id"], res["episode_id"], res["episode_instruction"]])
                    if get_rank() == 0:
                        sucs.append(res['success'])
                        spls.append(res['spl'])
                        oss.append(res['os'])
                        ones.append(res['ne'])
        
        for scene in sorted(scene_episode_dict.keys()):
            episodes = scene_episode_dict[scene]
            scene_id = scene.split('/')[-2]
            print(f"scene_id = {scene_id}")
            
            process_bar = tqdm.tqdm(range(len(episodes[idx::self.env_num])), desc=f"scene {scene_id}")
            for episode in episodes[idx::self.env_num]:
                episode_instruction = episode.instruction.instruction_text if 'objectnav' not in self.config_path else episode.object_category
                print("episode start: ",episode_instruction)
                episode_id = episode.episode_id
                if [scene_id, episode_id, episode_instruction] in done_res:
                    continue

                env.current_episode = episode
                observations = env.reset()

                vis_frames = []
                step_id = 0
                
                should_save_video = self.save_video and (random.random() < self.save_video_ratio)
                if should_save_video:
                    os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}'), exist_ok=True)
                
                rgb_list = []
                time_ids = []
                action_seq = []
                self.model.model.past_key_values_vggt = None
                
                while not env.episode_over:
                    
                    time_ids.append(step_id)
                    rgb = observations["rgb"]
                    
                    image = Image.fromarray(rgb).convert('RGB')
                    
                    # ================== 创新点：获取动态分数与距离 ==================
                    pred_dist, topo_score = self.model.get_frame_metrics(image)
                    # 将原来的只存 image 改为存字典
                    rgb_list.append({
                        "image": image, 
                        "score": topo_score, 
                        "step": step_id, 
                        "dist": pred_dist
                    })
                    # ==============================================================
                    
                    info = env.get_metrics()
                        
                    history_len = len(rgb_list) - 1 
                    
                    # ================== 创新点：拓扑感知动态记忆管理 ==================
                    if history_len <= self.num_history:
                        images = [x["image"] for x in rgb_list]
                    else:
                        current_frame = rgb_list[-1]
                        history_frames = rgb_list[:-1] # 提取除了当前帧以外的历史帧
                        
                        # 根据 topo_score 进行降序排序（门、路口分数最高）
                        sorted_history = sorted(history_frames, key=lambda x: x["score"], reverse=True)
                        top_k_history = sorted_history[:self.num_history]
                        
                        # 必须按时间(step)重新排序，确保 LLM 序列理解正确
                        top_k_history = sorted(top_k_history, key=lambda x: x["step"])
                        images = [x["image"] for x in top_k_history] + [current_frame["image"]]
                    # ================================================================
                    
                    action = self.model.call_model(images, episode_instruction,step_id)[0]
                    
                    # ================== 创新点：度量校准早停干预 ==================
                    cur_pred_dist = rgb_list[-1]["dist"]
                    if "STOP" in action and cur_pred_dist > 3.0:
                        print(f"🛑 [干预] 模型输出 STOP，但预测距离为 {cur_pred_dist:.2f}m，强制修改为 MOVE_FORWARD。")
                        action = "MOVE_FORWARD"
                    # ==============================================================

                    if info['top_down_map'] is not None and should_save_video:
                        frame = observations_to_image({'rgb':observations['rgb']}, info)
                        vis_frames.append(frame)
                    
                    if action in self.actions2idx:
                        action = self.actions2idx[action][0]
                    else:
                        action = 0

                    if step_id >= self.args.max_steps:
                        action = 0

                    observations = env.step(action)
                    step_id += 1

                process_bar.update(1)
                metrics = env.get_metrics()
                if should_save_video:
                    images_to_video(
                        vis_frames, os.path.join(self.output_path, f'vis_{self.epoch}'), f'{scene_id}_{episode_id}', fps=6, quality=9
                    )
                vis_frames.clear()
                sucs.append(metrics['success'])
                spls.append(metrics['spl'])
                oss.append(metrics['oracle_success'])
                ones.append(metrics['distance_to_goal'])
                print(f"scene_episode {scene_id}_{episode_id} success: {metrics['success']}, spl: {metrics['spl']}, os: {metrics['oracle_success']}, ne: {metrics['distance_to_goal']}")
                result = {
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "success": metrics["success"],
                    "spl": metrics["spl"],
                    "os": metrics['oracle_success'],
                    "ne": metrics["distance_to_goal"],
                    "steps": step_id,
                    "episode_instruction": episode_instruction
                }
                
                with open(os.path.join(self.output_path, f'result.json'), 'a') as f:
                    f.write(json.dumps(result) + "\n")

        env.close()
        return torch.tensor(sucs).to(self.device), torch.tensor(spls).to(self.device), torch.tensor(oss).to(self.device), torch.tensor(ones).to(self.device), torch.tensor(len(sucs)).to(self.device)    


class JanusVLN_Inference:
    # 🔴 修复 1：增加 model_max_length 参数接收
    def __init__(self, pretrained, device="cuda", model_max_length=8192):
        config = AutoConfig.from_pretrained(pretrained)
        self.model = Qwen2_5_VLForConditionalGenerationForJanusVLN.from_pretrained(
            pretrained,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            attn_implementation="flash_attention_2",
            mode='evaluation'
        ).eval()
        
        # 🔴 修复 1：强制传入 model_max_length 给 Tokenizer，防止长度截断
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained, 
            padding_side="left",
            model_max_length=model_max_length
        )
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels, padding_side="left")
        
        self.device = device

    # ================== 创新点：获取单帧度量与拓扑分数 ==================
    @torch.no_grad()
    def get_frame_metrics(self, image: Image.Image):
        """利用已训练的模型头提取当前画面的剩余距离和重要性分数"""
        inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
        pixel_values = inputs.pixel_values.type(torch.bfloat16)
        
        # 1. 提取底层视觉特征
        vision_features = self.model.visual(pixel_values, grid_thw=inputs.image_grid_thw)
        pooled_vision = vision_features.mean(dim=0, keepdim=True)
        
        # 2. 预测距离
        pred_distance = self.model.dist_head(pooled_vision).item()
        
        # 3. 计算拓扑分数 (假设 idx 2 是路口，idx 4 是门道)
        topo_logits = self.model.topo_head(pooled_vision)
        topo_probs = torch.softmax(topo_logits, dim=-1)
        topo_score = (topo_probs[0, 2] + topo_probs[0, 4]).item()
        
        return pred_distance, topo_score
    # =================================================================

    def call_model(
        self,
        observations, 
        task,
        step_id,
        add_frame_index: bool=False,
        gen_kwargs: dict = {},
    ):
        
        messages = []
        message = [
                {"role": "system", 
                "content": "You are a visual language navigation model, and your should go to the locations to complete the given task. Compare the observation and instruction to infer your current progress, and then select the correct direction from the candidates to go to the target location and finish the task."
                }
            ]
        context = f"These images are your historical observations and your current observation.\n Your task is to {task} \n You should take one of the following actions:\n MOVE_FORWARD\n TURN_LEFT\n TURN_RIGHT\n STOP."
        patch_size = self.processor.image_processor.patch_size
        merge_size = self.processor.image_processor.merge_size
        for i in enumerate([task]):
    
            visual = observations
            if isinstance(visual, Image.Image): 
                message.append({"role": "user", "content": [{"type": "image", "image": visual}, {"type": "text", "text": context}]})
            elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):  
                image_content = []
                image_count = 0
                for v in visual:
                    if add_frame_index:
                        image_content.append({"type": "text", "text": "Frame-{}: ".format(image_count)})    
                    image_content.append({"type": "image", "image": v})
                    image_count += 1
                message.append({"role": "user", "content": image_content + [{"type": "text", "text": context}]})
            else:
                message.append({"role": "user", "content": [{"type": "text", "text": context}]})

            messages.append(message)

        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
        images_vggt = []
        image_inputs = []
        for message in messages:
            vision_info = extract_vision_info(message)
            cur_images_vggt = []
            for i, ele in enumerate(vision_info):
                if "image" in ele:
                    image = ele["image"]
                    if isinstance(image, Image.Image):
                        pass
                    elif isinstance(image, str) and "base64," in image:
                        _, base64_data = image.split("base64,", 1)
                        data = base64.b64decode(base64_data)
                        with BytesIO(data) as bio:
                            image = copy.deepcopy(Image.open(bio))
                    else:
                        raise NotImplementedError("Unsupported image type")   
                else:
                    raise NotImplementedError("Unsupported vision info type")
    
                assert isinstance(image, Image.Image), f"Unsupported image type: {type(image)}"
                image = load_and_preprocess_images([image])[0]

                if i == len(vision_info) - 1:
                    cur_images_vggt.append(image)

                _, height, width = image.shape
                if (width // patch_size) % merge_size > 0:
                    width = width - (width // patch_size) % merge_size * patch_size
                if (height // patch_size) % merge_size > 0:
                    height = height - (height // patch_size) % merge_size * patch_size
                image = image[:, :height, :width]
                image_inputs.append(image)
    
            images_vggt.append(torch.stack(cur_images_vggt))
        
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
            do_rescale=False
        )
        device = self.model.device

        inputs["images_vggt"] = [feat.to(device) for feat in images_vggt]
        inputs = inputs.to(device)
    
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 24
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1
        
        pad_token_id = self.tokenizer.pad_token_id

        # 🔴 修复 2：清理生成参数，解决 HF 的 temperature 和 do_sample 冲突警告
        generate_params = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": pad_token_id,
            "max_new_tokens": gen_kwargs["max_new_tokens"],
            "num_beams": gen_kwargs["num_beams"],
        }
        
        if gen_kwargs["temperature"] > 0:
            generate_params["do_sample"] = True
            generate_params["temperature"] = gen_kwargs["temperature"]
            generate_params["top_p"] = gen_kwargs["top_p"]
        else:
            generate_params["do_sample"] = False

        cont = self.model.generate(
            **inputs,
            **generate_params
        )

        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
        answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        return answers


def eval():
    global local_rank
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--habitat_config_path", type=str, default='config/vln_r2r.yaml')
    parser.add_argument("--eval_split", type=str, default='val_unseen')
    parser.add_argument("--output_path", type=str, default='./results/val_unseen/streamvln')
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--model_max_length", type=int, default=8192,
                        help= "Maximum sequence length. Sequences will be right padded (and possibly truncated).")
    parser.add_argument("--save_video_ratio", type=float, default=0.05, help="0~1")
    
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int,
                        help='rank')
    parser.add_argument('--gpu', default=0, type=int,
                        help='gpu')
    parser.add_argument('--port', default='1111')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--max_steps', default=400, type=int,
                        help='max_steps')
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    
    args = parser.parse_args()
    set_seed(args.seed)
    init_distributed_mode(args)
    local_rank = args.local_rank

    # 🔴 修复 1：将 args.model_max_length 传递给模型初始化
    model = JanusVLN_Inference(
        args.model_path, 
        device=f"cuda:{local_rank}", 
        model_max_length=args.model_max_length
    )

    evaluate(model, args)


def evaluate(model, args):
    
    world_size = get_world_size()

    evaluator = VLNEvaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=world_size,
        output_path=args.output_path,
        model=model,
        epoch=0,
        args=args
    )
    sucs, spls, oss, ones, ep_num = evaluator.eval_action(get_rank()) 
    ep_num_all = [torch.zeros_like(ep_num) for _ in range(world_size)]
    dist.all_gather(ep_num_all, ep_num)
    sucs_all = [torch.zeros(ep_num_all[i], dtype=sucs.dtype).to(sucs.device) for i in range(world_size)]
    spls_all = [torch.zeros(ep_num_all[i], dtype=spls.dtype).to(spls.device) for i in range(world_size)]
    oss_all = [torch.zeros(ep_num_all[i], dtype=oss.dtype).to(oss.device) for i in range(world_size)]
    ones_all = [torch.zeros(ep_num_all[i], dtype=ones.dtype).to(ones.device) for i in range(world_size)]
    dist.barrier()
    dist.all_gather(sucs_all, sucs)
    dist.all_gather(spls_all, spls)
    dist.all_gather(oss_all, oss)
    dist.all_gather(ones_all, ones)
    dist.barrier()
    sucs_all = torch.cat(sucs_all, dim=0)
    spls_all = torch.cat(spls_all, dim=0)
    oss_all = torch.cat(oss_all, dim=0)
    ones_all = torch.cat(ones_all, dim=0)
    result_all = {
                    "sucs_all": (sum(sucs_all)/len(sucs_all)).item(),
                    "spls_all": (sum(spls_all)/len(spls_all)).item(),
                    "oss_all": (sum(oss_all)/len(oss_all)).item(),
                    "ones_all": (sum(ones_all)/len(ones_all)).item(),
                    'length': len(sucs_all)
                }
    
    print(result_all)
    if get_rank() == 0:
        with open(os.path.join(args.output_path, f'result.json'), 'a') as f:
            f.write(json.dumps(result_all))

if __name__ == "__main__":
    eval()
