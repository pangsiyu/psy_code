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

from typing import Any, Dict, List, Optional, Tuple
from omegaconf import OmegaConf
from PIL import Image, ImageFile
from collections import OrderedDict, defaultdict
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
from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationForJanusVLN
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


min_pixels: int = 28 * 28
max_pixels: int = 501760

ACTION_NAMES = ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP")
ACTION_PATTERN = re.compile(
    r"\b(MOVE[\s_-]*FORWARD|TURN[\s_-]*LEFT|TURN[\s_-]*RIGHT|STOP)\b",
    flags=re.IGNORECASE,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_action_text(action_text: Any) -> Optional[str]:
    if action_text is None:
        return None
    text = str(action_text).strip()
    if not text:
        return None

    match = ACTION_PATTERN.search(text)
    if match is None:
        return None

    normalized = re.sub(r"[\s-]+", "_", match.group(1).upper())
    if normalized in ACTION_NAMES:
        return normalized
    return None


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def empty_stop_metrics() -> Dict[str, float]:
    return {
        "early_stop_count": 0.0,
        "correct_stop_count": 0.0,
        "missed_stop_count": 0.0,
        "total_stop_count": 0.0,
        "gate_trigger_count": 0.0,
        "gate_correct_count": 0.0,
        "invalid_action_count": 0.0,
        "stop_distance_sum": 0.0,
        "stop_distance_count": 0.0,
        "model_stop_count": 0.0,
        "model_early_stop_count": 0.0,
        "model_correct_stop_count": 0.0,
        "model_stop_distance_sum": 0.0,
        "model_stop_distance_count": 0.0,
        "forced_stop_count": 0.0,
        "forced_early_stop_count": 0.0,
        "forced_correct_stop_count": 0.0,
        "forced_stop_distance_sum": 0.0,
        "forced_stop_distance_count": 0.0,
    }


STOP_METRIC_TENSOR_KEYS = (
    "early_stop_count",
    "correct_stop_count",
    "missed_stop_count",
    "total_stop_count",
    "gate_trigger_count",
    "gate_correct_count",
    "invalid_action_count",
    "stop_distance_sum",
    "stop_distance_count",
    "model_stop_count",
    "model_early_stop_count",
    "model_correct_stop_count",
    "model_stop_distance_sum",
    "model_stop_distance_count",
    "forced_stop_count",
    "forced_early_stop_count",
    "forced_correct_stop_count",
    "forced_stop_distance_sum",
    "forced_stop_distance_count",
)


def finalize_stop_metrics(metrics: Dict[str, float]) -> Dict[str, Any]:
    finalized = {
        "early_stop_count": int(metrics.get("early_stop_count", 0)),
        "correct_stop_count": int(metrics.get("correct_stop_count", 0)),
        "missed_stop_count": int(metrics.get("missed_stop_count", 0)),
        "total_stop_count": int(metrics.get("total_stop_count", 0)),
        "gate_trigger_count": int(metrics.get("gate_trigger_count", 0)),
        "gate_correct_count": int(metrics.get("gate_correct_count", 0)),
        "invalid_action_count": int(metrics.get("invalid_action_count", 0)),
        "model_stop_count": int(metrics.get("model_stop_count", 0)),
        "model_early_stop_count": int(metrics.get("model_early_stop_count", 0)),
        "model_correct_stop_count": int(metrics.get("model_correct_stop_count", 0)),
        "forced_stop_count": int(metrics.get("forced_stop_count", 0)),
        "forced_early_stop_count": int(metrics.get("forced_early_stop_count", 0)),
        "forced_correct_stop_count": int(metrics.get("forced_correct_stop_count", 0)),
    }

    def avg(sum_key: str, count_key: str) -> Optional[float]:
        count = metrics.get(count_key, 0.0)
        return metrics.get(sum_key, 0.0) / count if count > 0 else None

    def rate(num_key: str, den_key: str) -> float:
        return float(metrics.get(num_key, 0.0)) / max(float(metrics.get(den_key, 0.0)), 1.0)

    finalized["avg_stop_distance"] = avg("stop_distance_sum", "stop_distance_count")
    finalized["model_avg_stop_distance"] = avg("model_stop_distance_sum", "model_stop_distance_count")
    finalized["forced_avg_stop_distance"] = avg("forced_stop_distance_sum", "forced_stop_distance_count")
    finalized["early_stop_rate"] = rate("early_stop_count", "total_stop_count")
    finalized["model_early_stop_rate"] = rate("model_early_stop_count", "model_stop_count")
    finalized["forced_early_stop_rate"] = rate("forced_early_stop_count", "forced_stop_count")
    finalized["gate_correct_rate"] = rate("gate_correct_count", "gate_trigger_count")
    return finalized


def stop_metrics_to_tensor(metrics: Dict[str, float], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [float(metrics.get(key, 0.0)) for key in STOP_METRIC_TENSOR_KEYS],
        dtype=torch.float32,
        device=device,
    )


def tensor_to_stop_metrics(tensor: torch.Tensor) -> Dict[str, float]:
    values = tensor.detach().to("cpu").tolist()
    return {key: float(value) for key, value in zip(STOP_METRIC_TENSOR_KEYS, values)}



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
            self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = getattr(args, "gpu", 0)
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
        self.episode_subset_order = self._load_episode_subset(args.episode_subset_json)
        self.step_log_path = os.path.join(self.output_path, "step_logs.jsonl")

    @staticmethod
    def _normalize_scene_id(scene_id: Any) -> str:
        scene_id = str(scene_id)
        norm = os.path.normpath(scene_id)
        parts = norm.split(os.sep)
        if len(parts) >= 2 and parts[-1].endswith(".glb"):
            return parts[-2]
        return parts[-1]

    def _episode_key(self, scene_id: Any, episode_id: Any) -> Tuple[str, str]:
        return (self._normalize_scene_id(scene_id), str(episode_id))

    def _load_episode_subset(self, subset_path: Optional[str]) -> Optional[List[Tuple[str, str]]]:
        if not subset_path:
            return None
        with open(subset_path, "r") as f:
            subset = json.load(f)
        if not isinstance(subset, list):
            raise ValueError("--episode_subset_json must be a JSON list.")

        episode_keys = []
        for item in subset:
            if not isinstance(item, dict) or "scene_id" not in item or "episode_id" not in item:
                raise ValueError(
                    "--episode_subset_json entries must look like "
                    '{"scene_id": "...", "episode_id": "..."}'
                )
            episode_keys.append(self._episode_key(item["scene_id"], item["episode_id"]))
        print(f"Loaded {len(episode_keys)} episodes from subset: {subset_path}")
        return episode_keys

    def _select_episodes(self, scene_episode_dict):
        available = []
        available_by_key = {}
        for scene in sorted(scene_episode_dict.keys()):
            scene_id = self._normalize_scene_id(scene)
            for episode in scene_episode_dict[scene]:
                key = self._episode_key(scene_id, episode.episode_id)
                record = (scene, scene_id, episode)
                available.append(record)
                available_by_key[key] = record

        if self.episode_subset_order is not None:
            selected = []
            missing = []
            for key in self.episode_subset_order:
                record = available_by_key.get(key)
                if record is None:
                    missing.append(key)
                    continue
                selected.append(record)
            if missing:
                print(f"Warning: {len(missing)} subset episodes were not found in Habitat episodes.")
        else:
            selected = available

        episodes_per_scene = getattr(self.args, "episodes_per_scene", None)
        if episodes_per_scene is not None and episodes_per_scene > 0:
            scene_counts = defaultdict(int)
            limited = []
            for record in selected:
                scene_id = record[1]
                if scene_counts[scene_id] >= episodes_per_scene:
                    continue
                scene_counts[scene_id] += 1
                limited.append(record)
            selected = limited

        max_eval_episodes = getattr(self.args, "max_eval_episodes", None)
        if max_eval_episodes is not None and max_eval_episodes > 0:
            selected = selected[:max_eval_episodes]

        print(f"Selected {len(selected)} episodes for evaluation.")
        return selected

    def _merge_stop_metrics(self, target: Dict[str, float], source: Dict[str, Any]) -> None:
        for key in STOP_METRIC_TENSOR_KEYS:
            if key.endswith("_distance_sum") or key.endswith("_distance_count"):
                continue
            target[key] += float(source.get(key, 0) or 0)

        distance_specs = (
            ("stop_distance_sum", "stop_distance_count", "avg_stop_distance", "total_stop_count"),
            ("model_stop_distance_sum", "model_stop_distance_count", "model_avg_stop_distance", "model_stop_count"),
            ("forced_stop_distance_sum", "forced_stop_distance_count", "forced_avg_stop_distance", "forced_stop_count"),
        )
        for sum_key, count_key, avg_key, public_count_key in distance_specs:
            if sum_key in source and count_key in source:
                target[sum_key] += float(source.get(sum_key, 0) or 0)
                target[count_key] += float(source.get(count_key, 0) or 0)
                continue

            avg_distance = safe_float(source.get(avg_key))
            count = float(source.get(public_count_key, 0) or 0)
            if avg_distance is not None and count > 0:
                target[sum_key] += avg_distance * count
                target[count_key] += count

    def _write_step_log(self, step_log: Dict[str, Any]) -> None:
        if not self.args.save_step_logs:
            return
        with open(self.step_log_path, "a") as f:
            f.write(json.dumps(step_log, ensure_ascii=False) + "\n")

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

        selected_episodes = self._select_episodes(scene_episode_dict)
        selected_keys = {
            self._episode_key(scene_id, episode.episode_id)
            for _, scene_id, episode in selected_episodes
        }

        sucs, spls, oss, ones = [], [], [], []
        stop_metrics = empty_stop_metrics()
        done_res = set()
        result_path = os.path.join(self.output_path, "result.json")
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    res = json.loads(line)
                    if "scene_id" not in res or "episode_id" not in res:
                        continue
                    key = self._episode_key(res["scene_id"], res["episode_id"])
                    if key not in selected_keys:
                        continue
                    done_res.add(key)
                    if get_rank() == 0:
                        sucs.append(res["success"])
                        spls.append(res["spl"])
                        oss.append(res["os"])
                        ones.append(res["ne"])
                        self._merge_stop_metrics(stop_metrics, res)

        rank_episodes = selected_episodes[idx::self.env_num]
        process_bar = tqdm.tqdm(rank_episodes, desc=f"rank {idx} episodes")
        for _, scene_id, episode in process_bar:
            episode_instruction = (
                episode.instruction.instruction_text
                if "objectnav" not in self.config_path
                else episode.object_category
            )
            episode_id = str(episode.episode_id)
            episode_key = self._episode_key(scene_id, episode_id)
            if episode_key in done_res:
                continue

            print(f"episode start: scene={scene_id} episode={episode_id} instruction={episode_instruction}")
            env.current_episode = episode
            observations = env.reset()

            vis_frames = []
            step_id = 0
            episode_stop_metrics = empty_stop_metrics()

            should_save_video = self.save_video and (random.random() < self.save_video_ratio)
            if should_save_video:
                os.makedirs(os.path.join(self.output_path, f"vis_{self.epoch}"), exist_ok=True)

            rgb_list = []
            self.model.model.past_key_values_vggt = None

            while not env.episode_over:
                rgb = observations["rgb"]
                image = Image.fromarray(rgb).convert("RGB")
                rgb_list.append(image)

                info = env.get_metrics()
                distance_to_goal = safe_float(info.get("distance_to_goal"))

                history_len = len(rgb_list) - 1
                if history_len <= self.num_history:
                    history_images = rgb_list[:history_len]
                    images = history_images + [rgb_list[-1]]
                else:
                    indices = np.linspace(0, history_len, self.num_history + 1, dtype=int)
                    images = [rgb_list[i] for i in indices]

                if self.args.use_stop_progress_gate:
                    raw_actions, stop_progress = self.model.call_model(
                        images, episode_instruction, step_id, return_stop_progress=True
                    )
                else:
                    raw_actions = self.model.call_model(images, episode_instruction, step_id)
                    stop_progress = None

                raw_action = raw_actions[0].strip() if raw_actions else ""
                parsed_action = parse_action_text(raw_action)
                invalid_action = parsed_action is None
                final_action = parsed_action if parsed_action is not None else self.args.invalid_action_fallback
                stop_progress = safe_float(stop_progress)

                if info["top_down_map"] is not None and should_save_video:
                    frame = observations_to_image({"rgb": observations["rgb"]}, info)
                    vis_frames.append(frame)

                within_stop_radius = (
                    distance_to_goal is not None
                    and distance_to_goal <= self.args.stop_success_radius
                )
                gate_triggered = False
                gate_correct = False
                if (
                    self.args.use_stop_progress_gate
                    and parsed_action == "STOP"
                    and stop_progress is not None
                    and stop_progress < self.args.stop_progress_threshold
                ):
                    gate_triggered = True
                    gate_correct = distance_to_goal is not None and not within_stop_radius
                    final_action = self.args.stop_gate_fallback

                forced_max_step_stop = step_id >= self.args.max_steps
                if forced_max_step_stop:
                    final_action = "STOP"

                if final_action not in self.actions2idx:
                    final_action = "MOVE_FORWARD"

                is_model_stop = parsed_action == "STOP"
                is_forced_stop = forced_max_step_stop
                is_final_stop = final_action == "STOP"
                model_correct_stop = bool(is_model_stop and within_stop_radius)
                model_early_stop = bool(
                    is_model_stop and distance_to_goal is not None and not within_stop_radius
                )
                forced_correct_stop = bool(is_forced_stop and within_stop_radius)
                forced_early_stop = bool(
                    is_forced_stop and distance_to_goal is not None and not within_stop_radius
                )

                if invalid_action:
                    episode_stop_metrics["invalid_action_count"] += 1
                if gate_triggered:
                    episode_stop_metrics["gate_trigger_count"] += 1
                if gate_correct:
                    episode_stop_metrics["gate_correct_count"] += 1
                if within_stop_radius and final_action != "STOP":
                    episode_stop_metrics["missed_stop_count"] += 1
                if is_model_stop:
                    episode_stop_metrics["model_stop_count"] += 1
                    if distance_to_goal is not None:
                        episode_stop_metrics["model_stop_distance_sum"] += distance_to_goal
                        episode_stop_metrics["model_stop_distance_count"] += 1
                    if model_correct_stop:
                        episode_stop_metrics["model_correct_stop_count"] += 1
                    elif model_early_stop:
                        episode_stop_metrics["model_early_stop_count"] += 1
                if is_forced_stop:
                    episode_stop_metrics["forced_stop_count"] += 1
                    if distance_to_goal is not None:
                        episode_stop_metrics["forced_stop_distance_sum"] += distance_to_goal
                        episode_stop_metrics["forced_stop_distance_count"] += 1
                    if forced_correct_stop:
                        episode_stop_metrics["forced_correct_stop_count"] += 1
                    elif forced_early_stop:
                        episode_stop_metrics["forced_early_stop_count"] += 1

                correct_stop = False
                early_stop = False
                if is_final_stop:
                    episode_stop_metrics["total_stop_count"] += 1
                    if distance_to_goal is not None:
                        episode_stop_metrics["stop_distance_sum"] += distance_to_goal
                        episode_stop_metrics["stop_distance_count"] += 1
                    correct_stop = within_stop_radius
                    early_stop = distance_to_goal is not None and not within_stop_radius
                    if correct_stop:
                        episode_stop_metrics["correct_stop_count"] += 1
                    elif early_stop:
                        episode_stop_metrics["early_stop_count"] += 1

                step_log = {
                    "rank": get_rank(),
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "step_id": step_id,
                    "raw_action": raw_action,
                    "parsed_action": parsed_action,
                    "final_action": final_action,
                    "stop_progress": stop_progress,
                    "distance_to_goal": distance_to_goal,
                    "invalid_action": invalid_action,
                    "gate_triggered": gate_triggered,
                    "gate_correct": gate_correct,
                    "correct_stop": correct_stop,
                    "early_stop": early_stop,
                    "model_stop": is_model_stop,
                    "model_correct_stop": model_correct_stop,
                    "model_early_stop": model_early_stop,
                    "forced_stop": is_forced_stop,
                    "forced_correct_stop": forced_correct_stop,
                    "forced_early_stop": forced_early_stop,
                    "missed_stop": bool(within_stop_radius and final_action != "STOP"),
                    "forced_max_step_stop": forced_max_step_stop,
                    "early_stop_count": int(early_stop),
                    "correct_stop_count": int(correct_stop),
                    "missed_stop_count": int(within_stop_radius and final_action != "STOP"),
                    "total_stop_count": int(is_final_stop),
                    "model_stop_count": int(is_model_stop),
                    "model_early_stop_count": int(model_early_stop),
                    "model_correct_stop_count": int(model_correct_stop),
                    "model_avg_stop_distance": distance_to_goal if is_model_stop else None,
                    "forced_stop_count": int(is_forced_stop),
                    "forced_early_stop_count": int(forced_early_stop),
                    "forced_correct_stop_count": int(forced_correct_stop),
                    "forced_avg_stop_distance": distance_to_goal if is_forced_stop else None,
                    "gate_trigger_count": int(gate_triggered),
                    "gate_correct_count": int(gate_correct),
                    "invalid_action_count": int(invalid_action),
                    "avg_stop_distance": distance_to_goal if is_final_stop else None,
                    "stop_success_radius": self.args.stop_success_radius,
                    "stop_progress_threshold": self.args.stop_progress_threshold,
                }
                self._write_step_log(step_log)

                if (
                    self.args.use_stop_progress_gate
                    or invalid_action
                    or gate_triggered
                    or raw_action != final_action
                    or stop_progress is not None
                ):
                    print(
                        f"action_debug scene={scene_id} episode={episode_id} step={step_id} "
                        f"raw_action={raw_action} parsed_action={parsed_action} "
                        f"final_action={final_action} stop_progress={stop_progress} "
                        f"distance_to_goal={distance_to_goal}"
                    )

                action = self.actions2idx[final_action][0]
                observations = env.step(action)
                step_id += 1

            metrics = env.get_metrics()
            if should_save_video:
                images_to_video(
                    vis_frames,
                    os.path.join(self.output_path, f"vis_{self.epoch}"),
                    f"{scene_id}_{episode_id}",
                    fps=6,
                    quality=9,
                )
            vis_frames.clear()

            sucs.append(metrics["success"])
            spls.append(metrics["spl"])
            oss.append(metrics["oracle_success"])
            ones.append(metrics["distance_to_goal"])
            episode_stop_summary = finalize_stop_metrics(episode_stop_metrics)
            self._merge_stop_metrics(stop_metrics, episode_stop_metrics)

            print(
                f"scene_episode {scene_id}_{episode_id} success: {metrics['success']}, "
                f"spl: {metrics['spl']}, os: {metrics['oracle_success']}, "
                f"ne: {metrics['distance_to_goal']}, stop_metrics: {episode_stop_summary}"
            )
            result = {
                "scene_id": scene_id,
                "episode_id": episode_id,
                "success": metrics["success"],
                "spl": metrics["spl"],
                "os": metrics["oracle_success"],
                "ne": metrics["distance_to_goal"],
                "steps": step_id,
                "episode_instruction": episode_instruction,
                **episode_stop_summary,
            }

            with open(result_path, "a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        env.close()
        return (
            torch.tensor(sucs, dtype=torch.float32).to(self.device),
            torch.tensor(spls, dtype=torch.float32).to(self.device),
            torch.tensor(oss, dtype=torch.float32).to(self.device),
            torch.tensor(ones, dtype=torch.float32).to(self.device),
            torch.tensor(len(sucs), dtype=torch.long).to(self.device),
            stop_metrics_to_tensor(stop_metrics, self.device),
        )




class JanusVLN_Inference:
    def __init__(
        self,
        pretrained,
        device="cuda",
        device_map="single",
        max_memory_per_gpu=None,
        force_add_stop_progress_head=False,
        stop_head_hidden_dim=1024,
        stop_progress_head_path=None,
    ):
        config = AutoConfig.from_pretrained(pretrained)
        if force_add_stop_progress_head:
            setattr(config, "add_stop_progress_head", True)
            setattr(config, "stop_head_hidden_dim", stop_head_hidden_dim)
        model_kwargs = {
            "config": config,
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
            "mode": "evaluation",
        }
        if device_map != "single":
            if device_map == "janusvln_4gpu":
                model_kwargs["device_map"] = self._build_janusvln_4gpu_device_map(config)
            elif device_map == "janusvln_2gpu":
                model_kwargs["device_map"] = self._build_janusvln_2gpu_device_map(config)
            else:
                model_kwargs["device_map"] = device_map
            if max_memory_per_gpu and isinstance(model_kwargs["device_map"], str):
                model_kwargs["max_memory"] = {
                    gpu_idx: max_memory_per_gpu for gpu_idx in range(torch.cuda.device_count())
                }
        else:
            model_kwargs["device_map"] = {"": device}

        self.model = Qwen2_5_VLForConditionalGenerationForJanusVLN.from_pretrained(
            pretrained,
            **model_kwargs,
        ).eval()
        if stop_progress_head_path:
            self._load_stop_progress_head(stop_progress_head_path)
        if hasattr(self.model, "hf_device_map"):
            device_counts = {}
            for mapped_device in self.model.hf_device_map.values():
                device_counts[str(mapped_device)] = device_counts.get(str(mapped_device), 0) + 1
            print(f"hf_device_map devices: {device_counts}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, padding_side="left")
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels, padding_side="left")
        
        self.device = device
        self.input_device = self._infer_input_device()

    def _load_stop_progress_head(self, stop_progress_head_path: str):
        if not hasattr(self.model, "stop_progress_head"):
            raise ValueError(
                "stop_progress_head_path was provided, but the model has no stop_progress_head. "
                "Use --force_add_stop_progress_head or a checkpoint with add_stop_progress_head=True."
            )
        payload = torch.load(stop_progress_head_path, map_location="cpu")
        state_dict = payload.get("state_dict", payload)
        self.model.stop_progress_head.load_state_dict(state_dict, strict=True)
        self.model.stop_progress_head.eval()
        print(f"Loaded stop_progress_head from {stop_progress_head_path}")

    @staticmethod
    def _build_janusvln_4gpu_device_map(config):
        if torch.cuda.device_count() < 4:
            raise RuntimeError("janusvln_4gpu requires at least 4 visible CUDA devices.")

        num_layers = getattr(config, "num_hidden_layers", None)
        if num_layers is None and hasattr(config, "text_config"):
            num_layers = getattr(config.text_config, "num_hidden_layers", None)
        if num_layers is None:
            raise ValueError("Could not infer the number of decoder layers from the model config.")

        device_map = {
            "visual": 1,
            "vggt": 1,
            "merger": 1,
            "model.embed_tokens": 0,
        }
        layer_devices = [0, 2, 3]
        base_layers, extra_layers = divmod(num_layers, len(layer_devices))
        layer_idx = 0
        for group_idx, target_device in enumerate(layer_devices):
            group_size = base_layers + (1 if group_idx < extra_layers else 0)
            for _ in range(group_size):
                device_map[f"model.layers.{layer_idx}"] = target_device
                layer_idx += 1

        device_map["model.norm"] = 3
        device_map["model.rotary_emb"] = 3
        device_map["lm_head"] = 3
        if getattr(config, "add_ground_classifier", False):
            device_map["classifier"] = 3
        if getattr(config, "add_stop_progress_head", False):
            device_map["stop_progress_head"] = 3
        return device_map

    @staticmethod
    def _build_janusvln_2gpu_device_map(config):
        if torch.cuda.device_count() < 2:
            raise RuntimeError("janusvln_2gpu requires at least 2 visible CUDA devices.")

        num_layers = getattr(config, "num_hidden_layers", None)
        if num_layers is None and hasattr(config, "text_config"):
            num_layers = getattr(config.text_config, "num_hidden_layers", None)
        if num_layers is None:
            raise ValueError("Could not infer the number of decoder layers from the model config.")

        device_map = {
            "visual": 0,
            "vggt": 0,
            "merger": 0,
            "model.embed_tokens": 0,
        }

        # GPU 0 also hosts VGGT/visual modules, so keep fewer decoder layers there.
        first_device_layers = max(1, num_layers // 3)
        for layer_idx in range(num_layers):
            target_device = 0 if layer_idx < first_device_layers else 1
            device_map[f"model.layers.{layer_idx}"] = target_device

        device_map["model.norm"] = 1
        device_map["model.rotary_emb"] = 1
        device_map["lm_head"] = 1
        if getattr(config, "add_ground_classifier", False):
            device_map["classifier"] = 1
        if getattr(config, "add_stop_progress_head", False):
            device_map["stop_progress_head"] = 1
        return device_map

    def _infer_input_device(self):
        for param in self.model.parameters():
            if param.device.type != "meta":
                return param.device
        return torch.device(self.device)


    def call_model(
        self,
        observations, 
        task,
        step_id,
        add_frame_index: bool=False,
        gen_kwargs: dict = None,
        return_stop_progress: bool = False,
    ):
        gen_kwargs = {} if gen_kwargs is None else gen_kwargs
        
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
        device = self.input_device

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
        
        
        generate_params = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": gen_kwargs["temperature"] > 0,
            "num_beams": gen_kwargs["num_beams"],
            "max_new_tokens": gen_kwargs["max_new_tokens"],
        }
        if gen_kwargs["temperature"] > 0:
            generate_params["temperature"] = gen_kwargs["temperature"]
            generate_params["top_p"] = gen_kwargs["top_p"]
        else:
            generate_params["temperature"] = None
            generate_params["top_p"] = None

        prefill_stop_progress = None
        if return_stop_progress:
            with torch.no_grad():
                stop_outputs = self.model(
                    **inputs,
                    use_cache=False,
                    return_dict=True,
                    logits_to_keep=1,
                    output_hidden_states=True,
                )
            prefill_stop_progress = getattr(stop_outputs, "stop_progress", None)
            if prefill_stop_progress is None and hasattr(self.model, "stop_progress_head"):
                output_hidden_states = getattr(stop_outputs, "hidden_states", None)
                if output_hidden_states is not None:
                    hidden_states = output_hidden_states[-1]
                    attention_mask = inputs.get("attention_mask", None)
                    if attention_mask is not None:
                        decision_idx = (
                            attention_mask.to(hidden_states.device)
                            .long()
                            .sum(dim=1)
                            .sub(1)
                            .clamp(min=0, max=hidden_states.shape[1] - 1)
                        )
                    else:
                        decision_idx = torch.full(
                            (hidden_states.shape[0],),
                            hidden_states.shape[1] - 1,
                            device=hidden_states.device,
                            dtype=torch.long,
                        )
                    batch_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
                    decision_hidden = hidden_states[batch_idx, decision_idx]
                    decision_hidden = torch.nan_to_num(decision_hidden, nan=0.0, posinf=0.0, neginf=0.0)
                    prefill_stop_progress = self.model.stop_progress_head(decision_hidden)
                    prefill_stop_progress = torch.nan_to_num(prefill_stop_progress, nan=0.0, posinf=1.0, neginf=0.0)
                    self.model.last_stop_progress = prefill_stop_progress.detach()
        cont = self.model.generate(**inputs, **generate_params)

        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
        answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if return_stop_progress:
            stop_progress = getattr(self.model, "last_stop_progress", None)
            if stop_progress is None:
                stop_progress = prefill_stop_progress
            if stop_progress is not None:
                stop_progress = float(stop_progress.detach().flatten()[0].to("cpu"))
            return answers, stop_progress
        
        return answers




   
def eval():
    global local_rank, max_pixels
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--habitat_config_path", type=str, default='config/vln_r2r.yaml')
    parser.add_argument("--eval_split", type=str, default='val_unseen')
    parser.add_argument("--output_path", type=str, default='./results/val_unseen/streamvln')
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--max_pixels", type=int, default=max_pixels,
                        help="Maximum pixels per image for the Qwen-VL processor.")
    parser.add_argument("--model_max_length", type=int, default=4096,
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
    parser.add_argument('--model_device_map', default='single',
                        choices=['single', 'auto', 'balanced', 'balanced_low_0', 'sequential', 'janusvln_2gpu', 'janusvln_4gpu'],
                        help='single keeps one full model per process; auto shards one model across visible GPUs')
    parser.add_argument('--max_memory_per_gpu', default=None,
                        help='Per-GPU max memory for device_map=auto, e.g. 16GiB')
    parser.add_argument('--max_steps', default=400, type=int,
                        help='max_steps')
    parser.add_argument('--max_eval_episodes', default=None, type=int,
                        help='Evaluate at most this many selected episodes. Non-positive means no limit.')
    parser.add_argument('--episode_subset_json', default=None, type=str,
                        help='JSON list of {"scene_id": "...", "episode_id": "..."} entries to evaluate.')
    parser.add_argument('--episodes_per_scene', default=None, type=int,
                        help='Evaluate at most this many episodes per scene. Non-positive means no limit.')
    parser.add_argument('--save_step_logs', action='store_true', default=False,
                        help='Write per-step action and stop-progress diagnostics to step_logs.jsonl.')
    parser.add_argument('--stop_success_radius', default=3.0, type=float,
                        help='Distance threshold used for stop-specific metrics.')
    parser.add_argument('--force_add_stop_progress_head', action='store_true', default=False,
                        help='Instantiate a random stop_progress_head even if the checkpoint config does not enable it. Intended for smoke tests only.')
    parser.add_argument('--stop_head_hidden_dim', default=1024, type=int,
                        help='Hidden dimension used when --force_add_stop_progress_head instantiates the stop head.')
    parser.add_argument('--stop_progress_head_path', default=None,
                        help='Optional stop_progress_head.pt to load on top of model_path.')
    parser.add_argument('--use_stop_progress_gate', action='store_true', default=False,
                        help='Gate generated STOP actions with the stop progression head output.')
    parser.add_argument('--stop_progress_threshold', default=0.85, type=float,
                        help='STOP is executed only when stop_progress is at least this threshold.')
    parser.add_argument('--invalid_action_fallback', default='MOVE_FORWARD',
                        choices=['STOP', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT'],
                        help='Action to execute when generated text is not a valid navigation action.')
    parser.add_argument('--stop_gate_fallback', default='MOVE_FORWARD',
                        choices=['STOP', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT'],
                        help='Action to execute when STOP is blocked by the stop progression gate.')
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    
    args = parser.parse_args()
    max_pixels = args.max_pixels
    set_seed(args.seed)
    init_distributed_mode(args)
    local_rank = args.local_rank

    model = JanusVLN_Inference(
        args.model_path,
        device=f"cuda:{local_rank}",
        device_map=args.model_device_map,
        max_memory_per_gpu=args.max_memory_per_gpu,
        force_add_stop_progress_head=args.force_add_stop_progress_head,
        stop_head_hidden_dim=args.stop_head_hidden_dim,
        stop_progress_head_path=args.stop_progress_head_path,
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
    sucs, spls, oss, ones, ep_num, stop_metric_tensor = evaluator.eval_action(get_rank())
    metric_sums = torch.tensor(
        [
            sucs.sum().item(),
            spls.sum().item(),
            oss.sum().item(),
            ones.sum().item(),
            ep_num.item(),
        ],
        dtype=torch.float32,
        device=sucs.device,
    )

    if world_size > 1:
        dist.barrier()
        dist.all_reduce(metric_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(stop_metric_tensor, op=dist.ReduceOp.SUM)
        dist.barrier()

    count = int(metric_sums[4].item())
    result_all = {
        "sucs_all": (metric_sums[0] / count).item() if count > 0 else None,
        "spls_all": (metric_sums[1] / count).item() if count > 0 else None,
        "oss_all": (metric_sums[2] / count).item() if count > 0 else None,
        "ones_all": (metric_sums[3] / count).item() if count > 0 else None,
        "length": count,
        **finalize_stop_metrics(tensor_to_stop_metrics(stop_metric_tensor)),
    }
    
    print(result_all)
    if get_rank() == 0:
        with open(os.path.join(args.output_path, f'result.json'), 'a') as f:
            f.write(json.dumps(result_all) + "\n")

if __name__ == "__main__":
    eval()
