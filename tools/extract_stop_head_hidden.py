#!/usr/bin/env python
import argparse
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qwen_vl.data.data_qwen import (  # noqa: E402
    DataCollatorForSupervisedDataset,
    LazySupervisedDataset,
    read_jsonl,
)
from qwen_vl.data.rope2d import get_rope_index_25, get_rope_index_2  # noqa: E402
from qwen_vl.model.modeling_qwen2_5_vl import (  # noqa: E402
    Qwen2_5_VLForConditionalGenerationForJanusVLN,
)


def build_janusvln_2gpu_device_map(config):
    if torch.cuda.device_count() < 2:
        raise RuntimeError("janusvln_2gpu requires at least 2 visible CUDA devices.")
    num_layers = getattr(config, "num_hidden_layers", None)
    if num_layers is None and hasattr(config, "text_config"):
        num_layers = getattr(config.text_config, "num_hidden_layers", None)
    if num_layers is None:
        raise ValueError("Could not infer decoder layer count from config.")

    device_map = {
        "visual": 0,
        "vggt": 0,
        "merger": 0,
        "model.embed_tokens": 0,
    }
    first_device_layers = max(1, num_layers // 3)
    for layer_idx in range(num_layers):
        device_map[f"model.layers.{layer_idx}"] = 0 if layer_idx < first_device_layers else 1
    device_map["model.norm"] = 1
    device_map["model.rotary_emb"] = 1
    device_map["lm_head"] = 1
    if getattr(config, "add_ground_classifier", False):
        device_map["classifier"] = 1
    if getattr(config, "add_stop_progress_head", False):
        device_map["stop_progress_head"] = 1
    return device_map


def infer_input_device(model):
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def load_annotations(path: str, max_samples: int) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        annotations = read_jsonl(path, max_samples=max_samples)
    else:
        with open(path, "r") as f:
            annotations = json.load(f)
        if max_samples > 0:
            annotations = annotations[:max_samples]
    for idx, ann in enumerate(annotations):
        ann.setdefault("data_path", "")
        ann.setdefault("tag", "train_r2r_only_stop_progress")
        ann.setdefault("_source_index", idx)
    return annotations


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value != value:
        return None
    return value


def select_annotations(annotations: List[Dict[str, Any]], args) -> List[Dict[str, Any]]:
    if args.sample_strategy == "first_n":
        if args.max_samples > 0:
            return annotations[: args.max_samples]
        return annotations

    if args.sample_strategy != "balanced_stop_ready":
        raise ValueError(f"Unsupported sample_strategy={args.sample_strategy}")

    positives = [ann for ann in annotations if safe_float(ann.get("stop_ready")) == 1.0]
    negatives = [ann for ann in annotations if safe_float(ann.get("stop_ready")) == 0.0]
    if not positives or not negatives:
        raise ValueError(
            "balanced_stop_ready requires both stop_ready=1 and stop_ready=0 samples. "
            f"Got positives={len(positives)}, negatives={len(negatives)}."
        )

    generator = torch.Generator().manual_seed(args.seed)
    pos_perm = torch.randperm(len(positives), generator=generator).tolist()
    neg_perm = torch.randperm(len(negatives), generator=generator).tolist()

    if args.max_samples > 0:
        candidate_total = args.max_samples + max(args.max_samples, args.max_skipped)
        per_class = candidate_total // 2
        pos_count = min(len(positives), per_class)
        neg_count = min(len(negatives), candidate_total - pos_count)
        if pos_count < per_class:
            neg_count = min(len(negatives), pos_count)
        if neg_count < per_class:
            pos_count = min(len(positives), neg_count)
    else:
        pos_count = len(positives)
        neg_count = len(negatives)

    selected = [positives[i] for i in pos_perm[:pos_count]]
    selected.extend(negatives[i] for i in neg_perm[:neg_count])
    order = torch.randperm(len(selected), generator=generator).tolist()
    return [selected[i] for i in order]


def build_dataset(tokenizer, processor, annotations, args):
    data_args = SimpleNamespace(
        dataset_use="hidden_cache_inline",
        image_processor=processor.image_processor,
        model_type="qwen2.5vl",
        video_max_frames=args.video_max_frames,
        video_min_frames=min(args.video_max_frames, 4),
        video_max_total_pixels=1664 * 28 * 28,
        video_min_total_pixels=256 * 28 * 28,
        base_interval=2,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        video_max_frame_pixels=args.max_pixels,
        video_min_frame_pixels=args.min_pixels,
    )
    dataset = LazySupervisedDataset.__new__(LazySupervisedDataset)
    dataset.video_max_total_pixels = data_args.video_max_total_pixels
    dataset.video_min_total_pixels = data_args.video_min_total_pixels
    dataset.model_type = data_args.model_type
    dataset.get_rope_index = get_rope_index_25 if data_args.model_type == "qwen2.5vl" else get_rope_index_2
    dataset.tokenizer = tokenizer
    dataset.list_data_dict = annotations
    dataset.data_args = data_args
    data_args.image_processor.max_pixels = data_args.max_pixels
    data_args.image_processor.min_pixels = data_args.min_pixels
    data_args.image_processor.size["longest_edge"] = data_args.max_pixels
    data_args.image_processor.size["shortest_edge"] = data_args.min_pixels
    return dataset


def action_from_annotation(annotation: Dict[str, Any]) -> str:
    for turn in reversed(annotation.get("conversations", [])):
        if turn.get("from") == "gpt":
            return str(turn.get("value", "")).strip()
    return ""


def metadata_from_annotation(annotation: Dict[str, Any]) -> Dict[str, Any]:
    sample_id = str(annotation.get("id", ""))
    step_match = re.search(r"step[_-](\d+)", sample_id)
    traj_id = str(annotation.get("trajectory_id") or sample_id.split("/")[0] or "")
    return {
        "id": sample_id,
        "traj_id": traj_id,
        "episode_id": str(annotation.get("episode_id") or traj_id),
        "step_id": int(step_match.group(1)) if step_match else annotation.get("step_id"),
        "action": action_from_annotation(annotation),
    }


def save_shard(records: List[Dict[str, Any]], output_dir: Path, shard_idx: int, target_keys: List[str]) -> Dict[str, Any]:
    hidden = torch.stack([record["hidden"] for record in records], dim=0)
    metadata = [
        {key: value for key, value in record.items() if key != "hidden"}
        for record in records
    ]
    shard = {
        "hidden": hidden,
        "metadata": metadata,
    }
    for key in target_keys:
        values = [safe_float(record.get(key)) for record in records]
        if all(value is not None for value in values):
            shard[key] = torch.tensor(values, dtype=torch.float32)
    if "stop_progress" not in shard and target_keys:
        values = [safe_float(record.get(target_keys[0])) for record in records]
        if all(value is not None for value in values):
            shard["stop_progress"] = torch.tensor(values, dtype=torch.float32)
    shard_name = f"hidden_{shard_idx:05d}.pt"
    torch.save(shard, output_dir / shard_name)
    return {"file": shard_name, "num_samples": int(hidden.shape[0])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=256)
    parser.add_argument("--source_max_samples", type=int, default=0,
                        help="Number of source rows to scan. Defaults to max_samples * scan_multiplier.")
    parser.add_argument("--scan_multiplier", type=int, default=8,
                        help="Extra source rows to load so bad/overlong samples can be skipped while still filling the cache.")
    parser.add_argument("--max_skipped", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_pixels", type=int, default=501760)
    parser.add_argument("--min_pixels", type=int, default=28 * 28)
    parser.add_argument("--video_max_frames", type=int, default=4)
    parser.add_argument("--model_max_length", type=int, default=4096)
    parser.add_argument("--vggt_cache_start_size", type=int, default=4)
    parser.add_argument("--vggt_cache_recent_size", type=int, default=8)
    parser.add_argument("--device_map", default="janusvln_2gpu", choices=["single", "auto", "balanced", "sequential", "janusvln_2gpu"])
    parser.add_argument("--max_memory_per_gpu", default=None)
    parser.add_argument("--cpu_memory", default="80GiB")
    parser.add_argument("--offload_folder", default=None)
    parser.add_argument("--shard_size", type=int, default=256)
    parser.add_argument("--torch_dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--target_keys", default="stop_progress")
    parser.add_argument("--sample_strategy", default="first_n", choices=["first_n", "balanced_stop_ready"])
    parser.add_argument("--split_by", default=None, choices=[None, "traj_id"], help="Recorded in manifest for downstream train/val splitting.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    target_keys = parse_csv(args.target_keys)
    if not target_keys:
        raise ValueError("--target_keys must contain at least one key.")

    if args.max_samples <= 0:
        source_limit = args.source_max_samples if args.source_max_samples > 0 else -1
        target_samples = None
    else:
        source_limit = args.source_max_samples
        if source_limit <= 0:
            if args.sample_strategy == "balanced_stop_ready":
                source_limit = -1
            else:
                source_limit = max(args.max_samples, args.max_samples * max(args.scan_multiplier, 1))
        target_samples = args.max_samples

    os.environ["VGGT_CACHE_START_SIZE"] = str(args.vggt_cache_start_size)
    os.environ["VGGT_CACHE_RECENT_SIZE"] = str(args.vggt_cache_recent_size)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.torch_dtype]
    config = AutoConfig.from_pretrained(args.model_path)
    model_kwargs = {
        "config": config,
        "torch_dtype": dtype,
        "attn_implementation": "flash_attention_2",
        "mode": None,
    }
    if args.device_map == "janusvln_2gpu":
        model_kwargs["device_map"] = build_janusvln_2gpu_device_map(config)
    elif args.device_map != "single":
        model_kwargs["device_map"] = args.device_map
        if args.max_memory_per_gpu:
            model_kwargs["max_memory"] = {
                gpu_idx: args.max_memory_per_gpu for gpu_idx in range(torch.cuda.device_count())
            }
            model_kwargs["max_memory"]["cpu"] = args.cpu_memory
        if args.offload_folder:
            Path(args.offload_folder).mkdir(parents=True, exist_ok=True)
            model_kwargs["offload_folder"] = args.offload_folder
            model_kwargs["offload_state_dict"] = True
    else:
        model_kwargs["device_map"] = {"": "cuda:0" if torch.cuda.is_available() else "cpu"}

    model = Qwen2_5_VLForConditionalGenerationForJanusVLN.from_pretrained(
        args.model_path,
        **model_kwargs,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    source_annotations = load_annotations(args.data_path, source_limit)
    annotations = select_annotations(source_annotations, args)
    dataset = build_dataset(tokenizer, processor, annotations, args)
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    input_device = infer_input_device(model)

    manifest = {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "max_samples": args.max_samples,
        "source_max_samples": source_limit,
        "loaded_source_samples": len(source_annotations),
        "selected_samples": len(annotations),
        "target_keys": target_keys,
        "sample_strategy": args.sample_strategy,
        "split_by": args.split_by,
        "batch_size": args.batch_size,
        "max_pixels": args.max_pixels,
        "model_max_length": args.model_max_length,
        "video_max_frames": args.video_max_frames,
        "vggt_cache_start_size": args.vggt_cache_start_size,
        "vggt_cache_recent_size": args.vggt_cache_recent_size,
        "device_map": args.device_map,
        "shards": [],
        "skipped": [],
        "positive_count": sum(1 for ann in annotations if safe_float(ann.get("stop_ready")) == 1.0),
        "negative_count": sum(1 for ann in annotations if safe_float(ann.get("stop_ready")) == 0.0),
    }

    records: List[Dict[str, Any]] = []
    shard_idx = 0
    written = 0
    skipped = 0
    model_dtype = next(model.parameters()).dtype
    with torch.no_grad():
        for start in range(0, len(dataset), args.batch_size):
            if target_samples is not None and written >= target_samples:
                break
            batch_indices = list(range(start, min(start + args.batch_size, len(dataset))))
            try:
                instances = [dataset[i] for i in batch_indices]
                batch = collator(instances)
                labels = batch["labels"]
                forward_batch = {
                    key: value
                    for key, value in batch.items()
                    if key
                    in {
                        "input_ids",
                        "attention_mask",
                        "position_ids",
                        "pixel_values",
                        "image_grid_thw",
                        "pixel_values_videos",
                        "video_grid_thw",
                        "images_vggt",
                        "tag",
                    }
                }
                forward_batch = {key: move_to_device(value, input_device) for key, value in forward_batch.items()}
                if hasattr(model, "past_key_values_vggt"):
                    model.past_key_values_vggt = None
                outputs = model(
                    **forward_batch,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                    logits_to_keep=1,
                )
            except Exception as exc:
                skipped += len(batch_indices)
                error = f"{type(exc).__name__}: {exc}"
                manifest["skipped"].append(
                    {
                        "source_indices": batch_indices,
                        "error": error[:500],
                    }
                )
                print(f"Skipping source indices {batch_indices}: {error}", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if skipped > args.max_skipped:
                    raise RuntimeError(
                        f"Skipped {skipped} samples, exceeding --max_skipped={args.max_skipped}."
                    ) from exc
                continue
            hidden_states = outputs.hidden_states[-1]
            label_mask = labels.ne(-100)
            has_label = label_mask.any(dim=1)
            if not has_label.all():
                missing = [batch_indices[i] for i, ok in enumerate(has_label.tolist()) if not ok]
                print(f"Skipping samples with no supervised label after truncation: {missing}")
            first_label_idx = label_mask.float().argmax(dim=1).long()
            decision_idx = (first_label_idx - 1).clamp(min=0, max=hidden_states.shape[1] - 1).to(hidden_states.device)
            batch_arange = torch.arange(hidden_states.shape[0], device=hidden_states.device)
            decision_hidden = hidden_states[batch_arange, decision_idx]
            decision_hidden = torch.nan_to_num(decision_hidden, nan=0.0, posinf=0.0, neginf=0.0)

            for local_idx, sample_idx in enumerate(batch_indices):
                if target_samples is not None and written >= target_samples:
                    break
                if not bool(has_label[local_idx].item()):
                    continue
                annotation = annotations[sample_idx]
                metadata = metadata_from_annotation(annotation)
                metadata["source_index"] = int(annotation.get("_source_index", sample_idx))
                metadata["decision_token_index"] = int(decision_idx[local_idx].detach().cpu().item())
                for key in target_keys:
                    value = safe_float(annotation.get(key))
                    if value is None:
                        raise ValueError(f"Selected annotation is missing numeric target key {key}: {annotation.get('id')}")
                    metadata[key] = float(value)
                for key in (
                    "distance_to_goal",
                    "distance_progress",
                    "stop_ready",
                    "start_distance_to_goal",
                    "distance_type",
                    "distance_method",
                    "label_valid",
                    "scene_id",
                    "scene_key",
                    "current_position",
                    "goal_position",
                    "expert_stop_position",
                ):
                    if key in annotation and key not in metadata:
                        metadata[key] = annotation[key]
                records.append(
                    {
                        **metadata,
                        "hidden": decision_hidden[local_idx].detach().to("cpu", dtype=torch.float32),
                    }
                )
                written += 1
            if records and len(records) >= args.shard_size:
                manifest["shards"].append(save_shard(records, output_dir, shard_idx, target_keys))
                records = []
                shard_idx += 1
            target_label = target_samples if target_samples is not None else len(dataset)
            print(
                f"extracted {written}/{target_label} hidden states "
                f"(scanned {batch_indices[-1] + 1}/{len(dataset)}, skipped {skipped})",
                flush=True,
            )

    if records:
        manifest["shards"].append(save_shard(records, output_dir, shard_idx, target_keys))

    manifest["num_samples"] = int(sum(shard["num_samples"] for shard in manifest["shards"]))
    manifest["num_skipped"] = int(skipped)
    if target_samples is not None and manifest["num_samples"] < target_samples:
        print(
            f"WARNING: requested {target_samples} hidden states but only wrote {manifest['num_samples']} "
            f"after scanning {len(dataset)} source samples.",
            flush=True,
        )
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
