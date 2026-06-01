#!/usr/bin/env python3
import argparse
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


ACTION_PATTERN = re.compile(
    r"\b(MOVE[\s_-]*FORWARD|TURN[\s_-]*LEFT|TURN[\s_-]*RIGHT|STOP)\b",
    flags=re.IGNORECASE,
)
STEP_PATTERN = re.compile(r"step[_-]?(\d+)", flags=re.IGNORECASE)


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def load_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            records = json.load(f)
        else:
            records = [json.loads(line) for line in f if line.strip()]
    if not isinstance(records, list):
        raise ValueError("Input must be a JSONL file or a JSON list.")
    return records


def dump_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_action_text(text: Any) -> Optional[str]:
    if text is None:
        return None
    match = ACTION_PATTERN.search(str(text))
    if match is None:
        return None
    return re.sub(r"[\s-]+", "_", match.group(1).upper())


def get_assistant_action(record: Dict[str, Any]) -> Optional[str]:
    for key in ("action", "assistant_action", "target_action"):
        if key in record:
            return parse_action_text(record[key])

    conversations = record.get("conversations") or []
    for turn in reversed(conversations):
        role = turn.get("from", turn.get("role"))
        if role in {"gpt", "assistant"}:
            return parse_action_text(turn.get("value", turn.get("content")))
    return None


def derive_group_id(record: Dict[str, Any]) -> str:
    for key in ("trajectory_id", "traj_id", "path_id", "episode_id"):
        if record.get(key) is not None:
            return str(record[key])

    sample_id = record.get("id")
    if isinstance(sample_id, str) and "/" in sample_id:
        return sample_id.split("/", 1)[0]

    images = record.get("images") or record.get("image") or []
    if isinstance(images, str):
        images = [images]
    if images:
        parent = os.path.basename(os.path.dirname(str(images[0])))
        if parent:
            return parent

    return "__all__"


def derive_step_id(record: Dict[str, Any]) -> Optional[int]:
    for key in ("step_id", "step", "timestep", "time_step"):
        if record.get(key) is not None:
            try:
                return int(record[key])
            except (TypeError, ValueError):
                pass

    candidates = [record.get("id")]
    images = record.get("images") or record.get("image") or []
    if isinstance(images, str):
        images = [images]
    candidates.extend(images)

    for value in candidates:
        if value is None:
            continue
        match = STEP_PATTERN.search(str(value))
        if match is not None:
            return int(match.group(1))
    return None


def derive_traj_len(record: Dict[str, Any], group_len: int) -> int:
    for key in ("traj_len", "trajectory_len", "trajectory_length", "num_steps"):
        if record.get(key) is not None:
            try:
                return max(int(record[key]), 1)
            except (TypeError, ValueError):
                pass
    return max(group_len, 1)


def get_distance_to_goal(record: Dict[str, Any]) -> Optional[float]:
    for key in ("distance_to_goal", "dist_to_goal", "goal_distance"):
        value = safe_float(record.get(key))
        if value is not None:
            return value
    metrics = record.get("metrics") or {}
    if isinstance(metrics, dict):
        return safe_float(metrics.get("distance_to_goal"))
    return None


def infer_group_metadata(records: List[Dict[str, Any]]):
    groups = defaultdict(list)
    for idx, record in enumerate(records):
        groups[derive_group_id(record)].append(idx)

    metadata = {}
    for group_id, indices in groups.items():
        indexed_steps = [(idx, derive_step_id(records[idx])) for idx in indices]
        order_map = {idx: pos for pos, idx in enumerate(indices)}
        ordered = sorted(
            indexed_steps,
            key=lambda item: (item[1] is None, item[1] if item[1] is not None else order_map[item[0]]),
        )
        start_distance = None
        for idx, _ in ordered:
            start_distance = safe_float(records[idx].get("start_distance_to_goal"))
            if start_distance is None:
                start_distance = get_distance_to_goal(records[idx])
            if start_distance is not None and start_distance > 0:
                break

        for fallback_step_id, (idx, step_id) in enumerate(ordered):
            metadata[idx] = {
                "group_id": group_id,
                "step_id": step_id if step_id is not None else fallback_step_id,
                "traj_len": derive_traj_len(records[idx], len(indices)),
                "start_distance_to_goal": start_distance,
            }
    return metadata


def compute_stop_progress(
    record: Dict[str, Any],
    meta: Dict[str, Any],
    mode: str,
) -> Tuple[float, bool]:
    step_id = int(meta["step_id"])
    traj_len = int(meta["traj_len"])
    step_ratio = clip01(step_id / max(traj_len - 1, 1))

    if mode == "step_ratio":
        return step_ratio, False

    if mode == "binary_stop":
        return (1.0 if get_assistant_action(record) == "STOP" else 0.0), False

    if mode == "distance_progress":
        start_distance = safe_float(record.get("start_distance_to_goal"))
        if start_distance is None:
            start_distance = safe_float(meta.get("start_distance_to_goal"))
        distance = get_distance_to_goal(record)
        if start_distance is None or start_distance <= 0 or distance is None:
            return step_ratio, True
        return clip01(1.0 - distance / start_distance), False

    raise ValueError(f"Unsupported mode: {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl")
    parser.add_argument("output_jsonl")
    parser.add_argument(
        "--mode",
        choices=["step_ratio", "binary_stop", "distance_progress"],
        default="step_ratio",
    )
    args = parser.parse_args()

    if os.path.abspath(args.input_jsonl) == os.path.abspath(args.output_jsonl):
        raise ValueError("input_jsonl and output_jsonl must be different paths.")

    records = load_records(args.input_jsonl)
    metadata = infer_group_metadata(records)

    values = []
    fallback_count = 0
    for idx, record in enumerate(records):
        value, used_fallback = compute_stop_progress(record, metadata[idx], args.mode)
        record["stop_progress"] = value
        values.append(value)
        fallback_count += int(used_fallback)

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    dump_jsonl(records, args.output_jsonl)

    if values:
        mean_value = sum(values) / len(values)
        print(
            f"wrote={args.output_jsonl} count={len(values)} "
            f"min={min(values):.6f} max={max(values):.6f} mean={mean_value:.6f} "
            f"mode={args.mode} fallback_count={fallback_count}"
        )
    else:
        print(f"wrote={args.output_jsonl} count=0 mode={args.mode} fallback_count=0")


if __name__ == "__main__":
    main()
