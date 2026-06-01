#!/usr/bin/env python3
import argparse
import gzip
import json
import math
import os
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple


STEP_PATTERN = re.compile(r"step[_-]?(\d+)", flags=re.IGNORECASE)
ACTION_PATTERN = re.compile(
    r"\b(MOVE[\s_-]*FORWARD|TURN[\s_-]*LEFT|TURN[\s_-]*RIGHT|STOP)\b",
    flags=re.IGNORECASE,
)
MOVE_FORWARD_ACTION_ID = 1


def load_json_or_jsonl(path: str, max_records: int = -1) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"{path} must contain a JSON list.")
            return data[:max_records] if max_records > 0 else data

        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if max_records > 0 and len(records) >= max_records:
                break
    return records


def load_gzip_json(path: str) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def dump_jsonl(records: Iterable[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_traj_id(record: Dict[str, Any]) -> Optional[str]:
    for key in ("traj_id", "trajectory_id", "episode_id"):
        if record.get(key) is not None:
            return str(record[key])

    sample_id = str(record.get("id", ""))
    if "/" in sample_id:
        return sample_id.split("/", 1)[0]

    images = record.get("images") or record.get("image") or []
    if isinstance(images, str):
        images = [images]
    for image in images:
        parts = str(image).split("/")
        if len(parts) >= 2:
            parent = parts[-2]
            if parent:
                return parent
    return None


def parse_step_id(record: Dict[str, Any]) -> Optional[int]:
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


def parse_action_text(text: Any) -> Optional[str]:
    if text is None:
        return None
    match = ACTION_PATTERN.search(str(text))
    if match is None:
        return None
    return re.sub(r"[\s-]+", "_", match.group(1).upper())


def get_assistant_action(record: Dict[str, Any]) -> Optional[str]:
    for key in ("action", "assistant_action", "target_action"):
        action = parse_action_text(record.get(key))
        if action is not None:
            return action

    for turn in reversed(record.get("conversations") or []):
        role = turn.get("from", turn.get("role"))
        if role in {"gpt", "assistant"}:
            return parse_action_text(turn.get("value", turn.get("content")))
    return None


def euclidean(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def normalize_scene_id(scene_id: Any) -> str:
    if scene_id is None:
        return ""
    scene_id = str(scene_id)
    parts = os.path.normpath(scene_id).split(os.sep)
    if len(parts) >= 2 and parts[-1].endswith(".glb"):
        return parts[-2]
    return parts[-1]


def build_episode_index(r2r_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    episodes = r2r_json.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("R2R train json must contain an episodes list.")

    index: Dict[str, Dict[str, Any]] = {}
    for episode in episodes:
        episode_id = str(episode.get("episode_id"))
        index[episode_id] = episode
        if episode.get("trajectory_id") is not None:
            index.setdefault(str(episode["trajectory_id"]), episode)
    return index


def build_location_index(actions: List[int]) -> List[int]:
    """Map action index to the location before executing that action."""
    indices = []
    location_idx = 0
    for action in actions:
        indices.append(location_idx)
        if int(action) == MOVE_FORWARD_ACTION_ID:
            location_idx += 1
    return indices


def build_remaining_path_distances(locations: List[List[float]]) -> List[float]:
    if not locations:
        return []
    remaining = [0.0 for _ in locations]
    running = 0.0
    for idx in range(len(locations) - 2, -1, -1):
        running += euclidean(locations[idx], locations[idx + 1])
        remaining[idx] = running
    return remaining


def build_gt_precompute(gt_index: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    prepared: Dict[str, Dict[str, Any]] = {}
    for traj_id, gt in gt_index.items():
        item = dict(gt)
        locations = item.get("locations") or []
        actions = item.get("actions") or []
        if locations and actions:
            item["_action_to_location"] = build_location_index(actions)
            item["_remaining_path"] = build_remaining_path_distances(locations)
        prepared[str(traj_id)] = item
    return prepared


def label_record(
    record: Dict[str, Any],
    episode_index: Dict[str, Dict[str, Any]],
    gt_index: Dict[str, Dict[str, Any]],
    success_radius: float,
    distance_type: str,
    invalid_policy: str,
) -> Tuple[Dict[str, Any], bool, str]:
    out = dict(record)
    traj_id = parse_traj_id(record)
    step_id = parse_step_id(record)
    action_text = get_assistant_action(record)

    if traj_id is None or step_id is None:
        reason = "missing_traj_or_step"
        if invalid_policy == "raise":
            raise ValueError(f"{reason}: id={record.get('id')}")
        out.update({"label_valid": False, "label_error": reason})
        return out, False, reason

    gt = gt_index.get(str(traj_id))
    episode = episode_index.get(str(traj_id))
    if gt is None or episode is None:
        reason = "missing_r2r_or_gt"
        if invalid_policy == "raise":
            raise ValueError(f"{reason}: traj_id={traj_id}")
        out.update({"traj_id": str(traj_id), "step_id": int(step_id), "label_valid": False, "label_error": reason})
        return out, False, reason

    locations = gt.get("locations") or []
    actions = gt.get("actions") or []
    if not locations or not actions:
        reason = "empty_locations_or_actions"
        if invalid_policy == "raise":
            raise ValueError(f"{reason}: traj_id={traj_id}")
        out.update({"traj_id": str(traj_id), "step_id": int(step_id), "label_valid": False, "label_error": reason})
        return out, False, reason

    action_to_location = gt.get("_action_to_location")
    if action_to_location is None:
        action_to_location = build_location_index(actions)
    remaining_path = gt.get("_remaining_path")
    if remaining_path is None:
        remaining_path = build_remaining_path_distances(locations)
    action_idx = min(max(int(step_id), 0), max(len(action_to_location) - 1, 0))
    location_idx = min(action_to_location[action_idx], len(locations) - 1)
    current_position = locations[location_idx]
    expert_stop_position = locations[-1]
    goal_position = (episode.get("goals") or [{}])[0].get("position")
    if goal_position is None:
        goal_position = expert_stop_position

    if distance_type == "euclidean":
        distance_to_goal = euclidean(current_position, goal_position)
        start_distance_to_goal = euclidean(locations[0], goal_position)
        output_distance_type = "euclidean_debug"
        distance_method = "current_to_goal_euclidean"
    else:
        distance_to_goal = remaining_path[location_idx] if remaining_path else 0.0
        start_distance_to_goal = remaining_path[0] if remaining_path else distance_to_goal
        output_distance_type = "geodesic"
        distance_method = "r2r_train_gt_remaining_path"

    if start_distance_to_goal <= 0:
        distance_progress = 1.0
    else:
        distance_progress = max(0.0, min(1.0, 1.0 - distance_to_goal / start_distance_to_goal))
    stop_ready = 1 if distance_to_goal <= success_radius else 0

    episode_geodesic = None
    if isinstance(episode.get("info"), dict):
        try:
            episode_geodesic = float(episode["info"].get("geodesic_distance"))
        except (TypeError, ValueError):
            episode_geodesic = None

    out.update(
        {
            "traj_id": str(traj_id),
            "episode_id": str(episode.get("episode_id", traj_id)),
            "step_id": int(step_id),
            "action": action_text,
            "scene_id": episode.get("scene_id"),
            "scene_key": normalize_scene_id(episode.get("scene_id")),
            "current_position": current_position,
            "goal_position": goal_position,
            "expert_stop_position": expert_stop_position,
            "distance_to_goal": float(distance_to_goal),
            "start_distance_to_goal": float(start_distance_to_goal),
            "episode_geodesic_distance": episode_geodesic,
            "distance_progress": float(distance_progress),
            "stop_ready": int(stop_ready),
            "stop_success_radius": float(success_radius),
            "distance_type": output_distance_type,
            "distance_method": distance_method,
            "label_source": "r2r_train_gt",
            "label_valid": True,
            "location_index": int(location_idx),
            "action_index": int(action_idx),
            "num_locations": int(len(locations)),
            "num_actions": int(len(actions)),
        }
    )
    return out, True, ""


def summarize(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    n = 0
    pos = 0
    neg = 0
    missing_dist = 0
    invalid = 0
    dist: List[float] = []
    progress: List[float] = []
    distance_type = Counter()
    errors = Counter()
    step0_diffs: List[float] = []
    keys = Counter()

    for record in records:
        n += 1
        keys.update(record.keys())
        distance_type[str(record.get("distance_type"))] += 1
        if not record.get("label_valid", True):
            invalid += 1
            errors[str(record.get("label_error"))] += 1
        if record.get("stop_ready") == 1:
            pos += 1
        elif record.get("stop_ready") == 0:
            neg += 1
        d = record.get("distance_to_goal")
        if d is None:
            missing_dist += 1
        else:
            dist.append(float(d))
        if record.get("distance_progress") is not None:
            progress.append(float(record["distance_progress"]))
        if record.get("step_id") == 0 and record.get("episode_geodesic_distance") is not None and d is not None:
            step0_diffs.append(abs(float(d) - float(record["episode_geodesic_distance"])))

    return {
        "n": n,
        "missing_distance": missing_dist,
        "label_invalid": invalid,
        "label_errors": dict(errors),
        "distance_type": dict(distance_type),
        "stop_ready_pos": pos,
        "stop_ready_neg": neg,
        "stop_ready_pos_ratio": pos / max(n, 1),
        "distance_min": min(dist) if dist else None,
        "distance_max": max(dist) if dist else None,
        "distance_mean": sum(dist) / len(dist) if dist else None,
        "progress_min": min(progress) if progress else None,
        "progress_max": max(progress) if progress else None,
        "progress_mean": sum(progress) / len(progress) if progress else None,
        "step0_geodesic_abs_diff_mean": sum(step0_diffs) / len(step0_diffs) if step0_diffs else None,
        "step0_geodesic_abs_diff_max": max(step0_diffs) if step0_diffs else None,
        "top_keys": keys.most_common(80),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--janus_json", required=True)
    parser.add_argument("--r2r_json_gz", required=True)
    parser.add_argument("--r2r_gt_json_gz", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--success_radius", type=float, default=3.0)
    parser.add_argument("--distance_type", choices=["geodesic", "euclidean"], default="geodesic")
    parser.add_argument("--invalid_policy", choices=["skip", "keep", "raise"], default="keep")
    parser.add_argument("--max_records", type=int, default=-1)
    parser.add_argument("--habitat_config_path", default=None, help="Accepted for command compatibility; not needed for R2R GT-path labels.")
    args = parser.parse_args()

    janus_records = load_json_or_jsonl(args.janus_json, max_records=args.max_records)
    r2r_json = load_gzip_json(args.r2r_json_gz)
    r2r_gt = load_gzip_json(args.r2r_gt_json_gz)
    episode_index = build_episode_index(r2r_json)
    gt_index = build_gt_precompute(r2r_gt)

    output_records: List[Dict[str, Any]] = []
    for record in janus_records:
        labeled, valid, _ = label_record(
            record,
            episode_index=episode_index,
            gt_index=gt_index,
            success_radius=args.success_radius,
            distance_type=args.distance_type,
            invalid_policy=args.invalid_policy,
        )
        if valid or args.invalid_policy == "keep":
            output_records.append(labeled)

    dump_jsonl(output_records, args.output_jsonl)
    summary = summarize(output_records)
    summary.update(
        {
            "janus_json": args.janus_json,
            "r2r_json_gz": args.r2r_json_gz,
            "r2r_gt_json_gz": args.r2r_gt_json_gz,
            "output_jsonl": args.output_jsonl,
            "success_radius": args.success_radius,
            "requested_distance_type": args.distance_type,
        }
    )
    summary_path = args.summary_json or f"{args.output_jsonl}.summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
