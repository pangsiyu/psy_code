#!/usr/bin/env python3
import argparse
import json
from typing import Any, Dict, List, Optional


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def stat(values: List[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    return {
        "n": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_den = sum((x - x_mean) ** 2 for x in xs) ** 0.5
    y_den = sum((y - y_mean) ** 2 for y in ys) ** 0.5
    if x_den == 0 or y_den == 0:
        return None
    return num / (x_den * y_den)


def auc(scores: List[float], labels: List[int]) -> Optional[float]:
    pos = [(score, label) for score, label in zip(scores, labels) if label == 1]
    neg = [(score, label) for score, label in zip(scores, labels) if label == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for pos_score, _ in pos:
        for neg_score, _ in neg:
            if pos_score > neg_score:
                wins += 1.0
            elif pos_score == neg_score:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_logs", required=True)
    parser.add_argument("--thresholds", default="0.3,0.5,0.7,0.9")
    parser.add_argument("--stop_success_radius", type=float, default=3.0)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    stops = []
    all_scored = []
    with open(args.step_logs, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if safe_float(row.get("stop_progress")) is not None and safe_float(row.get("distance_to_goal")) is not None:
                all_scored.append(row)
            if row.get("parsed_action") == "STOP":
                stops.append(row)

    correct = [
        safe_float(row.get("stop_progress"))
        for row in stops
        if row.get("model_correct_stop") and safe_float(row.get("stop_progress")) is not None
    ]
    early = [
        safe_float(row.get("stop_progress"))
        for row in stops
        if row.get("model_early_stop") and safe_float(row.get("stop_progress")) is not None
    ]

    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    simulation = {}
    for tau in thresholds:
        blocked = [
            row for row in stops
            if safe_float(row.get("stop_progress")) is not None
            and safe_float(row.get("stop_progress")) < tau
        ]
        simulation[str(tau)] = {
            "blocked": len(blocked),
            "blocked_correct": sum(1 for row in blocked if row.get("model_correct_stop")),
            "blocked_early": sum(1 for row in blocked if row.get("model_early_stop")),
        }

    rows = []
    for row in stops:
        rows.append(
            {
                "scene_id": row.get("scene_id"),
                "episode_id": row.get("episode_id"),
                "step_id": row.get("step_id"),
                "score": row.get("stop_progress"),
                "distance_to_goal": row.get("distance_to_goal"),
                "model_correct_stop": row.get("model_correct_stop"),
                "model_early_stop": row.get("model_early_stop"),
            }
        )

    result = {
        "step_logs": args.step_logs,
        "parsed_stop_count": len(stops),
        "correct": stat([value for value in correct if value is not None]),
        "early": stat([value for value in early if value is not None]),
        "threshold_simulation": simulation,
        "stops": rows,
    }

    all_scores = [safe_float(row.get("stop_progress")) for row in all_scored]
    all_distances = [safe_float(row.get("distance_to_goal")) for row in all_scored]
    valid_pairs = [
        (score, distance)
        for score, distance in zip(all_scores, all_distances)
        if score is not None and distance is not None
    ]
    ready_scores = [score for score, distance in valid_pairs if distance <= args.stop_success_radius]
    far_scores = [score for score, distance in valid_pairs if distance > args.stop_success_radius]
    result["all_steps"] = {
        "count": len(valid_pairs),
        "ready_count": len(ready_scores),
        "far_count": len(far_scores),
        "ready_scores": stat(ready_scores),
        "far_scores": stat(far_scores),
        "pearson_score_vs_negative_distance": pearson(
            [score for score, _ in valid_pairs],
            [-distance for _, distance in valid_pairs],
        ),
        "ready_auc": auc(
            [score for score, _ in valid_pairs],
            [1 if distance <= args.stop_success_radius else 0 for _, distance in valid_pairs],
        ),
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
