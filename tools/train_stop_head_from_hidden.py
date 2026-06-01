#!/usr/bin/env python
import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class StopProgressHead(nn.Module):
    def __init__(self, hidden_dim: int, stop_head_hidden_dim: int):
        super().__init__()
        if hasattr(nn, "RMSNorm"):
            norm = nn.RMSNorm(hidden_dim, eps=1e-6)
        else:
            norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.net = nn.Sequential(
            norm,
            nn.Linear(hidden_dim, stop_head_hidden_dim),
            nn.GELU(),
            nn.Linear(stop_head_hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


def safe_float(value: Any) -> float:
    value = float(value)
    if value != value:
        raise ValueError("nan target value")
    return value


def load_cache(cache_dir: Path, target_key: str) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    hidden_parts = []
    label_parts = []
    metadata: List[Dict[str, Any]] = []
    shard_paths = sorted(cache_dir.glob("*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No .pt cache shards found in {cache_dir}")

    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location="cpu")
        if not isinstance(shard, dict) or "hidden" not in shard:
            raise ValueError(f"Unsupported cache shard format: {shard_path}")

        hidden = shard["hidden"].float()
        shard_metadata = shard.get("metadata", [{} for _ in range(hidden.shape[0])])
        if len(shard_metadata) != hidden.shape[0]:
            raise ValueError(f"metadata length mismatch in {shard_path}")

        if target_key in shard:
            labels = shard[target_key].float().view(-1)
        elif target_key == "stop_progress" and "stop_progress" in shard:
            labels = shard["stop_progress"].float().view(-1)
        else:
            values = []
            for item in shard_metadata:
                if target_key not in item:
                    raise KeyError(f"Missing target_key={target_key} in metadata for {shard_path}")
                values.append(safe_float(item[target_key]))
            labels = torch.tensor(values, dtype=torch.float32)

        hidden_parts.append(hidden)
        label_parts.append(labels)
        metadata.extend(shard_metadata)

    hidden = torch.cat(hidden_parts, dim=0)
    labels = torch.cat(label_parts, dim=0).float().view(-1)
    if hidden.shape[0] != labels.shape[0]:
        raise ValueError(f"Hidden/label size mismatch: {hidden.shape[0]} vs {labels.shape[0]}")
    return hidden, labels, metadata


def pearson_corr(pred: torch.Tensor, label: torch.Tensor) -> float:
    pred = pred.float().view(-1)
    label = label.float().view(-1)
    if pred.numel() < 2:
        return 0.0
    pred_centered = pred - pred.mean()
    label_centered = label - label.mean()
    denom = pred_centered.norm() * label_centered.norm()
    if denom.item() == 0:
        return 0.0
    return float((pred_centered * label_centered).sum().div(denom).item())


def binary_auc(pred: torch.Tensor, label: torch.Tensor) -> float:
    pred = pred.detach().float().view(-1)
    label = label.detach().float().view(-1)
    pos = label >= 0.5
    neg = ~pos
    n_pos = int(pos.sum().item())
    n_neg = int(neg.sum().item())
    if n_pos == 0 or n_neg == 0:
        return 0.0

    order = torch.argsort(pred)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, pred.numel() + 1, dtype=torch.float32)
    pos_rank_sum = ranks[pos].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / max(n_pos * n_neg, 1)
    return float(auc.item())


def binary_pr_auc(pred: torch.Tensor, label: torch.Tensor) -> float:
    pred = pred.detach().float().view(-1)
    label = (label.detach().float().view(-1) >= 0.5).float()
    positives = label.sum().item()
    if positives == 0:
        return 0.0

    order = torch.argsort(pred, descending=True)
    sorted_label = label[order]
    tp = torch.cumsum(sorted_label, dim=0)
    fp = torch.cumsum(1.0 - sorted_label, dim=0)
    precision = tp / torch.clamp(tp + fp, min=1.0)
    recall = tp / positives
    precision = torch.cat([torch.tensor([1.0]), precision])
    recall = torch.cat([torch.tensor([0.0]), recall])
    return float(torch.trapz(precision, recall).item())


def split_indices(
    num_samples: int,
    metadata: List[Dict[str, Any]],
    val_ratio: float,
    seed: int,
    split_by: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    if split_by == "traj_id":
        groups: Dict[str, List[int]] = {}
        for idx, item in enumerate(metadata):
            group = str(item.get("traj_id", item.get("episode_id", idx)))
            groups.setdefault(group, []).append(idx)
        group_ids = list(groups)
        perm = torch.randperm(len(group_ids), generator=generator).tolist()
        val_group_count = max(1, int(math.ceil(len(group_ids) * val_ratio))) if len(group_ids) > 1 else 0
        val_groups = {group_ids[i] for i in perm[:val_group_count]}
        val_idx = [idx for group in val_groups for idx in groups[group]]
        train_idx = [idx for group in group_ids if group not in val_groups for idx in groups[group]]
        if not train_idx or not val_idx:
            perm_samples = torch.randperm(num_samples, generator=generator)
            val_size = max(1, int(math.ceil(num_samples * val_ratio))) if num_samples > 1 else 0
            return perm_samples[val_size:], perm_samples[:val_size]
        return torch.tensor(train_idx, dtype=torch.long), torch.tensor(val_idx, dtype=torch.long)

    perm = torch.randperm(num_samples, generator=generator)
    val_size = max(1, int(math.ceil(num_samples * val_ratio))) if num_samples > 1 else 0
    return perm[val_size:], perm[:val_size]


def loss_fn(pred: torch.Tensor, labels: torch.Tensor, loss_type: str) -> torch.Tensor:
    labels = labels.view(-1, 1)
    if loss_type == "bce":
        return F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), labels)
    return F.mse_loss(pred, labels)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
    loss_type: str,
) -> Dict[str, float]:
    model.eval()
    preds = []
    for start in range(0, hidden.shape[0], batch_size):
        batch_hidden = hidden[start : start + batch_size].to(device)
        preds.append(model(batch_hidden).detach().cpu().view(-1))
    pred = torch.cat(preds, dim=0)
    labels = labels.cpu().view(-1)
    metrics = {
        "loss": float(loss_fn(pred.view(-1, 1), labels, loss_type).item()),
        "mse": float(F.mse_loss(pred, labels).item()),
        "bce": float(F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), labels.clamp(0, 1)).item()),
        "pred_mean": float(pred.mean().item()),
        "label_mean": float(labels.mean().item()),
        "pred_min": float(pred.min().item()),
        "pred_max": float(pred.max().item()),
        "label_min": float(labels.min().item()),
        "label_max": float(labels.max().item()),
        "pearson_corr": pearson_corr(pred, labels),
    }
    binary_labels = labels >= 0.5
    pos_pred = pred[binary_labels]
    neg_pred = pred[~binary_labels]
    metrics.update(
        {
            "auc": binary_auc(pred, labels),
            "pr_auc": binary_pr_auc(pred, labels),
            "pred_pos_mean": float(pos_pred.mean().item()) if pos_pred.numel() else None,
            "pred_neg_mean": float(neg_pred.mean().item()) if neg_pred.numel() else None,
            "positive_count": int(binary_labels.sum().item()),
            "negative_count": int((~binary_labels).sum().item()),
        }
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--target_key", default="stop_progress")
    parser.add_argument("--loss_type", default="mse", choices=["mse", "bce"])
    parser.add_argument("--split_by", default="random", choices=["random", "traj_id"])
    parser.add_argument("--stop_head_hidden_dim", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cache_dir = Path(args.cache_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hidden, labels, metadata = load_cache(cache_dir, args.target_key)
    if args.loss_type == "bce":
        labels = (labels >= 0.5).float()
    else:
        labels = labels.clamp(0.0, 1.0)

    num_samples = hidden.shape[0]
    hidden_dim = hidden.shape[1]
    train_idx, val_idx = split_indices(num_samples, metadata, args.val_ratio, args.seed, args.split_by)
    if train_idx.numel() == 0:
        raise ValueError("Need at least one training sample after validation split.")
    if val_idx.numel() == 0:
        val_idx = train_idx

    train_dataset = TensorDataset(hidden[train_idx], labels[train_idx])
    train_loader = DataLoader(train_dataset, batch_size=min(args.batch_size, len(train_dataset)), shuffle=True)
    val_hidden = hidden[val_idx]
    val_labels = labels[val_idx]

    device = torch.device(args.device)
    model = StopProgressHead(hidden_dim, args.stop_head_hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    logs: List[Dict[str, Any]] = []
    best_metric = -float("inf") if args.loss_type == "bce" else float("inf")
    best_state = None
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_hidden, batch_labels in train_loader:
            batch_hidden = batch_hidden.to(device)
            batch_labels = batch_labels.to(device).view(-1, 1)
            pred = model(batch_hidden)
            loss = loss_fn(pred, batch_labels, args.loss_type)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        train_metrics = evaluate(model, hidden[train_idx], labels[train_idx], args.batch_size, device, args.loss_type)
        val_metrics = evaluate(model, val_hidden, val_labels, args.batch_size, device, args.loss_type)
        log_entry: Dict[str, Any] = {
            "epoch": epoch,
            "target_key": args.target_key,
            "loss_type": args.loss_type,
            "train_mse": train_metrics["mse"],
            "val_mse": val_metrics["mse"],
            "train_bce": train_metrics["bce"],
            "val_bce": val_metrics["bce"],
            "val_auc": val_metrics["auc"],
            "val_pr_auc": val_metrics["pr_auc"],
            "pred_mean": val_metrics["pred_mean"],
            "label_mean": val_metrics["label_mean"],
            "pred_pos_mean": val_metrics["pred_pos_mean"],
            "pred_neg_mean": val_metrics["pred_neg_mean"],
            "pred_min": val_metrics["pred_min"],
            "pred_max": val_metrics["pred_max"],
            "label_min": val_metrics["label_min"],
            "label_max": val_metrics["label_max"],
            "pearson_corr": val_metrics["pearson_corr"],
            "positive_count": val_metrics["positive_count"],
            "negative_count": val_metrics["negative_count"],
        }
        logs.append(log_entry)
        print(json.dumps(log_entry), flush=True)

        current_metric = val_metrics["auc"] if args.loss_type == "bce" else val_metrics["mse"]
        is_better = current_metric > best_metric if args.loss_type == "bce" else current_metric < best_metric
        if is_better:
            best_metric = current_metric
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.net.state_dict().items()}

    if best_state is None:
        best_state = model.net.state_dict()

    torch.save(
        {
            "state_dict": best_state,
            "hidden_dim": hidden_dim,
            "stop_head_hidden_dim": args.stop_head_hidden_dim,
            "num_samples": num_samples,
            "cache_dir": str(cache_dir),
            "target_key": args.target_key,
            "loss_type": args.loss_type,
            "split_by": args.split_by,
            "best_epoch": best_epoch,
            "best_metric": best_metric,
        },
        output_path,
    )
    with open(output_path.parent / "train_log.json", "w") as f:
        json.dump(
            {
                "num_samples": num_samples,
                "hidden_dim": hidden_dim,
                "target_key": args.target_key,
                "loss_type": args.loss_type,
                "split_by": args.split_by,
                "train_size": int(train_idx.numel()),
                "val_size": int(val_idx.numel()),
                "positive_count": int((labels >= 0.5).sum().item()),
                "negative_count": int((labels < 0.5).sum().item()),
                "best_epoch": best_epoch,
                "best_metric": best_metric,
                "metadata_preview": metadata[:5],
                "epochs": logs,
            },
            f,
            indent=2,
        )
    print(f"Saved stop head to {output_path}")


if __name__ == "__main__":
    main()
