from __future__ import annotations

import math

import torch


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    return (predictions == labels).float().mean().item()


def binary_auc_from_scores(scores: torch.Tensor, labels: torch.Tensor) -> float:
    labels = labels.to(torch.long)
    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_count = int(pos_mask.sum().item())
    neg_count = int(neg_mask.sum().item())
    if pos_count == 0 or neg_count == 0:
        return float("nan")

    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, device=scores.device, dtype=torch.float32)
    pos_ranks = ranks[pos_mask].sum()
    auc = (pos_ranks - pos_count * (pos_count + 1) / 2.0) / (pos_count * neg_count)
    return float(auc.item())


def multiclass_auc_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    num_classes = logits.size(1)
    if num_classes == 2:
        probs = torch.softmax(logits, dim=-1)[:, 1]
        return binary_auc_from_scores(probs, labels)

    probs = torch.softmax(logits, dim=-1)
    auc_values: list[float] = []
    for class_idx in range(num_classes):
        class_scores = probs[:, class_idx]
        class_labels = (labels == class_idx).to(torch.long)
        auc = binary_auc_from_scores(class_scores, class_labels)
        if not math.isnan(auc):
            auc_values.append(auc)

    if not auc_values:
        return float("nan")
    return sum(auc_values) / len(auc_values)
