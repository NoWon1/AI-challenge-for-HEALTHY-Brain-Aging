"""Core metric wrappers."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
except ImportError:  # pragma: no cover
    average_precision_score = brier_score_loss = roc_auc_score = None


def _roc_auc_fallback(y_true: pd.Series, y_score: pd.Series) -> float:
    frame = pd.DataFrame({"y": y_true, "score": y_score}).sort_values("score")
    positives = int(frame["y"].sum())
    negatives = len(frame) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires at least one positive and one negative")
    ranks = np.arange(1, len(frame) + 1)
    positive_rank_sum = float(ranks[frame["y"].to_numpy() == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _average_precision_fallback(y_true: pd.Series, y_score: pd.Series) -> float:
    frame = pd.DataFrame({"y": y_true, "score": y_score}).sort_values("score", ascending=False)
    positives = int(frame["y"].sum())
    if positives == 0:
        raise ValueError("AUPRC requires at least one positive")
    tp = 0
    precisions = []
    for rank, y in enumerate(frame["y"], start=1):
        if y == 1:
            tp += 1
            precisions.append(tp / rank)
    return float(sum(precisions) / positives)


def binary_metrics(y_true: pd.Series, y_score: pd.Series) -> dict[str, float]:
    if roc_auc_score is None:
        return {
            "auroc": float(_roc_auc_fallback(y_true, y_score)),
            "auprc": float(_average_precision_fallback(y_true, y_score)),
            "brier": float(np.mean((y_true.to_numpy() - y_score.to_numpy()) ** 2)),
        }
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
    }
