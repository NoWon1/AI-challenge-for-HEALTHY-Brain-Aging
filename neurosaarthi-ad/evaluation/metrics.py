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
    y_arr = frame["y"].to_numpy()
    positives = int(y_arr.sum())
    if positives == 0:
        raise ValueError("AUPRC requires at least one positive")

    # ⚡ Bolt: Vectorized AUPRC computation avoiding slow O(N) explicit loop
    ranks = np.arange(1, len(y_arr) + 1)
    tp_cumsum = np.cumsum(y_arr)
    precisions = tp_cumsum[y_arr == 1] / ranks[y_arr == 1]
    return float(precisions.sum() / positives)


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
