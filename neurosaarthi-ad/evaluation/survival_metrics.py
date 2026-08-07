"""Survival-specific evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from sksurv.metrics import (
        concordance_index_censored,
        cumulative_dynamic_auc,
        integrated_brier_score as sksurv_ibs,
    )
    HAS_SKSURV = True
except ImportError:
    HAS_SKSURV = False

try:
    from lifelines.utils import concordance_index as ll_cindex
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False


def _numpy_cindex(event_times: np.ndarray, events: np.ndarray, risk_scores: np.ndarray) -> float:
    """Pure numpy fallback for Harrell's C-index."""
    n = len(event_times)
    concordant = 0.0
    total = 0.0
    for i in range(n):
        if not events[i]:
            continue
        for j in range(n):
            if event_times[i] < event_times[j]:
                total += 1.0
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1.0
                elif risk_scores[i] == risk_scores[j]:
                    concordant += 0.5
    if total == 0.0:
        return 0.5
    return float(concordant / total)


def concordance_index(
    event_times: np.ndarray | pd.Series,
    events: np.ndarray | pd.Series,
    risk_scores: np.ndarray | pd.Series,
) -> float:
    """
    Calculate Harrell's C-index.

    Args:
        event_times: Array of times to event or censoring.
        events: Boolean array indicating if event occurred.
        risk_scores: Array of predicted risk scores (higher means higher risk).

    Returns:
        C-index value between 0 and 1.
    """
    event_times = np.asarray(event_times)
    events = np.asarray(events, dtype=bool)
    risk_scores = np.asarray(risk_scores)

    if HAS_SKSURV:
        cindex, _, _, _, _ = concordance_index_censored(events, event_times, risk_scores)
        return float(cindex)
    if HAS_LIFELINES:
        # lifelines assumes higher score = longer survival by default, so we negate risk_scores
        return float(ll_cindex(event_times, -risk_scores, events))
    
    return _numpy_cindex(event_times, events, risk_scores)


def _to_sksurv_format(event_times: np.ndarray, events: np.ndarray) -> np.ndarray:
    return np.array(
        [(e, t) for e, t in zip(events, event_times)],
        dtype=[("event", bool), ("time", float)]
    )


def time_dependent_auc(
    event_times: np.ndarray | pd.Series,
    events: np.ndarray | pd.Series,
    risk_scores: np.ndarray | pd.Series,
    eval_times: np.ndarray | pd.Series,
) -> tuple[np.ndarray, float]:
    """
    Calculate Uno/IPCW time-dependent AUC at specified time points.

    Args:
        event_times: Array of times to event or censoring.
        events: Boolean array indicating if event occurred.
        risk_scores: Array of predicted risk scores.
        eval_times: Time points at which to evaluate AUC.

    Returns:
        Tuple of (AUC values at each eval time, mean AUC).
    """
    event_times = np.asarray(event_times)
    events = np.asarray(events, dtype=bool)
    risk_scores = np.asarray(risk_scores)
    eval_times = np.asarray(eval_times)

    if HAS_SKSURV:
        surv_y = _to_sksurv_format(event_times, events)
        # using the same data for train/test in IPCW (biased but common for simple eval)
        auc, mean_auc = cumulative_dynamic_auc(surv_y, surv_y, risk_scores, eval_times)
        return auc, float(mean_auc)
    
    # Fallback to simple binary AUC at each time point if sksurv is missing
    # Note: This is an approximation and ignores IPCW
    from sklearn.metrics import roc_auc_score
    aucs = []
    for t in eval_times:
        # Event happened before or at t
        y_true = (events & (event_times <= t)).astype(int)
        # Exclude censored before t
        valid = (event_times > t) | events
        if len(np.unique(y_true[valid])) > 1:
            try:
                aucs.append(roc_auc_score(y_true[valid], risk_scores[valid]))
            except ValueError:
                aucs.append(np.nan)
        else:
            aucs.append(np.nan)
    return np.array(aucs), float(np.nanmean(aucs))


def integrated_brier_score(
    event_times: np.ndarray | pd.Series,
    events: np.ndarray | pd.Series,
    survival_probs: np.ndarray,
    eval_times: np.ndarray | pd.Series,
) -> float:
    """
    Calculate Integrated Brier Score for survival probability calibration.

    Args:
        event_times: Array of times to event or censoring.
        events: Boolean array indicating if event occurred.
        survival_probs: 2D array of predicted survival probabilities (n_samples, n_eval_times).
        eval_times: Time points at which survival was evaluated.

    Returns:
        Integrated Brier Score.
    """
    event_times = np.asarray(event_times)
    events = np.asarray(events, dtype=bool)
    eval_times = np.asarray(eval_times)

    if HAS_SKSURV:
        surv_y = _to_sksurv_format(event_times, events)
        return float(sksurv_ibs(surv_y, surv_y, survival_probs, eval_times))

    # Fallback approximation
    from sklearn.metrics import brier_score_loss
    brier_scores = []
    for i, t in enumerate(eval_times):
        y_true = (events & (event_times <= t)).astype(int)
        valid = (event_times > t) | events
        if len(np.unique(y_true[valid])) > 1:
            # Survival prob to event prob
            event_probs = 1.0 - survival_probs[:, i]
            brier_scores.append(brier_score_loss(y_true[valid], event_probs[valid]))
    
    if not brier_scores:
        return np.nan
    return float(np.trapz(brier_scores, eval_times) / (eval_times[-1] - eval_times[0]))


def calibration_slope(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> tuple[float, float]:
    """
    Calculate logistic calibration slope and intercept.

    Args:
        y_true: True binary labels.
        y_pred: Predicted probabilities.

    Returns:
        Tuple of (slope, intercept).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Clip to avoid log(0)
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    logit_p = np.log(y_pred / (1 - y_pred))
    
    try:
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(penalty=None, solver='lbfgs')
        lr.fit(logit_p.reshape(-1, 1), y_true)
        return float(lr.coef_[0][0]), float(lr.intercept_[0])
    except ImportError:
        # Fallback using statsmodels if available, otherwise numpy polyfit (approximate)
        try:
            import statsmodels.api as sm
            X = sm.add_constant(logit_p)
            model = sm.Logit(y_true, X)
            result = model.fit(disp=0)
            return float(result.params[1]), float(result.params[0])
        except ImportError:
            # Linear approximation for slope/intercept
            slope, intercept = np.polyfit(logit_p, y_true, 1)
            return float(slope), float(intercept)
