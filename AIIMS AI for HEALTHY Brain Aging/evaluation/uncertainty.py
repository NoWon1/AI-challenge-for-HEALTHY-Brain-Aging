"""Uncertainty quantification utilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from .metrics import binary_metrics
except ImportError:  # pragma: no cover
    binary_metrics = None


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval calculator."""

    n_bootstrap: int = 200
    seed: int = 42
    ci_level: float = 0.95

    def compute(self, metric_fn: Callable[..., float], *data_args: np.ndarray) -> tuple[float, float, float]:
        """Compute point estimate and bootstrap confidence intervals.

        Args:
            metric_fn: Function that computes the metric.
            *data_args: Positional arguments to pass to metric_fn. They must all have the same length.

        Returns:
            Tuple of (point_estimate, lower_bound, upper_bound).
        """
        rng = np.random.default_rng(self.seed)
        point_estimate = metric_fn(*data_args)

        n_samples = len(data_args[0])
        bootstrapped_metrics = []
        for _ in range(self.n_bootstrap):
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            boot_args = [arg[indices] for arg in data_args]
            bootstrapped_metrics.append(metric_fn(*boot_args))

        lower_percentile = (1 - self.ci_level) / 2 * 100
        upper_percentile = (1 + self.ci_level) / 2 * 100

        lower = float(np.percentile(bootstrapped_metrics, lower_percentile))
        upper = float(np.percentile(bootstrapped_metrics, upper_percentile))

        return point_estimate, lower, upper

    def compute_metric_ci(
        self, y_true: pd.Series, y_score: pd.Series, metric_name: str = "auroc"
    ) -> dict[str, float]:
        """Compute CI for a specific metric.

        Args:
            y_true: True labels.
            y_score: Predicted scores.
            metric_name: Name of the metric (e.g. 'auroc', 'auprc', 'brier').

        Returns:
            Dictionary containing point, lower, and upper values.
        """

        def metric_fn(yt: np.ndarray, ys: np.ndarray) -> float:
            if binary_metrics is None:
                raise ImportError("binary_metrics could not be imported.")
            metrics = binary_metrics(pd.Series(yt), pd.Series(ys))
            return metrics[metric_name]

        point, lower, upper = self.compute(metric_fn, y_true.to_numpy(), y_score.to_numpy())
        return {"point": point, "lower": lower, "upper": upper}


@dataclass
class ConformalPredictor:
    """Conformal predictor for constructing valid prediction intervals."""

    def calibrate(self, residuals: np.ndarray | pd.Series, alpha: float = 0.1) -> float:
        """Compute the conformal quantile width.

        Args:
            residuals: Absolute residuals from a calibration set.
            alpha: Target error rate (default 0.1 for 90% coverage).

        Returns:
            The computed conformal quantile width.
        """
        if isinstance(residuals, pd.Series):
            residuals = residuals.to_numpy()
        residuals = np.abs(np.asarray(residuals, dtype=float))
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        if residuals.ndim != 1 or residuals.size == 0 or not np.isfinite(residuals).all():
            raise ValueError("residuals must be a non-empty finite one-dimensional array")
        n = len(residuals)
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        q_level = min(max(q_level, 0.0), 1.0)
        return float(np.quantile(residuals, q_level, method="higher"))

    def predict_interval(
        self, predictions: np.ndarray | pd.Series, width: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute prediction bands given base predictions and conformal width.

        Args:
            predictions: Base model predictions.
            width: Conformal quantile width from calibration.

        Returns:
            Tuple of (lower_bands, upper_bands).
        """
        if isinstance(predictions, pd.Series):
            predictions = predictions.to_numpy()
        return predictions - width, predictions + width

    def coverage(
        self, y_true: np.ndarray | pd.Series, lower: np.ndarray | pd.Series, upper: np.ndarray | pd.Series
    ) -> float:
        """Compute empirical coverage fraction.

        Args:
            y_true: True values.
            lower: Lower prediction bounds.
            upper: Upper prediction bounds.

        Returns:
            Fraction of true values falling within the prediction bounds.
        """
        if isinstance(y_true, pd.Series):
            y_true = y_true.to_numpy()
        if isinstance(lower, pd.Series):
            lower = lower.to_numpy()
        if isinstance(upper, pd.Series):
            upper = upper.to_numpy()

        covered = (y_true >= lower) & (y_true <= upper)
        return float(np.mean(covered))


@dataclass
class ReliabilityDiagram:
    """Utility for computing reliability diagrams and calibration metrics."""

    def compute(
        self, y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series, n_bins: int = 10
    ) -> pd.DataFrame:
        """Compute reliability diagram metrics.

        Args:
            y_true: True binary labels.
            y_pred: Predicted probabilities.
            n_bins: Number of bins.

        Returns:
            DataFrame containing bin midpoints, observed rates, predicted rates, counts, ECE, and MCE.
        """
        if isinstance(y_true, pd.Series):
            y_true = y_true.to_numpy()
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.to_numpy()

        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_pred, bins[1:-1])

        results = []
        ece = 0.0
        mce = 0.0
        total_samples = len(y_true)

        for i in range(n_bins):
            in_bin = bin_indices == i
            count = np.sum(in_bin)
            if count > 0:
                obs_rate = float(np.mean(y_true[in_bin]))
                pred_rate = float(np.mean(y_pred[in_bin]))
                diff = abs(obs_rate - pred_rate)
                ece += diff * (count / total_samples)
                mce = max(mce, diff)
            else:
                obs_rate = np.nan
                pred_rate = np.nan

            midpoint = (bins[i] + bins[i + 1]) / 2
            results.append(
                {
                    "bin_midpoint": midpoint,
                    "observed_rate": obs_rate,
                    "predicted_rate": pred_rate,
                    "count": count,
                }
            )

        df = pd.DataFrame(results)
        df["ece"] = ece
        df["mce"] = mce
        return df


def ood_score(
    query_features: np.ndarray | pd.DataFrame, training_features: np.ndarray | pd.DataFrame
) -> np.ndarray:
    """Compute dimension-normalised Mahalanobis out-of-distribution scores.

    Args:
        query_features: Features for the query samples.
        training_features: Features from the training manifold.

    Returns:
        Array of Mahalanobis distances for each query sample.
    """
    if isinstance(query_features, pd.DataFrame):
        query_features = query_features.to_numpy()
    if isinstance(training_features, pd.DataFrame):
        training_features = training_features.to_numpy()

    query_features = np.asarray(query_features, dtype=float)
    training_features = np.asarray(training_features, dtype=float)
    if query_features.ndim != 2 or training_features.ndim != 2:
        raise ValueError("query_features and training_features must be two-dimensional")
    if query_features.shape[1] != training_features.shape[1] or training_features.shape[0] < 2:
        raise ValueError("Feature dimensions must match and training_features needs at least two rows")
    if not np.isfinite(query_features).all() or not np.isfinite(training_features).all():
        raise ValueError("OOD features must be finite")

    mean = np.mean(training_features, axis=0)
    cov = np.cov(training_features, rowvar=False)

    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)

    diff = query_features - mean
    # Dimension normalisation makes scores comparable across differently sized
    # modality blocks. Any alert threshold must still be calibrated on held-out
    # in-distribution data; 1.0 is not a universal clinical cutoff.
    distances = np.sqrt(np.sum(np.dot(diff, inv_cov) * diff, axis=1)) / np.sqrt(training_features.shape[1])
    return distances
