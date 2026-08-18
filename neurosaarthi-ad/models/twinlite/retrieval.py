"""Nearest-neighbor trajectory retrieval for digital twin lite demos.

Supports multiple distance metrics:
- ``euclidean`` (default, backward-compatible with v0.1)
- ``mahalanobis`` (covariance-aware, recommended for multimodal features)
- ``importance_weighted`` (weight features by predictive importance)

The retriever also supports *cohort-balanced* retrieval so that returned
twins are drawn from diverse source cohorts rather than clustering on the
nearest single cohort.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


DistanceMetric = Literal["euclidean", "mahalanobis", "importance_weighted"]


@dataclass
class TwinLiteRetriever:
    """Retrieve the *k* most similar longitudinal prototypes from the
    training cohort and display their downstream trajectories.

    Parameters
    ----------
    feature_columns : list[str]
        Feature names used in the embedding.
    participant_col : str
        Column holding participant identifiers.
    metric : DistanceMetric
        Distance metric to use. ``"mahalanobis"`` accounts for feature
        correlations; ``"importance_weighted"`` applies user-supplied
        feature weights before Euclidean distance.
    feature_weights : dict[str, float] | None
        Per-feature importance weights for ``"importance_weighted"`` mode.
        Ignored for other metrics.  Missing keys default to 1.0.
    cohort_balanced : bool
        If ``True``, retrieve at most ``k // n_cohorts + 1`` neighbours
        from any single cohort to ensure cohort diversity.
    cohort_col : str
        Column holding cohort labels for balanced retrieval.
    """

    feature_columns: list[str]
    participant_col: str = "participant_id"
    metric: DistanceMetric = "euclidean"
    feature_weights: dict[str, float] | None = None
    cohort_balanced: bool = False
    cohort_col: str = "cohort"

    # Fitted state -------------------------------------------------------
    _reference: pd.DataFrame = field(default=None, repr=False, init=False)  # type: ignore[assignment]
    _matrix: np.ndarray = field(default=None, repr=False, init=False)  # type: ignore[assignment]
    _medians: pd.Series = field(default=None, repr=False, init=False)  # type: ignore[assignment]
    _means: pd.Series = field(default=None, repr=False, init=False)  # type: ignore[assignment]
    _stds: pd.Series = field(default=None, repr=False, init=False)  # type: ignore[assignment]
    _cov_inv: np.ndarray | None = field(default=None, repr=False, init=False)
    _weight_vector: np.ndarray | None = field(default=None, repr=False, init=False)

    # Backward compatibility aliases
    @property
    def reference_(self) -> pd.DataFrame:
        return self._reference

    @property
    def medians_(self) -> pd.Series:
        return self._medians

    @property
    def means_(self) -> pd.Series:
        return self._means

    @property
    def stds_(self) -> pd.Series:
        return self._stds

    @property
    def matrix_(self) -> np.ndarray:
        return self._matrix

    def fit(self, frame: pd.DataFrame) -> "TwinLiteRetriever":
        """Fit the retriever on the training *frame*."""
        self._reference = frame.reset_index(drop=True).copy()
        features = self._reference[self.feature_columns].astype(float)

        self._medians = features.median().fillna(0.0)
        imputed = features.fillna(self._medians)

        self._means = imputed.mean()
        self._stds = imputed.std(ddof=0).replace(0.0, 1.0).fillna(1.0)

        standardised = ((imputed - self._means) / self._stds).to_numpy()
        self._matrix = standardised

        # Mahalanobis: compute inverse covariance on standardised features
        if self.metric == "mahalanobis":
            cov = np.cov(standardised, rowvar=False)
            # Regularise for stability
            cov += np.eye(cov.shape[0]) * 1e-4
            try:
                self._cov_inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                self._cov_inv = np.linalg.pinv(cov)

        # Importance-weighted: build weight vector
        if self.metric == "importance_weighted" and self.feature_weights is not None:
            self._weight_vector = np.array(
                [self.feature_weights.get(col, 1.0) for col in self.feature_columns],
                dtype=float,
            )
            # Normalise so mean weight = 1
            self._weight_vector /= max(self._weight_vector.mean(), 1e-8)

        return self

    def __getstate__(self):
        """Sanitize participant data before pickling to prevent data exfiltration."""
        state = self.__dict__.copy()
        if "_reference" in state and state["_reference"] is not None:
            safe_ref = state["_reference"].copy()
            # Hash or obscure participant IDs to prevent leaking sensitive identifiers in exported artifacts
            import hashlib
            safe_ref[self.participant_col] = safe_ref[self.participant_col].astype(str).apply(
                lambda x: hashlib.sha256(x.encode()).hexdigest()[:12]
            )
            state["_reference"] = safe_ref
        return state

    def _compute_distances(self, vector: np.ndarray) -> np.ndarray:
        """Compute distances between *vector* and all reference rows."""
        diff = self._matrix - vector

        if self.metric == "mahalanobis" and self._cov_inv is not None:
            # d_M(x,y) = sqrt( (x-y)^T Σ^{-1} (x-y) )
            return np.sqrt(np.einsum("ij,jk,ik->i", diff, self._cov_inv, diff).clip(0.0))

        if self.metric == "importance_weighted" and self._weight_vector is not None:
            weighted_diff = diff * np.sqrt(self._weight_vector)
            return np.linalg.norm(weighted_diff, axis=1)

        # Default: Euclidean
        return np.linalg.norm(diff, axis=1)

    def query(
        self,
        row: pd.Series,
        k: int = 5,
        exclude_participant_id: str | None = None,
    ) -> pd.DataFrame:
        """Return the *k* nearest neighbours to *row*.

        Parameters
        ----------
        row : pd.Series
            Single participant feature vector.
        k : int
            Number of neighbours to return.
        exclude_participant_id : str | None
            Participant ID to exclude from results (typically the query
            participant itself).

        Returns
        -------
        pd.DataFrame
            Top-*k* neighbours with a ``distance`` column appended.
        """
        if self._matrix is None:
            raise RuntimeError("TwinLiteRetriever must be fitted before query")

        vector = row[self.feature_columns].astype(float).fillna(self._medians)
        vector = ((vector - self._means) / self._stds).to_numpy()
        distances = self._compute_distances(vector)

        result = self._reference.copy()
        result["distance"] = distances

        if exclude_participant_id is not None:
            result = result[result[self.participant_col] != exclude_participant_id]

        if self.cohort_balanced and self.cohort_col in result.columns:
            return self._balanced_select(result, k)

        return result.sort_values("distance").head(k).reset_index(drop=True)

    def _balanced_select(self, candidates: pd.DataFrame, k: int) -> pd.DataFrame:
        """Select top-*k* neighbours with at most proportional
        representation from each cohort."""
        candidates = candidates.sort_values("distance")
        cohorts = candidates[self.cohort_col].unique()
        max_per_cohort = max(1, (k // len(cohorts)) + 1)

        selected: list[pd.DataFrame] = []
        for _, group in candidates.groupby(self.cohort_col, sort=False):
            selected.append(group.head(max_per_cohort))
        combined = pd.concat(selected).sort_values("distance").head(k)
        return combined.reset_index(drop=True)
