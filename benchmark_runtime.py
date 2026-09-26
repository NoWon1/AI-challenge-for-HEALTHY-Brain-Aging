import time
import numpy as np
import pandas as pd
from typing import Iterable
import sys

# Mock for dependencies to run benchmark in isolation
class MockModel:
    def predict_distribution(self, frame):
        # returns (n_bootstrap, n_samples, n_horizons)
        return np.random.rand(50, len(frame), 3)

HORIZONS = [1, 3, 5]
MODALITY_WEIGHTS = {"clinical": 0.5, "mri": 0.3, "cog": 0.2}

def weighted_score_fusion(scores: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    weight_series = pd.Series(weights, dtype=float)
    aligned = scores[list(weights)].apply(pd.to_numeric, errors="coerce")
    available = aligned.notna()
    numerator = aligned.fillna(0.0).mul(weight_series, axis="columns").sum(axis=1)
    denominator = available.mul(weight_series, axis="columns").sum(axis=1)
    fused = numerator.div(denominator.where(denominator > 0))
    return pd.Series(fused, index=scores.index, name="fused_score")

class RuntimeMock:
    def __init__(self):
        self.n_bootstrap = 50
        self.risk_models = {
            "clinical": MockModel(),
            "mri": MockModel(),
            "cog": MockModel(),
        }

    def _available_mask(self, frame, modality):
        return pd.Series(np.random.choice([True, False], size=len(frame)), index=frame.index)

    def _risk_distribution_original(self, frame: pd.DataFrame, disabled_modalities: Iterable[str] = ()):
        disabled = set(disabled_modalities)
        modality_distributions = {
            modality: model.predict_distribution(frame)
            for modality, model in self.risk_models.items()
        }
        fused = np.full((self.n_bootstrap, len(frame), len(HORIZONS)), np.nan, dtype=float)
        for bootstrap_index in range(self.n_bootstrap):
            for horizon_index, _ in enumerate(HORIZONS):
                score_frame = pd.DataFrame(index=frame.index)
                for modality, distribution in modality_distributions.items():
                    values = distribution[bootstrap_index, :, horizon_index].copy()
                    if modality in disabled:
                        values[:] = np.nan
                    else:
                        values[~self._available_mask(frame, modality).to_numpy()] = np.nan
                    score_frame[modality] = values
                fused[bootstrap_index, :, horizon_index] = weighted_score_fusion(
                    score_frame, MODALITY_WEIGHTS
                ).to_numpy()
        return fused

    def _risk_distribution_optimized(self, frame: pd.DataFrame, disabled_modalities: Iterable[str] = ()):
        disabled = set(disabled_modalities)
        modality_distributions = {
            modality: model.predict_distribution(frame)
            for modality, model in self.risk_models.items()
        }

        fused = np.full((self.n_bootstrap, len(frame), len(HORIZONS)), np.nan, dtype=float)
        for bootstrap_index in range(self.n_bootstrap):
            for horizon_index, _ in enumerate(HORIZONS):
                score_dict = {}
                for modality, distribution in modality_distributions.items():
                    values = distribution[bootstrap_index, :, horizon_index].copy()
                    if modality in disabled:
                        values[:] = np.nan
                    else:
                        values[~self._available_mask(frame, modality).to_numpy()] = np.nan
                    score_dict[modality] = values
                score_frame = pd.DataFrame(score_dict, index=frame.index)
                fused[bootstrap_index, :, horizon_index] = weighted_score_fusion(
                    score_frame, MODALITY_WEIGHTS
                ).to_numpy()
        return fused

    def _risk_distribution_vectorized(self, frame: pd.DataFrame, disabled_modalities: Iterable[str] = ()):
        disabled = set(disabled_modalities)
        modality_distributions = {
            modality: model.predict_distribution(frame)
            for modality, model in self.risk_models.items()
        }

        M = len(modality_distributions)
        B, N, H = self.n_bootstrap, len(frame), len(HORIZONS)
        distributions_array = np.empty((M, B, N, H), dtype=float)
        weights_array = np.empty(M, dtype=float)

        for i, (modality, distribution) in enumerate(modality_distributions.items()):
            weights_array[i] = MODALITY_WEIGHTS[modality]
            vals = distribution.copy()
            if modality in disabled:
                vals[:] = np.nan
            else:
                mask = self._available_mask(frame, modality).to_numpy()
                vals[:, ~mask, :] = np.nan
            distributions_array[i] = vals

        available = ~np.isnan(distributions_array)
        weighted_vals = np.where(available, distributions_array * weights_array[:, None, None, None], 0.0)
        numerator = np.sum(weighted_vals, axis=0)
        denominator = np.sum(available * weights_array[:, None, None, None], axis=0)

        fused = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
        return fused

# Setup
frame = pd.DataFrame({"dummy": np.arange(100)})
runtime = RuntimeMock()

# Warmup
runtime._risk_distribution_original(frame)
runtime._risk_distribution_optimized(frame)
runtime._risk_distribution_vectorized(frame)

# Benchmark original
start = time.perf_counter()
runtime._risk_distribution_original(frame)
original_time = time.perf_counter() - start

# Benchmark optimized (dict of arrays)
start = time.perf_counter()
runtime._risk_distribution_optimized(frame)
optimized_time = time.perf_counter() - start

# Benchmark vectorized (numpy)
start = time.perf_counter()
runtime._risk_distribution_vectorized(frame)
vectorized_time = time.perf_counter() - start

print(f"Original: {original_time:.4f}s")
print(f"Optimized (dict of arrays): {optimized_time:.4f}s")
print(f"Vectorized (numpy): {vectorized_time:.4f}s")
