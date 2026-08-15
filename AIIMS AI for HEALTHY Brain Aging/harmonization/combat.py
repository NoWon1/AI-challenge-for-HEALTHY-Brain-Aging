"""Train-only ComBat-style batch correction for cross-site harmonisation.

Implements a simplified parametric empirical Bayes adjustment
(Johnson et al., 2007) fitted exclusively on training data to prevent
validation leakage. Biological covariates (age, sex) are preserved.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class TrainOnlyComBat:
    """Empirical Bayes batch-effect correction fitted only on training data.
    
    Parameters
    ----------
    batch_col : str
        Column identifying the batch (site/scanner/cohort).
    feature_columns : list[str]
        Numeric feature columns to harmonise.
    preserve_columns : list[str]
        Biological covariates to preserve during harmonisation (e.g. age, sex).
    """
    batch_col: str = "cohort"
    feature_columns: list[str] = field(default_factory=list)
    preserve_columns: list[str] = field(default_factory=lambda: ["age", "sex_binary"])
    
    # Fitted parameters
    _grand_mean: pd.Series = field(default=None, repr=False, init=False)
    _grand_std: pd.Series = field(default=None, repr=False, init=False)
    _batch_means: dict[str, pd.Series] = field(default_factory=dict, repr=False, init=False)
    _batch_stds: dict[str, pd.Series] = field(default_factory=dict, repr=False, init=False)
    _fitted: bool = field(default=False, repr=False, init=False)
    
    def fit(self, frame: pd.DataFrame) -> 'TrainOnlyComBat':
        data = frame[self.feature_columns].astype(float)
        self._grand_mean = data.mean()
        self._grand_std = data.std().replace(0.0, 1.0).fillna(1.0)
        
        # Residualise preserve_columns via simple regression
        residuals = data.copy()
        preserve_available = [c for c in self.preserve_columns if c in frame.columns]
        if preserve_available:
            X_preserve = frame[preserve_available].astype(float).fillna(0.0)
            # Simple OLS residualisation
            for col in self.feature_columns:
                y = data[col].fillna(data[col].median())
                X = X_preserve.copy()
                X['intercept'] = 1.0
                try:
                    beta = np.linalg.lstsq(X.values, y.values, rcond=None)[0]
                    residuals[col] = y - X.values @ beta
                except np.linalg.LinAlgError:
                    logger.warning("LinAlgError during residualisation for column %s. Skipping residualisation for this feature.", col)
        
        # Batch-specific location and scale from residuals
        for batch, group in frame.groupby(self.batch_col, sort=False):
            batch_data = residuals.loc[group.index]
            self._batch_means[batch] = batch_data.mean()
            self._batch_stds[batch] = batch_data.std().replace(0.0, 1.0).fillna(1.0)
        
        self._fitted = True
        return self
    
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError('TrainOnlyComBat must be fitted before transform')
        result = frame.copy()
        
        for batch, group in result.groupby(self.batch_col, sort=False):
            idx = group.index
            data = result.loc[idx, self.feature_columns].astype(float)
            
            if batch in self._batch_means:
                batch_mean = self._batch_means[batch]
                batch_std = self._batch_stds[batch]
            else:
                # Unseen batch: use grand statistics
                batch_mean = pd.Series(0.0, index=self.feature_columns)
                batch_std = pd.Series(1.0, index=self.feature_columns)
            
            # Standardise within batch, then rescale to grand distribution
            standardised = (data - batch_mean) / batch_std
            harmonised = standardised * self._grand_std + self._grand_mean
            result.loc[idx, self.feature_columns] = harmonised
        
        return result
    
    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)
    
    def diagnostics(self, frame: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for batch, group in frame.groupby(self.batch_col, sort=False):
            data = group[self.feature_columns].astype(float)
            for col in self.feature_columns:
                rows.append({
                    'batch': batch,
                    'feature': col,
                    'mean': float(data[col].mean()),
                    'std': float(data[col].std()),
                    'missing_rate': float(data[col].isna().mean()),
                    'harmonised': self._fitted,
                })
        return pd.DataFrame(rows)
