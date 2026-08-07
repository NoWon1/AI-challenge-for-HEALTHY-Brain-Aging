"""Mixed-effects trajectory model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    from statsmodels.regression.mixed_linear_model import MixedLM
except ImportError:  # pragma: no cover
    sm = MixedLM = None

try:
    from sklearn.linear_model import Ridge
except ImportError:  # pragma: no cover
    Ridge = None


@dataclass
class MixedEffectsTrajectory:
    """A mixed-effects trajectory model for disease progression."""
    
    feature_columns: list[str]
    group_col: str = "participant_id"
    time_col: str = "horizon_years"

    def fit(self, frame: pd.DataFrame, target_col: str = "future_score") -> "MixedEffectsTrajectory":
        """Fit the mixed-effects model.
        
        Uses statsmodels MixedLM if available, fitting random intercepts and slopes.
        Otherwise falls back to a per-participant Ridge regression approximation.
        """
        self._target_col = target_col
        self._is_statsmodels = MixedLM is not None

        if self._is_statsmodels:
            endog = frame[target_col].astype(float)
            exog = sm.add_constant(frame[self.feature_columns].astype(float).fillna(0.0))
            groups = frame[self.group_col]
            exog_re = sm.add_constant(frame[[self.time_col]].astype(float).fillna(0.0))
            
            self.model_ = MixedLM(endog, exog, groups, exog_re=exog_re)
            self.result_ = self.model_.fit()
        else:
            self.models_ = {}
            if Ridge is None:
                raise ImportError("Neither statsmodels nor sklearn is available for fallback.")
            
            for name, group in frame.groupby(self.group_col):
                X = group[self.feature_columns].astype(float).fillna(0.0)
                y = group[target_col].astype(float)
                # simple Ridge fallback
                m = Ridge(alpha=1.0).fit(X, y)
                self.models_[name] = m
            
            # Global model for unknown participants
            X_all = frame[self.feature_columns].astype(float).fillna(0.0)
            y_all = frame[target_col].astype(float)
            self.global_model_ = Ridge(alpha=1.0).fit(X_all, y_all)
            
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        """Predict the future scores for the given data."""
        if self._is_statsmodels:
            exog = sm.add_constant(frame[self.feature_columns].astype(float).fillna(0.0))
            exog_re = sm.add_constant(frame[[self.time_col]].astype(float).fillna(0.0))
            
            preds = []
            for i, row in frame.iterrows():
                group = row[self.group_col]
                fixed_pred = self.result_.predict(exog.loc[[i]])[i]
                
                if group in self.result_.random_effects:
                    re = self.result_.random_effects[group]
                    z = exog_re.loc[i]
                    re_pred = (z * re).sum()
                    preds.append(fixed_pred + re_pred)
                else:
                    preds.append(fixed_pred)
            return pd.Series(preds, index=frame.index, name="predicted_" + self._target_col)
        else:
            preds = []
            for i, row in frame.iterrows():
                group = row[self.group_col]
                X = pd.DataFrame([row[self.feature_columns].astype(float).fillna(0.0)])
                if group in self.models_:
                    preds.append(self.models_[group].predict(X)[0])
                else:
                    preds.append(self.global_model_.predict(X)[0])
            return pd.Series(preds, index=frame.index, name="predicted_" + self._target_col)

    def calibrate_conformal(self, residuals: np.ndarray | pd.Series, alpha: float = 0.1) -> "MixedEffectsTrajectory":
        """Compute the conformal quantile from held-out residuals."""
        res = np.asarray(residuals)
        n = len(res)
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        q_level = min(q_level, 1.0)
        self.conformal_q_ = np.quantile(np.abs(res), q_level)
        return self

    def predict_with_intervals(self, frame: pd.DataFrame, alpha: float = 0.1) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Predict future scores and return (predictions, lower_bound, upper_bound)."""
        preds = self.predict(frame)
        if not hasattr(self, "conformal_q_"):
            raise RuntimeError("calibrate_conformal must be called before predict_with_intervals")
        
        lower = preds - self.conformal_q_
        upper = preds + self.conformal_q_
        return preds, lower, upper
        
    def person_effects(self, participant_id: Any) -> pd.Series | dict:
        """Return the estimated random effects for a person."""
        if self._is_statsmodels:
            if participant_id in self.result_.random_effects:
                return self.result_.random_effects[participant_id]
            return pd.Series({"Group": 0.0, self.time_col: 0.0})
        else:
            if participant_id in self.models_:
                return {"coef": self.models_[participant_id].coef_}
            return {}
