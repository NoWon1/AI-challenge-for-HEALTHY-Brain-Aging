
import numpy as np
import pandas as pd


def validate_feature_matrix(X: pd.DataFrame | np.ndarray, allow_nan: bool = False) -> pd.DataFrame | np.ndarray:
    """
    Validates that the input feature matrix contains only finite numeric values (CWE-20 / CWE-682).
    Raises ValueError without leaking raw record contents.
    """
    if X is None:
        raise ValueError("Input feature matrix cannot be None.")

    # Convert to float to safely check inf/nan, this also handles object dtype with None
    if isinstance(X, pd.DataFrame):
        try:
            arr = X.to_numpy(dtype=float, na_value=np.nan)
        except (ValueError, TypeError):
             raise TypeError("Feature matrix must contain numeric types.")
    else:
        try:
            arr = np.asarray(X, dtype=float)
        except (ValueError, TypeError):
             raise TypeError("Feature matrix must contain numeric types.")

    if np.isinf(arr).any():
        raise ValueError("Input feature matrix contains non-finite values (NaN or Inf).")

    if not allow_nan and np.isnan(arr).any():
        raise ValueError("Input feature matrix contains NaN values but allow_nan=False.")

    return X
