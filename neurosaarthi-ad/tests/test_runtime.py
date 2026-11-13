import pandas as pd
import numpy as np
import pytest
from demo.runtime import _LegacyDiscreteTimeRiskEnsemble, GBMDiscreteTimeRiskEnsemble

def test_interval_frame_vectorized_equiv():
    # Generate dummy data
    N = 100
    np.random.seed(42)
    df = pd.DataFrame({
        'event_time_days': np.random.uniform(0, 2000, N),
        'event': np.random.randint(0, 2, N),
        'feature1': np.random.rand(N),
    })

    # Legacy implementation
    def legacy_interval_frame(frame: pd.DataFrame) -> pd.DataFrame:
        interval_rows: list[pd.DataFrame] = []
        for year in range(1, 6):
            interval_start = (year - 1) * 365.25
            interval_end = year * 365.25
            eligible = frame[frame["event_time_days"] > interval_start].copy()
            if eligible.empty:
                continue
            eligible["interval_year"] = float(year)
            eligible["interval_event"] = (
                (eligible["event"] == 1) & (eligible["event_time_days"] <= interval_end)
            ).astype(int)
            interval_rows.append(eligible)
        return pd.concat(interval_rows, ignore_index=True)

    # Use the class method
    ensemble = _LegacyDiscreteTimeRiskEnsemble(["feature1"])
    res_vectorized = ensemble._interval_frame(df)

    res_legacy = legacy_interval_frame(df)

    # Check shape
    assert res_legacy.shape == res_vectorized.shape

    # The vectorized approach order might be different depending on flat indexing,
    # so we sort by index/feature to compare
    res_legacy = res_legacy.sort_values(by=["interval_year", "feature1"]).reset_index(drop=True)
    res_vectorized = res_vectorized.sort_values(by=["interval_year", "feature1"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(res_legacy, res_vectorized, check_dtype=True)
