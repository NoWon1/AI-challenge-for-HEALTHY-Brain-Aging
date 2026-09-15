import pandas as pd
import numpy as np
import time

def slow_build_features(raw, cohort_name):
    feature_map = {
        'MMSE': ('cognition', 'cognitive_score', 'points'),
    }

    rows = []
    for _, record in raw.iterrows():
        for source_col, (modality, feature_name, unit) in feature_map.items():
            if source_col in record and pd.notna(record[source_col]):
                rows.append({
                    'feature_row_id': f"{record['visit_id']}-{feature_name}",
                    'participant_id': record['participant_id'],
                    'visit_id': record['visit_id'],
                    'cohort': cohort_name,
                    'modality': modality,
                    'feature_name': feature_name,
                    'value': float(record[source_col]),
                    'unit': unit,
                    'source_variable': source_col,
                    'qc_flag': 'pass',
                    'derived': False,
                })
    return pd.DataFrame(rows).sort_values(['participant_id', 'visit_id']).reset_index(drop=True) if rows else pd.DataFrame()

def fast_build_features(raw, cohort_name):
    feature_map = {
        'MMSE': ('cognition', 'cognitive_score', 'points'),
    }
    dfs = []
    for source_col, (modality, feature_name, unit) in feature_map.items():
        if source_col in raw.columns:
            valid_rows = raw[raw[source_col].notna()].copy()
            if not valid_rows.empty:
                df = pd.DataFrame({
                    'feature_row_id': valid_rows['visit_id'] + '-' + feature_name,
                    'participant_id': valid_rows['participant_id'],
                    'visit_id': valid_rows['visit_id'],
                    'cohort': cohort_name,
                    'modality': modality,
                    'feature_name': feature_name,
                    'value': valid_rows[source_col].astype(float),
                    'unit': unit,
                    'source_variable': source_col,
                    'qc_flag': 'pass',
                    'derived': False,
                    '_row_idx': valid_rows.index,
                    '_feat_idx': list(feature_map.keys()).index(source_col)
                })
                dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    result = pd.concat(dfs, ignore_index=True)
    # Sort by original row index then feature index to match iterrows order exactly
    result = result.sort_values(['_row_idx', '_feat_idx'])
    result = result.drop(columns=['_row_idx', '_feat_idx'])

    return result.sort_values(['participant_id', 'visit_id']).reset_index(drop=True)

# Generate dummy data
n = 10000
raw = pd.DataFrame({
    'participant_id': [f'P-{i}' for i in range(n)],
    'visit_id': [f'P-{i}-V0' for i in range(n)],
    'MMSE': np.random.rand(n) * 30
})
raw.loc[np.random.choice(n, int(n * 0.1), replace=False), 'MMSE'] = np.nan # Add some NaNs

print(f"Testing with {n} rows")

start = time.time()
slow_res = slow_build_features(raw, "TEST")
slow_time = time.time() - start
print(f"Slow: {slow_time:.4f}s")

start = time.time()
fast_res = fast_build_features(raw, "TEST")
fast_time = time.time() - start
print(f"Fast: {fast_time:.4f}s")

print(f"Speedup: {slow_time / fast_time:.1f}x")
print(f"Results match: {slow_res.equals(fast_res)}")
