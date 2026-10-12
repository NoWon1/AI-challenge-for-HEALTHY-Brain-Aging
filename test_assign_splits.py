import numpy as np
import pandas as pd
import time
import sys
sys.path.insert(0, "./neurosaarthi-ad")

from demo.runtime import _assign_splits

# Mock PUBLIC_COHORTS
import demo.runtime as runtime
runtime.PUBLIC_COHORTS = ["ADNI", "OASIS", "NACC", "AIBL"]

N = 100000
cohorts = np.random.choice(["ADNI", "OASIS", "TLSA", "UNKNOWN"], N)
participant_ids = np.arange(N)

baseline = pd.DataFrame({"cohort": cohorts, "participant_id": participant_ids})

start = time.time()
res1 = _assign_splits(baseline, 42)
end = time.time()
print("Original:", end - start)

def _assign_splits_fast(baseline: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 101)

    dfs = []
    for cohort, group in baseline.groupby("cohort", sort=False):
        ids = group["participant_id"].to_numpy(copy=True)
        rng.shuffle(ids)

        n_ids = len(ids)
        roles = np.empty(n_ids, dtype=object)

        if cohort in runtime.PUBLIC_COHORTS:
            validation_count = max(1, int(round(0.20 * n_ids)))
            roles[:validation_count] = "public_validation"
            roles[validation_count:] = "global_train"
        elif cohort == "TLSA":
            validation_count = max(1, int(round(0.30 * n_ids)))
            roles[:validation_count] = "india_validation"
            roles[validation_count:] = "tlsa_adaptation"
        else:
            roles[:] = "external_validation"

        dfs.append(pd.DataFrame({
            "participant_id": ids,
            "cohort": cohort,
            "role": roles
        }))

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["participant_id", "cohort", "role"])

start = time.time()
res2 = _assign_splits_fast(baseline, 42)
end = time.time()
print("Fast:", end - start)

print("Equality:", res1.sort_values('participant_id').reset_index(drop=True).equals(res2.sort_values('participant_id').reset_index(drop=True)))
