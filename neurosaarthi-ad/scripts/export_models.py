"""Export fitted NeuroSaarthi-AD models to disk for Hugging Face upload."""

import contextlib
import joblib
import json
import os
from pathlib import Path

# ── Import your demo runtime which trains the models on synthetic data ──
from demo.runtime import build_demo_runtime
from demo.synthetic import generate_demo_cohort


@contextlib.contextmanager
def secure_umask(mask: int = 0o077):
    """Ensure newly created files are only accessible by the current user."""
    old_mask = os.umask(mask)
    try:
        yield
    finally:
        os.umask(old_mask)

with secure_umask(0o077):
    OUT_DIR = Path("hf_upload")
    OUT_DIR.mkdir(exist_ok=True)

# 1. Build the runtime (trains all sub-models on synthetic data)
bundle = generate_demo_cohort(seed=42, n_per_cohort=120)
runtime = build_demo_runtime(bundle=bundle, n_bootstrap=1)

with secure_umask(0o077):
    # 2. Save classification pipelines (LightGBM)
    for modality, ensemble in getattr(runtime, 'risk_models', {}).items():
        path = OUT_DIR / f"risk_{modality.replace('+', '').replace('/', '_').replace(' ', '_')}.joblib"
        joblib.dump(ensemble, path)

    # 3. Save survival models (RSF / CoxBoost)
    for model_name, model in getattr(runtime, 'survival_models', {}).items():
        path = OUT_DIR / f"survival_{model_name}.joblib"
        joblib.dump(model, path)

    # 4. Save the cognitive-trajectory regressor
    path = OUT_DIR / "progression_regressor.joblib"
    joblib.dump(runtime.trajectory_model, path)

    # 5. Save the twin-lite retrieval index
    path = OUT_DIR / "twinlite_retriever.joblib"
    joblib.dump(runtime.twin_retriever, path)

# 6. Save config / feature metadata
config = {
    "horizons": [1, 3, 5],
    "modality_features": {
        "Cognition + clinical": [
            "age", "education_years", "sex_binary", "rural_indicator",
            "cognitive_score", "memory_score", "executive_score",
        ],
        "MRI": ["hippocampal_volume_mm3", "wmh_burden_ml"],
        "Blood": ["hba1c_percent", "hs_crp_mg_l"],
        "OCT/OCTA": ["rnfl_um", "vessel_density_percent"],
        "Genomics": ["apoe_e4_count", "ancestry_pc1"],
    },
    "modality_weights": {
        "Cognition + clinical": 0.40,
        "MRI": 0.23,
        "Blood": 0.14,
        "OCT/OCTA": 0.11,
        "Genomics": 0.12,
    },
    "framework": "scikit-learn",
    "python_requires": ">=3.10",
}
path = OUT_DIR / "config.json"
with secure_umask(0o077):
    with open(path, "w") as f:
        json.dump(config, f, indent=2)

print(f" All artifacts saved to {OUT_DIR.resolve()}")
